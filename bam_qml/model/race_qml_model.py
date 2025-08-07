from flax import linen as nn 
import e3nn_jax as e3nn

import jax.numpy as jnp
from typing import Callable
import jax 
import pennylane as qml
from bam_qml.model.qml_circuit import (circuit_basic,
                                       circuit_cx,
                                       circuit_cz,
                                       circuit_qft_cx,
                                       circuit_qft_cz,
                                       circuit_cx_noise)

circuit_map = {
    'basic': circuit_basic,
    'cx' : circuit_cx,
    'cz' : circuit_cz,
    'qft_cx' : circuit_qft_cx,
    'qft_cz' : circuit_qft_cz,
    'cx_noise' : circuit_cx_noise
}


def default_radial_basis (r, n: int):
    """Default radial basis function.
    r: input distance,
    n: number of basis functions
    r_max: max(r)
    Polynomial envelop with p = 2
    e3nn.poly_envelope (p-1, 2) (r) = e3nn.radial.u (p, r)
    """
    return e3nn.bessel(r, n) * e3nn.poly_envelope(1, 2)(r)[:, None]
    


class LayerFlax (nn.Module):
    hidden_irreps: e3nn.Irreps
    output_irreps: e3nn.Irreps 
    num_species: int 
    node_species: jax.Array
    x_node_feats: e3nn.IrrepsArray
    radial_embedding: e3nn.IrrepsArray
    edges_attr: e3nn.IrrepsArray
    senders: jax.Array # [n_edges]
    receivers: jax.Array #[n_edges]
    avg_num_neighbors: float
    # Radial Embedding
    mlp_n_hidden: int = 64
    mlp_n_layers: int = 2
    mlp_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.silu
    # QML
    qml_circuit : str="cx"
    qml_num_layers: int = 3
    qml_qubits: int = 8

    @nn.compact
    def __call__(self,
                 node_feats: e3nn.IrrepsArray, # [n_nodes, irreps]
                 ):
        
        #num_edges = Rab.shape[0]

        # Skip connection 
        
        skip = e3nn.flax.Linear(self.hidden_irreps,
                                num_indexed_weights=self.num_species,
                                name="skip_tp",
                                force_irreps_out=True)(
                                       self.node_species,
                                       node_feats
                                )
        
        #### InteractionBlock ####
        # 
        node_feats = e3nn.flax.Linear (self.hidden_irreps,
                                       name='linear_up') (node_feats)
        # [[[MessagePassingConvolution]]]
        # Regroup the target irreps to make sure that gate activation
        # has the same irreps as the target
        
        messages = node_feats[self.senders]
        # Angular part
        
        hidden_irreps = e3nn.Irreps(self.hidden_irreps).regroup()
        messages = e3nn.concatenate (
            [
                messages.filter (hidden_irreps+"0e"),
                e3nn.tensor_product(
                    messages,
                    self.edges_attr,
                    filter_ir_out=hidden_irreps+"0e",
                ),
            ]
        ).regroup () # [n_edges, irreps]

        # Radial Part 
        mix = e3nn.flax.MultiLayerPerceptron (
            self.mlp_n_layers *(self.mlp_n_hidden,) + (messages.irreps.num_irreps,),
            self.mlp_activation,
            output_activation=False,
        ) (self.radial_embedding)

        
        messages = messages * mix 
        # MessagePassing of convoluted information
        zeros = e3nn.IrrepsArray.zeros (
            messages.irreps, node_feats.shape[:1], messages.dtype
        )
        node_feats = zeros.at[self.receivers].add (messages)
        node_feats = node_feats / jnp.sqrt (self.avg_num_neighbors)
        ##[[ Done MessagePassingConvolution]]
        node_feats = e3nn.flax.Linear(self.hidden_irreps, name='linear_down') (node_feats)
        ### (DONE InteractionBlock)

        # MACE rescaling
        node_feats /= jnp.sqrt (self.avg_num_neighbors)        
        # 
        node_feats = e3nn.tensor_product (self.x_node_feats, node_feats, 
                                       filter_ir_out=hidden_irreps)
        node_feats = e3nn.flax.Linear(self.hidden_irreps,
                                      force_irreps_out=True) (node_feats)
        node_feats = node_feats + skip 

        weight_qml = self.param (f'weight_qml',
                        nn.initializers.lecun_normal(),
                        (self.qml_num_layers, self.qml_qubits))
        features = e3nn.flax.Linear (f"{self.qml_qubits}x0e") (node_feats)
        features = circuit_map[self.qml_circuit] (self.qml_qubits, features.array, weight_qml)
        features = e3nn.IrrepsArray(f"{self.qml_qubits}x0e", jnp.array(features).T)
        
        node_outputs = e3nn.flax.Linear(self.output_irreps, biases=True) (features)
        
        return node_outputs, node_feats
    

    

class GraphNN (nn.Module):
    cutoff: float
    avg_num_neighbors: float
    num_species: int 
    max_ell: int = 3
    num_basis_func: int = 8
    # small GPU memory
    hidden_irreps: e3nn.Irreps = e3nn.Irreps ("32x0e+8x1o+4x2e")
    nlayers: int = 4
    features_dim : int = 128
    output_irreps: e3nn.Irreps = e3nn.Irreps("1x0e")
    qml_circuit: str="cx"

    @nn.compact
    def __call__ (self, Rij, data_graph):
        # Rij : (num_edges, 3)
        assert Rij.ndim == 2 and Rij.shape[1] == 3
        # iatoms ==> senders
        # jatoms ==> receivers
        iatoms = data_graph.senders
        jatoms = data_graph.receivers
        
        Rij = Rij/self.cutoff
        Rij = e3nn.IrrepsArray ("1o", Rij)
        
        # (Embedding)
        # num_embeddings = (the number of atomic species)
        species = data_graph.nodes['species']
        node_feats = nn.Embed (num_embeddings=self.num_species, 
                              features=self.features_dim) (species)
        node_feats = e3nn.IrrepsArray(f"{node_feats.shape[-1]}x0e", node_feats)
        
        lengths = e3nn.norm(Rij).array[:,0]
        #
        # Note that there are length == 0 owing to jraph.pad_with_graphs.
        #
        radial_embedding = jnp.where (
            (lengths == 0.0) [:,None], 0.0,
            default_radial_basis(lengths, self.num_basis_func) )
                                            
        interaction_irreps = e3nn.Irreps.spherical_harmonics(self.max_ell)

        # Interactions
        outputs = []
        node_logvar = []

        x_node_feats = e3nn.flax.Linear (node_feats.irreps) (node_feats)
        
        edges_attr = e3nn.spherical_harmonics(interaction_irreps, #range(1, self.max_ell+1),
                                            Rij,
                                            normalize=True,
                                            normalization="component"
                    )
        for _ in range(self.nlayers):
            
            layer = LayerFlax(
                hidden_irreps = self.hidden_irreps,
                output_irreps = self.output_irreps,
                num_species = self.num_species,
                node_species = species,
                x_node_feats = x_node_feats,
                radial_embedding = radial_embedding,
                edges_attr = edges_attr,
                senders = iatoms,
                receivers = jatoms,
                avg_num_neighbors = self.avg_num_neighbors,
                qml_circuit=self.qml_circuit
            )
            node_outputs, node_feats = layer(node_feats)

            outputs += [node_outputs.array[:,0]]
            if self.output_irreps == e3nn.Irreps ('2x0e'):
                node_logvar += node_outputs.array[:,1] 
        
        # 
        node_energy = jnp.stack(outputs, axis=1) # [n_nodes, num_interactions]
        node_energy = node_energy.sum(axis=-1) #(n_nodes)
        # Global Pooling
        # Average the features (energy prediction) over the nodes of each graph
        graph_energy = e3nn.scatter_sum(node_energy, nel=data_graph.n_node)
        
        if self.output_irreps == e3nn.Irreps ('2x0e'):
            node_logvar = jnp.stack(node_logvar, axis=1) # [n_nodes, num_interactions]
            node_logvar = node_logvar.mean(axis=-1) #(n_nodes)
        else: 
            node_logvar = jnp.zeros (node_feats.shape[0])
        node_ener_var = jnp.exp(node_logvar) # (n_nodes)
        graph_ener_var = e3nn.scatter_sum (node_ener_var, nel=data_graph.n_node)/data_graph.n_node[:]
        
        return graph_energy.reshape(-1), graph_ener_var.reshape(-1)



