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

# Map circuit names to quantum circuit functions
circuit_map = {
    'basic': circuit_basic,
    'cx': circuit_cx,
    'cz': circuit_cz,
    'qft_cx': circuit_qft_cx,
    'qft_cz': circuit_qft_cz,
    'cx_noise': circuit_cx_noise
}


def default_radial_basis(r, n: int):
    """
    Compute radial basis functions for distance encoding.
    
    Uses Bessel functions with a polynomial envelope for smooth cutoff.
    This provides a learnable representation of interatomic distances.
    
    Args:
        r: Input distance values (can be scalar or array)
        n: Number of basis functions to generate
        
    Returns:
        Array of shape (..., n) containing basis function values
        
    Notes:
        - Bessel functions provide oscillatory basis
        - Polynomial envelope with p=2 ensures smooth decay to zero
        - e3nn.poly_envelope(1, 2)(r) gives (1-r)^2 * (1+2r) for 0 <= r <= 1
    """
    return e3nn.bessel(r, n) * e3nn.poly_envelope(1, 2)(r)[:, None]
    

class LayerFlax(nn.Module):
    """
    Quantum-classical hybrid equivariant GNN layer.
    
    This layer extends the classical GNN with a quantum circuit component:
    1. Classical message passing (same as race_model.py)
    2. Quantum circuit processes features instead of classical gating
    3. Final linear projection to outputs
    
    The quantum circuit acts as a trainable nonlinearity that may capture
    correlations beyond classical activation functions.
    
    Attributes:
        hidden_irreps: Hidden layer irreducible representations
        output_irreps: Output irreducible representations  
        num_species: Number of distinct atomic species
        node_species: Array of species indices for each node
        x_node_feats: Initial node features for tensor product
        radial_embedding: Radial basis function values for each edge
        edges_attr: Spherical harmonic edge attributes
        senders: Edge source node indices
        receivers: Edge target node indices
        avg_num_neighbors: Average neighbor count for normalization
        mlp_n_hidden: Hidden layer size for radial MLP (default: 64)
        mlp_n_layers: Number of layers in radial MLP (default: 2)
        mlp_activation: Activation function for MLP (default: silu)
        qml_circuit: Type of quantum circuit to use (default: "cx")
        qml_num_layers: Number of quantum circuit layers (default: 3)
        qml_qubits: Number of qubits in quantum circuit (default: 8)
    """
    hidden_irreps: e3nn.Irreps
    output_irreps: e3nn.Irreps 
    num_species: int 
    node_species: jax.Array
    x_node_feats: e3nn.IrrepsArray
    radial_embedding: e3nn.IrrepsArray
    edges_attr: e3nn.IrrepsArray
    senders: jax.Array  # [n_edges]
    receivers: jax.Array  # [n_edges]
    avg_num_neighbors: float
    # Radial Embedding
    mlp_n_hidden: int = 64
    mlp_n_layers: int = 2
    mlp_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.silu
    # QML parameters
    qml_circuit: str = "cx"
    qml_num_layers: int = 3
    qml_qubits: int = 8

    @nn.compact
    def __call__(self, node_feats: e3nn.IrrepsArray):
        """
        Apply quantum-classical hybrid graph convolution layer.
        
        Args:
            node_feats: Input node features as IrrepsArray [n_nodes, irreps]
            
        Returns:
            node_outputs: Output predictions for each node [n_nodes, output_irreps]
            node_feats: Updated node features for next layer [n_nodes, hidden_irreps]
            
        Process (same as classical until quantum circuit step):
            1. Compute skip connection with species-dependent weights
            2. Classical message passing with angular and radial components
            3. Normalize by sqrt(avg_num_neighbors) twice (MACE rescaling)
            4. Apply tensor product with initial node features
            5. Add skip connection
            6. **QUANTUM**: Process features through quantum circuit
            7. Final linear projection to outputs
            
        The key difference from race_model.py is step 6, where quantum
        circuits replace classical gated nonlinearities.
        """
        # Skip connection with learnable species-dependent transformation
        skip = e3nn.flax.Linear(self.hidden_irreps,
                                num_indexed_weights=self.num_species,
                                name="skip_tp",
                                force_irreps_out=True)(
                                       self.node_species,
                                       node_feats)
        
        #### InteractionBlock (Classical Message Passing) ####
        node_feats = e3nn.flax.Linear(self.hidden_irreps,
                                       name='linear_up')(node_feats)
        
        # Gather features from sender nodes
        messages = node_feats[self.senders]
        
        # Angular part: tensor product with spherical harmonics
        hidden_irreps = e3nn.Irreps(self.hidden_irreps).regroup()
        messages = e3nn.concatenate(
            [
                messages.filter(hidden_irreps+"0e"),
                e3nn.tensor_product(
                    messages,
                    self.edges_attr,
                    filter_ir_out=hidden_irreps+"0e",
                ),
            ]
        ).regroup()  # [n_edges, irreps]

        # Radial Part: distance-dependent weighting
        mix = e3nn.flax.MultiLayerPerceptron(
            self.mlp_n_layers * (self.mlp_n_hidden,) + (messages.irreps.num_irreps,),
            self.mlp_activation,
            output_activation=False,
        )(self.radial_embedding)

        messages = messages * mix 
        
        # Message passing aggregation
        zeros = e3nn.IrrepsArray.zeros(
            messages.irreps, node_feats.shape[:1], messages.dtype
        )
        node_feats = zeros.at[self.receivers].add(messages)
        node_feats = node_feats / jnp.sqrt(self.avg_num_neighbors)
        
        node_feats = e3nn.flax.Linear(self.hidden_irreps, name='linear_down')(node_feats)

        # MACE rescaling
        node_feats /= jnp.sqrt(self.avg_num_neighbors)        
        
        # Tensor product with initial features
        node_feats = e3nn.tensor_product(self.x_node_feats, node_feats, 
                                       filter_ir_out=hidden_irreps)
        node_feats = e3nn.flax.Linear(self.hidden_irreps,
                                      force_irreps_out=True)(node_feats)
        node_feats = node_feats + skip 

        #### Quantum Circuit Processing ####
        # Initialize trainable quantum circuit parameters
        weight_qml = self.param(f'weight_qml',
                        nn.initializers.lecun_normal(),
                        (self.qml_num_layers, self.qml_qubits))
        
        # Project features to match number of qubits
        features = e3nn.flax.Linear(f"{self.qml_qubits}x0e")(node_feats)
        
        # Apply quantum circuit (replaces classical gating)
        # Each row of features is encoded and processed by quantum circuit
        features = circuit_map[self.qml_circuit](self.qml_qubits, 
                                                  features.array, 
                                                  weight_qml)
        
        # Convert back to IrrepsArray format
        features = e3nn.IrrepsArray(f"{self.qml_qubits}x0e", jnp.array(features).T)
        
        # Final projection to output dimension
        node_outputs = e3nn.flax.Linear(self.output_irreps, biases=True)(features)
        
        return node_outputs, node_feats
    

class GraphNN(nn.Module):
    """
    Quantum-classical hybrid Graph Neural Network for molecular properties.
    
    This model combines classical equivariant graph convolutions with
    quantum circuit processing. The architecture is identical to the
    classical model (race_model.GraphNN) except each layer uses quantum
    circuits for feature transformation instead of classical gates.
    
    Potential advantages of quantum processing:
    - Ability to represent exponentially complex feature spaces
    - Natural encoding of quantum mechanical correlations
    - Novel nonlinearities beyond classical activation functions
    
    Attributes:
        cutoff: Distance cutoff for neighbor list (Angstroms)
        avg_num_neighbors: Average number of neighbors per atom
        num_species: Number of distinct atomic species
        max_ell: Maximum angular momentum for spherical harmonics (default: 3)
        num_basis_func: Number of radial basis functions (default: 8)
        hidden_irreps: Hidden layer irreps (default: "32x0e+8x1o+4x2e")
        nlayers: Number of graph convolution layers (default: 4)
        features_dim: Dimension of initial species embedding (default: 128)
        output_irreps: Output irreps per node (default: "1x0e")
        qml_circuit: Type of quantum circuit (default: "cx")
    """
    cutoff: float
    avg_num_neighbors: float
    num_species: int 
    max_ell: int = 3
    num_basis_func: int = 8
    hidden_irreps: e3nn.Irreps = e3nn.Irreps("32x0e+8x1o+4x2e")
    nlayers: int = 4
    features_dim: int = 128
    output_irreps: e3nn.Irreps = e3nn.Irreps("1x0e")
    qml_circuit: str = "cx"

    @nn.compact
    def __call__(self, Rij, data_graph):
        """
        Forward pass through quantum-classical hybrid GNN.
        
        Args:
            Rij: Edge displacement vectors [num_edges, 3]
            data_graph: jraph.GraphsTuple containing graph structure and features
            
        Returns:
            graph_energy: Predicted energy per graph [n_graphs]
            graph_ener_var: Predicted energy variance per graph [n_graphs]
            
        Process (identical to classical model):
            1. Normalize distances and embed species
            2. Compute radial and angular edge features
            3. Apply nlayers quantum-classical hybrid layers
            4. Sum node predictions to graph-level outputs
            5. Optionally compute uncertainty
            
        The only difference from race_model.GraphNN is that LayerFlax
        here uses quantum circuits internally.
        """
        # Validate input
        assert Rij.ndim == 2 and Rij.shape[1] == 3
        
        # Get edge connectivity
        iatoms = data_graph.senders
        jatoms = data_graph.receivers
        
        # Normalize by cutoff
        Rij = Rij / self.cutoff
        Rij = e3nn.IrrepsArray("1o", Rij)
        
        # Embed atomic species
        species = data_graph.nodes['species']
        node_feats = nn.Embed(num_embeddings=self.num_species, 
                              features=self.features_dim)(species)
        node_feats = e3nn.IrrepsArray(f"{node_feats.shape[-1]}x0e", node_feats)
        
        # Compute edge lengths
        lengths = e3nn.norm(Rij).array[:, 0]
        
        # Radial basis functions (handle padded edges with length=0)
        radial_embedding = jnp.where(
            (lengths == 0.0)[:, None], 0.0,
            default_radial_basis(lengths, self.num_basis_func))
                                            
        # Spherical harmonics for angular features
        interaction_irreps = e3nn.Irreps.spherical_harmonics(self.max_ell)

        # Storage for layer outputs
        outputs = []
        node_logvar = []

        # Initial features for tensor products
        x_node_feats = e3nn.flax.Linear(node_feats.irreps)(node_feats)
        
        # Compute edge angular attributes
        edges_attr = e3nn.spherical_harmonics(interaction_irreps,
                                            Rij,
                                            normalize=True,
                                            normalization="component")
        
        # Apply multiple quantum-classical hybrid layers
        for _ in range(self.nlayers):
            layer = LayerFlax(
                hidden_irreps=self.hidden_irreps,
                output_irreps=self.output_irreps,
                num_species=self.num_species,
                node_species=species,
                x_node_feats=x_node_feats,
                radial_embedding=radial_embedding,
                edges_attr=edges_attr,
                senders=iatoms,
                receivers=jatoms,
                avg_num_neighbors=self.avg_num_neighbors,
                qml_circuit=self.qml_circuit  # Pass quantum circuit type
            )
            node_outputs, node_feats = layer(node_feats)

            # Collect outputs
            outputs += [node_outputs.array[:, 0]]
            if self.output_irreps == e3nn.Irreps('2x0e'):
                node_logvar += node_outputs.array[:, 1] 
        
        # Aggregate node predictions
        node_energy = jnp.stack(outputs, axis=1)  # [n_nodes, nlayers]
        node_energy = node_energy.sum(axis=-1)  # [n_nodes]
        
        # Global pooling: sum to graph level
        graph_energy = e3nn.scatter_sum(node_energy, nel=data_graph.n_node)
        
        # Compute uncertainty if requested
        if self.output_irreps == e3nn.Irreps('2x0e'):
            node_logvar = jnp.stack(node_logvar, axis=1)
            node_logvar = node_logvar.mean(axis=-1)
        else: 
            node_logvar = jnp.zeros(node_feats.shape[0])
        
        node_ener_var = jnp.exp(node_logvar)
        graph_ener_var = e3nn.scatter_sum(node_ener_var, nel=data_graph.n_node) / data_graph.n_node[:]
        
        return graph_energy.reshape(-1), graph_ener_var.reshape(-1)
