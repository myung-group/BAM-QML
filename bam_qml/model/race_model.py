from flax import linen as nn 
import e3nn_jax as e3nn

import jax.numpy as jnp
from typing import Callable
import jax 


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
    Single equivariant graph neural network layer with message passing.
    
    This layer implements:
    1. Skip connection with species-dependent linear transformation
    2. Message passing with angular (spherical harmonics) and radial components
    3. MACE-style rescaling by neighbor count
    4. Tensor product interaction between node features
    5. Gated nonlinearity for feature mixing
    
    The layer is E(3)-equivariant, meaning it respects rotational symmetry.
    
    Attributes:
        hidden_irreps: Hidden layer irreducible representations (e.g., "32x0e+8x1o+4x2e")
        output_irreps: Output irreducible representations
        num_species: Number of distinct atomic species
        node_species: Array of species indices for each node
        x_node_feats: Initial node features for tensor product
        radial_embedding: Radial basis function values for each edge
        edges_attr: Spherical harmonic edge attributes (angular features)
        senders: Edge source node indices
        receivers: Edge target node indices
        avg_num_neighbors: Average neighbor count for normalization
        mlp_n_hidden: Hidden layer size for radial MLP (default: 64)
        mlp_n_layers: Number of layers in radial MLP (default: 2)
        mlp_activation: Activation function for MLP (default: silu)
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

    @nn.compact
    def __call__(self, node_feats: e3nn.IrrepsArray):
        """
        Apply equivariant graph convolution layer.
        
        Args:
            node_feats: Input node features as IrrepsArray [n_nodes, irreps]
            
        Returns:
            node_outputs: Output predictions for each node [n_nodes, output_irreps]
            node_feats: Updated node features for next layer [n_nodes, hidden_irreps]
            
        Process:
            1. Compute skip connection with species-dependent weights
            2. Project node features to hidden dimension
            3. Gather features at sender nodes for message construction
            4. Compute angular features via tensor product with edge attributes
            5. Weight messages by radial MLP output
            6. Aggregate messages at receiver nodes (message passing)
            7. Normalize by sqrt(avg_num_neighbors) (MACE rescaling, done twice)
            8. Apply tensor product with initial node features
            9. Add skip connection
            10. Apply gated nonlinearity and produce outputs
        """
        # Skip connection with learnable species-dependent transformation
        skip = e3nn.flax.Linear(self.hidden_irreps,
                                num_indexed_weights=self.num_species,
                                name="skip_tp",
                                force_irreps_out=True)(
                                       self.node_species,
                                       node_feats)
        
        #### InteractionBlock ####
        # Project to hidden dimension
        node_feats = e3nn.flax.Linear(self.hidden_irreps,
                                       name='linear_up')(node_feats)
        
        # [[[MessagePassingConvolution]]]
        # Gather features from sender nodes
        messages = node_feats[self.senders]
        
        # Angular part: combine node features with edge spherical harmonics
        hidden_irreps = e3nn.Irreps(self.hidden_irreps).regroup()
        messages = e3nn.concatenate(
            [
                messages.filter(hidden_irreps+"0e"),  # Keep scalar part
                e3nn.tensor_product(
                    messages,
                    self.edges_attr,
                    filter_ir_out=hidden_irreps+"0e",  # Tensor product for angular info
                ),
            ]
        ).regroup()  # [n_edges, irreps]

        # Radial Part: Learn distance-dependent weights
        mix = e3nn.flax.MultiLayerPerceptron(
            self.mlp_n_layers * (self.mlp_n_hidden,) + (messages.irreps.num_irreps,),
            self.mlp_activation,
            output_activation=False,
        )(self.radial_embedding)

        # Apply radial weights to angular messages
        messages = messages * mix 
        
        # MessagePassing: aggregate messages at receiver nodes
        zeros = e3nn.IrrepsArray.zeros(
            messages.irreps, node_feats.shape[:1], messages.dtype
        )
        node_feats = zeros.at[self.receivers].add(messages)
        node_feats = node_feats / jnp.sqrt(self.avg_num_neighbors)
        ##[[ Done MessagePassingConvolution]]
        
        node_feats = e3nn.flax.Linear(self.hidden_irreps, force_irreps_out=True, 
                                      name='linear_down')(node_feats)
        ### (DONE InteractionBlock)

        # MACE rescaling: second normalization by neighbor count
        node_feats /= jnp.sqrt(self.avg_num_neighbors)        
        
        # Tensor product with initial node features (multi-body interactions)
        node_feats = e3nn.tensor_product(self.x_node_feats, node_feats, 
                                       filter_ir_out=hidden_irreps)
        node_feats = e3nn.flax.Linear(self.hidden_irreps,
                                      force_irreps_out=True)(node_feats)
        
        # Add skip connection
        node_feats = node_feats + skip 

        # Gated nonlinearity: mix features with learned gates
        features = e3nn.flax.Linear("32x0e")(node_feats)
        features = e3nn.gate(features,
                                  even_act=jax.nn.silu,
                                  odd_act=jax.nn.tanh,
                                  even_gate_act=jax.nn.silu)
        
        # Final output projection
        node_outputs = e3nn.flax.Linear(self.output_irreps, biases=True)(features)
        
        return node_outputs, node_feats
    

class GraphNN(nn.Module):
    """
    E(3)-equivariant Graph Neural Network for molecular property prediction.
    
    This model implements a multi-layer message passing neural network that:
    - Respects rotational and translational symmetry (E(3) equivariance)
    - Uses spherical harmonics for angular information
    - Employs radial basis functions for distance encoding
    - Predicts per-atom contributions that sum to global properties
    
    Architecture:
    1. Embed atomic species into feature vectors
    2. Compute edge features (distances, directions, spherical harmonics)
    3. Apply multiple equivariant graph convolution layers
    4. Sum node predictions to get graph-level output (e.g., total energy)
    5. Optionally predict uncertainty via variance
    
    Attributes:
        cutoff: Distance cutoff for neighbor list (Angstroms)
        avg_num_neighbors: Average number of neighbors per atom (for normalization)
        num_species: Number of distinct atomic species
        max_ell: Maximum angular momentum for spherical harmonics (default: 3)
        num_basis_func: Number of radial basis functions (default: 8)
        hidden_irreps: Hidden layer irreps (default: "32x0e+8x1o+4x2e")
        nlayers: Number of graph convolution layers (default: 4)
        features_dim: Dimension of initial species embedding (default: 128)
        output_irreps: Output irreps per node (default: "1x0e" for energy)
        qml_circuit: Quantum circuit type (not used in this classical model)
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
        Forward pass through the graph neural network.
        
        Args:
            Rij: Edge displacement vectors [num_edges, 3]
            data_graph: jraph.GraphsTuple containing:
                - nodes['species']: Atomic species indices
                - nodes['positions']: Atomic coordinates
                - senders: Edge source indices
                - receivers: Edge target indices
                - n_node: Number of nodes per graph
                
        Returns:
            graph_energy: Predicted energy per graph [n_graphs]
            graph_ener_var: Predicted energy variance per graph [n_graphs]
            
        Process:
            1. Normalize distances by cutoff
            2. Embed atomic species into feature vectors
            3. Compute radial basis functions from distances
            4. Compute spherical harmonic edge attributes
            5. Apply nlayers equivariant convolutions
            6. Sum node predictions across graphs
            7. Compute variance if output_irreps has 2 channels
        """
        # Validate input shape
        assert Rij.ndim == 2 and Rij.shape[1] == 3
        
        # Get edge connectivity
        iatoms = data_graph.senders
        jatoms = data_graph.receivers
        
        # Normalize by cutoff and convert to IrrepsArray
        Rij = Rij / self.cutoff
        Rij = e3nn.IrrepsArray("1o", Rij)
        
        # (Embedding) Map atomic species to learned feature vectors
        species = data_graph.nodes['species']
        node_feats = nn.Embed(num_embeddings=self.num_species, 
                              features=self.features_dim)(species)
        node_feats = e3nn.IrrepsArray(f"{node_feats.shape[-1]}x0e", node_feats)
        
        # Compute edge lengths
        lengths = e3nn.norm(Rij).array[:, 0]
        
        # Radial embedding with handling for padded edges (length == 0)
        # Note: jraph.pad_with_graphs adds dummy edges with zero length
        radial_embedding = jnp.where(
            (lengths == 0.0)[:, None], 0.0,
            default_radial_basis(lengths, self.num_basis_func))
        
        # Compute spherical harmonics up to max_ell for angular information
        interaction_irreps = e3nn.Irreps.spherical_harmonics(self.max_ell)

        # Prepare for multi-layer message passing
        outputs = []
        node_logvar = []

        # Initial node features for tensor product operations
        x_node_feats = e3nn.flax.Linear(node_feats.irreps)(node_feats)

        # Compute edge angular features
        edges_attr = e3nn.spherical_harmonics(interaction_irreps,
                                            Rij,
                                            normalize=True,
                                            normalization="component")

        # Apply multiple graph convolution layers
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
            )
            node_outputs, node_feats = layer(node_feats)

            # Collect outputs from each layer
            outputs += [node_outputs.array[:, 0]]
            if self.output_irreps == e3nn.Irreps('2x0e'):
                # Second channel is log variance for uncertainty
                node_logvar += node_outputs.array[:, 1] 
        
        # Sum contributions from all layers
        node_energy = jnp.stack(outputs, axis=1)  # [n_nodes, num_interactions]
        node_energy = node_energy.sum(axis=-1)  # [n_nodes]
        
        # Global Pooling: sum node energies to get graph energy
        graph_energy = e3nn.scatter_sum(node_energy, nel=data_graph.n_node)
        
        # Compute variance if model outputs uncertainty
        if self.output_irreps == e3nn.Irreps('2x0e'):
            node_logvar = jnp.stack(node_logvar, axis=1)
            node_logvar = node_logvar.mean(axis=-1)
        else: 
            node_logvar = jnp.zeros(node_feats.shape[0])
        
        # Convert log variance to variance
        node_ener_var = jnp.exp(node_logvar)  # [n_nodes]
        # Average variance across nodes in each graph
        graph_ener_var = e3nn.scatter_sum(node_ener_var, nel=data_graph.n_node) / data_graph.n_node[:]
        
        return graph_energy.reshape(-1), graph_ener_var.reshape(-1)
