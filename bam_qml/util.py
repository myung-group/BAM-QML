from ase.io import read
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
import pickle
import os

from matscipy.neighbours import neighbour_list
from datetime import datetime
import jraph


def get_graphset(data, cutoff, nbatch, uniq_element, enr_avg_per_element):
    """
    Convert atomic structure data into batched graph datasets for GNN training.

    This function processes a list of atomic structures and converts them into
    graph representations suitable for message-passing neural networks. Each atom
    becomes a node, and edges are created between atoms within the cutoff distance.

    Args:
        data: List of ASE Atoms objects containing atomic structures
        cutoff: Distance cutoff (Angstroms) for determining neighbor connections
        nbatch: Number of graphs to batch together
        uniq_element: Dictionary mapping atomic numbers to species indices
        enr_avg_per_element: Dictionary of average energy per element for reference

    Returns:
        dataset_list: List of batched and padded jraph.GraphsTuple objects

    Notes:
        - Energy is shifted by removing per-element average energies
        - Graphs are padded to uniform size for efficient batching
        - Sender/receiver indexing follows ASE neighbor list convention
    """
    # iatoms ==> sender
    # jatoms ==> receiver
    # Dij = R[jatom] - R[iatom] + jnp.einsum('ei,ij->ej', Sij, cell)
    # Dij = R[jatom] - R[iatom] + Sij.dot(cell)
    graph_list = []
    for atoms in data:
        crds = atoms.get_positions()

        # Calculate node-averaged reference energy
        node_enr_avg = jnp.array([enr_avg_per_element[uniq_element[iz]]
                                  for iz in atoms.numbers])

        # Shift total energy by removing atomic reference energies
        enr = atoms.get_potential_energy() - node_enr_avg.sum()
        frc = atoms.get_forces()
        cell = atoms.get_cell()
        volume = atoms.get_volume()
        stress = jnp.zeros(6)
        if 'stress' in atoms._calc.results.keys():
            stress = atoms.get_stress()

        # Build neighbor list: iatoms (senders) -> jatoms (receivers)
        # Sij contains periodic boundary shift vectors
        iatoms, jatoms, Sij = neighbour_list(quantities='ijS',
                                              atoms=atoms,
                                              cutoff=cutoff)

        species = jnp.array([uniq_element[iz] for iz in atoms.numbers])
        num_nodes = crds.shape[0]
        num_edges = iatoms.shape[0]

        # Create graph structure with node, edge, and global features
        graph = jraph.GraphsTuple(
            nodes={
                "positions": jnp.array(crds),
                "species": species,
                "forces": jnp.array(frc)
            },
            edges=dict(Sij=jnp.array(Sij)),
            globals=dict(energy=jnp.array([enr]),
                         cell=jnp.array([cell.array]),
                         volume=jnp.array([volume]),
                         stress=jnp.array([stress])),
            senders=jnp.array(iatoms),
            receivers=jnp.array(jatoms),
            n_node=jnp.array([num_nodes]),
            n_edge=jnp.array([num_edges]),
        )
        graph_list.append(graph)

    # Determine maximum padding sizes by scanning through batches
    pad_nodes_to = 0
    pad_edges_to = 0
    n_data = len(data)
    for ist0 in range(0, n_data, nbatch):
        ied0 = jnp.where(ist0+nbatch < n_data, ist0+nbatch, n_data)
        graph = jraph.batch(graph_list[ist0:ied0])
        pad_nodes_to = max(graph.n_node.sum(), pad_nodes_to)
        pad_edges_to = max(graph.n_edge.sum(), pad_edges_to)

    pad_nodes_to = pad_nodes_to + 1
    pad_graphs_to = nbatch + 1

    # Create padded batches for uniform tensor sizes
    dataset_list = []
    for ist0 in range(0, n_data, nbatch):
        ied0 = jnp.where(ist0+nbatch < n_data, ist0+nbatch, n_data)
        graph = jraph.batch(graph_list[ist0:ied0])
        graph = jraph.pad_with_graphs(graph,
                                      pad_nodes_to,
                                      pad_edges_to,
                                      pad_graphs_to)
        dataset_list.append(graph)

    return dataset_list


def get_enr_avg_per_element(traj, element):
    """
    Calculate average energy per element type using least-squares optimization.

    This function fits atomic reference energies by minimizing the difference
    between predicted and actual total energies across all structures. The
    predicted energy is a linear combination of atomic counts weighted by
    per-element energies.

    Args:
        traj: List of ASE Atoms objects with computed energies
        element: List of unique atomic numbers in the dataset

    Returns:
        enr_avg_per_element: Dictionary mapping species index to average energy
        uniq_element: Dictionary mapping atomic number to species index

    Algorithm:
        Minimize: sum_i (E_i - sum_j n_ij * w_j)^2
        where E_i is structure energy, n_ij is count of element j in structure i,
        and w_j is the fitted energy for element j.
    """
    tgt_enr = np.array([atoms.get_potential_energy()
                    for atoms in traj])

    # Create mapping from atomic number to species index
    uniq_element = {int(e): i for i, e in enumerate(element)}

    # Count occurrences of each element in each structure
    element_counts = {i: np.array([(atoms.numbers == e).sum()
                                   for atoms in traj])
                                for e, i in uniq_element.items()}

    # Initialize weights with mean energy per atom
    c0 = jnp.array([element_counts[i] for i in element_counts.keys()])
    m0 = tgt_enr.sum() / c0.sum()
    w0 = jnp.array([m0 for _ in element])

    def loss_fn(weight, count):
        """
        Compute mean squared error between predicted and actual energies.

        Args:
            weight: (nspec,) array of per-element energies
            count: (nspec, ndata) array of element counts per structure

        Returns:
            Mean squared error
        """
        def objective_mean(w0, c0):
            # Predicted energy = sum over elements (weight * count)
            return np.einsum('i,ij->j', w0, c0)

        prd_enr = objective_mean(weight, count)
        diff = (tgt_enr - prd_enr)
        return (diff*diff).mean()

    # Optimize using BFGS
    results = minimize(loss_fn, x0=w0, args=(c0,), method='BFGS')
    w0 = results.x

    # Convert to dictionary format
    enr_avg_per_element = {}
    for i, e in enumerate(element):
        enr_avg_per_element[i] = w0[i]

    return enr_avg_per_element, uniq_element


def get_trajectory(fname_train, fname_test, cutoff, nbatch, element, w0):
    """
    Load training and test trajectories and convert to graph datasets.

    This is the main data loading function that reads atomic structures from
    files and converts them into batched graph representations for neural
    network training and evaluation.

    Args:
        fname_train: Path to training data file (any ASE-readable format)
        fname_test: Path to test/validation data file
        cutoff: Distance cutoff for neighbor list construction (Angstroms)
        nbatch: Number of structures per batch
        element: List of unique atomic numbers present in dataset
        w0: List of per-element reference energies

    Returns:
        train_graphset: List of batched training graphs
        test_graphset: List of batched test graphs
        uniq_element: Dictionary mapping atomic number to species index
        enr_avg_per_element: Dictionary of reference energies per element

    Notes:
        Uses provided w0 energies rather than computing from data.
        Prints dataset sizes for verification.
    """
    train_data = read(fname_train, index=slice(None))
    test_data = read(fname_test, index=slice(None))
    print(f'\nntrain: {len(train_data)} | ntest: {len(test_data)}\n')
    traj = train_data + test_data

    # Create element mapping
    uniq_element = {int(e): i for i, e in enumerate(element)}

    # Use provided reference energies instead of computing
    enr_avg_per_element = {}
    for i, e in enumerate(element):
        enr_avg_per_element[i] = w0[i]

    print('enr_avg_per_element', enr_avg_per_element)

    # Convert to graph datasets
    train_graphset = get_graphset(train_data, cutoff, nbatch,
                                   uniq_element, enr_avg_per_element)
    test_graphset = get_graphset(test_data, cutoff, nbatch,
                                  uniq_element, enr_avg_per_element)

    return train_graphset, test_graphset, uniq_element, enr_avg_per_element


def get_graphset_to_predict_from_atoms(atoms, cutoff, uniq_element, max_edges):
    """
    Convert a single atomic structure to a padded graph for inference.

    This function creates a graph representation for prediction on a single
    structure. Unlike training graphs, it doesn't include target properties
    (energy, forces, stress) since these are unknown.

    Args:
        atoms: ASE Atoms object to convert
        cutoff: Distance cutoff for neighbor list (Angstroms)
        uniq_element: Dictionary mapping atomic numbers to species indices
        max_edges: Maximum number of edges for padding (from molecular dynamics simulations)

    Returns:
        graph: Padded jraph.GraphsTuple ready for model inference
        pad_edges_to: Actual padding size used for edges

    Notes:
        - Only includes structural information (positions, cell, species)
        - Padded to allow batching with other structures if needed
        - Edge padding ensures tensor shape compatibility with trained model
    """
    # iatoms ==> sender
    # jatoms ==> receiver
    # Dij = R[jatom] - R[iatom] + jnp.einsum('ei,ij->ej', Sij, cell)
    # Dij = R[jatom] - R[iatom] + Sij.dot(cell)
    crds = atoms.get_positions()
    cell = atoms.get_cell()
    volume = atoms.get_volume()

    # Build neighbor list (no target properties needed)
    iatoms, jatoms, Sij = neighbour_list(quantities='ijS',
                                        atoms=atoms,
                                        cutoff=cutoff)

    species = jnp.array([uniq_element[iz] for iz in atoms.numbers])
    num_nodes = crds.shape[0]
    num_edges = iatoms.shape[0]

    # Create graph with only structural features
    graph = jraph.GraphsTuple(nodes={"positions": jnp.array(crds),
                                     "species": species},
                                edges=dict(Sij=jnp.array(Sij)),
                                senders=jnp.array(iatoms),
                                receivers=jnp.array(jatoms),
                                n_node=jnp.array([num_nodes]),
                                n_edge=jnp.array([num_edges]),
                                globals=dict(cell=jnp.array([cell.array]),
                                             volume=jnp.array([volume])))

    # Pad to ensure compatibility with model
    pad_nodes_to = num_nodes + 1
    pad_edges_to = max(num_edges, max_edges)
    pad_graphs_to = 2
    graph = jraph.pad_with_graphs(graph,
                                      pad_nodes_to,
                                      pad_edges_to,
                                      pad_graphs_to)
    return graph, pad_edges_to


def checkpoint_save(fname, ckpt):
    """
    Save model checkpoint to disk using pickle.

    Args:
        fname: Output filename for checkpoint
        ckpt: Dictionary containing model state (params, optimizer state, etc.)
    """
    with open(fname, 'wb') as fp:
        pickle.dump(ckpt, fp)


def checkpoint_load(fname):
    """
    Load model checkpoint from disk.

    Args:
        fname: Checkpoint filename to load

    Returns:
        Dictionary containing saved model state
    """
    with open(fname, 'rb') as fp:
        return pickle.load(fp)


def check_elapsed_time(start_time: str, end_time: str):
    """
    Calculate time difference between two datetime strings.

    Args:
        start_time: Start time in format "MM/DD/YYYY HH:MM:SS"
        end_time: End time in same format

    Returns:
        timedelta object representing elapsed time
    """
    start_time = datetime.strptime(start_time, "%m/%d/%Y %H:%M:%S")
    end_time = datetime.strptime(end_time, "%m/%d/%Y %H:%M:%S")
    time_difference = end_time - start_time
    return time_difference


def date(fmt="%m/%d/%Y %H:%M:%S"):
    """
    Get current date and time as formatted string.

    Args:
        fmt: strftime format string (default: MM/DD/YYYY HH:MM:SS)

    Returns:
        Formatted datetime string
    """
    return datetime.now().strftime(fmt)


def mae(value1, value2):
    """
    Compute mean absolute error between two values.

    Args:
        value1: First value or array
        value2: Second value or array

    Returns:
        Absolute difference |value1 - value2|
    """
    return jnp.abs(value1 - value2)


def mse(value1, value2):
    """
    Compute mean squared error between two values.

    Args:
        value1: First value or array
        value2: Second value or array

    Returns:
        Squared difference (value1 - value2)^2
    """
    return (value1 - value2)**2


def nll(exact, pred, pred_std):
    """
    Compute negative log likelihood for Gaussian distribution.

    This is useful for uncertainty quantification in models that predict
    both mean and variance/standard deviation.

    Args:
        exact: True/target value
        pred: Predicted mean value
        pred_std: Predicted standard deviation

    Returns:
        Negative log likelihood: 0.5 * (log(var) + (exact-pred)^2/var)

    Notes:
        Assumes Gaussian likelihood: p(exact|pred,std) = N(pred, std^2)
        Ignores constant terms from normalization.
    """
    diff = exact - pred
    diff2 = diff * diff
    pred_var = pred_std * pred_std
    return 0.5 * (jnp.log(pred_var) + (diff2 / pred_var))
