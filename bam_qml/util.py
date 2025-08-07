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


def get_graphset (data, cutoff, nbatch, uniq_element, enr_avg_per_element ):
    # iatoms ==> sender
    # jatoms ==> receiver
    # Dij = R[jatom] - R[iatom] + jnp.einsum('ei,ij->ej', Sij, cell)
    # Dij = R[jatom] - R[iatom] + Sij.dot(cell)
    graph_list = []
    for atoms in data:
        crds = atoms.get_positions()
        
        node_enr_avg = jnp.array([enr_avg_per_element[uniq_element[iz]] 
                                  for iz in atoms.numbers])
        
        enr = atoms.get_potential_energy() - node_enr_avg.sum() 
        frc = atoms.get_forces ()
        cell = atoms.get_cell ()
        volume = atoms.get_volume ()
        stress = jnp.zeros (6)
        if 'stress' in atoms._calc.results.keys():
            stress = atoms.get_stress ()

        iatoms, jatoms, Sij = neighbour_list (quantities='ijS',
                                              atoms=atoms,
                                              cutoff=cutoff,
                                              )

        species = jnp.array([uniq_element[iz] for iz in atoms.numbers])
        num_nodes = crds.shape[0]
        num_edges = iatoms.shape[0]

        # positions, species, forces depend on nodes
        # energy, cell, volume, stress depend on each configuration (globals)
        graph = jraph.GraphsTuple (
            nodes={
                "positions":jnp.array (crds),
                "species":species,
                "forces" : jnp.array(frc)
            },
            edges=dict(Sij=jnp.array(Sij)),
            globals=dict(energy=jnp.array([enr]), 
                         cell=jnp.array([cell.array]),
                         volume=jnp.array([volume]),
                         stress=jnp.array([stress])),
            ###
            senders=jnp.array(iatoms),
            receivers=jnp.array(jatoms),
            n_node=jnp.array([num_nodes]),
            n_edge=jnp.array([num_edges]),
        )
        graph_list.append (graph)


    pad_nodes_to = 0 #nbatch*max_nodes+1
    pad_edges_to = 0 #nbatch*max_edges
    n_data = len(data)
    for ist0 in range (0, n_data, nbatch):
        ied0 = jnp.where (ist0+nbatch < n_data, ist0+nbatch, n_data)
        graph = jraph.batch (graph_list[ist0:ied0])
        pad_nodes_to = max(graph.n_node.sum(), pad_nodes_to)
        pad_edges_to = max(graph.n_edge.sum(), pad_edges_to)

    pad_nodes_to = pad_nodes_to + 1
    pad_graphs_to = nbatch + 1
    dataset_list = []
    for ist0 in range (0, n_data, nbatch):
        ied0 = jnp.where (ist0+nbatch < n_data, ist0+nbatch, n_data)
        graph = jraph.batch (graph_list[ist0:ied0])
        graph = jraph.pad_with_graphs(graph, 
                                      pad_nodes_to, 
                                      pad_edges_to,
                                      pad_graphs_to)
        dataset_list.append (graph)

    return dataset_list



def get_enr_avg_per_element (traj, element):

    tgt_enr = np.array([atoms.get_potential_energy() 
                    for atoms in traj])
    
    uniq_element = {int(e): i for i, e in enumerate(element)}
    element_counts = {i: np.array([ (atoms.numbers == e).sum()
                                   for atoms in traj])
                                for e, i in uniq_element.items()}
    c0 = jnp.array ([element_counts[i] for i in element_counts.keys()])
    m0 = tgt_enr.sum()/c0.sum()
    w0 = jnp.array ([m0 for _ in element])
    
    def loss_fn (weight, count):
        # weight:  (nspec)
        # count:  (nspec, ndata)
        
        def objective_mean (w0, c0): 
            # w0: weight (nspec)
            # c0: count  (nspec, ndata)
            return np.einsum('i,ij->j', w0, c0)
    
        prd_enr = objective_mean (weight, count)
        diff = (tgt_enr - prd_enr)
        return (diff*diff).mean()
    
    results = minimize (loss_fn, x0=w0, args=(c0,), method='BFGS')
    w0 = results.x
    
    enr_avg_per_element = {}
    for i, e in enumerate(element):
        enr_avg_per_element[i] = w0[i]
    
    return enr_avg_per_element, uniq_element


def get_trajectory (fname_train, fname_test, cutoff, nbatch, element, w0):
    
    train_data = read (fname_train, index=slice(None))
    test_data = read (fname_test, index=slice(None))
    print(f'\nntrain: {len(train_data)} | ntest: {len(test_data)}\n')
    traj = train_data + test_data
    
    uniq_element = {int(e): i for i, e in enumerate(element)}
    #enr_avg_per_element, uniq_element = \
    #    get_enr_avg_per_element(traj, element)
    enr_avg_per_element = {}
    for i, e in enumerate(element):
        enr_avg_per_element[i] = w0[i]
    
    print ('enr_avg_per_element', enr_avg_per_element)

    
    train_graphset = get_graphset (train_data, cutoff, nbatch, 
                                   uniq_element, enr_avg_per_element)
    test_graphset = get_graphset (test_data, cutoff, nbatch, 
                                  uniq_element, enr_avg_per_element)
                                  
    return train_graphset, test_graphset, uniq_element, enr_avg_per_element



def get_graphset_to_predict_from_atoms(atoms, cutoff, 
                                       uniq_element,
                                       max_edges):
    # iatoms ==> sender
    # jatoms ==> receiver
    # Dij = R[jatom] - R[iatom] + jnp.einsum('ei,ij->ej', Sij, cell)
    # Dij = R[jatom] - R[iatom] + Sij.dot(cell)
    crds = atoms.get_positions()
    cell = atoms.get_cell ()
    volume = atoms.get_volume ()
    # No information of energy, forces, stress
    iatoms, jatoms, Sij = neighbour_list(quantities='ijS',
                                        atoms=atoms,
                                        cutoff=cutoff)
    
    species = jnp.array([uniq_element[iz] for iz in atoms.numbers])
    num_nodes = crds.shape[0]
    num_edges = iatoms.shape[0]

    graph = jraph.GraphsTuple(nodes={"positions":jnp.array(crds),
                                     "species":species},
                                edges=dict(Sij=jnp.array(Sij)),
                                senders=jnp.array(iatoms),
                                receivers=jnp.array(jatoms),
                                n_node=jnp.array([num_nodes]),
                                n_edge=jnp.array([num_edges]),
                                globals=dict(cell=jnp.array([cell.array]), 
                                             volume=jnp.array([volume]))
                                )
    pad_nodes_to = num_nodes + 1
    pad_edges_to = max(num_edges, max_edges)
    pad_graphs_to = 2
    graph = jraph.pad_with_graphs(graph, 
                                      pad_nodes_to, 
                                      pad_edges_to,
                                      pad_graphs_to)
    return graph, pad_edges_to



def checkpoint_save (fname, ckpt):
    with open(fname, 'wb') as fp:        
        pickle.dump (ckpt, fp)

def checkpoint_load (fname):
    with open (fname, 'rb') as fp:
        return pickle.load (fp)
    	
def check_elapsed_time(start_time:str, end_time:str):
    start_time = datetime.strptime(start_time, "%m/%d/%Y %H:%M:%S")
    end_time = datetime.strptime(end_time, "%m/%d/%Y %H:%M:%S")
    time_difference = end_time - start_time
    return time_difference

def date(fmt="%m/%d/%Y %H:%M:%S"):
    return datetime.now().strftime(fmt)


def mae(value1, value2):
    # value1, value2 are constant
    return jnp.abs (value1-value2)
    

def mse(value1, value2):
    # value1, value2 are constant
    return (value1-value2)**2
    
def nll(exact, pred, pred_std):
    # exact, pred, pred_std are constant
    diff = exact - pred 
    diff2 = diff*diff
    pred_var = pred_std*pred_std
    return 0.5*(jnp.log(pred_var)+(diff2/pred_var))
    
