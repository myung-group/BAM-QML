import ase
from ase.calculators.calculator import Calculator, all_changes
import numpy as np
import pickle
import sys

import jax
import jax.numpy as jnp
import jraph

from bam_qml.util import get_graphset_to_predict_from_atoms
import e3nn_jax as e3nn


def get_edge_relative_vectors (R, cell, data_graph):
    # iatoms ==> senders
    # jatoms ==> receivers
    iatoms = data_graph.senders
    jatoms = data_graph.receivers
    Sij = data_graph.edges['Sij']
    num_edges = Sij.shape[0]
    Sij = jnp.einsum(
            'ei,eij->ej',
            Sij,
            jnp.repeat(
                cell, # [n_graph, 3, 3]
                data_graph.n_edge, #[n_graph]
                axis=0,
                total_repeat_length=num_edges,
            ) # [n_edges, 3, 3]
        ) # [n_edges, 3]

    Rij = (R[jatoms] - R[iatoms] + Sij)

    return Rij # (num_edges, 3)

#@jax.jit
def energy_fn (graphset, params, apply_fn):

    def loc_energy_fn (R, cell, params):
        Rij = get_edge_relative_vectors (R, cell, graphset)
        energy, _ = apply_fn({'params':params}, Rij, graphset)
        mask = jraph.get_graph_padding_mask (graphset)
        energy = energy*mask 
        return energy
    
    R = graphset.nodes['positions']
    cell = graphset.globals['cell']
    graph_energy = loc_energy_fn (R, cell, params)
    
    return graph_energy.reshape(-1)[:-1]



class BaseCalculator(Calculator):
    implemented_properties = ['energy']
    
    def __init__(self, json_data, model=None):
        Calculator.__init__(self)
        self.cutoff = json_data['cutoff']
        self.avg_num_neighbors = json_data["avg_num_neighbors"]
        
        fname_model_pkl = json_data['NN']["fname_pkl"]
        fd_ckpt = open(fname_model_pkl, 'rb')
        model_ckpt = pickle.load(fd_ckpt)
        self.params = model_ckpt['params']
        self.uniq_element = model_ckpt['uniq_element']
        self.enr_avg_per_element = model_ckpt['enr_avg_per_element']
        self.logfile = 'ase_model.log'
        
        if json_data["model"] in ['race_qml']:
            from bam_qml.model.race_qml_model import GraphNN
            print ('using race_qml import')
        else:
            from bam_qml.model.race_model import GraphNN
            print ('using race model import')

        output_irreps = e3nn.Irreps ("1x0e")
        hidden_irreps = e3nn.Irreps (json_data['hidden_irreps'])
        G_model = GraphNN (json_data['cutoff'],
                     json_data['avg_num_neighbors'],
                     json_data['num_species'],
                     num_basis_func=json_data['num_radial_basis'],
                     hidden_irreps=hidden_irreps,
                     nlayers = json_data['nlayers'],
                     features_dim = json_data['features_dim'],
                     output_irreps=output_irreps,
                     qml_circuit=json_data['qml_circuit'])
        
        self.model_apply_fn = jax.jit (G_model.apply)
        self.step = 0
        self.pad_edges_to = 0
        self.log ("BAMCalculatorEnergy says Hello!", mode='w')


    def calculate(self, atoms, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        graph, self.pad_edges_to = get_graphset_to_predict_from_atoms(atoms, 
                                                   self.cutoff,
                                                   self.uniq_element,
                                                   self.pad_edges_to)
        
        species = graph.nodes['species']
        node_enr_avg = jnp.array([self.enr_avg_per_element[int(iz)] for iz in species])
        graph_enr_avg = node_enr_avg.sum()
        enr = energy_fn(graph, self.params, self.model_apply_fn)
        self.results["energy"] = float(enr[0]) + graph_enr_avg

        self.step += 1
        #self.log ("{} ".format (enr[0]))
        #sys.exit()


    def log(self, mssge, mode='a'):
        if self.logfile:
            with open(self.logfile, mode) as f:
                f.write ("{}  {}\n".format( self.step, mssge))
