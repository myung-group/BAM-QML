import json
import jax 
import jax.numpy as jnp 
import optax 
from bam_qml.training.train_state import TrainState 
from bam_qml.util import (get_trajectory, 
                      checkpoint_load, 
                      checkpoint_save, 
                      date)

import jraph
import e3nn_jax as e3nn
from functools import partial


gamma_frc = 100.0


def l2_regularization (params):
    wgt, _ = jax.flatten_util.ravel_pytree (params) # or state.params instead of params
    return jnp.einsum('i,i->i', wgt, wgt).mean()


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


def energy_forces_stress_fn (graphset, params, apply_fn):

    def energy_fn (R, cell, params):
        Rij = get_edge_relative_vectors (R, cell, graphset)
        energy, _ = apply_fn({'params':params}, Rij, graphset)
        mask = jraph.get_graph_padding_mask (graphset)
        energy = energy*mask 
        return energy.sum(), (energy)
    
    R = graphset.nodes['positions']
    cell = graphset.globals['cell']
    grad, (graph_energy) = \
        jax.grad (energy_fn, argnums=(0), has_aux=True) (R, cell, params)
    #
    # graph_energy (n_graph)
    # frc (n_graph*n_atoms*3)
    # stress (n_graph*6)
    # enr_var (n_graph)
    # Remove the padded information (+1 graph, +1 node) 
    return graph_energy.reshape(-1)[:-1], \
          -grad.reshape(-1)[:-3]


def loss_value (predicts, targets):
    """
    enr_var is 1 (constant, not trainable value) if output_irreps = 1x0e
    enr_var becomes a trainable parameter if output_irreps = 2x0e
    """
    prd_enr, prd_frc = predicts
    tgt_enr, tgt_frc = targets
    
    diff_enr = (tgt_enr - prd_enr)#/n_node
    diff_frc = (tgt_frc - prd_frc)
    
    loss_enr2 = jnp.einsum('i,i->i', diff_enr, diff_enr).mean()
    loss_frc2 = jnp.einsum('i,i->i', diff_frc, diff_frc).mean()
    
    return loss_enr2, loss_frc2


#@partial(jax.jit, static_argnames=['lambd'])
@jax.jit
def train_step (state, graphset, lambd):
    
    def loss_fn (params, apply_fn):
        predicts = energy_forces_stress_fn (graphset, params, apply_fn)
        
        ref_enr=graphset.globals['energy'][:-1]
        ref_frc=graphset.nodes['forces'].reshape(-1)[:-3]
        
        targets = (ref_enr, ref_frc) 
        loss_enr2, loss_frc2 = \
                loss_value(predicts, targets)
        loss_l2 = l2_regularization (params)

        return loss_enr2 + gamma_frc*loss_frc2 + lambd * loss_l2
        
    grads = jax.grad (loss_fn) (state.params, state.apply_fn) 
    
    return state.apply_gradients(grads=grads) 


@jax.jit
def get_loss (state, graphset):
        
    prd_enr, prd_frc = energy_forces_stress_fn (graphset, state.params, state.apply_fn)
    ref_enr=graphset.globals['energy'][:-1]
    ref_frc=graphset.nodes['forces'].reshape(-1)[:-3]
    
    shift_enr = ref_enr.mean() - prd_enr.mean()
    prd_enr = prd_enr + shift_enr 

    predicts = (prd_enr, prd_frc)
    targets  = (ref_enr, ref_frc) 
    loss_enr2, loss_frc2 = loss_value(predicts, targets)
    
    return loss_enr2, loss_frc2

    
def get_dataset_loss (state, dataset):
    
    loss_mse = jnp.array([get_loss (state, x) 
                          for x in dataset]).mean(axis=0)

    return loss_mse[0], loss_mse[1]
    

def rase_trainer (json_data):
    """
    Restratified atomic cluster expansion (RACE)
    """
    
    fout = open(json_data['train']['fname_log'], 'w', 1)
    x_train, x_test, uniq_element, enr_avg_per_element = \
          get_trajectory (json_data['fname_train'],
                            json_data['fname_val'],
                            json_data['cutoff'],
                            json_data['nbatch'],
                            json_data['element'],
                            json_data['enr_avg_per_element'])
    
    

    if json_data["model"] in ['race_qml']:
        from bam_qml.model.race_qml_model import GraphNN
        print ('using race_qml import')
    else:
        from bam_qml.model.race_model import GraphNN
        print ('using race model import')

    output_irreps = e3nn.Irreps ("1x0e")
    hidden_irreps = e3nn.Irreps (json_data['hidden_irreps'])
    model = GraphNN (json_data['cutoff'],
                     json_data['avg_num_neighbors'],
                     json_data['num_species'],
                     num_basis_func=json_data['num_radial_basis'],
                     hidden_irreps=hidden_irreps,
                     nlayers = json_data['nlayers'],
                     features_dim = json_data['features_dim'],
                     output_irreps=output_irreps,
                     qml_circuit=json_data['qml_circuit'])
    
    rng = jax.random.key(json_data['NN']['rng_seed'])
    rng, key = jax.random.split (rng)
    graphset = x_train[0]
    R = graphset.nodes['positions']
    cell = graphset.globals['cell']
    Rij = get_edge_relative_vectors (R, cell, graphset)
    params = model.init (key, Rij, graphset)['params']

    lr = json_data['NN']['learning_rate']
    ema_decay=json_data['NN']['ema_decay']
    opt_method = optax.chain (optax.clip_by_global_norm (0.5),
                              optax.clip (0.5),
                              optax.inject_hyperparams(optax.amsgrad) (learning_rate=lr))
    
    decay_factor = json_data['NN']['decay_factor']
    lr_min = 0.2*lr 
    itolerate = 0

    state = TrainState.create (
        apply_fn = model.apply,
        params = params,
        tx=opt_method,
        ema_decay=ema_decay
    )

    lambd = json_data['NN']['l2_lambda']
    print(date())
    
    
    loss_dict = {'train': [], 'test': []} 
    l_ckpt_saved = False
    ckpt = {
        'params': state.params,
        'opt_state': state.opt_state,
        'uniq_element': uniq_element,
        'enr_avg_per_element': enr_avg_per_element,
        'loss': loss_dict
    }
    if json_data['NN']['restart']:
        ckpt = checkpoint_load (json_data['NN']['fname_pkl'])
        opt_state = ckpt['opt_state']
        #opt_state = optax.tree_utils.tree_set (opt_state, learning_rate=lr)
        state = state.replace (params=ckpt['params'],
                               opt_state=opt_state)    
        l_ckpt_saved = True
    
    loss_l2 = l2_regularization(state.params)
    loss_enr2_test, loss_frc2_test = \
        get_dataset_loss (state, x_test)
    
    loss_test_min = loss_enr2_test + gamma_frc*loss_frc2_test + lambd * loss_l2
    loss_test_local = loss_test_min

    print (f'                    \tTRAIN_loss____________________________________________| TEST_loss', file=fout)
    line = f"MM/DD/YYYY HH/MM/SS\t {'EPOCH':7}{'LOSS':14}{'RMSE_E':11}{'RMSE_F':11}{'L2':10}| "
    line = line + f"{'LOSS':14}{'RMSE_E':11}{'RMSE_F':11}|{'LR':7}"
    print (line, file=fout)
    print ('----------------------------------------------------------------------------------', file=fout)

    print(date())

    
    nepoch = json_data['NN']['nepoch']
    for epch in range (nepoch): 
        for graphset in x_train:
            state = train_step (state, graphset, lambd=lambd)
        
        if (epch+1)%5 == 0:
            # Estimate LOSS
            loss_l2 = l2_regularization(state.params)

            loss_enr2, loss_frc2 = get_dataset_loss (state, x_train)
            loss_enr2_test, loss_frc2_test = get_dataset_loss (state, x_test)
                
            loss = loss_enr2 + gamma_frc*loss_frc2 + lambd*loss_l2
            loss_test = loss_enr2_test + gamma_frc*loss_frc2_test + lambd*loss_l2
            
            lambd = lambd*loss_l2
            if loss_test < loss_test_min:
                loss_test_min = loss_test 
                ckpt['params'] = state.params
                ckpt['opt_state'] = state.opt_state
                ckpt['loss'] = loss_dict
                l_ckpt_saved = False
                

            if loss_test < loss_test_local:
                itolerate = 0
                loss_test_local = loss_test
            else:
                itolerate = itolerate + 1

            lr = optax.tree_utils.tree_get (state.opt_state, "learning_rate")
            if itolerate >= 50:
                lr_new = jnp.where (lr*decay_factor > lr_min, lr*decay_factor, lr_min)
                opt_state = optax.tree_utils.tree_set (state.opt_state, learning_rate=lr_new)
                state = state.replace (opt_state=opt_state)
                itolerate = 0
                lr = lr_new
                loss_test_local = loss_test + 0.01

            loss_dict['train'].append (loss)
            loss_dict['test'].append (loss_test)

            rmse_E = jnp.sqrt (loss_enr2)
            rmse_F = jnp.sqrt (loss_frc2)
            
            rmse_E_test = jnp.sqrt (loss_enr2_test)
            rmse_F_test = jnp.sqrt (loss_frc2_test)
            
            line = f'{date()}\t {epch+1:<7}{loss:<14.6f}{rmse_E:<11.6f}{rmse_F:<11.6f}{loss_l2:<10.5f}'
            line += f'{loss_test:<14.6f}{rmse_E_test:<11.6f}{rmse_F_test:<11.6f}{lr:<7.4f}'
            print (line, file=fout)

            
        if (epch+1)%json_data['NN']['nsave'] == 0 and \
            not l_ckpt_saved:
            checkpoint_save (json_data['NN']['fname_pkl'], ckpt)
            l_ckpt_saved = True
            
        if (epch+1)%20 == 0:
            train_step._clear_cache()
    

if __name__ == '__main__':
    import getopt 
    import sys 

    argv = sys.argv[1:]
    opts, args = getopt.getopt (
        argv, "hi:", ["help=", "input="]
    )
    
    fname_json = "input_qml.json"
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("python -m bam_qml.training.race_trainer -i <input_file.json>")
            sys.exit(1)
        elif opt in ("-i", "--input"):
            fname_json = arg
    print ('fname_json', fname_json)

    with open(fname_json) as f:
        json_data = json.load(f)
        
        if json_data['float64']:
            jax.config.update ("jax_enable_x64", True)
            print ("running with float64")
        else:
            jax.config.update ("jax_enable_x64", False)
            print ("running with float32")
        
        rase_trainer (json_data)
        
        
    print(date())
