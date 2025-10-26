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

# Force weight for multi-task learning (energy + forces)
gamma_frc = 100.0


def l2_regularization(params):
    """
    Compute L2 regularization penalty on model parameters.
    
    Encourages small parameter values to prevent overfitting and improve
    generalization. The penalty is the mean squared magnitude of all parameters.
    
    Args:
        params: PyTree of model parameters
        
    Returns:
        Scalar L2 penalty = mean(sum(w_i^2))
        
    Notes:
        - Flattens entire parameter tree into single vector
        - Computes mean of squared values for scale-invariance
        - Added to loss as: total_loss = prediction_loss + lambda * L2_penalty
    """
    wgt, _ = jax.flatten_util.ravel_pytree(params)
    return jnp.einsum('i,i->i', wgt, wgt).mean()


def get_edge_relative_vectors(R, cell, data_graph):
    """
    Compute edge displacement vectors with periodic boundary conditions.
    
    For each edge i→j, computes the vector from atom i to atom j, accounting
    for periodic boundaries. The formula is:
        R_ij = R_j - R_i + S_ij @ cell
    where S_ij is the periodic shift vector.
    
    Args:
        R: Node positions [n_nodes, 3]
        cell: Lattice cell vectors [n_graphs, 3, 3]
        data_graph: jraph.GraphsTuple containing edge connectivity
        
    Returns:
        Rij: Edge displacement vectors [n_edges, 3]
        
    Implementation details:
        - Senders (iatoms): Source nodes of edges
        - Receivers (jatoms): Target nodes of edges
        - Sij: Periodic shift vectors for each edge
        - Cell must be repeated for each edge in the batch
    """
    iatoms = data_graph.senders
    jatoms = data_graph.receivers
    Sij = data_graph.edges['Sij']
    num_edges = Sij.shape[0]
    
    # Apply periodic shift: Sij @ cell
    Sij = jnp.einsum(
            'ei,eij->ej',
            Sij,
            jnp.repeat(
                cell,  # [n_graph, 3, 3]
                data_graph.n_edge,  # [n_graph]
                axis=0,
                total_repeat_length=num_edges,
            )  # [n_edges, 3, 3]
        )  # [n_edges, 3]

    # Compute displacement with periodic boundaries
    Rij = (R[jatoms] - R[iatoms] + Sij)

    return Rij  # [num_edges, 3]


def energy_forces_stress_fn(graphset, params, apply_fn):
    """
    Compute energy and forces using automatic differentiation.
    
    This function:
    1. Computes graph energy from model predictions
    2. Differentiates energy w.r.t. positions to get forces: F = -∇E
    3. Removes padded elements from outputs
    
    Args:
        graphset: Batched graph data structure
        params: Model parameters
        apply_fn: Model forward function
        
    Returns:
        graph_energy: Energy per graph [n_graphs]
        forces: Atomic forces [n_atoms * 3]
        
    Notes:
        - Forces are negative gradient of energy: F = -dE/dR
        - Padding mask ensures dummy nodes/graphs don't contribute
        - Last element in outputs is padding and must be removed
        - Stress computation would require differentiation w.r.t. cell
    """
    def energy_fn(R, cell, params):
        """
        Inner function for energy computation and differentiation.
        
        Args:
            R: Atomic positions [n_nodes, 3]
            cell: Lattice cell [n_graphs, 3, 3]
            params: Model parameters
            
        Returns:
            total_energy: Sum of energies (for gradient computation)
            graph_energy: Energy per graph (auxiliary output)
        """
        # Compute edge vectors
        Rij = get_edge_relative_vectors(R, cell, graphset)
        
        # Model prediction (also returns variance, which we ignore here)
        energy, _ = apply_fn({'params': params}, Rij, graphset)
        
        # Mask out padded graphs
        mask = jraph.get_graph_padding_mask(graphset)
        energy = energy * mask 
        
        return energy.sum(), (energy)
    
    R = graphset.nodes['positions']
    cell = graphset.globals['cell']
    
    # Compute gradient of energy w.r.t. positions
    # grad is dE/dR, which gives -forces
    grad, (graph_energy) = \
        jax.grad(energy_fn, argnums=(0), has_aux=True)(R, cell, params)
    
    # Remove padding: last graph and last 3 coordinates (1 dummy node)
    return graph_energy.reshape(-1)[:-1], \
          -grad.reshape(-1)[:-3]


def loss_value(predicts, targets):
    """
    Compute multi-task loss for energy and forces.
    
    Combines mean squared errors for both energy and force predictions.
    This joint training helps the model learn both thermodynamic properties
    (energy) and local chemical environment (forces).
    
    Args:
        predicts: Tuple of (predicted_energy, predicted_forces)
        targets: Tuple of (target_energy, target_forces)
        
    Returns:
        loss_enr2: Mean squared error for energy
        loss_frc2: Mean squared error for forces
        
    Notes:
        - Energy loss: mean((E_true - E_pred)^2)
        - Force loss: mean((F_true - F_pred)^2) over all components
        - Forces are weighted by gamma_frc in total loss
    """
    prd_enr, prd_frc = predicts
    tgt_enr, tgt_frc = targets
    
    diff_enr = (tgt_enr - prd_enr)
    diff_frc = (tgt_frc - prd_frc)
    
    loss_enr2 = jnp.einsum('i,i->i', diff_enr, diff_enr).mean()
    loss_frc2 = jnp.einsum('i,i->i', diff_frc, diff_frc).mean()
    
    return loss_enr2, loss_frc2


@jax.jit
def train_step(state, graphset, lambd):
    """
    Single training step with gradient descent.
    
    Computes loss, gradients, and updates model parameters using the optimizer.
    This function is JIT-compiled for efficient execution.
    
    Args:
        state: TrainState containing params, optimizer, and apply_fn
        graphset: Batch of training graphs
        lambd: L2 regularization coefficient
        
    Returns:
        Updated TrainState with new parameters
        
    Loss function:
        L = MSE(energy) + gamma_frc * MSE(forces) + lambd * L2(params)
        
    Notes:
        - JIT compilation caches function for repeated calls
        - Automatic differentiation computes parameter gradients
        - Optimizer handles parameter updates (e.g., Adam, AMSGrad)
    """
    def loss_fn(params, apply_fn):
        """
        Combined loss for energy, forces, and regularization.
        
        Args:
            params: Model parameters
            apply_fn: Model forward function
            
        Returns:
            Scalar loss value
        """
        # Compute predictions
        predicts = energy_forces_stress_fn(graphset, params, apply_fn)
        
        # Get reference values (remove padding)
        ref_enr = graphset.globals['energy'][:-1]
        ref_frc = graphset.nodes['forces'].reshape(-1)[:-3]
        
        targets = (ref_enr, ref_frc) 
        
        # Compute losses
        loss_enr2, loss_frc2 = loss_value(predicts, targets)
        loss_l2 = l2_regularization(params)

        # Combined loss with force weighting and regularization
        return loss_enr2 + gamma_frc*loss_frc2 + lambd * loss_l2
        
    # Compute gradients
    grads = jax.grad(loss_fn)(state.params, state.apply_fn) 
    
    # Update parameters using optimizer
    return state.apply_gradients(grads=grads) 


@jax.jit
def get_loss(state, graphset):
    """
    Evaluate model performance without computing gradients.
    
    Used for validation/test set evaluation. Includes energy shift
    correction to align predicted and reference energy scales.
    
    Args:
        state: TrainState with current parameters
        graphset: Batch of graphs to evaluate
        
    Returns:
        loss_enr2: Mean squared error for energy
        loss_frc2: Mean squared error for forces
        
    Notes:
        - Energy shift aligns mean predictions with mean targets
        - This is valid since absolute energy is arbitrary
        - Forces are shift-invariant (derivatives cancel constant)
        - No gradients computed (evaluation only)
    """
    # Get predictions
    prd_enr, prd_frc = energy_forces_stress_fn(graphset, state.params, state.apply_fn)
    
    # Get targets
    ref_enr = graphset.globals['energy'][:-1]
    ref_frc = graphset.nodes['forces'].reshape(-1)[:-3]
    
    # Shift energy to match reference scale
    shift_enr = ref_enr.mean() - prd_enr.mean()
    prd_enr = prd_enr + shift_enr 

    predicts = (prd_enr, prd_frc)
    targets = (ref_enr, ref_frc) 
    
    loss_enr2, loss_frc2 = loss_value(predicts, targets)
    
    return loss_enr2, loss_frc2

    
def get_dataset_loss(state, dataset):
    """
    Compute average loss over entire dataset.
    
    Evaluates model on all batches and averages the results.
    Used for computing training and validation metrics.
    
    Args:
        state: Current model state
        dataset: List of batched graphs
        
    Returns:
        avg_loss_enr2: Average energy MSE across all batches
        avg_loss_frc2: Average force MSE across all batches
    """
    loss_mse = jnp.array([get_loss(state, x) 
                          for x in dataset]).mean(axis=0)

    return loss_mse[0], loss_mse[1]
    

def rase_trainer(json_data):
    """
    Main training loop for RACE (Restratified Atomic Cluster Expansion) model.
    
    This function orchestrates the entire training process:
    1. Load and prepare data
    2. Initialize model and optimizer
    3. Training loop with periodic evaluation
    4. Learning rate scheduling
    5. Checkpoint saving
    
    Args:
        json_data: Dictionary containing all configuration parameters:
            - fname_train/fname_val: Data file paths
            - cutoff: Neighbor list cutoff distance
            - nbatch: Batch size
            - element: List of atomic numbers
            - model: 'race_qml' or 'race' (classical)
            - hidden_irreps: Hidden layer configuration
            - NN: Neural network hyperparameters
                - learning_rate: Initial learning rate
                - nepoch: Number of training epochs
                - l2_lambda: L2 regularization strength
                - ema_decay: Exponential moving average decay
                - decay_factor: Learning rate decay factor
                - nsave: Checkpoint save frequency
                - restart: Whether to load from checkpoint
                
    Training features:
        - Multi-task learning (energy + forces)
        - L2 regularization with adaptive lambda
        - Learning rate decay on plateau
        - Checkpointing best model
        - Exponential moving average of parameters
        
    Output:
        Saves checkpoints to disk and logs training progress
    """
    # Open log file for training progress
    fout = open(json_data['train']['fname_log'], 'w', 1)
    
    # Load and prepare data
    x_train, x_test, uniq_element, enr_avg_per_element = \
          get_trajectory(json_data['fname_train'],
                            json_data['fname_val'],
                            json_data['cutoff'],
                            json_data['nbatch'],
                            json_data['element'],
                            json_data['enr_avg_per_element'])
    
    # Select model architecture (quantum or classical)
    if json_data["model"] in ['race_qml']:
        from bam_qml.model.race_qml_model import GraphNN
        print('using race_qml import')
    else:
        from bam_qml.model.race_model import GraphNN
        print('using race model import')

    # Configure model
    output_irreps = e3nn.Irreps("1x0e")
    hidden_irreps = e3nn.Irreps(json_data['hidden_irreps'])
    model = GraphNN(json_data['cutoff'],
                     json_data['avg_num_neighbors'],
                     json_data['num_species'],
                     num_basis_func=json_data['num_radial_basis'],
                     hidden_irreps=hidden_irreps,
                     nlayers=json_data['nlayers'],
                     features_dim=json_data['features_dim'],
                     output_irreps=output_irreps,
                     qml_circuit=json_data['qml_circuit'])
    
    # Initialize model parameters
    rng = jax.random.key(json_data['NN']['rng_seed'])
    rng, key = jax.random.split(rng)
    graphset = x_train[0]
    R = graphset.nodes['positions']
    cell = graphset.globals['cell']
    Rij = get_edge_relative_vectors(R, cell, graphset)
    params = model.init(key, Rij, graphset)['params']

    # Configure optimizer with gradient clipping and AMSGrad
    lr = json_data['NN']['learning_rate']
    ema_decay = json_data['NN']['ema_decay']
    opt_method = optax.chain(optax.clip_by_global_norm(0.5),
                              optax.clip(0.5),
                              optax.inject_hyperparams(optax.amsgrad)(learning_rate=lr))
    
    # Learning rate scheduling parameters
    decay_factor = json_data['NN']['decay_factor']
    lr_min = 0.2 * lr 
    itolerate = 0  # Patience counter for learning rate decay

    # Create training state
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=opt_method,
        ema_decay=ema_decay
    )

    # L2 regularization coefficient
    lambd = json_data['NN']['l2_lambda']
    print(date())
    
    # Initialize tracking and checkpointing
    loss_dict = {'train': [], 'test': []} 
    l_ckpt_saved = False
    ckpt = {
        'params': state.params,
        'opt_state': state.opt_state,
        'uniq_element': uniq_element,
        'enr_avg_per_element': enr_avg_per_element,
        'loss': loss_dict
    }
    
    # Optionally restart from checkpoint
    if json_data['NN']['restart']:
        ckpt = checkpoint_load(json_data['NN']['fname_pkl'])
        opt_state = ckpt['opt_state']
        state = state.replace(params=ckpt['params'],
                               opt_state=opt_state)    
        l_ckpt_saved = True
    
    # Initial evaluation
    loss_l2 = l2_regularization(state.params)
    loss_enr2_test, loss_frc2_test = \
        get_dataset_loss(state, x_test)
    
    # Track best model
    loss_test_min = loss_enr2_test + gamma_frc*loss_frc2_test + lambd * loss_l2
    loss_test_local = loss_test_min

    # Print header for training log
    print(f'                    \tTRAIN_loss____________________________________________| TEST_loss', file=fout)
    line = f"MM/DD/YYYY HH/MM/SS\t {'EPOCH':7}{'LOSS':14}{'RMSE_E':11}{'RMSE_F':11}{'L2':10}| "
    line = line + f"{'LOSS':14}{'RMSE_E':11}{'RMSE_F':11}|{'LR':7}"
    print(line, file=fout)
    print('----------------------------------------------------------------------------------', file=fout)

    print(date())

    # Main training loop
    nepoch = json_data['NN']['nepoch']
    for epch in range(nepoch): 
        # Train on all batches
        for graphset in x_train:
            state = train_step(state, graphset, lambd=lambd)
        
        # Periodic evaluation
        if (epch+1) % 5 == 0:
            # Compute current losses
            loss_l2 = l2_regularization(state.params)

            loss_enr2, loss_frc2 = get_dataset_loss(state, x_train)
            loss_enr2_test, loss_frc2_test = get_dataset_loss(state, x_test)
                
            loss = loss_enr2 + gamma_frc*loss_frc2 + lambd*loss_l2
            loss_test = loss_enr2_test + gamma_frc*loss_frc2_test + lambd*loss_l2
            
            # Adaptive L2 regularization (decay with loss magnitude)
            lambd = lambd * loss_l2
            
            # Save if best model so far
            if loss_test < loss_test_min:
                loss_test_min = loss_test 
                ckpt['params'] = state.params
                ckpt['opt_state'] = state.opt_state
                ckpt['loss'] = loss_dict
                l_ckpt_saved = False
                
            # Learning rate scheduling: decay if no improvement
            if loss_test < loss_test_local:
                itolerate = 0
                loss_test_local = loss_test
            else:
                itolerate = itolerate + 1

            lr = optax.tree_utils.tree_get(state.opt_state, "learning_rate")
            
            # Decay learning rate after 50 epochs without improvement
            if itolerate >= 50:
                lr_new = jnp.where(lr*decay_factor > lr_min, lr*decay_factor, lr_min)
                opt_state = optax.tree_utils.tree_set(state.opt_state, learning_rate=lr_new)
                state = state.replace(opt_state=opt_state)
                itolerate = 0
                lr = lr_new
                loss_test_local = loss_test + 0.01  # Small margin for continued improvement

            # Store losses
            loss_dict['train'].append(loss)
            loss_dict['test'].append(loss_test)

            # Compute RMSEs for interpretability
            rmse_E = jnp.sqrt(loss_enr2)
            rmse_F = jnp.sqrt(loss_frc2)
            
            rmse_E_test = jnp.sqrt(loss_enr2_test)
            rmse_F_test = jnp.sqrt(loss_frc2_test)
            
            # Log progress
            line = f'{date()}\t {epch+1:<7}{loss:<14.6f}{rmse_E:<11.6f}{rmse_F:<11.6f}{loss_l2:<10.5f}'
            line += f'{loss_test:<14.6f}{rmse_E_test:<11.6f}{rmse_F_test:<11.6f}{lr:<7.4f}'
            print(line, file=fout)

        # Periodic checkpointing
        if (epch+1) % json_data['NN']['nsave'] == 0 and \
            not l_ckpt_saved:
            checkpoint_save(json_data['NN']['fname_pkl'], ckpt)
            l_ckpt_saved = True
            
        # Clear JIT cache periodically to manage memory
        if (epch+1) % 20 == 0:
            train_step._clear_cache()
    

if __name__ == '__main__':
    """
    Command-line interface for training script.
    
    Usage:
        python -m bam_qml.training.race_trainer -i input.json
        
    Arguments:
        -i, --input: Path to JSON configuration file
        -h, --help: Display usage information
    """
    import getopt 
    import sys 

    argv = sys.argv[1:]
    opts, args = getopt.getopt(
        argv, "hi:", ["help=", "input="]
    )
    
    fname_json = "input_qml.json"
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("python -m bam_qml.training.race_trainer -i <input_file.json>")
            sys.exit(1)
        elif opt in ("-i", "--input"):
            fname_json = arg
    print('fname_json', fname_json)

    with open(fname_json) as f:
        json_data = json.load(f)
        
        # Configure JAX precision
        if json_data['float64']:
            jax.config.update("jax_enable_x64", True)
            print("running with float64")
        else:
            jax.config.update("jax_enable_x64", False)
            print("running with float32")
        
        # Run training
        rase_trainer(json_data)
        
    print(date())
