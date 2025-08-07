### Hybrid Quantum-Classical Machine Learning Potential with Variational Circuits

This code integrates a classical E(3)-equivariant message-passing machine learning potential with variational quantum circuits. 
The classical component ensures rotational, translational, and reflectional equivariant in the learned potential, while the quantum component introduces trainable quantum gates that can capture complex correlations in atomic local chemical environments. 


### Installation
Before using the packages, ensure that Python 3.9+ and the required dependences (jax, flax, optax, e3nn-jax, matscipy, and pennylane) are installed.
You may install the package in editable mode to enable direct modification of the source code:
```bash
pip install -e .
```

### How to use
#### 1. Traning the Hybrid Model

Sample input files are provided in two directories:
* `input_race` -- for training without quantum components.
* `input_race_qml_cx` -- for training with quantum components using the CX (controlled-X) gate.

If you wish to experiment with different quantum circuits, you can modify `quantum circuit` setting in the configuration file (`input.json`). Supported options include:
* `cz` -- controlled-Z gate.
* `qft_cx` -- quantum Fourier transform using CX gates.
* `qft_cz` -- quantum Fourier transform using CZ gates.
* `basic` -- a simple baseline circuit without entangling gates. 

To initiate training with the CX-based circuit configuration:
```bash
python -m bam_qml.training.race_trainer -i input_race_qml_cx/input.json &
```
The `input.json` file specifies parameters such as learning rate, batch size, number of epoches, and quantum circuit.

#### 2. Molecular Dynamics (MD) simulations
Once the model is trained, its performance can be evaluated through molecular dynamics simulations. These simulations propagate atomic trajectories using forces predicted by the trained hybrid potential. To run MD simulations with the hybrid model using the CX gate:
```bash
python -m bam_qml.jase.md_base -i input_race_qml_cx/input.json &
``` 
This step helps verify whether the model conserves energy, reproduces expected structural dynamics, and remains stable over extended simulation times.

#### 3. Monte Carlo (MC) simulations
To explore model behavior under realistic quantum noise, you can perform Monte Carlo simulations. 
This approach is useful when deploying the model on real quantum hardware, where measurement errors and gate imperfections can affect predictions.
In this case, the simulation introduces stochatic noise into the predicted energies. Run the MC simulations with:
```bash
python -m bam_qml.jase.mc_base -i input_race_qml_cx_noise/input.json &
``` 

### License
MIT License
