### Hybrid Quantum-Classical Machine Learning Potential with Variational Circuits

We introduce a hybrid model that combines a classical E(3)-equivariant message-passing machine learning potential with variational quantum circuits.

### Installation
To install the package, run:
```bash
pip install -e .
```

### How to use
#### 1. Traning
Sample input files are available in the `input_race` and `input_race_qml_cx` folders. 

You can also test other types of quantum circuits by chaning the value of `quantum circuit` to one of the following: `cz`, `qft_cx`, `qft_cz`, or `basic`. 

To start training a model, use: 
```bash
python -m bam_qml.training.race_trainer -i input_race_qml_cx/input.json &
```

#### 2. Molecular Dynamics (MD) simulations
* To test the trained model, run:
```bash
python -m bam_qml.jase.md_base -i input_race_qml_cx/input.json &
``` 

#### 3. Monte Carlo (MC) simulations
* To simulate the model on the real hardware -- accounting for noise in predicted values --, use:
```bash
python -m bam_qml.jase.mc_base -i input_race_qml_cx_noise/input.json &
``` 

### License
MIT License
