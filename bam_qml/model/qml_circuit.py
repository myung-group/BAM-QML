import pennylane as qml
from bam_qml.model.qml_layers import (BasicLayers,
                                      BasicEntanglerCZLayers,
                                      BasicQftEntanglerCXLayers,
                                      BasicQftEntanglerCZLayers)

# Quantum circuit configuration
num_qubits = 8  # Number of physical qubits
num_qft = 3  # Number of auxiliary qubits for Quantum Fourier Transform

# Create quantum devices (simulators)
dev = qml.device("default.qubit", wires=num_qubits)


@qml.qnode(dev)
def circuit_basic(num_qubits, param_feat, param_ansatz):
    """
    Basic quantum circuit without entanglement.

    This is the simplest quantum circuit that only applies single-qubit
    rotations without any entangling gates. Useful as a baseline.

    Args:
        num_qubits: Number of qubits (must match device)
        param_feat: Feature parameters to encode [n_samples, num_qubits]
        param_ansatz: Trainable ansatz parameters [n_layers, num_qubits]

    Returns:
        List of expectation values <Z_i> for each qubit i

    Circuit structure:
        1. AngleEmbedding: Encode classical features as qubit rotations
        2. BasicLayers: Apply trainable single-qubit rotations (no entanglement)
        3. Measure: Pauli-Z expectation values
    """
    # Feature map: encode input data into quantum state
    qml.AngleEmbedding(features=param_feat, wires=range(num_qubits), rotation='Y')

    # Ansatz: trainable quantum circuit (no entanglement in this version)
    BasicLayers(weights=param_ansatz, wires=range(num_qubits))

    # Measurements: return expectation values in computational basis
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_qubits)]


@qml.qnode(dev)
def circuit_cx(num_qubits, param_feat, param_ansatz):
    """
    Quantum circuit with CNOT entanglement (BasicEntanglerLayers).

    This circuit uses CNOT gates to create entanglement between qubits,
    allowing the quantum state to capture correlations between features.

    Args:
        num_qubits: Number of qubits
        param_feat: Feature parameters [n_samples, num_qubits]
        param_ansatz: Trainable ansatz parameters [n_layers, num_qubits]

    Returns:
        List of expectation values <Z_i> for each qubit

    Circuit structure:
        1. AngleEmbedding: Encode features as Y-rotations
        2. BasicEntanglerLayers: Trainable rotations + ring of CNOTs
        3. Measure: Pauli-Z expectation values

    Note: BasicEntanglerLayers creates a ring of CNOTs connecting
    neighboring qubits: 0→1, 1→2, ..., (n-1)→0
    """
    # Feature map
    qml.AngleEmbedding(features=param_feat, wires=range(num_qubits), rotation='Y')

    # Ansatz with CNOT entanglement
    qml.BasicEntanglerLayers(weights=param_ansatz, wires=range(num_qubits))

    # Measurements
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_qubits)]


@qml.qnode(dev)
def circuit_cz(num_qubits, param_feat, param_ansatz):
    """
    Quantum circuit with CZ (Controlled-Z) entanglement.

    Similar to circuit_cx but uses CZ gates instead of CNOT. CZ gates
    are phase gates that are more natural for some quantum hardware.

    Args:
        num_qubits: Number of qubits
        param_feat: Feature parameters [n_samples, num_qubits]
        param_ansatz: Trainable ansatz parameters [n_layers, num_qubits]

    Returns:
        List of expectation values <Z_i> for each qubit

    CZ gate: Applies phase flip to |11⟩ state
    |00⟩ → |00⟩, |01⟩ → |01⟩, |10⟩ → |10⟩, |11⟩ → -|11⟩
    """
    # Feature map
    qml.AngleEmbedding(features=param_feat, wires=range(num_qubits), rotation='Y')

    # Ansatz with CZ entanglement
    BasicEntanglerCZLayers(weights=param_ansatz, wires=range(num_qubits))

    # Measurements
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_qubits)]


# Device with additional qubits for QFT
dev_qft = qml.device("default.qubit", wires=num_qubits+num_qft)


@qml.qnode(dev_qft)
def circuit_qft_cx(num_qubits, param_feat, param_ansatz):
    """
    Quantum circuit with Quantum Fourier Transform and CNOT entanglement.

    The Quantum Fourier Transform (QFT) is a quantum analog of the discrete
    Fourier transform. It can extract frequency-domain features from quantum
    states, potentially improving model expressiveness.

    Args:
        num_qubits: Number of primary qubits for features
        param_feat: Feature parameters [n_samples, num_qubits]
        param_ansatz: Trainable parameters [n_layers, num_qubits+num_qft]

    Returns:
        List of expectation values <Z_i> for primary qubits only

    Circuit structure:
        1. AngleEmbedding: Encode features in primary qubits
        2. QFT: Apply quantum Fourier transform to auxiliary qubits
        3. BasicQftEntanglerCXLayers: Entangle all qubits (primary + auxiliary)
        4. Measure: Only primary qubits (auxiliary provide extra entanglement)

    The auxiliary qubits in QFT state provide additional quantum resources
    that may help capture complex patterns.
    """
    # Feature map on primary qubits
    qml.AngleEmbedding(features=param_feat, wires=range(num_qubits), rotation='Y')

    # QFT on auxiliary qubits [num_qubits, num_qubits+num_qft)
    qml.QFT(wires=range(num_qubits, num_qubits+num_qft))

    # Ansatz with entanglement across all qubits
    BasicQftEntanglerCXLayers(weights=param_ansatz, wires=range(num_qubits+num_qft))

    # Measure only primary qubits (auxiliary qubits enhance entanglement)
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_qubits)]


@qml.qnode(dev_qft)
def circuit_qft_cz(num_qubits, param_feat, param_ansatz):
    """
    Quantum circuit with QFT and CZ entanglement.

    Combines Quantum Fourier Transform with CZ-based entangling layers.
    Similar to circuit_qft_cx but uses CZ gates.

    Args:
        num_qubits: Number of primary qubits
        param_feat: Feature parameters [n_samples, num_qubits]
        param_ansatz: Trainable parameters [n_layers, num_qubits+num_qft]

    Returns:
        List of expectation values <Z_i> for primary qubits

    This variant may be preferable on hardware where CZ gates have
    lower error rates than CNOT gates.
    """
    # Feature map
    qml.AngleEmbedding(features=param_feat, wires=range(num_qubits), rotation='Y')

    # QFT on auxiliary qubits
    qml.QFT(wires=range(num_qubits, num_qubits+num_qft))

    # Ansatz with CZ entanglement
    BasicQftEntanglerCZLayers(weights=param_ansatz, wires=range(num_qubits+num_qft))

    # Measurements
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_qubits)]


# Device for noisy simulation
dev_noise = qml.device("default.qubit", wires=num_qubits, shots=50000)


@qml.qnode(dev_noise)
def circuit_cx_noise(num_qubits, param_feat, param_ansatz):
    """
    Quantum circuit with finite sampling (shot noise).

    This circuit is identical to circuit_cx but uses finite sampling
    (50000 shots) to simulate measurement noise. This is important for
    understanding how the model performs on real quantum hardware where
    perfect measurements are impossible.

    Args:
        num_qubits: Number of qubits
        param_feat: Feature parameters [n_samples, num_qubits]
        param_ansatz: Trainable parameters [n_layers, num_qubits]

    Returns:
        List of expectation values <Z_i> estimated from finite samples

    Differences from circuit_cx:
        - Uses finite shots (50000) instead of exact simulation
        - Results have sampling noise (statistical uncertainty)
        - More realistic model of quantum hardware behavior

    Statistical uncertainty scales as 1/sqrt(shots), so with 50000 shots
    the standard error is approximately 1/sqrt(50000) ≈ 0.0045
    """
    # Feature map
    qml.AngleEmbedding(features=param_feat, wires=range(num_qubits), rotation='Y')

    # Ansatz with CNOT entanglement
    qml.BasicEntanglerLayers(weights=param_ansatz, wires=range(num_qubits))

    # Measurements with shot noise
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_qubits)]
