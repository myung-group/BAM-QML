import pennylane as qml 
from bam_qml.model.qml_basic_layers import BasicLayers
from bam_qml.model.qml_basic_entangler_cz import BasicEntanglerCZLayers
from bam_qml.model.qml_basic_qft_entangler_cx import BasicQftEntanglerCXLayers
from bam_qml.model.qml_basic_qft_entangler_cz import BasicQftEntanglerCZLayers

num_qubits=8 
num_qft = 3 # quantum fourier transform (QFT)
dev = qml.device ("default.qubit", wires=num_qubits)

@qml.qnode (dev)
def circuit_basic (num_qubits, param_feat, param_ansatz):
    # feature map
    qml.AngleEmbedding (features=param_feat, wires=range(num_qubits), rotation='Y')

    # ansatz map
    BasicLayers (weights=param_ansatz, wires=range(num_qubits))

    # measures
    return [qml.expval(qml.PauliZ(wires=i)) for i in range (num_qubits)]



@qml.qnode (dev)
def circuit_cx (num_qubits, param_feat, param_ansatz):
    # feature map
    qml.AngleEmbedding (features=param_feat, wires=range(num_qubits), rotation='Y')

    # ansatz map
    qml.BasicEntanglerLayers (weights=param_ansatz, wires=range(num_qubits))

    # measures
    return [qml.expval(qml.PauliZ(wires=i)) for i in range (num_qubits)]

@qml.qnode (dev)
def circuit_cz (num_qubits, param_feat, param_ansatz):
    # feature map
    qml.AngleEmbedding (features=param_feat, wires=range(num_qubits), rotation='Y')

    # ansatz map
    BasicEntanglerCZLayers (weights=param_ansatz, wires=range(num_qubits))

    # measures
    return [qml.expval(qml.PauliZ(wires=i)) for i in range (num_qubits)]

dev_qft = qml.device ("default.qubit", wires=num_qubits+num_qft)
@qml.qnode (dev_qft)
def circuit_qft_cx (num_qubits, param_feat, param_ansatz):
    # feature map
    qml.AngleEmbedding (features=param_feat, wires=range(num_qubits), rotation='Y')

    # QFT
    qml.QFT (wires=range(num_qubits,num_qubits+num_qft))  # [num_qubits: num_qubits + num_nqft]
    
    # ansatz map
    BasicQftEntanglerCXLayers (weights=param_ansatz, wires=range(num_qubits+num_qft))

    # measures
    return [qml.expval(qml.PauliZ(wires=i)) for i in range (num_qubits)]


@qml.qnode (dev_qft)
def circuit_qft_cz (num_qubits, param_feat, param_ansatz):
    # feature map
    qml.AngleEmbedding (features=param_feat, wires=range(num_qubits), rotation='Y')

    # QFT
    qml.QFT (wires=range(num_qubits,num_qubits+num_qft))  # [num_qubits: num_qubits + num_nqft]
    
    # ansatz map
    BasicQftEntanglerCZLayers (weights=param_ansatz, wires=range(num_qubits+num_qft))

    # measures
    return [qml.expval(qml.PauliZ(wires=i)) for i in range (num_qubits)]


dev_noise = qml.device ("default.qubit", wires=num_qubits, shots=50000)
@qml.qnode (dev_noise)
def circuit_cx_noise (num_qubits, param_feat, param_ansatz):
    # feature map
    qml.AngleEmbedding (features=param_feat, wires=range(num_qubits), rotation='Y')

    # ansatz map
    qml.BasicEntanglerLayers (weights=param_ansatz, wires=range(num_qubits))

    # measures
    return [qml.expval(qml.PauliZ(wires=i)) for i in range (num_qubits)]
