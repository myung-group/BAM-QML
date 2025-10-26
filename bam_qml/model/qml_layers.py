# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Quantum circuit layer templates for variational quantum algorithms.

This module provides customizable quantum circuit layers combining:
- Single-qubit rotations for parameterized feature encoding
- Two-qubit entangling gates for quantum correlations
- Support for Quantum Fourier Transform (QFT) auxiliary qubits

These layers can be used as building blocks in quantum machine learning models.
"""

# pylint: disable=consider-using-enumerate,too-many-arguments
from typing import Optional, Iterable, Any, List
import pennylane as qml
from pennylane.operation import Operation


# ============================================================================
# CONSTANTS
# ============================================================================

# Number of auxiliary qubits used in QFT-enhanced circuits
NUM_QFT_QUBITS = 3

# Gradient computation method (None means parameter-shift rule)
GRAD_METHOD = None


# ============================================================================
# BASE CLASS
# ============================================================================

class BaseQuantumLayer(Operation):
    """
    Abstract base class for quantum circuit layers.
    
    Provides common functionality for:
    - Weight tensor validation
    - Shape checking for batched/unbatched inputs
    - Rotation gate configuration
    
    All concrete layer classes should inherit from this base.
    """
    
    grad_method = GRAD_METHOD
    
    def __init__(
        self,
        weights,
        wires: Optional[Iterable] = None,
        rotation: Optional[Operation] = None,
        id: Optional[str] = None,
        num_auxiliary_qubits: int = 0
    ):
        """
        Initialize a quantum layer with trainable parameters.
        
        Args:
            weights: Trainable parameters with shape (n_layers, n_wires - num_auxiliary_qubits)
                    Can be 2D (single sample) or 3D (batched)
            wires: Quantum wires (qubits) this operation acts on
            rotation: Single-qubit rotation gate (default: RX)
            id: Custom identifier for this operation
            num_auxiliary_qubits: Number of qubits excluded from parameterization
                                 (used in QFT-enhanced circuits)
        
        Raises:
            ValueError: If weight tensor shape is incompatible with wire count
        """
        # Ensure weights are in proper tensor format
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        
        # Validate weight tensor dimensions
        self._validate_weights(weights, wires, num_auxiliary_qubits)
        
        # Store rotation gate (default to RX if not specified)
        self._hyperparameters = {"rotation": rotation or qml.RX}
        
        super().__init__(weights, wires=wires, id=id)
    
    def _validate_weights(
        self,
        weights,
        wires: Iterable,
        num_auxiliary_qubits: int
    ) -> None:
        """
        Validate weight tensor shape matches circuit requirements.
        
        Args:
            weights: Weight tensor to validate
            wires: Circuit wires
            num_auxiliary_qubits: Number of non-parameterized qubits
            
        Raises:
            ValueError: If tensor dimensions are incorrect
        """
        shape = qml.math.shape(weights)
        
        # Check rank: 2D (no batching) or 3D (batched)
        if len(shape) not in (2, 3):
            raise ValueError(
                f"Weights tensor must be 2D (no batching) or 3D (batched); "
                f"got {len(shape)}D with shape {shape}"
            )
        
        # Check last dimension matches parameterized qubit count
        expected_size = len(wires) - num_auxiliary_qubits
        if shape[-1] != expected_size:
            raise ValueError(
                f"Weights tensor last dimension must be {expected_size} "
                f"(len(wires) - num_auxiliary_qubits); got {shape[-1]}"
            )
    
    @property
    def num_params(self) -> int:
        """Number of trainable parameter tensors."""
        return 1
    
    @staticmethod
    def shape(n_layers: int, n_wires: int) -> tuple:
        """
        Compute required weight tensor shape.
        
        Args:
            n_layers: Number of circuit layers
            n_wires: Number of qubits (excluding auxiliary qubits if applicable)
            
        Returns:
            Tuple (n_layers, n_wires) specifying weight tensor shape
        """
        return (n_layers, n_wires)


# ============================================================================
# CONCRETE LAYER IMPLEMENTATIONS
# ============================================================================

class BasicLayers(BaseQuantumLayer):
    """
    Basic quantum layers with only single-qubit rotations (no entanglement).
    
    Circuit structure per layer:
        1. Apply parameterized rotation to each qubit
        
    This is the simplest variational circuit, useful as a baseline or
    when entanglement is not desired. The lack of entanglement means
    the quantum state remains separable, limiting expressiveness but
    potentially improving trainability.
    
    Mathematical form:
        U(θ) = ⊗ᵢ R(θᵢ)
        where R is the rotation gate (RX, RY, RZ, etc.)
    
    Args:
        weights: Trainable parameters of shape (n_layers, n_wires)
        wires: Quantum wires to act on
        rotation: Single-parameter rotation gate (default: RX)
        
    Example:
        >>> import pennylane as qml
        >>> dev = qml.device('default.qubit', wires=3)
        >>> @qml.qnode(dev)
        ... def circuit(weights):
        ...     BasicLayers(weights, wires=range(3))
        ...     return qml.expval(qml.PauliZ(0))
        >>> weights = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        >>> circuit(weights)
    """
    
    def __init__(
        self,
        weights,
        wires: Optional[Iterable] = None,
        rotation: Optional[Operation] = None,
        id: Optional[str] = None
    ):
        super().__init__(weights, wires, rotation, id, num_auxiliary_qubits=0)
    
    @staticmethod
    def compute_decomposition(
        weights,
        wires: Iterable,
        rotation: Operation
    ) -> List[Operation]:
        """
        Decompose the layer into elementary quantum gates.
        
        Creates a sequence of single-qubit rotations, one per layer per qubit.
        No entangling gates are applied in this basic version.
        
        Args:
            weights: Parameter tensor of shape (n_layers, n_wires)
            wires: Quantum wires
            rotation: Single-qubit rotation gate to apply
            
        Returns:
            List of quantum operations (only rotation gates)
            
        Circuit diagram (2 qubits, 1 layer):
            q0: ──R(θ₀)──
            q1: ──R(θ₁)──
        """
        n_layers = qml.math.shape(weights)[-2]
        n_wires = len(wires)
        
        operations = []
        for layer in range(n_layers):
            for qubit in range(n_wires):
                operations.append(
                    rotation(weights[..., layer, qubit], wires=wires[qubit:qubit + 1])
                )
        
        return operations


class BasicEntanglerCZLayers(BaseQuantumLayer):
    """
    Quantum layers with single-qubit rotations and CZ entangling gates.
    
    Circuit structure per layer:
        1. Apply parameterized rotation to each qubit
        2. Apply CZ gates in a ring topology
        
    The CZ (Controlled-Z) gate creates entanglement between qubits, allowing
    the circuit to represent correlations beyond separable states. The ring
    topology connects each qubit to its neighbor, with periodic boundary
    conditions (last qubit connects to first).
    
    CZ gate action:
        |00⟩ → |00⟩, |01⟩ → |01⟩, |10⟩ → |10⟩, |11⟩ → -|11⟩
        Equivalent to: CNOT conjugated by Hadamards on target
    
    Mathematical form:
        U(θ) = (⊗ᵢⱼ CZᵢⱼ) · (⊗ᵢ R(θᵢ))
        
    Args:
        weights: Trainable parameters of shape (n_layers, n_wires)
        wires: Quantum wires to act on
        rotation: Single-parameter rotation gate (default: RX)
        
    Notes:
        - CZ gates are symmetric (order of qubits doesn't matter)
        - Ring topology ensures all qubits are connected
        - For 2 qubits, periodic boundary is dropped (only 1 CZ gate)
        
    Example:
        >>> dev = qml.device('default.qubit', wires=4)
        >>> @qml.qnode(dev)
        ... def circuit(weights):
        ...     BasicEntanglerCZLayers(weights, wires=range(4))
        ...     return [qml.expval(qml.PauliZ(i)) for i in range(4)]
    """
    
    def __init__(
        self,
        weights,
        wires: Optional[Iterable] = None,
        rotation: Optional[Operation] = None,
        id: Optional[str] = None
    ):
        super().__init__(weights, wires, rotation, id, num_auxiliary_qubits=0)
    
    @staticmethod
    def compute_decomposition(
        weights,
        wires: Iterable,
        rotation: Operation
    ) -> List[Operation]:
        """
        Decompose into rotations followed by CZ entangling gates.
        
        Each layer consists of:
        1. Single-qubit rotations on all wires
        2. CZ gates connecting neighboring qubits in a ring
        
        Args:
            weights: Parameter tensor of shape (n_layers, n_wires)
            wires: Quantum wires
            rotation: Single-qubit rotation gate
            
        Returns:
            List of quantum operations (rotations + CZ gates)
            
        Circuit diagram (3 qubits, 1 layer):
            q0: ──R(θ₀)──●────────●──
                          │        │
            q1: ──R(θ₁)──●──●─────┼──
                             │     │
            q2: ──R(θ₂)─────●─────●──
            
            Where ● represents CZ gates
        """
        n_layers = qml.math.shape(weights)[-2]
        n_wires = len(wires)
        
        operations = []
        for layer in range(n_layers):
            # Step 1: Apply rotations to all qubits
            for qubit in range(n_wires):
                operations.append(
                    rotation(weights[..., layer, qubit], wires=wires[qubit:qubit + 1])
                )
            
            # Step 2: Apply CZ gates in ring topology
            for qubit in range(n_wires):
                # periodic_boundary=True creates ring: last connects to first
                wire_pair = wires.subset([qubit, qubit + 1], periodic_boundary=True)
                operations.append(qml.CZ(wires=wire_pair))
        
        return operations


class BasicQftEntanglerCXLayers(BaseQuantumLayer):
    """
    Quantum layers with CNOT entanglement and QFT auxiliary qubits.
    
    This variant is designed for circuits that use Quantum Fourier Transform
    (QFT) on auxiliary qubits. The main qubits are parameterized, while
    auxiliary qubits provide additional quantum resources for entanglement.
    
    Circuit structure per layer:
        1. Apply rotations to main qubits only (not QFT qubits)
        2. Apply CNOT gates across ALL qubits (main + auxiliary)
        
    The QFT auxiliary qubits are prepared in a Fourier basis state before
    this layer is applied, providing frequency-domain features that may
    improve model expressiveness.
    
    CNOT gate action:
        |00⟩ → |00⟩, |01⟩ → |01⟩, |10⟩ → |11⟩, |11⟩ → |10⟩
        Flips target qubit if control is |1⟩
    
    Args:
        weights: Parameters of shape (n_layers, n_main_qubits)
                where n_main_qubits = total_wires - NUM_QFT_QUBITS
        wires: All quantum wires (main + auxiliary)
        rotation: Single-parameter rotation gate (default: RX)
        
    Notes:
        - Only main qubits are parameterized
        - QFT qubits participate in entanglement but not rotation
        - Requires NUM_QFT_QUBITS auxiliary qubits
        
    Example:
        >>> # 8 main qubits + 3 QFT auxiliary = 11 total wires
        >>> dev = qml.device('default.qubit', wires=11)
        >>> @qml.qnode(dev)
        ... def circuit(weights):
        ...     # Prepare QFT state on auxiliary qubits [8, 9, 10]
        ...     qml.QFT(wires=range(8, 11))
        ...     # Apply parameterized layer
        ...     BasicQftEntanglerCXLayers(weights, wires=range(11))
        ...     return [qml.expval(qml.PauliZ(i)) for i in range(8)]
    """
    
    def __init__(
        self,
        weights,
        wires: Optional[Iterable] = None,
        rotation: Optional[Operation] = None,
        id: Optional[str] = None
    ):
        super().__init__(
            weights, wires, rotation, id,
            num_auxiliary_qubits=NUM_QFT_QUBITS
        )
    
    @staticmethod
    def compute_decomposition(
        weights,
        wires: Iterable,
        rotation: Operation
    ) -> List[Operation]:
        """
        Decompose into rotations on main qubits + CNOT gates on all qubits.
        
        Args:
            weights: Parameters of shape (n_layers, n_main_qubits)
            wires: All wires (main + auxiliary)
            rotation: Single-qubit rotation gate
            
        Returns:
            List of quantum operations
            
        Circuit diagram (8 main + 3 aux = 11 qubits, 1 layer):
            Main qubits (0-7):
            q0: ──R(θ₀)──●─────...
            q1: ──R(θ₁)──┼──●──...
            ...
            q7: ──R(θ₇)──┼──┼──...
            
            QFT auxiliary qubits (8-10):
            q8: ──────────●──┼──...  (no rotation, only entanglement)
            q9: ─────────────●──...
            q10: ────────────────...
        """
        n_layers = qml.math.shape(weights)[-2]
        n_main_qubits = len(wires) - NUM_QFT_QUBITS
        n_total_wires = len(wires)
        
        operations = []
        for layer in range(n_layers):
            # Step 1: Rotations on main qubits only
            for qubit in range(n_main_qubits):
                operations.append(
                    rotation(weights[..., layer, qubit], wires=wires[qubit:qubit + 1])
                )
            
            # Step 2: CNOT entanglement across all qubits (including auxiliary)
            for qubit in range(n_total_wires - 1):
                # No periodic boundary for QFT circuits
                wire_pair = wires.subset([qubit, qubit + 1], periodic_boundary=False)
                operations.append(qml.CNOT(wires=wire_pair))
        
        return operations


class BasicQftEntanglerCZLayers(BaseQuantumLayer):
    """
    Quantum layers with CZ entanglement and QFT auxiliary qubits.
    
    Similar to BasicQftEntanglerCXLayers but uses CZ gates instead of CNOTs.
    CZ gates may be preferable on certain quantum hardware due to lower
    error rates or native gate sets.
    
    Circuit structure per layer:
        1. Apply rotations to main qubits only
        2. Apply CZ gates across ALL qubits (main + auxiliary)
        
    CZ vs CNOT:
        - CZ is symmetric: CZ(i,j) = CZ(j,i)
        - CNOT has control/target distinction
        - CZ = H₂ · CNOT · H₂ (Hadamard on target)
        - For many algorithms, CZ and CNOT are interchangeable
    
    Args:
        weights: Parameters of shape (n_layers, n_main_qubits)
        wires: All quantum wires (main + auxiliary)
        rotation: Single-parameter rotation gate (default: RX)
        
    Example:
        >>> dev = qml.device('default.qubit', wires=11)
        >>> @qml.qnode(dev)
        ... def circuit(weights):
        ...     qml.QFT(wires=range(8, 11))
        ...     BasicQftEntanglerCZLayers(weights, wires=range(11))
        ...     return [qml.expval(qml.PauliZ(i)) for i in range(8)]
    """
    
    def __init__(
        self,
        weights,
        wires: Optional[Iterable] = None,
        rotation: Optional[Operation] = None,
        id: Optional[str] = None
    ):
        super().__init__(
            weights, wires, rotation, id,
            num_auxiliary_qubits=NUM_QFT_QUBITS
        )
    
    @staticmethod
    def compute_decomposition(
        weights,
        wires: Iterable,
        rotation: Operation
    ) -> List[Operation]:
        """
        Decompose into rotations on main qubits + CZ gates on all qubits.
        
        Args:
            weights: Parameters of shape (n_layers, n_main_qubits)
            wires: All wires (main + auxiliary)
            rotation: Single-qubit rotation gate
            
        Returns:
            List of quantum operations
            
        Circuit diagram: Same topology as BasicQftEntanglerCXLayers
        but with CZ gates (●) instead of CNOTs
        """
        n_layers = qml.math.shape(weights)[-2]
        n_main_qubits = len(wires) - NUM_QFT_QUBITS
        n_total_wires = len(wires)
        
        operations = []
        for layer in range(n_layers):
            # Step 1: Rotations on main qubits only
            for qubit in range(n_main_qubits):
                operations.append(
                    rotation(weights[..., layer, qubit], wires=wires[qubit:qubit + 1])
                )
            
            # Step 2: CZ entanglement across all qubits
            for qubit in range(n_total_wires - 1):
                wire_pair = wires.subset([qubit, qubit + 1], periodic_boundary=False)
                operations.append(qml.CZ(wires=wire_pair))
        
        return operations


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_layer_class(circuit_type: str) -> type:
    """
    Get the appropriate layer class for a given circuit type.
    
    Args:
        circuit_type: One of 'basic', 'cz', 'qft_cx', 'qft_cz'
        
    Returns:
        Layer class
        
    Raises:
        ValueError: If circuit_type is not recognized
    """
    layer_map = {
        'basic': BasicLayers,
        'cz': BasicEntanglerCZLayers,
        'qft_cx': BasicQftEntanglerCXLayers,
        'qft_cz': BasicQftEntanglerCZLayers
    }
    
    if circuit_type not in layer_map:
        raise ValueError(
            f"Unknown circuit type '{circuit_type}'. "
            f"Choose from: {list(layer_map.keys())}"
        )
    
    return layer_map[circuit_type]
