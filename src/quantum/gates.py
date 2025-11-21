import torch
import math
from typing import Callable, cast

def _complex_matrix(data: list[list[complex | int | float]]) -> torch.Tensor:
    return torch.tensor(data, dtype=torch.complex64)

expand_diagonal = cast(Callable[..., torch.Tensor], torch.block_diag)

class Gate:
    tensor: torch.Tensor
    targets: list[int]

    def __init__(self, tensor: torch.Tensor, *targets: int):
        if len(targets) != math.log2(tensor.shape[0]):
            raise ValueError(f"Number of targets ({len(targets)}) does not match rank ({math.log2(tensor.shape[0])})")

        self.tensor = tensor
        self.targets = list(targets)

    def if_(self, classical_bit: int) -> 'ConditionalGate':
        """Make this gate conditional on a classical bit being 1.

        Usage: H(0).if_(classical_bit=0)
        """
        return ConditionalGate(self, classical_bit)

    def __eq__(self, other):
        if not isinstance(other, Gate):
            return False
        return torch.allclose(self.tensor, other.tensor) and self.targets == other.targets

    def __hash__(self):
        return hash((tuple(self.tensor.flatten().tolist()), tuple(self.targets)))

class GateType:
    tensor: torch.Tensor

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __call__(self, *targets: int) -> Gate:
        return Gate(self.tensor, *targets)

class ParametricGateType:
    matrix_fn: Callable[[torch.Tensor], torch.Tensor]

    def __init__(self, matrix_fn: Callable[[torch.Tensor], torch.Tensor]):
        self.matrix_fn = matrix_fn

    def __call__(self, param: float):
        # When called with theta, return a GateType that can then be called with targets
        param_tensor = torch.tensor(param, dtype=torch.float32)
        matrix = self.matrix_fn(param_tensor)
        return GateType(matrix)

class ControlledGateType:
    tensor: torch.Tensor

    def __init__(self, base_gate: 'GateType | ControlledGateType'):
        # Controlled gate = identity on control=0, base gate on control=1
        # This expands the gate matrix by one qubit
        eye = torch.eye(base_gate.tensor.shape[0], dtype=torch.complex64)
        self.tensor = expand_diagonal(eye, base_gate.tensor)

    def __call__(self, *targets: int) -> Gate:
        return Gate(self.tensor, *targets)


I = GateType(_complex_matrix([[1, 0], [0, 1]]))
H = GateType(_complex_matrix([[1, 1], [1, -1]]) / math.sqrt(2))
X = GateType(_complex_matrix([[0, 1], [1, 0]]))
Y = GateType(_complex_matrix([[0, -1j], [1j, 0]]))
Z = GateType(_complex_matrix([[1, 0], [0, -1]]))
S = GateType(_complex_matrix([[1, 0], [0, 1j]]))
T = GateType(_complex_matrix([[1, 0], [0, (1 + 1j) / math.sqrt(2)]]))

# Parametric rotation gates
RX = ParametricGateType(lambda theta: torch.tensor(
    [[torch.cos(theta / 2), -1j * torch.sin(theta / 2)],
     [-1j * torch.sin(theta / 2), torch.cos(theta / 2)]],
    dtype=torch.complex64))

RY = ParametricGateType(lambda theta: torch.tensor(
    [[torch.cos(theta / 2), -torch.sin(theta / 2)],
     [torch.sin(theta / 2), torch.cos(theta / 2)]],
    dtype=torch.complex64))

RZ = ParametricGateType(lambda theta: torch.tensor(
    [[torch.exp(-1j * theta / 2), 0],
     [0, torch.exp(1j * theta / 2)]],
    dtype=torch.complex64))

# Common controlled gates
CX = ControlledGateType(X)   # Controlled-NOT (CNOT)
CCX = ControlledGateType(CX)  # Toffoli gate (CCNOT)

class Measurement:
    qubit: int
    bit: int
    def __init__(self, qubit: int, bit: int):
        self.qubit = qubit
        self.bit = bit

class ConditionalGate:
    """A gate that only executes if the classical register equals the condition value."""
    gate: Gate
    condition: int

    def __init__(self, gate: Gate, condition: int):
        self.gate = gate
        self.condition = condition
