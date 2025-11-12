import torch
from typing import Callable

class Measurement:
    target: int
    output: int
    def __init__(self, target: int, output: int):
        self.target = target
        self.output = output

class Gate:
    tensor: torch.Tensor
    targets: list[int]

    def __init__(self, tensor: torch.Tensor, targets: list[int]):
        self.tensor = tensor
        self.targets = targets

class ConditionalGate:
    gate: Gate
    classical_target: int

    def __init__(self, gate: Gate, classical_target: int):
        self.gate = gate
        self.classical_target = classical_target

# Base class for single-qubit gates
class SingleQubitGate:
    _matrix: torch.Tensor

    def __init__(self, matrix: torch.Tensor):
        self._matrix = matrix

    def __call__(self, target: int) -> Gate:
        return Gate(self._matrix, [target])

# Base class for two-qubit gates
class TwoQubitGate:
    _matrix: torch.Tensor

    def __init__(self, matrix: torch.Tensor):
        self._matrix = matrix

    def __call__(self, targets: list[int]) -> Gate:
        if len(targets) != 2:
            raise ValueError(f"Two-qubit gate requires exactly 2 targets, got {len(targets)}")
        return Gate(self._matrix, targets)

# Base class for three-qubit gates
class ThreeQubitGate:
    _matrix: torch.Tensor

    def __init__(self, matrix: torch.Tensor):
        self._matrix = matrix

    def __call__(self, targets: list[int]) -> Gate:
        if len(targets) != 3:
            raise ValueError(f"Three-qubit gate requires exactly 3 targets, got {len(targets)}")
        return Gate(self._matrix, targets)

# Base class for parametric single-qubit gates
class ParametricSingleQubitGate:
    _matrix_fn: Callable[[torch.Tensor], torch.Tensor]

    def __init__(self, matrix_fn: Callable[[torch.Tensor], torch.Tensor]):
        self._matrix_fn = matrix_fn

    def __call__(self, theta: torch.Tensor) -> SingleQubitGate:
        """Bind the parameter and return a SingleQubitGate that can be applied to a target."""
        matrix = self._matrix_fn(theta)
        return SingleQubitGate(matrix)

# identity gate
I = SingleQubitGate(torch.tensor(
    [[1, 0],
     [0, 1]],
    dtype=torch.complex64))

# hadamard gate
H = SingleQubitGate((1 / torch.sqrt(torch.tensor(2.0))) * torch.tensor(
    [[1, 1],
     [1, -1]],
    dtype=torch.complex64))

# phase gate
S = SingleQubitGate(torch.tensor(
    [[1, 0],
     [0, 1j]],
    dtype=torch.complex64))

# pi/8 aka T gate
T = SingleQubitGate(torch.tensor(
    [[1, 0],
     [0, torch.exp(torch.tensor(1j * torch.pi / 4))]],
    dtype=torch.complex64))

# pauli-x gate
X = SingleQubitGate(torch.tensor(
    [[0, 1],
     [1, 0]],
    dtype=torch.complex64))

# pauli-y gate
Y = SingleQubitGate(torch.tensor(
    [[0, -1j],
     [1j, 0]],
    dtype=torch.complex64))

# pauli-z gate
Z = SingleQubitGate(torch.tensor(
    [[1, 0],
     [0, -1]],
    dtype=torch.complex64))

# rotate X gate
RX = ParametricSingleQubitGate(lambda theta: torch.tensor(
    [[torch.cos(theta / 2), -1j * torch.sin(theta / 2)],
    [-1j * torch.sin(theta / 2), torch.cos(theta / 2)]],
    dtype=torch.complex64))

# rotate Y gate
RY = ParametricSingleQubitGate(lambda theta: torch.tensor(
    [[torch.cos(theta / 2), -torch.sin(theta / 2)],
    [torch.sin(theta / 2), torch.cos(theta / 2)]],
    dtype=torch.complex64))

# rotate Z gate
RZ = ParametricSingleQubitGate(lambda theta: torch.tensor(
    [[torch.cos(theta / 2) - 1j * torch.sin(theta / 2), 0],
    [0, torch.cos(theta / 2) + 1j * torch.sin(theta / 2)]],
    dtype=torch.complex64))

# CNOT aka CX
CX = TwoQubitGate(torch.tensor(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 1, 0]],
    dtype=torch.complex64))

CZ = TwoQubitGate(torch.tensor(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, -1]],
    dtype=torch.complex64))

SWAP = TwoQubitGate(torch.tensor(
    [[1, 0, 0, 0],
     [0, 0, 1, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1]],
    dtype=torch.complex64))

# CCNOT aka Toffoli aka CCX
CCX = ThreeQubitGate(torch.tensor(
    [[1, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0, 1, 0]],
    dtype=torch.complex64))

# Controlled-U gate, ie. controlled version of any other gate
class ControlledGate:
    def __call__(self, gate_matrix: torch.Tensor, targets: list[int]) -> Gate:
        if len(targets) != 2:
            raise ValueError(f"Controlled gate requires exactly 2 targets (control, target), got {len(targets)}")
        controlled_matrix = torch.tensor(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, gate_matrix[0, 0], gate_matrix[0, 1]],
             [0, 0, gate_matrix[1, 0], gate_matrix[1, 1]]],
            dtype=torch.complex64)
        return Gate(controlled_matrix, targets)

Controlled = ControlledGate()
