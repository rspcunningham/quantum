from __future__ import annotations

import torch
import math
from collections.abc import Iterator, Sequence
from typing import Callable, cast, overload, override

def _complex_matrix(data: list[list[complex | int | float]]) -> torch.Tensor:
    return torch.tensor(data, dtype=torch.complex64)

expand_diagonal = cast(Callable[..., torch.Tensor], torch.block_diag)


class QuantumRegister:
    """A named group of qubits with auto-assigned indices."""
    _offset: int
    _size: int

    def __init__(self, size: int, offset: int = 0):
        self._offset = offset
        self._size = size

    @overload
    def __getitem__(self, key: int) -> int: ...
    @overload
    def __getitem__(self, key: slice) -> QuantumRegister: ...
    def __getitem__(self, key: int | slice) -> int | QuantumRegister:
        if isinstance(key, int):
            if key < 0:
                key = self._size + key
            if key < 0 or key >= self._size:
                raise IndexError(f"Register index {key} out of range [0, {self._size})")
            return self._offset + key
        # key is a slice
        indices = range(self._size)[key]
        if len(indices) == 0:
            return QuantumRegister(0, self._offset)
        return QuantumRegister(len(indices), self._offset + indices[0])

    def __iter__(self) -> Iterator[int]:
        return iter(range(self._offset, self._offset + self._size))

    def __len__(self) -> int:
        return self._size


def registers(*sizes: int) -> tuple[QuantumRegister, ...]:
    """Create multiple contiguous quantum registers with auto-assigned indices.

    input_reg, working_reg, ancilla = registers(4, 4, 1)
    # input_reg: qubits 0-3, working_reg: qubits 4-7, ancilla: qubit 8
    """
    offset = 0
    regs: list[QuantumRegister] = []
    for size in sizes:
        regs.append(QuantumRegister(size, offset=offset))
        offset += size
    return tuple(regs)


class Gate:
    tensor: torch.Tensor
    targets: list[int]

    def __init__(self, tensor: torch.Tensor, *targets: int):
        if len(targets) != math.log2(tensor.shape[0]):
            raise ValueError(f"Number of targets ({len(targets)}) does not match rank ({math.log2(tensor.shape[0])})")

        self.tensor = tensor
        self.targets = list(targets)

    def if_(self, classical_bit: int) -> ConditionalGate:
        """Make this gate conditional on a classical bit being 1.

        Usage: H(0).if_(classical_bit=0)
        """
        return ConditionalGate(self, classical_bit)

    def __add__(self, other: Gate | Measurement | ConditionalGate | Circuit) -> Circuit:
        return Circuit([self, other])

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Gate):
            return False
        return bool(torch.allclose(self.tensor, other.tensor)) and self.targets == other.targets

    @override
    def __hash__(self) -> int:
        data: list[float] = [float(x) for x in self.tensor.flatten()]
        return hash((tuple(data), tuple(self.targets)))

class GateType:
    tensor: torch.Tensor

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __call__(self, *targets: int) -> Gate:
        return Gate(self.tensor, *targets)

    def on(self, *qubits: int | QuantumRegister) -> Circuit:
        """Apply this single-qubit gate to each qubit independently.

        H.on(qr)         # all qubits in register
        H.on(qr[0:3])    # slice of register
        H.on(0, 1, 2)    # raw ints still work
        """
        if self.tensor.shape[0] != 2:
            raise ValueError("on() is only supported for single-qubit gates")
        expanded: list[int] = []
        for q in qubits:
            if isinstance(q, QuantumRegister):
                expanded.extend(q)
            else:
                expanded.append(q)
        return Circuit([Gate(self.tensor, q) for q in expanded])

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

    def __init__(self, base_gate: GateType | ControlledGateType):
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

    def __add__(self, other: Gate | Measurement | ConditionalGate | Circuit) -> Circuit:
        return Circuit([self, other])

class ConditionalGate:
    """A gate that only executes if the classical register equals the condition value."""
    gate: Gate
    condition: int

    def __init__(self, gate: Gate, condition: int):
        self.gate = gate
        self.condition = condition

    def __add__(self, other: Gate | Measurement | ConditionalGate | Circuit) -> Circuit:
        return Circuit([self, other])


class Circuit:
    operations: list[Gate | ConditionalGate | Measurement | Circuit]

    def __init__(self, operations: Sequence[Gate | ConditionalGate | Measurement | Circuit]):
        self.operations = list(operations)

    def inverse(self) -> Circuit:
        reversed_operations = self.operations[::-1]
        new_operations: list[Gate | ConditionalGate | Measurement | Circuit] = []
        for op in reversed_operations:
            if isinstance(op, Circuit):
                new_operations.append(op.inverse())
            else:
                new_operations.append(op)

        return Circuit(new_operations)

    def __add__(self, other: Gate | Measurement | ConditionalGate | Circuit) -> Circuit:
        return Circuit([self, other])

    def __radd__(self, other: Gate | Measurement | ConditionalGate) -> Circuit:
        return Circuit([other, self])

    def __mul__(self, n: int) -> Circuit:
        return Circuit([self] * n)

    def __rmul__(self, n: int) -> Circuit:
        return Circuit([self] * n)

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Circuit):
            return NotImplemented
        return self.operations == other.operations
