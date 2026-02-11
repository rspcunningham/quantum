from __future__ import annotations

import torch
import math
from collections.abc import Iterator, Sequence
from typing import Callable, cast, overload, override

def _complex_matrix(data: list[list[complex | int | float]]) -> torch.Tensor:
    return torch.tensor(data, dtype=torch.complex64)

expand_diagonal = cast(Callable[..., torch.Tensor], torch.block_diag)


def _num_targets_for_dimension(dim: int) -> int:
    if dim <= 0 or (dim & (dim - 1)) != 0:
        raise ValueError(f"Gate dimension must be a positive power of two, got {dim}")
    return dim.bit_length() - 1


def _infer_diagonal(tensor: torch.Tensor) -> torch.Tensor | None:
    """Return diagonal entries if tensor is diagonal, otherwise None."""
    if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
        return None
    dim = tensor.shape[0]
    if dim == 1:
        return tensor.reshape(1)

    mask = ~torch.eye(dim, dtype=torch.bool, device=tensor.device)
    off_diag = tensor.masked_select(mask)
    if torch.allclose(off_diag, torch.zeros_like(off_diag)):
        return torch.diagonal(tensor)
    return None


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
    _tensor: torch.Tensor | None
    _diagonal: torch.Tensor | None
    targets: list[int]

    def __init__(self, tensor: torch.Tensor | None, *targets: int, diagonal: torch.Tensor | None = None):
        if tensor is None and diagonal is None:
            raise ValueError("Gate must define either a dense matrix or diagonal entries")

        dim: int
        if tensor is not None:
            if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
                raise ValueError(f"Gate matrix must be square, got shape {tuple(tensor.shape)}")
            dim = int(tensor.shape[0])
        else:
            assert diagonal is not None
            if diagonal.ndim != 1:
                raise ValueError(f"Gate diagonal must be 1D, got shape {tuple(diagonal.shape)}")
            dim = int(diagonal.shape[0])

        if diagonal is None and tensor is not None:
            diagonal = _infer_diagonal(tensor)

        if diagonal is not None and diagonal.ndim != 1:
            raise ValueError(f"Gate diagonal must be 1D, got shape {tuple(diagonal.shape)}")
        if diagonal is not None and int(diagonal.shape[0]) != dim:
            raise ValueError(f"Gate diagonal length ({int(diagonal.shape[0])}) must match gate dimension ({dim})")

        n_targets = _num_targets_for_dimension(dim)
        if len(targets) != n_targets:
            raise ValueError(f"Number of targets ({len(targets)}) does not match rank ({n_targets})")

        self._tensor = tensor
        self._diagonal = diagonal
        self.targets = list(targets)

    @property
    def tensor(self) -> torch.Tensor:
        if self._tensor is None:
            assert self._diagonal is not None
            self._tensor = torch.diag(self._diagonal)
        return self._tensor

    @tensor.setter
    def tensor(self, value: torch.Tensor) -> None:
        self._tensor = value
        self._diagonal = _infer_diagonal(value)

    @property
    def diagonal(self) -> torch.Tensor | None:
        return self._diagonal

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
        data = tuple(self.tensor.flatten().tolist())
        return hash((tuple(data), tuple(self.targets)))

class GateType:
    _tensor: torch.Tensor | None
    _diagonal: torch.Tensor | None

    def __init__(self, tensor: torch.Tensor | None = None, *, diagonal: torch.Tensor | None = None):
        if tensor is None and diagonal is None:
            raise ValueError("GateType must define either a dense matrix or diagonal entries")
        if tensor is not None and (tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]):
            raise ValueError(f"GateType matrix must be square, got shape {tuple(tensor.shape)}")
        if diagonal is not None and diagonal.ndim != 1:
            raise ValueError(f"GateType diagonal must be 1D, got shape {tuple(diagonal.shape)}")

        if diagonal is None and tensor is not None:
            diagonal = _infer_diagonal(tensor)

        if diagonal is not None and tensor is not None and int(diagonal.shape[0]) != int(tensor.shape[0]):
            raise ValueError("GateType diagonal length must match matrix dimension")

        if tensor is not None:
            _num_targets_for_dimension(int(tensor.shape[0]))
        else:
            assert diagonal is not None
            _num_targets_for_dimension(int(diagonal.shape[0]))

        self._tensor = tensor
        self._diagonal = diagonal

    @property
    def tensor(self) -> torch.Tensor:
        if self._tensor is None:
            assert self._diagonal is not None
            self._tensor = torch.diag(self._diagonal)
        return self._tensor

    @property
    def diagonal(self) -> torch.Tensor | None:
        return self._diagonal

    @property
    def dimension(self) -> int:
        if self._tensor is not None:
            return int(self._tensor.shape[0])
        assert self._diagonal is not None
        return int(self._diagonal.shape[0])

    def __call__(self, *targets: int) -> Gate:
        return Gate(self._tensor, *targets, diagonal=self._diagonal)

    def on(self, *qubits: int | QuantumRegister) -> Circuit:
        """Apply this single-qubit gate to each qubit independently.

        H.on(qr)         # all qubits in register
        H.on(qr[0:3])    # slice of register
        H.on(0, 1, 2)    # raw ints still work
        """
        if self.dimension != 2:
            raise ValueError("on() is only supported for single-qubit gates")
        expanded: list[int] = []
        for q in qubits:
            if isinstance(q, QuantumRegister):
                expanded.extend(q)
            else:
                expanded.append(q)
        return Circuit([self(q) for q in expanded])

class ParametricGateType:
    matrix_fn: Callable[[torch.Tensor], torch.Tensor] | None
    diagonal_fn: Callable[[torch.Tensor], torch.Tensor] | None

    def __init__(
        self,
        matrix_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        *,
        diagonal_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        if matrix_fn is None and diagonal_fn is None:
            raise ValueError("ParametricGateType requires matrix_fn or diagonal_fn")
        self.matrix_fn = matrix_fn
        self.diagonal_fn = diagonal_fn

    def __call__(self, param: float):
        # When called with theta, return a GateType that can then be called with targets
        param_tensor = torch.tensor(param, dtype=torch.float32)
        if self.diagonal_fn is not None:
            diagonal = self.diagonal_fn(param_tensor).to(torch.complex64)
            return GateType(diagonal=diagonal)
        assert self.matrix_fn is not None
        matrix = self.matrix_fn(param_tensor).to(torch.complex64)
        return GateType(matrix)

class ControlledGateType:
    _tensor: torch.Tensor | None
    _diagonal: torch.Tensor | None

    def __init__(self, base_gate: GateType | ControlledGateType):
        # Controlled gate = identity on control=0, base gate on control=1
        # This expands the gate matrix by one qubit
        base_diagonal = base_gate.diagonal
        if base_diagonal is not None:
            self._tensor = None
            self._diagonal = torch.cat((torch.ones_like(base_diagonal), base_diagonal))
        else:
            base_tensor = base_gate.tensor
            eye = torch.eye(base_tensor.shape[0], dtype=base_tensor.dtype)
            self._tensor = expand_diagonal(eye, base_tensor)
            self._diagonal = _infer_diagonal(self._tensor)

    @property
    def tensor(self) -> torch.Tensor:
        if self._tensor is None:
            assert self._diagonal is not None
            self._tensor = torch.diag(self._diagonal)
        return self._tensor

    @property
    def diagonal(self) -> torch.Tensor | None:
        return self._diagonal

    def __call__(self, *targets: int) -> Gate:
        return Gate(self._tensor, *targets, diagonal=self._diagonal)


I = GateType(diagonal=torch.tensor([1, 1], dtype=torch.complex64))
H = GateType(_complex_matrix([[1, 1], [1, -1]]) / math.sqrt(2))
X = GateType(_complex_matrix([[0, 1], [1, 0]]))
Y = GateType(_complex_matrix([[0, -1j], [1j, 0]]))
Z = GateType(diagonal=torch.tensor([1, -1], dtype=torch.complex64))
S = GateType(diagonal=torch.tensor([1, 1j], dtype=torch.complex64))
T = GateType(diagonal=torch.tensor([1, (1 + 1j) / math.sqrt(2)], dtype=torch.complex64))

# Parametric rotation gates
RX = ParametricGateType(lambda theta: torch.tensor(
    [[torch.cos(theta / 2), -1j * torch.sin(theta / 2)],
     [-1j * torch.sin(theta / 2), torch.cos(theta / 2)]],
    dtype=torch.complex64))

RY = ParametricGateType(lambda theta: torch.tensor(
    [[torch.cos(theta / 2), -torch.sin(theta / 2)],
     [torch.sin(theta / 2), torch.cos(theta / 2)]],
    dtype=torch.complex64))

RZ = ParametricGateType(
    diagonal_fn=lambda theta: torch.tensor(
        [torch.exp(-1j * theta / 2), torch.exp(1j * theta / 2)],
        dtype=torch.complex64,
    )
)

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
