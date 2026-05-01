from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import override

import quantum_native_runtime as native


# ---------------------------------------------------------------------------
# Gate kind constants — must match GateKind enum in py_module.cpp
# ---------------------------------------------------------------------------

_I = 0
_H = 1
_X = 2
_Y = 3
_Z = 4
_S = 5
_Sdg = 6
_T = 7
_Tdg = 8
_SX = 9
_RX = 10
_RY = 11
_RZ = 12
_CX = 13
_CZ = 14
_CCX = 15
_SWAP = 16
_CP = 17


# ---------------------------------------------------------------------------
# Quantum register (pure Python, no native dependency)
# ---------------------------------------------------------------------------

class QuantumRegister:
    """A named group of qubits with auto-assigned indices."""
    _offset: int
    _size: int

    def __init__(self, size: int, offset: int = 0):
        self._offset = offset
        self._size = size

    def __getitem__(self, key: int | slice) -> int | QuantumRegister:
        if isinstance(key, int):
            if key < 0:
                key = self._size + key
            if key < 0 or key >= self._size:
                raise IndexError(f"Register index {key} out of range [0, {self._size})")
            return self._offset + key
        indices = range(self._size)[key]
        if len(indices) == 0:
            return QuantumRegister(0, self._offset)
        return QuantumRegister(len(indices), self._offset + indices[0])

    def __iter__(self) -> Iterator[int]:
        return iter(range(self._offset, self._offset + self._size))

    def __len__(self) -> int:
        return self._size


def registers(*sizes: int) -> tuple[QuantumRegister, ...]:
    """Create multiple contiguous quantum registers."""
    offset = 0
    regs: list[QuantumRegister] = []
    for size in sizes:
        regs.append(QuantumRegister(size, offset=offset))
        offset += size
    return tuple(regs)


# ---------------------------------------------------------------------------
# Gate — thin wrapper around native.NativeGate
# ---------------------------------------------------------------------------

class Gate:
    __slots__ = ('_native',)

    def __init__(self, native_gate: native.NativeGate):
        self._native = native_gate

    @property
    def targets(self) -> list[int]:
        return self._native.targets()

    @property
    def tensor(self):
        """Return the gate's unitary matrix as a numpy array."""
        return self._native.matrix()

    def if_(self, classical_bit: int) -> ConditionalGate:
        return ConditionalGate(self, classical_bit)

    def __add__(self, other: Gate | Measurement | ConditionalGate | Circuit) -> Circuit:
        return Circuit([self, other])

    def __getstate__(self):
        return {'matrix': self._native.matrix(), 'targets': self._native.targets()}

    def __setstate__(self, state):
        import numpy as np
        self._native = native.make_dense_gate(
            np.asarray(state['matrix'], dtype=np.complex64),
            state['targets'],
        )


# ---------------------------------------------------------------------------
# Gate type factories
# ---------------------------------------------------------------------------

class GateType:
    """Factory that creates Gate instances for given target qubits."""
    __slots__ = ('_kind', '_param')

    def __init__(self, kind: int, param: float = 0.0):
        self._kind = kind
        self._param = param

    def __call__(self, *targets: int) -> Gate:
        return Gate(native.make_gate(self._kind, list(targets), self._param))

    @property
    def dimension(self) -> int:
        """Number of target qubits this gate operates on."""
        if self._kind in (_CX, _CZ, _SWAP, _CP):
            return 4
        if self._kind == _CCX:
            return 8
        return 2

    def on(self, *qubits: int | QuantumRegister) -> Circuit:
        """Apply this single-qubit gate to each qubit independently."""
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
    """Factory for parametric gates: RX(theta)(qubit)."""
    __slots__ = ('_kind',)

    def __init__(self, kind: int):
        self._kind = kind

    def __call__(self, param: float) -> GateType:
        return GateType(self._kind, float(param))


# ---------------------------------------------------------------------------
# Standard gates
# ---------------------------------------------------------------------------

I = GateType(_I)
H = GateType(_H)
X = GateType(_X)
Y = GateType(_Y)
Z = GateType(_Z)
S = GateType(_S)
Sdg = GateType(_Sdg)
T = GateType(_T)
Tdg = GateType(_Tdg)
SX = GateType(_SX)

RX = ParametricGateType(_RX)
RY = ParametricGateType(_RY)
RZ = ParametricGateType(_RZ)

CX = GateType(_CX)
CZ = GateType(_CZ)
CCX = GateType(_CCX)
SWAP = GateType(_SWAP)
CP = ParametricGateType(_CP)


# ---------------------------------------------------------------------------
# Custom gate support (for QASM parser and arbitrary unitaries)
# ---------------------------------------------------------------------------

class CustomGateType:
    """Gate type from a custom numpy matrix or diagonal."""
    __slots__ = ('_matrix', '_diagonal')

    def __init__(self, matrix=None, *, diagonal=None):
        if matrix is None and diagonal is None:
            raise ValueError("CustomGateType requires matrix or diagonal")
        self._matrix = matrix
        self._diagonal = diagonal

    def __call__(self, *targets: int) -> Gate:
        import numpy as np
        if self._diagonal is not None:
            return Gate(native.make_diagonal_gate(
                np.ascontiguousarray(self._diagonal, dtype=np.complex64),
                list(targets),
            ))
        return Gate(native.make_dense_gate(
            np.ascontiguousarray(self._matrix, dtype=np.complex64),
            list(targets),
        ))


# ---------------------------------------------------------------------------
# Measurement & conditional gate
# ---------------------------------------------------------------------------

class Measurement:
    __slots__ = ('qubit', 'bit')

    def __init__(self, qubit: int, bit: int):
        self.qubit = qubit
        self.bit = bit

    def __add__(self, other: Gate | Measurement | ConditionalGate | Circuit) -> Circuit:
        return Circuit([self, other])


class ConditionalGate:
    """A gate that only executes if the classical register equals the condition value."""
    __slots__ = ('gate', 'condition')

    def __init__(self, gate: Gate, condition: int):
        self.gate = gate
        self.condition = condition

    def __add__(self, other: Gate | Measurement | ConditionalGate | Circuit) -> Circuit:
        return Circuit([self, other])


# ---------------------------------------------------------------------------
# Circuit
# ---------------------------------------------------------------------------

class Circuit:
    operations: list[Gate | ConditionalGate | Measurement | Circuit]

    def __init__(self, operations: Sequence[Gate | ConditionalGate | Measurement | Circuit]):
        self.operations = list(operations)

    def append(self, operation: Gate | ConditionalGate | Measurement | Circuit) -> None:
        self.operations.append(operation)

    def extend(self, operations: Sequence[Gate | ConditionalGate | Measurement | Circuit]) -> None:
        self.operations.extend(operations)

    def _flatten_operations(self) -> list[Gate | ConditionalGate | Measurement]:
        out: list[Gate | ConditionalGate | Measurement] = []

        def _visit(op: Gate | ConditionalGate | Measurement | Circuit) -> None:
            if isinstance(op, Circuit):
                for child in op.operations:
                    _visit(child)
                return
            out.append(op)

        for op in self.operations:
            _visit(op)
        return out

    def flatten_native(self) -> list[native.NativeGate | native.NativeMeasurement]:
        """Flatten a static circuit to native objects for the C++ runtime."""
        flat = self._flatten_operations()
        out: list[native.NativeGate | native.NativeMeasurement] = []
        seen_measurement = False
        for op in flat:
            if isinstance(op, Gate):
                if seen_measurement:
                    raise ValueError(
                        "Static circuits require all measurements to be terminal."
                    )
                out.append(op._native)
            elif isinstance(op, Measurement):
                seen_measurement = True
                out.append(native.NativeMeasurement(op.qubit, op.bit))
            elif isinstance(op, ConditionalGate):
                raise ValueError(
                    "Conditional gates are dynamic and are not supported by quantum.compile()."
                )
            else:
                raise RuntimeError(f"Unknown operation type: {type(op)}")
        return out

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
