"""Backend-neutral benchmark IR utilities."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch

from quantum import infer_resources
from quantum.gates import Circuit, ConditionalGate, Gate, Measurement


@dataclass(frozen=True)
class IRGate:
    """A gate operation with explicit matrix and target ordering."""

    tensor: torch.Tensor
    targets: tuple[int, ...]


@dataclass(frozen=True)
class IRMeasurement:
    """A measurement operation mapping qubit -> classical bit."""

    qubit: int
    bit: int


@dataclass(frozen=True)
class IRConditional:
    """A conditional gate on full classical-register integer equality."""

    gate: IRGate
    condition: int


IROperation = IRGate | IRMeasurement | IRConditional


@dataclass(frozen=True)
class CircuitIR:
    """Flattened circuit IR and derived structural properties."""

    n_qubits: int
    n_bits: int
    operations: list[IROperation]
    has_conditional: bool
    has_non_terminal_measurement: bool
    source_circuit: Circuit

    @property
    def is_dynamic(self) -> bool:
        return self.has_conditional or self.has_non_terminal_measurement


def _flatten_ops(
    ops: Sequence[Gate | Measurement | ConditionalGate | Circuit],
    out: list[Gate | Measurement | ConditionalGate],
) -> None:
    for op in ops:
        if isinstance(op, Circuit):
            _flatten_ops(op.operations, out)
        else:
            out.append(op)


def build_circuit_ir(
    circuit: Circuit,
    *,
    n_qubits: int | None = None,
    n_bits: int | None = None,
) -> CircuitIR:
    """Construct backend-neutral IR from a circuit."""
    if n_qubits is None or n_bits is None:
        inferred_qubits, inferred_bits = infer_resources(circuit)
        if n_qubits is None:
            n_qubits = inferred_qubits
        if n_bits is None:
            n_bits = inferred_bits

    flat_ops: list[Gate | Measurement | ConditionalGate] = []
    _flatten_ops(circuit.operations, flat_ops)

    ir_ops: list[IROperation] = []
    has_conditional = False
    has_non_terminal_measurement = False
    seen_measurement = False

    for op in flat_ops:
        if isinstance(op, Gate):
            if seen_measurement:
                has_non_terminal_measurement = True
            ir_ops.append(IRGate(tensor=op.tensor, targets=tuple(op.targets)))
            continue

        if isinstance(op, Measurement):
            seen_measurement = True
            ir_ops.append(IRMeasurement(qubit=op.qubit, bit=op.bit))
            continue

        has_conditional = True
        seen_measurement = True  # conditionals are inherently non-terminal behavior
        ir_ops.append(
            IRConditional(
                gate=IRGate(tensor=op.gate.tensor, targets=tuple(op.gate.targets)),
                condition=op.condition,
            )
        )

    return CircuitIR(
        n_qubits=n_qubits,
        n_bits=n_bits,
        operations=ir_ops,
        has_conditional=has_conditional,
        has_non_terminal_measurement=has_non_terminal_measurement,
        source_circuit=circuit,
    )
