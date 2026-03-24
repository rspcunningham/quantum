"""Quantum system state management."""

from __future__ import annotations

from collections.abc import Sequence

from quantum.gates import Circuit, ConditionalGate, Gate, Measurement, QuantumRegister


def infer_resources(circuit: Circuit) -> tuple[int, int]:
    """Infer (n_qubits, n_bits) by walking all operations recursively."""
    max_qubit = -1
    max_bit = -1

    def _walk(ops: Sequence[Gate | ConditionalGate | Measurement | Circuit]) -> None:
        nonlocal max_qubit, max_bit
        for op in ops:
            if isinstance(op, Circuit):
                _walk(op.operations)
            elif isinstance(op, Gate):
                for target in op.targets:
                    if target > max_qubit:
                        max_qubit = target
            elif isinstance(op, Measurement):
                if op.qubit > max_qubit:
                    max_qubit = op.qubit
                if op.bit > max_bit:
                    max_bit = op.bit
            else:
                for target in op.gate.targets:
                    if target > max_qubit:
                        max_qubit = target
                if op.condition > 0:
                    needed_bits = op.condition.bit_length()
                    if needed_bits - 1 > max_bit:
                        max_bit = needed_bits - 1

    _walk(circuit.operations)
    return (max_qubit + 1, max_bit + 1)


def measure_all(qubits: QuantumRegister | list[int] | int) -> Circuit:
    """Measure qubits into sequential classical bits starting at 0."""
    if isinstance(qubits, int):
        indices = list(range(qubits))
    elif isinstance(qubits, QuantumRegister):
        indices = list(qubits)
    else:
        indices = list(qubits)
    return Circuit([Measurement(qubit, bit) for bit, qubit in enumerate(indices)])


def run_simulation(
    circuit: Circuit,
    num_shots: int,
    *,
    n_qubits: int | None = None,
    n_bits: int | None = None,
) -> dict[str, int]:
    """Run a quantum circuit simulation and return measurement counts."""
    if n_qubits is None or n_bits is None:
        inferred_qubits, inferred_bits = infer_resources(circuit)
        if n_qubits is None:
            n_qubits = inferred_qubits
        if n_bits is None:
            n_bits = inferred_bits

    from quantum.metal_exec import execute_static_circuit

    return execute_static_circuit(
        circuit,
        n_qubits=n_qubits,
        n_bits=n_bits,
        num_shots=num_shots,
    )
