"""Toffoli-heavy oracle round-trip: deep non-diagonal multi-qubit gate stress."""

from quantum import Circuit, registers, H, CX, CCX, measure_all
from benchmarks.cases import BenchmarkCase

INPUT_BITS = 6
WORK_BITS = 4
ANCILLA_BITS = 1


def _oracle_block(input_reg, work_reg, ancilla):
    anc = ancilla[0]
    ops = []

    # Build reusable nonlinear terms in work bits.
    for i in range(12):
        a = input_reg[i % INPUT_BITS]
        b = input_reg[(i + 1) % INPUT_BITS]
        w = work_reg[i % WORK_BITS]
        ops.append(CCX(a, b, w))
        ops.append(CX(w, input_reg[(i + 2) % INPUT_BITS]))

    # Mix work bits back into ancilla and input.
    for i in range(10):
        w0 = work_reg[i % WORK_BITS]
        w1 = work_reg[(i + 1) % WORK_BITS]
        tgt = input_reg[(i + 3) % INPUT_BITS]
        ops.append(CCX(w0, w1, anc))
        ops.append(CCX(anc, tgt, w0))

    return Circuit(ops)


def toffoli_oracle() -> BenchmarkCase:
    input_reg, work_reg, ancilla = registers(INPUT_BITS, WORK_BITS, ANCILLA_BITS)

    oracle = _oracle_block(input_reg, work_reg, ancilla)
    all_qubits = [*input_reg, *work_reg, *ancilla]
    n_qubits = len(all_qubits)

    # Oracle + adjoint (all gates in oracle are self-inverse) -> identity.
    circuit = H.on(input_reg) + oracle + oracle.inverse() + H.on(input_reg) + measure_all(all_qubits)

    return BenchmarkCase(
        name="toffoli_oracle",
        circuit=circuit,
        expected={"0" * n_qubits: 1.0},
        n_qubits=n_qubits,
    )
