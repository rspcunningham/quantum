"""GHZ state: 12 qubits, shallow circuit. Tests qubit scaling with simple gates."""

from quantum import Circuit, QuantumRegister, H, CX, measure_all
from benchmarks.cases import BenchmarkCase

N_QUBITS = 12


def ghz_state() -> BenchmarkCase:
    qr = QuantumRegister(N_QUBITS)
    ops = [H(qr[0])] + [CX(qr[0], qr[i]) for i in range(1, N_QUBITS)]
    circuit = Circuit(ops) + measure_all(qr)
    return BenchmarkCase(
        name="ghz_state",
        circuit=circuit,
        expected={"0" * N_QUBITS: 0.5, "1" * N_QUBITS: 0.5},
    )
