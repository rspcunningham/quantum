"""Bell state: 2 qubits, 2 gates. Minimal circuit baseline."""

from quantum import QuantumRegister, H, CX, measure_all
from benchmarks.cases import BenchmarkCase


def bell_state() -> BenchmarkCase:
    qr = QuantumRegister(2)
    circuit = H(qr[0]) + CX(qr[0], qr[1]) + measure_all(qr)
    return BenchmarkCase(
        name="bell_state",
        circuit=circuit,
        expected={"00": 0.5, "11": 0.5},
    )
