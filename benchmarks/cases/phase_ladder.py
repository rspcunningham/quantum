"""Phase-heavy round-trip: diagonal-gate stress with known final state."""

from quantum import Circuit, QuantumRegister, H, RZ, CP, measure_all
from benchmarks.cases import BenchmarkCase

N_QUBITS = 11
N_LAYERS = 6


def _phase_ladder(qr: QuantumRegister, *, sign: float) -> Circuit:
    ops = []

    for layer in range(N_LAYERS):
        for q in range(N_QUBITS):
            angle = sign * (0.031 * (q + 1) * (layer + 1))
            ops.append(RZ(angle)(qr[q]))

        for q in range(N_QUBITS - 1):
            angle = sign * (0.017 * (layer + 1) * (q + 1))
            ops.append(CP(angle)(qr[q], qr[q + 1]))

        for q in range(N_QUBITS - 2):
            angle = sign * (0.011 * (layer + 1) * (q + 1))
            ops.append(CP(angle)(qr[q], qr[q + 2]))

    return Circuit(ops)


def phase_ladder() -> BenchmarkCase:
    qr = QuantumRegister(N_QUBITS)
    forward = _phase_ladder(qr, sign=1.0)
    backward = _phase_ladder(qr, sign=-1.0)
    circuit = H.on(qr) + forward + backward + H.on(qr) + measure_all(qr)

    return BenchmarkCase(
        name="phase_ladder",
        circuit=circuit,
        expected={"0" * N_QUBITS: 1.0},
        n_qubits=N_QUBITS,
    )
