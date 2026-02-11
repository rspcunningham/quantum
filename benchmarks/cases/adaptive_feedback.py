"""Adaptive measurement/conditional stress: repeated mid-circuit feedback."""

from quantum import Circuit, QuantumRegister, H, X, Measurement
from benchmarks.cases import BenchmarkCase

N_ROUNDS = 40


def adaptive_feedback() -> BenchmarkCase:
    qr = QuantumRegister(2)

    ops = []
    for _ in range(N_ROUNDS):
        # Randomize q0, measure to bit 0, then conditionally reset q0 to |1>.
        ops.append(H(qr[0]))
        ops.append(Measurement(qr[0], 0))
        ops.append(X(qr[0]).if_(0))

        # Two identical conditional toggles on q1 cancel each other out.
        # This adds conditional-gate pressure without changing expected output.
        ops.append(X(qr[1]).if_(1))
        ops.append(X(qr[1]).if_(1))

    # Final check: q0 should deterministically be |1>.
    ops.append(Measurement(qr[0], 0))

    return BenchmarkCase(
        name="adaptive_feedback",
        circuit=Circuit(ops),
        expected={"1": 1.0},
        n_qubits=2,
    )
