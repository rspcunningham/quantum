"""QFT round-trip: 10 qubits. Forward QFT then inverse QFT on |0...0>."""

import math

from quantum import Circuit, QuantumRegister, H, CP, measure_all
from benchmarks.cases import BenchmarkCase

N_QUBITS = 10


def build_qft(qr: QuantumRegister, n: int) -> Circuit:
    ops = []
    for i in range(n):
        ops.append(H(qr[i]))
        for j in range(i + 1, n):
            k = j - i + 1
            phi = 2 * math.pi / (1 << k)
            ops.append(CP(phi)(qr[j], qr[i]))
    return Circuit(ops)


def build_inverse_qft(qr: QuantumRegister, n: int) -> Circuit:
    ops = []
    for i in range(n - 1, -1, -1):
        for j in range(n - 1, i, -1):
            k = j - i + 1
            phi = -2 * math.pi / (1 << k)
            ops.append(CP(phi)(qr[j], qr[i]))
        ops.append(H(qr[i]))
    return Circuit(ops)


def qft() -> BenchmarkCase:
    qr = QuantumRegister(N_QUBITS)
    circuit = build_qft(qr, N_QUBITS) + build_inverse_qft(qr, N_QUBITS) + measure_all(qr)
    return BenchmarkCase(
        name="qft",
        circuit=circuit,
        expected={"0" * N_QUBITS: 1.0},
        n_qubits=N_QUBITS,
    )
