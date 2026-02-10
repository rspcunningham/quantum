"""QFT round-trip: 10 qubits. Forward QFT then inverse QFT on |0...0>.

Tests parametric controlled phase gates. The round-trip should return
to the initial state, so expected output is 100% on |0...0>.
"""

import cmath
import math

import torch

from quantum import Circuit, QuantumRegister, H, GateType, ControlledGateType, measure_all
from benchmarks.cases import BenchmarkCase

N_QUBITS = 10


def _phase_gate(phi: float) -> GateType:
    matrix = torch.tensor([[1, 0], [0, cmath.exp(1j * phi)]], dtype=torch.complex64)
    return GateType(matrix)


def _build_qft(qr: QuantumRegister, n: int) -> Circuit:
    """Forward QFT without final swaps."""
    ops = []
    for i in range(n):
        ops.append(H(qr[i]))
        for j in range(i + 1, n):
            k = j - i + 1
            phi = 2 * math.pi / (1 << k)
            cp = ControlledGateType(_phase_gate(phi))
            ops.append(cp(qr[j], qr[i]))
    return Circuit(ops)


def _build_inverse_qft(qr: QuantumRegister, n: int) -> Circuit:
    """Inverse QFT without initial swaps (adjoint of _build_qft)."""
    ops = []
    for i in range(n - 1, -1, -1):
        for j in range(n - 1, i, -1):
            k = j - i + 1
            phi = -2 * math.pi / (1 << k)
            cp = ControlledGateType(_phase_gate(phi))
            ops.append(cp(qr[j], qr[i]))
        ops.append(H(qr[i]))
    return Circuit(ops)


def qft() -> BenchmarkCase:
    qr = QuantumRegister(N_QUBITS)
    circuit = _build_qft(qr, N_QUBITS) + _build_inverse_qft(qr, N_QUBITS) + measure_all(qr)
    return BenchmarkCase(
        name="qft",
        circuit=circuit,
        expected={"0" * N_QUBITS: 1.0},
        n_qubits=N_QUBITS,
    )
