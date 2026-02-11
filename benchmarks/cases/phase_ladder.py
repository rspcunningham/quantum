"""Phase-heavy round-trip: diagonal-gate stress with known final state."""

import cmath

import torch

from quantum import Circuit, QuantumRegister, H, RZ, GateType, ControlledGateType, measure_all
from benchmarks.cases import BenchmarkCase

N_QUBITS = 11
N_LAYERS = 6


def _phase_gate(phi: float) -> GateType:
    matrix = torch.tensor([[1, 0], [0, cmath.exp(1j * phi)]], dtype=torch.complex64)
    return GateType(matrix)


def _phase_ladder(qr: QuantumRegister, *, sign: float) -> Circuit:
    ops = []

    for layer in range(N_LAYERS):
        # Single-qubit diagonal phases.
        for q in range(N_QUBITS):
            angle = sign * (0.031 * (q + 1) * (layer + 1))
            ops.append(RZ(angle)(qr[q]))

        # Nearest-neighbor controlled phases.
        for q in range(N_QUBITS - 1):
            angle = sign * (0.017 * (layer + 1) * (q + 1))
            cp = ControlledGateType(_phase_gate(angle))
            ops.append(cp(qr[q], qr[q + 1]))

        # Longer-range controlled phases.
        for q in range(N_QUBITS - 2):
            angle = sign * (0.011 * (layer + 1) * (q + 1))
            cp = ControlledGateType(_phase_gate(angle))
            ops.append(cp(qr[q], qr[q + 2]))

    return Circuit(ops)


def phase_ladder() -> BenchmarkCase:
    qr = QuantumRegister(N_QUBITS)

    forward = _phase_ladder(qr, sign=1.0)
    backward = _phase_ladder(qr, sign=-1.0)

    # All phase operations are diagonal, so +phi then -phi returns identity.
    circuit = H.on(qr) + forward + backward + H.on(qr) + measure_all(qr)

    return BenchmarkCase(
        name="phase_ladder",
        circuit=circuit,
        expected={"0" * N_QUBITS: 1.0},
        n_qubits=N_QUBITS,
    )
