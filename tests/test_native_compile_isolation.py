"""Smoke tests for simulator behavior without benchmark package dependency."""

from __future__ import annotations

import unittest
from typing import cast

from quantum import (
    Circuit,
    ConditionalGate,
    CX,
    Gate,
    H,
    Measurement,
    QuantumRegister,
    X,
    measure_all,
    run_simulation,
)


class SimulatorSmokeTests(unittest.TestCase):
    def test_bell_state_distribution(self):
        qubits = QuantumRegister(2)
        q0 = cast(int, qubits[0])
        q1 = cast(int, qubits[1])
        circuit = H(q0) + CX(q0, q1) + measure_all(qubits)

        result = run_simulation(circuit, 2_000, n_qubits=2, n_bits=2)
        total = sum(result.values())

        self.assertAlmostEqual(result.get("00", 0) / total, 0.5, delta=0.08)
        self.assertAlmostEqual(result.get("11", 0) / total, 0.5, delta=0.08)

    def test_mid_circuit_feedback(self):
        qubits = QuantumRegister(2)
        q0 = cast(int, qubits[0])
        q1 = cast(int, qubits[1])
        ops: list[Gate | ConditionalGate | Measurement | Circuit] = []
        for _ in range(5):
            ops.append(H(q0))
            ops.append(Measurement(q0, 0))
            ops.append(X(q0).if_(0))
            ops.append(X(q1).if_(1))
            ops.append(X(q1).if_(1))
        ops.append(Measurement(q0, 0))

        result = run_simulation(Circuit(ops), 2_000, n_qubits=2, n_bits=1)
        total = sum(result.values())

        self.assertEqual(result.get("1", 0), total)


if __name__ == "__main__":
    _ = unittest.main()
