"""Smoke tests for the static compiled simulator API."""

from __future__ import annotations

import unittest
from typing import cast

import quantum_native_runtime as native
from quantum import (
    Circuit,
    CX,
    H,
    Measurement,
    QuantumRegister,
    X,
    compile as compile_circuit,
    measure_all,
)


class CompiledStaticApiTests(unittest.TestCase):
    def test_bell_state_distribution(self):
        qubits = QuantumRegister(2)
        q0 = cast(int, qubits[0])
        q1 = cast(int, qubits[1])
        circuit = H(q0) + CX(q0, q1) + measure_all(qubits)

        with compile_circuit(circuit, n_qubits=2, n_bits=2) as compiled:
            result = compiled.run(2_000, seed=7)

        total = sum(result.values())
        self.assertAlmostEqual(result.get("00", 0) / total, 0.5, delta=0.08)
        self.assertAlmostEqual(result.get("11", 0) / total, 0.5, delta=0.08)

    def test_seeded_runs_are_deterministic(self):
        circuit = H(0) + Measurement(0, 0)

        with compile_circuit(circuit, n_qubits=1, n_bits=1) as compiled:
            first = compiled.run(256, seed=123)
            second = compiled.run(256, seed=123)

        self.assertEqual(first, second)

    def test_zero_shots_returns_empty_counts(self):
        with compile_circuit(H(0) + Measurement(0, 0), n_qubits=1, n_bits=1) as compiled:
            self.assertEqual(compiled.run(0), {})

    def test_close_is_idempotent_and_blocks_run(self):
        compiled = compile_circuit(H(0) + Measurement(0, 0), n_qubits=1, n_bits=1)
        compiled.close()
        compiled.close()

        with self.assertRaisesRegex(RuntimeError, "closed"):
            _ = compiled.run(1)

    def test_mid_circuit_measurement_is_rejected(self):
        circuit = Circuit([H(0), Measurement(0, 0), X(0), Measurement(0, 0)])

        with self.assertRaisesRegex(ValueError, "terminal"):
            _ = compile_circuit(circuit, n_qubits=1, n_bits=1)

    def test_conditional_gate_is_rejected(self):
        circuit = Circuit([X(0).if_(1), Measurement(0, 0)])

        with self.assertRaisesRegex(ValueError, "Conditional"):
            _ = compile_circuit(circuit, n_qubits=1, n_bits=1)

    def test_dynamic_native_entrypoints_are_removed(self):
        self.assertFalse(hasattr(native, "run_circuit"))
        self.assertFalse(hasattr(native, "make_conditional"))
        self.assertFalse(hasattr(native, "NativeConditionalGate"))


if __name__ == "__main__":
    _ = unittest.main()
