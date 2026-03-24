"""Smoke tests: run core benchmark cases and verify correctness."""

from __future__ import annotations

import unittest

from quantum import run_simulation
from benchmarks.cases import CORE_CASES


class CoreCaseTests(unittest.TestCase):
    pass


def _make_test(case_fn):
    def test(self):
        case = case_fn()
        result = run_simulation(case.circuit, 2000, n_qubits=case.n_qubits)
        total = sum(result.values())
        for bitstring, expected_prob in case.expected.items():
            actual_prob = result.get(bitstring, 0) / total
            self.assertAlmostEqual(
                actual_prob, expected_prob, delta=case.tolerance,
                msg=f"{case.name}: {bitstring} expected {expected_prob:.3f}, got {actual_prob:.3f}",
            )
    return test


for _case_fn in CORE_CASES:
    setattr(CoreCaseTests, f"test_{_case_fn.__name__}", _make_test(_case_fn))


if __name__ == "__main__":
    unittest.main()
