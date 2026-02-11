"""Generate expected distributions by running .qasm circuits on Aer.

Run: uv run python benchmarks/generate_expected.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CIRCUITS_DIR = Path(__file__).parent / "circuits"
EXPECTED_DIR = Path(__file__).parent / "expected"
AER_SHOTS = 100_000


def _reverse_bitstring(bs: str) -> str:
    """Reverse bitstring: Qiskit LE → our BE."""
    return bs[::-1]


def _auto_tolerance(expected: dict[str, float]) -> float:
    max_prob = max(expected.values()) if expected else 0
    if max_prob > 0.95:
        return 0.02
    if max_prob > 0.7:
        return 0.05
    return 0.10


def _load_qasm(qasm_str: str):
    """Load QASM with custom instructions Qiskit's exporter produces but loader doesn't know."""
    from qiskit import qasm2
    from qiskit.qasm2 import CustomInstruction
    from qiskit.circuit.library import SwapGate, CPhaseGate
    return qasm2.loads(qasm_str, custom_instructions=[
        CustomInstruction("cp", 1, 2, CPhaseGate, builtin=True),
        CustomInstruction("swap", 0, 2, SwapGate, builtin=True),
    ])


def generate_one(qasm_path: Path) -> dict:
    from qiskit_aer import AerSimulator

    qc = _load_qasm(qasm_path.read_text())
    sim = AerSimulator()
    result = sim.run(qc, shots=AER_SHOTS).result()
    counts = result.get_counts()

    # Reverse bitstrings and normalize
    total = sum(counts.values())
    expected: dict[str, float] = {}
    for bs, count in counts.items():
        prob = count / total
        if prob >= 0.001:
            # Remove spaces Qiskit sometimes inserts between registers
            bs_clean = bs.replace(" ", "")
            expected[_reverse_bitstring(bs_clean)] = round(prob, 6)

    tolerance = _auto_tolerance(expected)
    return {
        "expected": expected,
        "tolerance": tolerance,
        "aer_shots": AER_SHOTS,
    }


def main() -> None:
    EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
    qasm_files = sorted(CIRCUITS_DIR.glob("*.qasm"))

    if not qasm_files:
        print(f"No .qasm files found in {CIRCUITS_DIR}")
        sys.exit(1)

    print(f"Found {len(qasm_files)} circuits to process")
    generated = 0
    skipped = 0
    failed = 0

    for qasm_path in qasm_files:
        name = qasm_path.stem
        json_path = EXPECTED_DIR / f"{name}.json"

        if json_path.exists():
            skipped += 1
            continue

        try:
            data = generate_one(qasm_path)
            json_path.write_text(json.dumps(data, indent=2) + "\n")
            n_outcomes = len(data["expected"])
            print(f"  {name}: {n_outcomes} outcomes, tol={data['tolerance']}")
            generated += 1
        except Exception as e:
            print(f"  FAILED {name}: {e}")
            failed += 1

    print(f"\nDone: {generated} generated, {skipped} skipped (existing), {failed} failed")


if __name__ == "__main__":
    main()
