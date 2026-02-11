"""Auto-discover .qasm circuits and pair with .json expected distributions."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path


def discover_qasm_cases() -> list[Callable]:
    from benchmarks.cases import BenchmarkCase

    circuits_dir = Path(__file__).parent.parent / "circuits"
    expected_dir = Path(__file__).parent.parent / "expected"

    if not circuits_dir.is_dir():
        return []

    factories: list[Callable] = []
    for qasm_path in sorted(circuits_dir.glob("*.qasm")):
        name = qasm_path.stem
        json_path = expected_dir / f"{name}.json"
        if json_path.exists():
            factories.append(_make_factory(name, qasm_path, json_path))
    return factories


def _make_factory(name: str, qasm_path: Path, json_path: Path) -> Callable:
    def factory():
        from benchmarks.cases import BenchmarkCase
        from quantum.qasm import parse_qasm

        parsed = parse_qasm(qasm_path.read_text())
        data = json.loads(json_path.read_text())
        return BenchmarkCase(
            name=name,
            circuit=parsed.circuit,
            expected=data["expected"],
            n_qubits=parsed.n_qubits,
            tolerance=data.get("tolerance", 0.05),
        )

    return factory
