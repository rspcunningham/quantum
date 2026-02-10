"""Benchmark case registry."""

from dataclasses import dataclass
from collections.abc import Callable

from quantum import Circuit


@dataclass
class BenchmarkCase:
    name: str
    circuit: Circuit
    expected: dict[str, float]
    n_qubits: int | None = None
    tolerance: float = 0.05


# Import after BenchmarkCase is defined (case files import it from here)
from benchmarks.cases.bell_state import bell_state
from benchmarks.cases.simple_grovers import simple_grovers
from benchmarks.cases.real_grovers import real_grovers
from benchmarks.cases.ghz_state import ghz_state
from benchmarks.cases.qft import qft
from benchmarks.cases.teleportation import teleportation

ALL_CASES: list[Callable[[], BenchmarkCase]] = [
    bell_state,
    simple_grovers,
    real_grovers,
    ghz_state,
    qft,
    teleportation,
]
