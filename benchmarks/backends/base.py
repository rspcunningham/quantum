"""Backend adapter interfaces for cross-simulator comparisons."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from benchmarks.ir import CircuitIR


@dataclass(frozen=True)
class BackendAvailability:
    available: bool
    reason: str | None = None


class BackendAdapter(ABC):
    """Abstract adapter contract used by compare harness."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def availability(self) -> BackendAvailability:
        raise NotImplementedError

    @abstractmethod
    def version_info(self) -> dict[str, str]:
        raise NotImplementedError

    @abstractmethod
    def supports(self, case_ir: CircuitIR) -> tuple[bool, str | None]:
        raise NotImplementedError

    @abstractmethod
    def prepare(self, case_ir: CircuitIR) -> Any:
        raise NotImplementedError

    @abstractmethod
    def run(self, prepared_case: Any, shots: int, *, warmup: bool = False) -> dict[str, int]:
        raise NotImplementedError

