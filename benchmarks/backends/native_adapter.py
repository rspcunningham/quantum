"""Native simulator backend adapter."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from benchmarks.backends.base import BackendAdapter, BackendAvailability
from benchmarks.ir import CircuitIR
from quantum import run_simulation
from quantum.gates import Circuit


@dataclass(frozen=True)
class NativePreparedCase:
    circuit: Circuit
    n_qubits: int
    n_bits: int
    device: torch.device


class NativeAdapter(BackendAdapter):
    """Adapter that delegates to the project's `run_simulation`."""

    def __init__(self, *, device: torch.device | None = None):
        self._device = device

    def _resolve_device(self) -> torch.device:
        if self._device is not None:
            return self._device
        return torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )

    @property
    def name(self) -> str:
        return "native"

    def availability(self) -> BackendAvailability:
        return BackendAvailability(available=True)

    def version_info(self) -> dict[str, str]:
        return {"torch": torch.__version__, "device": self._resolve_device().type}

    def supports(self, case_ir: CircuitIR) -> tuple[bool, str | None]:
        device = self._resolve_device()
        if device.type == "mps" and (case_ir.n_qubits + 1) > 16:
            return (
                False,
                "native backend on MPS cannot run circuits requiring tensor rank > 16 (batch + qubit axes).",
            )
        return True, None

    def prepare(self, case_ir: CircuitIR) -> NativePreparedCase:
        circuit = case_ir.source_circuit
        if not isinstance(circuit, Circuit):
            raise RuntimeError("Invalid native circuit payload.")

        device = self._resolve_device()
        return NativePreparedCase(
            circuit=circuit,
            n_qubits=case_ir.n_qubits,
            n_bits=case_ir.n_bits,
            device=device,
        )

    def run(self, prepared_case: NativePreparedCase, shots: int, *, warmup: bool = False) -> dict[str, int]:
        _ = warmup
        return run_simulation(
            prepared_case.circuit,
            shots,
            n_qubits=prepared_case.n_qubits,
            n_bits=prepared_case.n_bits,
            device=prepared_case.device,
        )
