"""Native simulator backend adapter."""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.backends.base import BackendAdapter, BackendAvailability
from benchmarks.ir import CircuitIR
from quantum import run_simulation
from quantum.gates import Circuit


@dataclass(frozen=True)
class NativePreparedCase:
    circuit: Circuit
    n_qubits: int
    n_bits: int


class NativeAdapter(BackendAdapter):

    @property
    def name(self) -> str:
        return "native"

    def availability(self) -> BackendAvailability:
        return BackendAvailability(available=True)

    def version_info(self) -> dict[str, str]:
        return {"backend": "metal"}

    def supports(self, case_ir: CircuitIR) -> tuple[bool, str | None]:
        return True, None

    def prepare(self, case_ir: CircuitIR) -> NativePreparedCase:
        circuit = case_ir.source_circuit
        if not isinstance(circuit, Circuit):
            raise RuntimeError("Invalid native circuit payload.")
        return NativePreparedCase(circuit=circuit, n_qubits=case_ir.n_qubits, n_bits=case_ir.n_bits)

    def run(self, prepared_case: NativePreparedCase, shots: int, *, warmup: bool = False) -> dict[str, int]:
        _ = warmup
        return run_simulation(prepared_case.circuit, shots, n_qubits=prepared_case.n_qubits, n_bits=prepared_case.n_bits)
