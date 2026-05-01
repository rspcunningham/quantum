"""Native simulator backend adapter."""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.backends.base import BackendAdapter, BackendAvailability
from benchmarks.ir import CircuitIR
from quantum import CompiledCircuit, compile as compile_circuit
from quantum.gates import Circuit


@dataclass(frozen=True)
class NativePreparedCase:
    compiled: CompiledCircuit


class NativeAdapter(BackendAdapter):

    @property
    def name(self) -> str:
        return "native"

    def availability(self) -> BackendAvailability:
        return BackendAvailability(available=True)

    def version_info(self) -> dict[str, str]:
        return {"backend": "metal"}

    def supports(self, case_ir: CircuitIR) -> tuple[bool, str | None]:
        if case_ir.is_dynamic:
            return False, "dynamic circuits are not supported by the static compiled native backend"
        return True, None

    def prepare(self, case_ir: CircuitIR) -> NativePreparedCase:
        circuit = case_ir.source_circuit
        if not isinstance(circuit, Circuit):
            raise RuntimeError("Invalid native circuit payload.")
        compiled = compile_circuit(circuit, n_qubits=case_ir.n_qubits, n_bits=case_ir.n_bits)
        return NativePreparedCase(compiled=compiled)

    def run(self, prepared_case: NativePreparedCase, shots: int, *, warmup: bool = False) -> dict[str, int]:
        _ = warmup
        return prepared_case.compiled.run(shots)
