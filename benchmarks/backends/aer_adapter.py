"""Qiskit Aer backend adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from benchmarks.backends.base import BackendAdapter, BackendAvailability
from benchmarks.ir import CircuitIR, IRConditional, IRGate, IRMeasurement


@dataclass(frozen=True)
class AerPreparedCase:
    circuit: Any
    creg: Any
    n_bits: int


class AerAdapter(BackendAdapter):
    """Adapter for Qiskit AerSimulator (statevector mode)."""

    def __init__(self):
        self._import_error: str | None = None
        self._qiskit: Any = None
        self._QuantumCircuit: Any = None
        self._QuantumRegister: Any = None
        self._ClassicalRegister: Any = None
        self._Operator: Any = None
        self._AerSimulator: Any = None
        self._simulator: Any = None

        try:
            from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister  # type: ignore[import-not-found]
            from qiskit.quantum_info import Operator  # type: ignore[import-not-found]
            from qiskit_aer import AerSimulator  # type: ignore[import-not-found]

            self._QuantumCircuit = QuantumCircuit
            self._QuantumRegister = QuantumRegister
            self._ClassicalRegister = ClassicalRegister
            self._Operator = Operator
            self._AerSimulator = AerSimulator
            self._simulator = AerSimulator(method="statevector")
        except Exception as error:  # pragma: no cover - exercised by env differences
            self._import_error = str(error)

    @property
    def name(self) -> str:
        return "aer"

    def availability(self) -> BackendAvailability:
        if self._import_error is not None:
            return BackendAvailability(False, f"Qiskit Aer unavailable: {self._import_error}")
        return BackendAvailability(True)

    def version_info(self) -> dict[str, str]:
        info: dict[str, str] = {}
        if self._import_error is not None:
            return info
        try:
            import qiskit  # type: ignore[import-not-found]
            import qiskit_aer  # type: ignore[import-not-found]

            info["qiskit"] = getattr(qiskit, "__version__", "unknown")
            info["qiskit_aer"] = getattr(qiskit_aer, "__version__", "unknown")
        except Exception:
            pass
        return info

    def supports(self, case_ir: CircuitIR) -> tuple[bool, str | None]:
        if self._import_error is not None:
            return False, f"Qiskit Aer unavailable: {self._import_error}"
        _ = case_ir
        return True, None

    def _append_gate(self, circuit: Any, qubits: list[Any], op: IRGate) -> Any:
        assert self._Operator is not None
        matrix = np.asarray(op.tensor.detach().cpu().numpy(), dtype=np.complex128)
        matrix = self._to_qiskit_endianness(matrix)
        gate_op = self._Operator(matrix)
        return circuit.unitary(gate_op, [qubits[t] for t in op.targets])

    def _to_qiskit_endianness(self, matrix: np.ndarray) -> np.ndarray:
        """Convert from project big-endian gate indexing to Qiskit's convention."""
        dim = int(matrix.shape[0])
        if dim <= 2:
            return matrix

        n_targets = int(np.log2(dim))
        if (1 << n_targets) != dim:
            raise ValueError(f"Gate matrix dimension {dim} is not a power of two.")

        perm = [int(format(i, f"0{n_targets}b")[::-1], 2) for i in range(dim)]
        return matrix[np.ix_(perm, perm)]

    def _append_conditional_gate(self, circuit: Any, qubits: list[Any], creg: Any, op: IRConditional) -> None:
        # Prefer context-manager API when available.
        if hasattr(circuit, "if_test"):
            with circuit.if_test((creg, op.condition)):
                _ = self._append_gate(circuit, qubits, op.gate)
            return

        # Fallback for older APIs.
        instruction_set = self._append_gate(circuit, qubits, op.gate)
        if hasattr(instruction_set, "c_if"):
            _ = instruction_set.c_if(creg, op.condition)
            return
        raise RuntimeError("Aer adapter could not attach classical condition to gate.")

    def _project_bit_to_qiskit(self, bit: int, n_bits: int) -> int:
        """Map project bit index (MSB-first) to Qiskit classical index (LSB-first integer)."""
        return n_bits - 1 - bit

    def prepare(self, case_ir: CircuitIR) -> AerPreparedCase:
        if self._import_error is not None:
            raise RuntimeError(f"Qiskit Aer unavailable: {self._import_error}")
        assert self._QuantumCircuit is not None
        assert self._QuantumRegister is not None
        assert self._ClassicalRegister is not None

        qreg = self._QuantumRegister(case_ir.n_qubits, "q")
        creg = self._ClassicalRegister(case_ir.n_bits, "c")
        circuit = self._QuantumCircuit(qreg, creg)
        qubits = list(qreg)

        for op in case_ir.operations:
            if isinstance(op, IRGate):
                _ = self._append_gate(circuit, qubits, op)
                continue
            if isinstance(op, IRMeasurement):
                cidx = self._project_bit_to_qiskit(op.bit, case_ir.n_bits)
                circuit.measure(qubits[op.qubit], creg[cidx])
                continue
            self._append_conditional_gate(circuit, qubits, creg, op)

        return AerPreparedCase(circuit=circuit, creg=creg, n_bits=case_ir.n_bits)

    def _normalize_counts(self, raw_counts: dict[str, int], n_bits: int, shots: int) -> dict[str, int]:
        if n_bits == 0:
            return {"": shots}

        counts: dict[str, int] = {}
        for key, value in raw_counts.items():
            key_clean = key.replace(" ", "")
            key_clean = key_clean.zfill(n_bits)[-n_bits:]
            counts[key_clean] = counts.get(key_clean, 0) + int(value)
        return counts

    def run(self, prepared_case: AerPreparedCase, shots: int, *, warmup: bool = False) -> dict[str, int]:
        _ = warmup
        if self._simulator is None:
            raise RuntimeError("Qiskit Aer simulator is unavailable.")

        result = self._simulator.run(prepared_case.circuit, shots=shots).result()
        raw_counts = result.get_counts(prepared_case.circuit)
        if isinstance(raw_counts, list):
            if not raw_counts:
                return {"": shots} if prepared_case.n_bits == 0 else {}
            raw_counts = raw_counts[0]
        if not isinstance(raw_counts, dict):
            return {"": shots} if prepared_case.n_bits == 0 else {}
        return self._normalize_counts(raw_counts, prepared_case.n_bits, shots)
