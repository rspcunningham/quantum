"""qsim (via Cirq) backend adapter for static circuits."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np

from benchmarks.backends.base import BackendAdapter, BackendAvailability
from benchmarks.ir import CircuitIR, IRConditional, IRGate, IRMeasurement


@dataclass(frozen=True)
class QsimPreparedCase:
    circuit: Any
    n_qubits: int
    n_bits: int
    measurements: list[IRMeasurement]


class QsimAdapter(BackendAdapter):
    """Adapter for qsim static-circuit statevector comparison."""

    def __init__(self):
        self._import_error: str | None = None
        self._cirq: Any = None
        self._qsimcirq: Any = None
        self._simulator: Any = None
        try:
            import cirq  # type: ignore[import-not-found]
            import qsimcirq  # type: ignore[import-not-found]

            self._cirq = cirq
            self._qsimcirq = qsimcirq
            self._simulator = qsimcirq.QSimSimulator()
        except Exception as error:  # pragma: no cover - depends on environment packages
            self._import_error = str(error)

    @property
    def name(self) -> str:
        return "qsim"

    def availability(self) -> BackendAvailability:
        if self._import_error is not None:
            return BackendAvailability(False, f"qsim unavailable: {self._import_error}")
        return BackendAvailability(True)

    def version_info(self) -> dict[str, str]:
        info: dict[str, str] = {}
        if self._import_error is not None:
            return info
        try:
            import cirq  # type: ignore[import-not-found]
            import qsimcirq  # type: ignore[import-not-found]

            info["cirq"] = getattr(cirq, "__version__", "unknown")
            info["qsimcirq"] = getattr(qsimcirq, "__version__", "unknown")
            info["KMP_DUPLICATE_LIB_OK"] = os.environ.get("KMP_DUPLICATE_LIB_OK", "")
        except Exception:
            pass
        return info

    def supports(self, case_ir: CircuitIR) -> tuple[bool, str | None]:
        if self._import_error is not None:
            return False, f"qsim unavailable: {self._import_error}"
        if case_ir.is_dynamic:
            return False, "qsim adapter currently supports static terminal-measurement circuits only."
        return True, None

    def prepare(self, case_ir: CircuitIR) -> QsimPreparedCase:
        if self._import_error is not None:
            raise RuntimeError(f"qsim unavailable: {self._import_error}")
        if self._cirq is None:
            raise RuntimeError("Cirq unavailable.")

        qubits = self._cirq.LineQubit.range(case_ir.n_qubits)
        circuit = self._cirq.Circuit()
        measurements: list[IRMeasurement] = []

        for op in case_ir.operations:
            if isinstance(op, IRGate):
                matrix = np.asarray(op.tensor.detach().cpu().numpy(), dtype=np.complex128)
                gate = self._cirq.MatrixGate(matrix)
                circuit.append(gate.on(*[qubits[t] for t in op.targets]))
            elif isinstance(op, IRMeasurement):
                measurements.append(op)
            else:
                raise RuntimeError("qsim adapter does not support conditional operations.")

        return QsimPreparedCase(
            circuit=circuit,
            n_qubits=case_ir.n_qubits,
            n_bits=case_ir.n_bits,
            measurements=measurements,
        )

    def _counts_from_samples(
        self,
        samples: np.ndarray,
        *,
        n_qubits: int,
        n_bits: int,
        measurements: list[IRMeasurement],
    ) -> dict[str, int]:
        if n_bits == 0:
            return {"": int(samples.shape[0])}

        register_codes = np.zeros(samples.shape[0], dtype=np.int64)
        for measurement in measurements:
            measured_bit = (samples >> (n_qubits - 1 - measurement.qubit)) & 1
            shift = n_bits - 1 - measurement.bit
            bit_mask = 1 << shift
            register_codes = (register_codes & ~bit_mask) | (measured_bit << shift)

        counts: dict[str, int] = {}
        if n_bits <= 20:
            histogram = np.bincount(register_codes, minlength=1 << n_bits)
            for code, count in enumerate(histogram):
                count_int = int(count)
                if count_int:
                    counts[format(code, f"0{n_bits}b")] = count_int
            return counts

        unique_codes, unique_counts = np.unique(register_codes, return_counts=True)
        for code, count in zip(unique_codes, unique_counts, strict=True):
            count_int = int(count)
            if count_int:
                counts[format(int(code), f"0{n_bits}b")] = count_int
        return counts

    def run(self, prepared_case: QsimPreparedCase, shots: int, *, warmup: bool = False) -> dict[str, int]:
        _ = warmup
        if self._simulator is None:
            raise RuntimeError("qsim simulator unavailable.")

        result = self._simulator.simulate(prepared_case.circuit)
        state_vector = np.asarray(result.final_state_vector, dtype=np.complex128)
        probs = np.abs(state_vector) ** 2
        probs = probs / np.sum(probs)

        rng = np.random.default_rng()
        basis_samples = rng.choice(probs.shape[0], size=shots, p=probs).astype(np.int64)
        return self._counts_from_samples(
            basis_samples,
            n_qubits=prepared_case.n_qubits,
            n_bits=prepared_case.n_bits,
            measurements=prepared_case.measurements,
        )
