from __future__ import annotations

import os
from typing import Any

from quantum.gates import Circuit

try:
    import quantum_native_runtime as _native_runtime
except Exception as exc:  # pragma: no cover - surfaced on first use
    _native_runtime = None
    _native_import_error = exc
else:
    _native_import_error = None


_native_runtime_initialized = False


def _ensure_native_runtime() -> Any:
    global _native_runtime_initialized

    if _native_runtime is None:
        raise RuntimeError(
            "Native Metal runtime extension is unavailable. "
            "Reinstall/build the project so `quantum_native_runtime` is present."
        ) from _native_import_error

    if not _native_runtime_initialized:
        module_path = getattr(_native_runtime, "__file__", "")
        if module_path:
            _native_runtime.set_module_file_path(module_path)

        override_path = os.environ.get("QUANTUM_METAL_METALLIB")
        if override_path:
            _native_runtime.set_metallib_path_override(override_path)

        _native_runtime_initialized = True

    return _native_runtime


def execute_static_circuit(
    circuit: Any,
    *,
    n_qubits: int,
    n_bits: int,
    num_shots: int,
    seed: int | None = None,
) -> dict[str, int]:
    """Compile + execute a static circuit via native Metal runtime."""
    if num_shots == 0:
        return {}

    native = _ensure_native_runtime()
    if not isinstance(circuit, Circuit):
        raise RuntimeError("Circuit input must be a quantum.gates.Circuit instance.")

    flat_ops = circuit.flatten_native()
    raw = native.run_circuit(
        flat_ops,
        int(n_qubits),
        int(n_bits),
        int(num_shots),
        None if seed is None else int(seed),
    )
    return {str(bits): int(count) for bits, count in raw.items()}
