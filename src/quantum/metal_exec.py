from __future__ import annotations

import os
from collections.abc import Mapping
from types import TracebackType
from typing import Protocol, Self, cast, final

from quantum.gates import Circuit
from quantum.system import infer_resources


class _NativeRuntime(Protocol):
    def set_module_file_path(self, module_file_path: str) -> None: ...
    def set_metallib_path_override(self, metallib_path: str) -> None: ...
    def compile_circuit(self, flat_ops: list[object], n_qubits: int, n_bits: int) -> int: ...
    def execute_static_program(
        self,
        handle: int,
        num_shots: int,
        seed: int | None,
        timeout: float,
    ) -> Mapping[str, int]: ...
    def get_program_stats(self, handle: int) -> Mapping[str, int]: ...
    def free_program(self, handle: int) -> None: ...


try:
    import quantum_native_runtime as _imported_native_runtime
except Exception as exc:  # pragma: no cover - surfaced on first use
    _native_runtime: _NativeRuntime | None = None
    _native_import_error = exc
else:
    _native_runtime = cast(_NativeRuntime, _imported_native_runtime)  # pyright: ignore[reportInvalidCast]
    _native_import_error = None


_native_runtime_initialized = False


def _ensure_native_runtime() -> _NativeRuntime:
    global _native_runtime_initialized

    if _native_runtime is None:
        raise RuntimeError(
            "Native Metal runtime extension is unavailable. Reinstall/build the " +
            "project so `quantum_native_runtime` is present."
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


@final
class CompiledCircuit:
    """A static circuit compiled to a native Metal program."""

    __slots__: tuple[str, ...] = ("_native", "_handle", "n_qubits", "n_bits")
    _native: _NativeRuntime
    _handle: int
    n_qubits: int
    n_bits: int

    def __init__(self, native: _NativeRuntime, handle: int, *, n_qubits: int, n_bits: int):
        self._native = native
        self._handle = int(handle)
        self.n_qubits = int(n_qubits)
        self.n_bits = int(n_bits)

    @property
    def closed(self) -> bool:
        return self._handle == 0

    def run(
        self,
        shots: int,
        *,
        seed: int | None = None,
        timeout: float = 0.0,
    ) -> dict[str, int]:
        """Execute this compiled circuit and return measurement counts."""
        if self._handle == 0:
            raise RuntimeError("CompiledCircuit is closed.")
        if shots < 0:
            raise ValueError("shots must be non-negative")
        if shots == 0:
            return {}

        raw = self._native.execute_static_program(
            self._handle,
            int(shots),
            None if seed is None else int(seed),
            float(timeout),
        )
        return {str(bits): int(count) for bits, count in raw.items()}

    def stats(self) -> dict[str, int]:
        if self._handle == 0:
            raise RuntimeError("CompiledCircuit is closed.")
        raw = self._native.get_program_stats(self._handle)
        return {str(key): int(value) for key, value in raw.items()}

    def close(self) -> None:
        handle = self._handle
        if handle == 0:
            return
        self._handle = 0
        self._native.free_program(handle)

    def __enter__(self) -> Self:
        if self._handle == 0:
            raise RuntimeError("CompiledCircuit is closed.")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        _ = exc_type, exc, tb
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def compile(
    circuit: object,
    *,
    n_qubits: int | None = None,
    n_bits: int | None = None,
) -> CompiledCircuit:
    """Compile a static circuit to a reusable native Metal program."""
    if not isinstance(circuit, Circuit):
        raise RuntimeError("Circuit input must be a quantum.gates.Circuit instance.")

    if n_qubits is None or n_bits is None:
        inferred_qubits, inferred_bits = infer_resources(circuit)
        if n_qubits is None:
            n_qubits = inferred_qubits
        if n_bits is None:
            n_bits = inferred_bits

    if n_qubits < 0:
        raise ValueError("n_qubits must be non-negative")
    if n_bits < 0:
        raise ValueError("n_bits must be non-negative")

    native = _ensure_native_runtime()
    flat_ops = cast(list[object], circuit.flatten_native())
    handle = native.compile_circuit(flat_ops, int(n_qubits), int(n_bits))
    return CompiledCircuit(native, handle, n_qubits=int(n_qubits), n_bits=int(n_bits))
