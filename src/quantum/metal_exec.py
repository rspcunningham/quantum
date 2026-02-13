from __future__ import annotations

import os
import struct
import weakref
from typing import Any

import numpy as np

from quantum.gates import Circuit
from quantum.metal_program import MetalProgram

try:
    import quantum_native_runtime as _native_runtime
except Exception as exc:  # pragma: no cover - surfaced on first use
    _native_runtime = None
    _native_import_error = exc
else:
    _native_import_error = None


_native_runtime_initialized = False
def _safe_free(native: Any, handle: int) -> None:
    try:
        native.free_program(int(handle))
    except Exception:
        # best-effort cleanup on interpreter shutdown
        return


class _NativeProgramHandle:
    __slots__ = ("handle", "_finalizer", "__weakref__")

    def __init__(self, *, native: Any, handle: int) -> None:
        self.handle = int(handle)
        self._finalizer = weakref.finalize(self, _safe_free, native, self.handle)

    def close(self) -> None:
        self._finalizer()


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


def _pack_monomial_blob(monomial_specs: tuple[dict[str, Any], ...]) -> bytes:
    blob = bytearray()
    blob.extend(struct.pack("<I", len(monomial_specs)))

    for spec in monomial_specs:
        gate_ks = [int(v) for v in spec.get("gate_ks", [])]
        gate_targets = spec.get("gate_targets", [])
        gate_permutations = spec.get("gate_permutations", [])
        gate_factors_re = spec.get("gate_factors_re", [])
        gate_factors_im = spec.get("gate_factors_im", [])

        blob.extend(struct.pack("<I", len(gate_ks)))
        for gate_idx, k_val in enumerate(gate_ks):
            blob.extend(struct.pack("<i", int(k_val)))

            targets_row = gate_targets[gate_idx] if gate_idx < len(gate_targets) else []
            t0 = int(targets_row[0]) if len(targets_row) > 0 else 0
            t1 = int(targets_row[1]) if len(targets_row) > 1 else 0
            blob.extend(struct.pack("<ii", t0, t1))

            perm_row = list(gate_permutations[gate_idx] if gate_idx < len(gate_permutations) else [])
            while len(perm_row) < 4:
                perm_row.append(len(perm_row))
            for value in perm_row[:4]:
                blob.extend(struct.pack("<i", int(value)))

            factors_re_row = list(gate_factors_re[gate_idx] if gate_idx < len(gate_factors_re) else [])
            factors_im_row = list(gate_factors_im[gate_idx] if gate_idx < len(gate_factors_im) else [])
            while len(factors_re_row) < 4:
                factors_re_row.append(1.0)
            while len(factors_im_row) < 4:
                factors_im_row.append(0.0)
            for value in factors_re_row[:4]:
                blob.extend(struct.pack("<f", float(value)))
            for value in factors_im_row[:4]:
                blob.extend(struct.pack("<f", float(value)))

    return bytes(blob)


def _compile_static_native_program(program: MetalProgram) -> _NativeProgramHandle:
    native = _ensure_native_runtime()

    op_table = np.empty(len(program.op_table) * 8, dtype=np.int32)
    for i, op in enumerate(program.op_table):
        base = i * 8
        op_table[base + 0] = int(op.opcode)
        op_table[base + 1] = int(op.target_offset)
        op_table[base + 2] = int(op.target_len)
        op_table[base + 3] = int(op.coeff_offset)
        op_table[base + 4] = int(op.coeff_len)
        op_table[base + 5] = int(op.flags)
        op_table[base + 6] = int(op.aux0)
        op_table[base + 7] = int(op.aux1)

    dispatch_table = np.empty(len(program.dispatch_table) * 4, dtype=np.int32)
    for i, group in enumerate(program.dispatch_table):
        base = i * 4
        dispatch_table[base + 0] = int(group.kernel_id)
        dispatch_table[base + 1] = int(group.op_start)
        dispatch_table[base + 2] = int(group.op_count)
        dispatch_table[base + 3] = int(group.lane_id)

    terminal = np.empty(len(program.terminal_measurements) * 2, dtype=np.int32)
    for i, (qubit, bit) in enumerate(program.terminal_measurements):
        base = i * 2
        terminal[base + 0] = int(qubit)
        terminal[base + 1] = int(bit)

    handle = native.compile_static_program(
        int(program.manifest.n_qubits),
        int(program.manifest.n_bits),
        op_table,
        dispatch_table,
        np.asarray(program.target_pool, dtype=np.int32),
        np.asarray(program.diag_pool_re, dtype=np.float32),
        np.asarray(program.diag_pool_im, dtype=np.float32),
        np.asarray(program.perm_pool, dtype=np.int32),
        np.asarray(program.phase_pool_re, dtype=np.float32),
        np.asarray(program.phase_pool_im, dtype=np.float32),
        np.asarray(program.dense_pool_re, dtype=np.float32),
        np.asarray(program.dense_pool_im, dtype=np.float32),
        _pack_monomial_blob(program.monomial_specs),
        terminal,
    )
    return _NativeProgramHandle(native=native, handle=int(handle))


def _get_native_handle(program: MetalProgram) -> _NativeProgramHandle:
    cached = program.runtime_plan
    if isinstance(cached, _NativeProgramHandle):
        return cached

    handle = _compile_static_native_program(program)
    program.runtime_plan = handle
    return handle


def execute_metal_program(
    program: MetalProgram,
    num_shots: int,
    *,
    seed: int | None = None,
) -> dict[str, int]:
    """Execute a compiled static MetalProgram via native ObjC++ runtime."""
    if num_shots == 0:
        return {}

    handle = _get_native_handle(program)
    native = _ensure_native_runtime()

    raw = native.execute_static_program(
        int(handle.handle),
        int(num_shots),
        None if seed is None else int(seed),
    )
    return {str(bits): int(count) for bits, count in raw.items()}


def execute_static_circuit(
    circuit: Any,
    *,
    n_qubits: int,
    n_bits: int,
    num_shots: int,
    seed: int | None = None,
) -> dict[str, int]:
    """Compile + execute a static circuit via native packed ABI runtime."""
    if num_shots == 0:
        return {}

    native = _ensure_native_runtime()
    if not isinstance(circuit, Circuit):
        raise RuntimeError("Circuit input must be a quantum.gates.Circuit instance.")

    payload = circuit.build_native_static_payload(
        n_qubits=int(n_qubits),
        n_bits=int(n_bits),
    )
    raw = native.run_static_packed(
        payload,
        int(num_shots),
        None if seed is None else int(seed),
    )
    return {str(bits): int(count) for bits, count in raw.items()}
