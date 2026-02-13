from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from quantum.gates import Circuit, ConditionalGate, Gate, Measurement
from quantum.metal_program import (
    BlockKind,
    BlockPlan,
    CanonicalGraph,
    CanonicalOp,
    DispatchGroup,
    MetalProgram,
    OpCode,
    ProgramManifest,
    TerminatorKind,
)

ABI_VERSION = 1
PLANNER_VERSION = 2


@dataclass(frozen=True, slots=True)
class _PoolOffset:
    offset: int
    length: int


def _mps_device_signature() -> str:
    return f"torch-{torch.__version__}|mps"


def _complex_tensor_to_np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.complex64, copy=False).reshape(-1)


def _hash_gate(hasher: hashlib._Hash, gate: Gate) -> None:
    hasher.update(b"G")
    hasher.update(len(gate.targets).to_bytes(2, "little", signed=False))
    for target in gate.targets:
        hasher.update(int(target).to_bytes(2, "little", signed=True))

    if gate.diagonal is not None:
        hasher.update(b"D")
        diag = _complex_tensor_to_np(gate.diagonal)
        hasher.update(diag.tobytes())
        return

    if gate.permutation is not None:
        hasher.update(b"P")
        perm = gate.permutation.detach().cpu().numpy().astype(np.int32, copy=False).reshape(-1)
        hasher.update(perm.tobytes())
        if gate.permutation_factors is not None:
            hasher.update(b"F")
            factors = _complex_tensor_to_np(gate.permutation_factors)
            hasher.update(factors.tobytes())
        else:
            hasher.update(b"N")
        return

    hasher.update(b"T")
    dense = _complex_tensor_to_np(gate.tensor)
    hasher.update(dense.tobytes())


def _hash_operation(hasher: hashlib._Hash, operation: Gate | ConditionalGate | Measurement | Circuit) -> None:
    if isinstance(operation, Circuit):
        hasher.update(b"C[")
        for child in operation.operations:
            _hash_operation(hasher, child)
        hasher.update(b"]")
        return

    if isinstance(operation, Measurement):
        hasher.update(b"M")
        hasher.update(int(operation.qubit).to_bytes(2, "little", signed=True))
        hasher.update(int(operation.bit).to_bytes(2, "little", signed=True))
        return

    if isinstance(operation, ConditionalGate):
        hasher.update(b"Q")
        hasher.update(int(operation.condition).to_bytes(8, "little", signed=False))
        _hash_gate(hasher, operation.gate)
        return

    _hash_gate(hasher, operation)


def _structural_circuit_hash(circuit: Circuit, *, n_qubits: int, n_bits: int) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"quantum-metal-program-v1")
    hasher.update(int(n_qubits).to_bytes(4, "little", signed=False))
    hasher.update(int(n_bits).to_bytes(4, "little", signed=False))
    for operation in circuit.operations:
        _hash_operation(hasher, operation)
    return hasher.hexdigest()


def _pool_add_complex(
    values: np.ndarray,
    *,
    pool_re: list[float],
    pool_im: list[float],
    cache: dict[bytes, _PoolOffset],
) -> _PoolOffset:
    values = values.astype(np.complex64, copy=False).reshape(-1)
    key = values.tobytes()
    cached = cache.get(key)
    if cached is not None:
        return cached

    offset = len(pool_re)
    pool_re.extend(float(v.real) for v in values)
    pool_im.extend(float(v.imag) for v in values)
    result = _PoolOffset(offset=offset, length=int(values.shape[0]))
    cache[key] = result
    return result


def _pool_add_int(
    values: np.ndarray,
    *,
    pool: list[int],
    cache: dict[bytes, _PoolOffset],
) -> _PoolOffset:
    values = values.astype(np.int32, copy=False).reshape(-1)
    key = values.tobytes()
    cached = cache.get(key)
    if cached is not None:
        return cached

    offset = len(pool)
    pool.extend(int(v) for v in values)
    result = _PoolOffset(offset=offset, length=int(values.shape[0]))
    cache[key] = result
    return result


def _targets_offset(targets: tuple[int, ...], target_pool: list[int], cache: dict[tuple[int, ...], _PoolOffset]) -> _PoolOffset:
    cached = cache.get(targets)
    if cached is not None:
        return cached

    offset = len(target_pool)
    target_pool.extend(int(t) for t in targets)
    result = _PoolOffset(offset=offset, length=len(targets))
    cache[targets] = result
    return result


def _kernel_id_for_op(opcode: int, target_len: int) -> int:
    return (int(opcode) << 4) | int(target_len)


def _serialize_monomial_spec(spec: Any) -> dict[str, Any]:
    return {
        "gate_ks": [int(v) for v in spec.gate_ks.detach().cpu().tolist()],
        "gate_targets": [[int(x) for x in pair] for pair in spec.gate_targets.detach().cpu().tolist()],
        "gate_permutations": [[int(x) for x in row] for row in spec.gate_permutations.detach().cpu().tolist()],
        "gate_factors_re": [[float(complex(x).real) for x in row] for row in spec.gate_factors.detach().cpu().tolist()],
        "gate_factors_im": [[float(complex(x).imag) for x in row] for row in spec.gate_factors.detach().cpu().tolist()],
    }


def _lower_compiled_to_program(
    *,
    compiled: Any,
    n_qubits: int,
    n_bits: int,
    use_split_static_executor: bool,
    cache_key: str,
    program_hash: str,
    device_signature: str,
) -> MetalProgram:
    from quantum import system as _system

    target_pool: list[int] = []
    target_pool_cache: dict[tuple[int, ...], _PoolOffset] = {}

    diag_pool_re: list[float] = []
    diag_pool_im: list[float] = []
    diag_cache: dict[bytes, _PoolOffset] = {}

    perm_pool: list[int] = []
    perm_cache: dict[bytes, _PoolOffset] = {}

    phase_pool_re: list[float] = []
    phase_pool_im: list[float] = []
    phase_cache: dict[bytes, _PoolOffset] = {}

    dense_pool_re: list[float] = []
    dense_pool_im: list[float] = []
    dense_cache: dict[bytes, _PoolOffset] = {}

    monomial_specs: list[dict[str, Any]] = []
    monomial_cache: dict[str, int] = {}

    op_table: list[CanonicalOp] = []
    dispatch_table: list[DispatchGroup] = []
    blocks: list[BlockPlan] = []

    lane_id = 1 if use_split_static_executor else 0

    for block_index, step in enumerate(compiled.steps):
        op_start = len(op_table)
        block_kind = BlockKind.SEGMENT
        condition = -1
        measurement_qubit = -1
        measurement_bit = -1

        if isinstance(step, _system._DynamicConditionalStep):
            block_kind = BlockKind.CONDITIONAL
            condition = int(step.condition)
            gates = step.gates
        elif isinstance(step, _system._DynamicSegmentStep):
            block_kind = BlockKind.SEGMENT
            gates = step.gates
        else:
            block_kind = BlockKind.MEASUREMENT
            gates = ()
            measurement_qubit = int(step.measurement.qubit)
            measurement_bit = int(step.measurement.bit)

        for gate in gates:
            stream_spec = _system._gate_monomial_stream_spec(gate)
            targets = tuple(int(t) for t in gate.targets)
            target_meta = _targets_offset(targets, target_pool, target_pool_cache)

            if stream_spec is not None:
                payload = _serialize_monomial_spec(stream_spec)
                payload_key = json.dumps(payload, sort_keys=True, separators=(",", ":"))
                spec_idx = monomial_cache.get(payload_key)
                if spec_idx is None:
                    spec_idx = len(monomial_specs)
                    monomial_specs.append(payload)
                    monomial_cache[payload_key] = spec_idx
                op_table.append(
                    CanonicalOp(
                        opcode=int(OpCode.MONOMIAL_STREAM),
                        target_offset=target_meta.offset,
                        target_len=target_meta.length,
                        coeff_offset=spec_idx,
                        coeff_len=1,
                    )
                )
                continue

            if gate.diagonal is not None:
                diag_values = _complex_tensor_to_np(gate.diagonal)
                diag_meta = _pool_add_complex(
                    diag_values,
                    pool_re=diag_pool_re,
                    pool_im=diag_pool_im,
                    cache=diag_cache,
                )
                opcode = OpCode.DIAG_FULL if len(targets) == n_qubits else OpCode.DIAG_SUBSET
                op_table.append(
                    CanonicalOp(
                        opcode=int(opcode),
                        target_offset=target_meta.offset,
                        target_len=target_meta.length,
                        coeff_offset=diag_meta.offset,
                        coeff_len=diag_meta.length,
                    )
                )
                continue

            if gate.permutation is not None:
                perm_values = gate.permutation.detach().cpu().numpy().astype(np.int32, copy=False)
                perm_meta = _pool_add_int(
                    perm_values,
                    pool=perm_pool,
                    cache=perm_cache,
                )
                phase_offset = -1
                phase_len = 0
                if gate.permutation_factors is not None:
                    phase_values = _complex_tensor_to_np(gate.permutation_factors)
                    phase_meta = _pool_add_complex(
                        phase_values,
                        pool_re=phase_pool_re,
                        pool_im=phase_pool_im,
                        cache=phase_cache,
                    )
                    phase_offset = phase_meta.offset
                    phase_len = phase_meta.length
                opcode = OpCode.PERM_FULL if len(targets) == n_qubits else OpCode.PERM_SUBSET
                op_table.append(
                    CanonicalOp(
                        opcode=int(opcode),
                        target_offset=target_meta.offset,
                        target_len=target_meta.length,
                        coeff_offset=perm_meta.offset,
                        coeff_len=perm_meta.length,
                        aux0=phase_offset,
                        aux1=phase_len,
                    )
                )
                continue

            dense_values = _complex_tensor_to_np(gate.tensor)
            dense_meta = _pool_add_complex(
                dense_values,
                pool_re=dense_pool_re,
                pool_im=dense_pool_im,
                cache=dense_cache,
            )
            op_table.append(
                CanonicalOp(
                    opcode=int(OpCode.DENSE),
                    target_offset=target_meta.offset,
                    target_len=target_meta.length,
                    coeff_offset=dense_meta.offset,
                    coeff_len=dense_meta.length,
                )
            )

        op_count = len(op_table) - op_start
        dispatch_start = len(dispatch_table)

        if op_count > 0:
            group_start = op_start
            prev_kernel = _kernel_id_for_op(op_table[op_start].opcode, op_table[op_start].target_len)
            for idx in range(op_start + 1, op_start + op_count):
                prev_op = op_table[idx - 1]
                curr_op = op_table[idx]
                prev_dense_like = prev_op.opcode in (int(OpCode.DENSE), int(OpCode.MONOMIAL_STREAM))
                curr_dense_like = curr_op.opcode in (int(OpCode.DENSE), int(OpCode.MONOMIAL_STREAM))
                kernel_id = _kernel_id_for_op(curr_op.opcode, curr_op.target_len)
                if prev_dense_like or curr_dense_like or kernel_id != prev_kernel:
                    dispatch_table.append(
                        DispatchGroup(
                            kernel_id=prev_kernel,
                            op_start=group_start,
                            op_count=idx - group_start,
                            lane_id=lane_id,
                        )
                    )
                    group_start = idx
                    prev_kernel = kernel_id
            dispatch_table.append(
                DispatchGroup(
                    kernel_id=prev_kernel,
                    op_start=group_start,
                    op_count=(op_start + op_count) - group_start,
                    lane_id=lane_id,
                )
            )

        dispatch_count = len(dispatch_table) - dispatch_start
        next_block = block_index + 1 if block_index + 1 < len(compiled.steps) else -1

        if block_kind == BlockKind.MEASUREMENT:
            terminator = TerminatorKind.MEASURE_SPLIT if next_block >= 0 else TerminatorKind.RETURN_COUNTS
        else:
            terminator = TerminatorKind.NEXT if next_block >= 0 else TerminatorKind.RETURN_COUNTS

        blocks.append(
            BlockPlan(
                block_index=block_index,
                block_kind=int(block_kind),
                op_start=op_start,
                op_count=op_count,
                condition=condition,
                measurement_qubit=measurement_qubit,
                measurement_bit=measurement_bit,
                terminator=int(terminator),
                next_block=next_block,
                lane_id=lane_id,
                dispatch_start=dispatch_start,
                dispatch_count=dispatch_count,
            )
        )

    graph = CanonicalGraph(blocks=tuple(blocks), entry_block=0)

    switch_count = 0
    host_sync_points = 0
    for block in blocks:
        if block.dispatch_count > 0:
            switch_count += max(0, block.dispatch_count - 1)
        if block.block_kind == int(BlockKind.MEASUREMENT):
            host_sync_points += 1

    manifest = ProgramManifest(
        program_hash=program_hash,
        checksum_sha256="",
        abi_version=ABI_VERSION,
        planner_version=PLANNER_VERSION,
        n_qubits=n_qubits,
        n_bits=n_bits,
        device_signature=device_signature,
        dispatch_count=len(dispatch_table),
        block_count=len(blocks),
        switch_count=switch_count,
        host_sync_points=host_sync_points,
        cache_key=cache_key,
    )

    terminal_measurements = tuple((int(m.qubit), int(m.bit)) for m in compiled.terminal_measurements)

    return MetalProgram(
        manifest=manifest,
        graph=graph,
        op_table=tuple(op_table),
        dispatch_table=tuple(dispatch_table),
        target_pool=tuple(target_pool),
        diag_pool_re=tuple(diag_pool_re),
        diag_pool_im=tuple(diag_pool_im),
        perm_pool=tuple(perm_pool),
        phase_pool_re=tuple(phase_pool_re),
        phase_pool_im=tuple(phase_pool_im),
        dense_pool_re=tuple(dense_pool_re),
        dense_pool_im=tuple(dense_pool_im),
        monomial_specs=tuple(monomial_specs),
        terminal_measurements=terminal_measurements,
        use_split_static_executor=use_split_static_executor,
    )


def _buffer_payload_bytes(program: MetalProgram) -> bytes:
    payload = program.to_buffer_payload()
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _materialize_program(program: MetalProgram) -> MetalProgram:
    buffer_bytes = _buffer_payload_bytes(program)
    checksum = hashlib.sha256(buffer_bytes).hexdigest()

    manifest = ProgramManifest(
        program_hash=program.manifest.program_hash,
        checksum_sha256=checksum,
        abi_version=program.manifest.abi_version,
        planner_version=program.manifest.planner_version,
        n_qubits=program.manifest.n_qubits,
        n_bits=program.manifest.n_bits,
        device_signature=program.manifest.device_signature,
        dispatch_count=program.manifest.dispatch_count,
        block_count=program.manifest.block_count,
        switch_count=program.manifest.switch_count,
        host_sync_points=program.manifest.host_sync_points,
        cache_key=program.manifest.cache_key,
    )

    materialized = MetalProgram(
        manifest=manifest,
        graph=program.graph,
        op_table=program.op_table,
        dispatch_table=program.dispatch_table,
        target_pool=program.target_pool,
        diag_pool_re=program.diag_pool_re,
        diag_pool_im=program.diag_pool_im,
        perm_pool=program.perm_pool,
        phase_pool_re=program.phase_pool_re,
        phase_pool_im=program.phase_pool_im,
        dense_pool_re=program.dense_pool_re,
        dense_pool_im=program.dense_pool_im,
        monomial_specs=program.monomial_specs,
        terminal_measurements=program.terminal_measurements,
        use_split_static_executor=program.use_split_static_executor,
    )

    return materialized


def compile_to_metal_program(
    circuit: Circuit,
    n_qubits: int,
    n_bits: int,
    device_caps: dict[str, Any] | None = None,
) -> MetalProgram:
    """Compile a deterministic static-only MetalProgram for this circuit."""
    del device_caps
    if not torch.backends.mps.is_available():
        raise RuntimeError("Metal runtime required; MPS backend unavailable")

    structural_hash = _structural_circuit_hash(circuit, n_qubits=n_qubits, n_bits=n_bits)
    device_signature = _mps_device_signature()
    cache_key = f"{structural_hash}-q{n_qubits}-b{n_bits}-pv{PLANNER_VERSION}-abi{ABI_VERSION}-{device_signature}"

    from quantum import system as _system

    compilation = _system._compile_circuit_for_metal_program(
        circuit=circuit,
        n_qubits=n_qubits,
        n_bits=n_bits,
        device=torch.device("mps"),
        allow_cpu_fallback=False,
        allow_initial_state_precompute=False,
        cold_static_minimal_fusions=True,
    )
    if compilation.compiled.node_count != 0:
        raise RuntimeError("Dynamic circuits are temporarily unsupported in static-only Metal build.")

    for step in compilation.compiled.steps:
        if not isinstance(step, _system._DynamicSegmentStep):
            raise RuntimeError("Dynamic circuits are temporarily unsupported in static-only Metal build.")

    program = _lower_compiled_to_program(
        compiled=compilation.compiled,
        n_qubits=n_qubits,
        n_bits=n_bits,
        use_split_static_executor=compilation.use_split_static_executor,
        cache_key=cache_key,
        program_hash=structural_hash,
        device_signature=device_signature,
    )

    materialized = _materialize_program(program)
    if os.environ.get("QUANTUM_METAL_DUMP_PLAN") == "1":
        print(
            "[metal-compile] compiled "
            f"dispatch={materialized.manifest.dispatch_count} "
            f"blocks={materialized.manifest.block_count} "
            f"switches={materialized.manifest.switch_count} "
            "cache=disabled"
        )
    return materialized
