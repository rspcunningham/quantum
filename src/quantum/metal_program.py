from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class OpCode(IntEnum):
    DIAG_SUBSET = 1
    DIAG_FULL = 2
    PERM_SUBSET = 3
    PERM_FULL = 4
    DENSE = 5
    MONOMIAL_STREAM = 6


class BlockKind(IntEnum):
    SEGMENT = 1
    CONDITIONAL = 2
    MEASUREMENT = 3


class TerminatorKind(IntEnum):
    NEXT = 1
    MEASURE_SPLIT = 2
    RETURN_COUNTS = 3


@dataclass(frozen=True, slots=True)
class CanonicalOp:
    opcode: int
    target_offset: int
    target_len: int
    coeff_offset: int
    coeff_len: int
    flags: int = 0
    aux0: int = -1
    aux1: int = -1

    def to_dict(self) -> dict[str, int]:
        return {
            "opcode": self.opcode,
            "target_offset": self.target_offset,
            "target_len": self.target_len,
            "coeff_offset": self.coeff_offset,
            "coeff_len": self.coeff_len,
            "flags": self.flags,
            "aux0": self.aux0,
            "aux1": self.aux1,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, int]) -> "CanonicalOp":
        return cls(
            opcode=int(payload["opcode"]),
            target_offset=int(payload["target_offset"]),
            target_len=int(payload["target_len"]),
            coeff_offset=int(payload["coeff_offset"]),
            coeff_len=int(payload["coeff_len"]),
            flags=int(payload.get("flags", 0)),
            aux0=int(payload.get("aux0", -1)),
            aux1=int(payload.get("aux1", -1)),
        )


@dataclass(frozen=True, slots=True)
class DispatchGroup:
    kernel_id: int
    op_start: int
    op_count: int
    lane_id: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "kernel_id": self.kernel_id,
            "op_start": self.op_start,
            "op_count": self.op_count,
            "lane_id": self.lane_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, int]) -> "DispatchGroup":
        return cls(
            kernel_id=int(payload["kernel_id"]),
            op_start=int(payload["op_start"]),
            op_count=int(payload["op_count"]),
            lane_id=int(payload.get("lane_id", 0)),
        )


@dataclass(frozen=True, slots=True)
class BlockPlan:
    block_index: int
    block_kind: int
    op_start: int
    op_count: int
    condition: int = -1
    measurement_qubit: int = -1
    measurement_bit: int = -1
    terminator: int = int(TerminatorKind.NEXT)
    next_block: int = -1
    lane_id: int = 0
    dispatch_start: int = 0
    dispatch_count: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "block_index": self.block_index,
            "block_kind": self.block_kind,
            "op_start": self.op_start,
            "op_count": self.op_count,
            "condition": self.condition,
            "measurement_qubit": self.measurement_qubit,
            "measurement_bit": self.measurement_bit,
            "terminator": self.terminator,
            "next_block": self.next_block,
            "lane_id": self.lane_id,
            "dispatch_start": self.dispatch_start,
            "dispatch_count": self.dispatch_count,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, int]) -> "BlockPlan":
        return cls(
            block_index=int(payload["block_index"]),
            block_kind=int(payload["block_kind"]),
            op_start=int(payload["op_start"]),
            op_count=int(payload["op_count"]),
            condition=int(payload.get("condition", -1)),
            measurement_qubit=int(payload.get("measurement_qubit", -1)),
            measurement_bit=int(payload.get("measurement_bit", -1)),
            terminator=int(payload.get("terminator", int(TerminatorKind.NEXT))),
            next_block=int(payload.get("next_block", -1)),
            lane_id=int(payload.get("lane_id", 0)),
            dispatch_start=int(payload.get("dispatch_start", 0)),
            dispatch_count=int(payload.get("dispatch_count", 0)),
        )


@dataclass(frozen=True, slots=True)
class CanonicalGraph:
    blocks: tuple[BlockPlan, ...]
    entry_block: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_block": self.entry_block,
            "blocks": [block.to_dict() for block in self.blocks],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CanonicalGraph":
        return cls(
            blocks=tuple(BlockPlan.from_dict(block) for block in payload["blocks"]),
            entry_block=int(payload.get("entry_block", 0)),
        )


@dataclass(frozen=True, slots=True)
class ProgramManifest:
    program_hash: str
    checksum_sha256: str
    abi_version: int
    planner_version: int
    n_qubits: int
    n_bits: int
    device_signature: str
    dispatch_count: int
    block_count: int
    switch_count: int
    host_sync_points: int
    cache_key: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_hash": self.program_hash,
            "checksum_sha256": self.checksum_sha256,
            "abi_version": self.abi_version,
            "planner_version": self.planner_version,
            "n_qubits": self.n_qubits,
            "n_bits": self.n_bits,
            "device_signature": self.device_signature,
            "dispatch_count": self.dispatch_count,
            "block_count": self.block_count,
            "switch_count": self.switch_count,
            "host_sync_points": self.host_sync_points,
            "cache_key": self.cache_key,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProgramManifest":
        return cls(
            program_hash=str(payload["program_hash"]),
            checksum_sha256=str(payload["checksum_sha256"]),
            abi_version=int(payload["abi_version"]),
            planner_version=int(payload["planner_version"]),
            n_qubits=int(payload["n_qubits"]),
            n_bits=int(payload["n_bits"]),
            device_signature=str(payload["device_signature"]),
            dispatch_count=int(payload.get("dispatch_count", 0)),
            block_count=int(payload.get("block_count", 0)),
            switch_count=int(payload.get("switch_count", 0)),
            host_sync_points=int(payload.get("host_sync_points", 0)),
            cache_key=str(payload.get("cache_key", "")),
        )


@dataclass(slots=True)
class MetalProgram:
    manifest: ProgramManifest
    graph: CanonicalGraph
    op_table: tuple[CanonicalOp, ...]
    dispatch_table: tuple[DispatchGroup, ...]
    target_pool: tuple[int, ...]
    diag_pool_re: tuple[float, ...]
    diag_pool_im: tuple[float, ...]
    perm_pool: tuple[int, ...]
    phase_pool_re: tuple[float, ...]
    phase_pool_im: tuple[float, ...]
    dense_pool_re: tuple[float, ...]
    dense_pool_im: tuple[float, ...]
    monomial_specs: tuple[dict[str, Any], ...]
    terminal_measurements: tuple[tuple[int, int], ...]
    use_split_static_executor: bool

    # runtime-only ephemeral fields (not serialized)
    runtime_plan: Any | None = field(default=None, repr=False)

    def to_buffer_payload(self) -> dict[str, Any]:
        return {
            "graph": self.graph.to_dict(),
            "op_table": [op.to_dict() for op in self.op_table],
            "dispatch_table": [group.to_dict() for group in self.dispatch_table],
            "target_pool": list(self.target_pool),
            "diag_pool_re": list(self.diag_pool_re),
            "diag_pool_im": list(self.diag_pool_im),
            "perm_pool": list(self.perm_pool),
            "phase_pool_re": list(self.phase_pool_re),
            "phase_pool_im": list(self.phase_pool_im),
            "dense_pool_re": list(self.dense_pool_re),
            "dense_pool_im": list(self.dense_pool_im),
            "monomial_specs": list(self.monomial_specs),
            "terminal_measurements": [[int(q), int(b)] for q, b in self.terminal_measurements],
            "use_split_static_executor": bool(self.use_split_static_executor),
        }

    @classmethod
    def from_buffer_payload(
        cls,
        *,
        manifest: ProgramManifest,
        payload: dict[str, Any],
    ) -> "MetalProgram":
        return cls(
            manifest=manifest,
            graph=CanonicalGraph.from_dict(payload["graph"]),
            op_table=tuple(CanonicalOp.from_dict(op) for op in payload["op_table"]),
            dispatch_table=tuple(DispatchGroup.from_dict(group) for group in payload["dispatch_table"]),
            target_pool=tuple(int(v) for v in payload["target_pool"]),
            diag_pool_re=tuple(float(v) for v in payload["diag_pool_re"]),
            diag_pool_im=tuple(float(v) for v in payload["diag_pool_im"]),
            perm_pool=tuple(int(v) for v in payload["perm_pool"]),
            phase_pool_re=tuple(float(v) for v in payload["phase_pool_re"]),
            phase_pool_im=tuple(float(v) for v in payload["phase_pool_im"]),
            dense_pool_re=tuple(float(v) for v in payload["dense_pool_re"]),
            dense_pool_im=tuple(float(v) for v in payload["dense_pool_im"]),
            monomial_specs=tuple(dict(spec) for spec in payload.get("monomial_specs", [])),
            terminal_measurements=tuple((int(q), int(b)) for q, b in payload.get("terminal_measurements", [])),
            use_split_static_executor=bool(payload.get("use_split_static_executor", False)),
        )
