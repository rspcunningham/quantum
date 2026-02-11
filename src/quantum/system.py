"""Quantum system state management."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Callable

import torch

from quantum.gates import Circuit, ConditionalGate, Gate, Measurement, QuantumRegister


def infer_resources(circuit: Circuit) -> tuple[int, int]:
    """Infer (n_qubits, n_bits) by walking all operations recursively."""
    max_qubit = -1
    max_bit = -1

    def _walk(ops: Sequence[Gate | ConditionalGate | Measurement | Circuit]) -> None:
        nonlocal max_qubit, max_bit
        for op in ops:
            if isinstance(op, Circuit):
                _walk(op.operations)
            elif isinstance(op, Gate):
                for target in op.targets:
                    if target > max_qubit:
                        max_qubit = target
            elif isinstance(op, Measurement):
                if op.qubit > max_qubit:
                    max_qubit = op.qubit
                if op.bit > max_bit:
                    max_bit = op.bit
            else:
                for target in op.gate.targets:
                    if target > max_qubit:
                        max_qubit = target

    _walk(circuit.operations)
    return (max_qubit + 1, max_bit + 1)


def _flatten_circuit_operations(
    operations: Sequence[Gate | ConditionalGate | Measurement | Circuit],
    out: list[Gate | ConditionalGate | Measurement],
) -> None:
    """Flatten nested circuits into a linear operation list."""
    for operation in operations:
        if isinstance(operation, Circuit):
            _flatten_circuit_operations(operation.operations, out)
        else:
            out.append(operation)


@dataclass(frozen=True, slots=True)
class _DynamicSegmentStep:
    """Unitary edge: contiguous unconditional gate sequence."""

    gates: tuple[Gate, ...]


@dataclass(frozen=True, slots=True)
class _DynamicConditionalStep:
    """Control edge: contiguous gates for one classical condition."""

    condition: int
    gates: tuple[Gate, ...]


@dataclass(frozen=True, slots=True)
class _DynamicMeasurementStep:
    """Control node: measurement barrier that may split branch paths."""

    measurement: Measurement


type _DynamicGraphStep = _DynamicSegmentStep | _DynamicConditionalStep | _DynamicMeasurementStep


@dataclass(frozen=True, slots=True)
class _DynamicCompiledGraph:
    """Compiled execution graph with explicit sink-phase terminal measurements."""

    steps: tuple[_DynamicGraphStep, ...]
    node_count: int
    terminal_measurements: tuple[Measurement, ...]


def _compile_execution_graph(circuit: Circuit) -> _DynamicCompiledGraph:
    """Compile any circuit into segment/conditional/measurement steps."""
    flattened: list[Gate | ConditionalGate | Measurement] = []
    _flatten_circuit_operations(circuit.operations, flattened)

    terminal_start = len(flattened)
    while terminal_start > 0 and isinstance(flattened[terminal_start - 1], Measurement):
        terminal_start -= 1
    terminal_measurements = tuple(flattened[terminal_start:])
    active_operations = flattened[:terminal_start]

    steps: list[_DynamicGraphStep] = []
    node_count = 0

    pending_gates: list[Gate] = []
    pending_condition: int | None = None
    pending_conditional_gates: list[Gate] = []

    def _flush_segment() -> None:
        nonlocal pending_gates
        if pending_gates:
            steps.append(_DynamicSegmentStep(gates=tuple(pending_gates)))
            pending_gates = []

    def _flush_conditional() -> None:
        nonlocal pending_condition, pending_conditional_gates
        if pending_condition is not None and pending_conditional_gates:
            steps.append(
                _DynamicConditionalStep(
                    condition=pending_condition,
                    gates=tuple(pending_conditional_gates),
                )
            )
        pending_condition = None
        pending_conditional_gates = []

    for operation in active_operations:
        if isinstance(operation, Gate):
            _flush_conditional()
            pending_gates.append(operation)
            continue

        if isinstance(operation, ConditionalGate):
            _flush_segment()
            if pending_condition != operation.condition:
                _flush_conditional()
                pending_condition = operation.condition
            pending_conditional_gates.append(operation.gate)
            continue

        _flush_segment()
        _flush_conditional()
        steps.append(_DynamicMeasurementStep(measurement=operation))
        node_count += 1

    _flush_segment()
    _flush_conditional()

    return _DynamicCompiledGraph(
        steps=tuple(steps),
        node_count=node_count,
        terminal_measurements=tuple(terminal_measurements),
    )


def _set_classical_bit(value: int, *, bit: int, outcome: int, n_bits: int) -> int:
    """Overwrite one classical register bit in big-endian indexing."""
    if n_bits == 0:
        return value
    shift = n_bits - 1 - bit
    bit_mask = 1 << shift
    return (value & ~bit_mask) | (outcome << shift)


def _counts_from_dynamic_branch_paths(
    branches: dict[tuple[int, int], int],
    *,
    n_bits: int,
    num_shots: int,
) -> dict[str, int]:
    """Convert dynamic path counts keyed by (state_id, classical_value)."""
    if num_shots == 0:
        return {}
    if n_bits == 0:
        return {"": num_shots}

    counts: dict[str, int] = {}
    for (_, classical_value), shots in branches.items():
        if shots <= 0:
            continue
        key = format(classical_value, f"0{n_bits}b")
        counts[key] = counts.get(key, 0) + shots
    return counts


def _counts_from_register_codes(
    register_codes: torch.Tensor,
    *,
    n_bits: int,
    num_shots: int,
) -> dict[str, int]:
    """Convert encoded classical-register integers to output count dict."""
    if num_shots == 0:
        return {}
    if n_bits == 0:
        return {"": num_shots}

    counts: dict[str, int] = {}

    if n_bits <= 20:
        histogram = torch.bincount(register_codes, minlength=1 << n_bits)
        nonzero_codes = torch.nonzero(histogram, as_tuple=False).flatten()
        if nonzero_codes.numel() == 0:
            return counts

        values = histogram[nonzero_codes].to(dtype=torch.int64)
        for code, count in zip(nonzero_codes.cpu().tolist(), values.cpu().tolist(), strict=True):
            counts[format(int(code), f"0{n_bits}b")] = int(count)
        return counts

    unique_codes, unique_counts = torch.unique(register_codes, return_counts=True, sorted=True)
    for code, count in zip(unique_codes.cpu().tolist(), unique_counts.cpu().tolist(), strict=True):
        count_int = int(count)
        if count_int:
            counts[format(int(code), f"0{n_bits}b")] = count_int
    return counts


def _accumulate_count_dict(dst: dict[str, int], src: dict[str, int]) -> None:
    for key, value in src.items():
        dst[key] = dst.get(key, 0) + value


def _sample_terminal_measurements_from_branches(
    *,
    branches: dict[tuple[int, int], int],
    state_arena: dict[int, torch.Tensor],
    terminal_measurements: tuple[Measurement, ...],
    num_shots: int,
    n_qubits: int,
    n_bits: int,
) -> dict[str, int]:
    """Sample terminal measurements from branch-weighted final states."""
    if not terminal_measurements:
        return _counts_from_dynamic_branch_paths(branches, n_bits=n_bits, num_shots=num_shots)

    sampled_counts: dict[str, int] = {}
    single_terminal_measurement = terminal_measurements[0] if len(terminal_measurements) == 1 else None

    if single_terminal_measurement is not None:
        measurement = single_terminal_measurement
        qubit_shift = n_qubits - 1 - measurement.qubit
        bit_shift = n_bits - 1 - measurement.bit
        bit_mask = 1 << bit_shift
        any_state = next(iter(state_arena.values()))
        indices = torch.arange(1 << n_qubits, device=any_state.device, dtype=torch.int64)
        measured_mask = ((indices >> qubit_shift) & 1).to(torch.float32)
        state_prob_cache: dict[int, float] = {}

        for (state_id, classical_value), shots in branches.items():
            if shots <= 0:
                continue

            p1 = state_prob_cache.get(state_id)
            if p1 is None:
                probs = torch.abs(state_arena[state_id][0]) ** 2
                p1 = float((probs @ measured_mask).clamp(0.0, 1.0).item())
                state_prob_cache[state_id] = p1

            shots_1 = _sample_binomial_count(shots=shots, p1=p1)
            shots_0 = shots - shots_1

            if shots_0 > 0:
                classical_0 = classical_value & ~bit_mask
                key_0 = format(classical_0, f"0{n_bits}b")
                sampled_counts[key_0] = sampled_counts.get(key_0, 0) + shots_0

            if shots_1 > 0:
                classical_1 = classical_value | bit_mask
                key_1 = format(classical_1, f"0{n_bits}b")
                sampled_counts[key_1] = sampled_counts.get(key_1, 0) + shots_1

        return sampled_counts

    state_prob_cache: dict[int, torch.Tensor] = {}

    measurement_specs = tuple(
        (n_qubits - 1 - measurement.qubit, n_bits - 1 - measurement.bit)
        for measurement in terminal_measurements
    )

    for (state_id, classical_value), shots in branches.items():
        if shots <= 0:
            continue

        probs = state_prob_cache.get(state_id)
        if probs is None:
            probs = torch.abs(state_arena[state_id][0]) ** 2
            probs = probs / probs.sum().clamp_min(1e-12)
            state_prob_cache[state_id] = probs

        sampling_probs = probs
        if sampling_probs.device.type == "mps":
            sampling_probs = sampling_probs.to("cpu")

        samples = torch.multinomial(sampling_probs, shots, replacement=True).to(dtype=torch.int64)
        register_codes = torch.full((shots,), classical_value, dtype=torch.int64, device=samples.device)

        for qubit_shift, bit_shift in measurement_specs:
            measured_bit = (samples >> qubit_shift) & 1
            bit_mask = 1 << bit_shift
            register_codes = (register_codes & ~bit_mask) | (measured_bit << bit_shift)

        _accumulate_count_dict(
            sampled_counts,
            _counts_from_register_codes(register_codes, n_bits=n_bits, num_shots=shots),
        )

    return sampled_counts


def _state_merge_signature(
    state_vector: torch.Tensor,
    *,
    sig_vector_a: torch.Tensor,
    sig_vector_b: torch.Tensor,
    scale: int,
    signature_weights: torch.Tensor,
) -> int:
    """Compact state fingerprint used to bucket equivalent collapsed states."""
    state = state_vector[0]
    projections = torch.stack(
        (
            torch.sum(state * sig_vector_a),
            torch.sum(state * sig_vector_b),
            state[0],
            state[-1],
        )
    )
    values = torch.view_as_real(projections).reshape(-1).to(torch.float32)
    quantized = torch.round(values * scale).to(torch.int64)
    signature = torch.sum(quantized * signature_weights)
    return int(signature.item())


def _sample_binomial_count(*, shots: int, p1: float) -> int:
    """Sample Binomial(shots, p1) with a scalar CPU draw."""
    if shots <= 0:
        return 0
    if p1 <= 0.0:
        return 0
    if p1 >= 1.0:
        return shots

    sample = torch.binomial(
        torch.tensor(float(shots), dtype=torch.float32),
        torch.tensor(float(p1), dtype=torch.float32),
    )
    return int(sample.item())


def _advance_branches_with_gate_sequence(
    *,
    branches: dict[tuple[int, int], int],
    state_arena: dict[int, torch.Tensor],
    gate_system: "BatchedQuantumSystem",
    gates: tuple[Gate, ...],
    add_state: Callable[[torch.Tensor], int],
    condition: int | None,
) -> dict[tuple[int, int], int]:
    """Advance branch states through a contiguous gate run."""
    transition_cache: dict[int, int] = {}
    next_branches: dict[tuple[int, int], int] = {}

    for (state_id, classical_value), shots in branches.items():
        if shots <= 0:
            continue

        if condition is not None and classical_value != condition:
            key = (state_id, classical_value)
            next_branches[key] = next_branches.get(key, 0) + shots
            continue

        next_state_id = transition_cache.get(state_id)
        if next_state_id is None:
            gate_system.state_vectors = state_arena[state_id]
            for gate in gates:
                _ = gate_system.apply_gate(gate)
            next_state_id = add_state(gate_system.state_vectors)
            transition_cache[state_id] = next_state_id

        key = (next_state_id, classical_value)
        next_branches[key] = next_branches.get(key, 0) + shots

    return next_branches


def _advance_branches_with_measurement(
    *,
    branches: dict[tuple[int, int], int],
    state_arena: dict[int, torch.Tensor],
    gate_system: "BatchedQuantumSystem",
    measurement: Measurement,
    n_bits: int,
    measurement_masks: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    add_state: Callable[[torch.Tensor], int],
    sig_vector_a: torch.Tensor,
    sig_vector_b: torch.Tensor,
    signature_scale: int,
    signature_weights: torch.Tensor,
) -> dict[tuple[int, int], int]:
    """Advance branches through one non-terminal measurement step."""
    mask_entry = measurement_masks.get(measurement.qubit)
    if mask_entry is None:
        mask_1 = gate_system._measurement_mask_for_qubit(measurement.qubit)
        weight_1 = gate_system._measurement_weight_for_qubit(measurement.qubit)
        keep_0 = (~mask_1).to(torch.complex64).unsqueeze(0)
        keep_1 = mask_1.to(torch.complex64).unsqueeze(0)
        measurement_masks[measurement.qubit] = (weight_1, keep_0, keep_1)
    else:
        weight_1, keep_0, keep_1 = mask_entry

    state_prob_cache: dict[int, float] = {}
    outcome_state_cache: dict[tuple[int, int], int] = {}
    merged_state_by_signature: dict[int, int] = {}
    next_branches: dict[tuple[int, int], int] = {}

    for (state_id, classical_value), shots in branches.items():
        if shots <= 0:
            continue

        p1 = state_prob_cache.get(state_id)
        if p1 is None:
            probs = torch.abs(state_arena[state_id][0]) ** 2
            p1_tensor = (probs @ weight_1).clamp(0.0, 1.0)
            p1 = float(p1_tensor.item())
            state_prob_cache[state_id] = p1

        shots_1 = _sample_binomial_count(shots=shots, p1=p1)
        shots_0 = shots - shots_1

        if shots_0 > 0:
            outcome_key = (state_id, 0)
            state_0_id = outcome_state_cache.get(outcome_key)
            if state_0_id is None:
                state_0 = state_arena[state_id] * keep_0
                norm_0 = max(1.0 - p1, 1e-12) ** 0.5
                state_0 = state_0 / norm_0
                signature_0 = _state_merge_signature(
                    state_0,
                    sig_vector_a=sig_vector_a,
                    sig_vector_b=sig_vector_b,
                    scale=signature_scale,
                    signature_weights=signature_weights,
                )
                state_0_id = merged_state_by_signature.get(signature_0)
                if state_0_id is None:
                    state_0_id = add_state(state_0)
                    merged_state_by_signature[signature_0] = state_0_id
                outcome_state_cache[outcome_key] = state_0_id

            classical_0 = _set_classical_bit(
                classical_value,
                bit=measurement.bit,
                outcome=0,
                n_bits=n_bits,
            )
            key = (state_0_id, classical_0)
            next_branches[key] = next_branches.get(key, 0) + shots_0

        if shots_1 > 0:
            outcome_key = (state_id, 1)
            state_1_id = outcome_state_cache.get(outcome_key)
            if state_1_id is None:
                state_1 = state_arena[state_id] * keep_1
                norm_1 = max(p1, 1e-12) ** 0.5
                state_1 = state_1 / norm_1
                signature_1 = _state_merge_signature(
                    state_1,
                    sig_vector_a=sig_vector_a,
                    sig_vector_b=sig_vector_b,
                    scale=signature_scale,
                    signature_weights=signature_weights,
                )
                state_1_id = merged_state_by_signature.get(signature_1)
                if state_1_id is None:
                    state_1_id = add_state(state_1)
                    merged_state_by_signature[signature_1] = state_1_id
                outcome_state_cache[outcome_key] = state_1_id

            classical_1 = _set_classical_bit(
                classical_value,
                bit=measurement.bit,
                outcome=1,
                n_bits=n_bits,
            )
            key = (state_1_id, classical_1)
            next_branches[key] = next_branches.get(key, 0) + shots_1

    return next_branches


def _compact_state_arena(
    state_arena: dict[int, torch.Tensor],
    branches: dict[tuple[int, int], int],
) -> None:
    """Drop unreferenced states to keep memory bounded."""
    live_state_ids = {state_id for (state_id, _), shots in branches.items() if shots > 0}
    stale_state_ids = [state_id for state_id in state_arena if state_id not in live_state_ids]
    for state_id in stale_state_ids:
        del state_arena[state_id]


@torch.inference_mode()
def _run_compiled_simulation(
    *,
    circuit: Circuit,
    num_shots: int,
    n_qubits: int,
    n_bits: int,
    device: torch.device,
) -> dict[str, int]:
    """Unified compiled executor for static and dynamic circuits."""
    if num_shots == 0:
        return {}

    compiled = _compile_execution_graph(circuit)

    gate_system = BatchedQuantumSystem(
        n_qubits=n_qubits,
        n_bits=n_bits,
        batch_size=1,
        device=device,
    )

    state_arena: dict[int, torch.Tensor] = {0: gate_system.state_vectors}
    next_state_id = 1

    def _add_state(state_vector: torch.Tensor) -> int:
        nonlocal next_state_id
        state_id = next_state_id
        state_arena[state_id] = state_vector
        next_state_id += 1
        return state_id

    branches: dict[tuple[int, int], int] = {(0, 0): num_shots}

    signature_scale = 1_000_000
    signature_weights: torch.Tensor | None = None
    sig_vector_a: torch.Tensor | None = None
    sig_vector_b: torch.Tensor | None = None
    measurement_masks: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    if compiled.node_count > 0:
        dim = 1 << n_qubits
        indices = torch.arange(dim, device=device, dtype=torch.float32)
        phase_a = 0.731 * indices + 0.119 * (indices * indices)
        phase_b = 1.213 * indices + 0.071 * (indices * indices)
        sig_vector_a = (torch.cos(phase_a) + 1j * torch.sin(phase_a)).to(torch.complex64)
        sig_vector_b = (torch.cos(phase_b) + 1j * torch.sin(phase_b)).to(torch.complex64)
        signature_weights = torch.tensor(
            [
                1_000_003,
                1_000_033,
                1_000_037,
                1_000_039,
                1_000_081,
                1_000_099,
                1_000_117,
                1_000_121,
            ],
            dtype=torch.int64,
            device=device,
        )

    for step in compiled.steps:
        if isinstance(step, _DynamicSegmentStep):
            branches = _advance_branches_with_gate_sequence(
                branches=branches,
                state_arena=state_arena,
                gate_system=gate_system,
                gates=step.gates,
                add_state=_add_state,
                condition=None,
            )

        elif isinstance(step, _DynamicConditionalStep):
            branches = _advance_branches_with_gate_sequence(
                branches=branches,
                state_arena=state_arena,
                gate_system=gate_system,
                gates=step.gates,
                add_state=_add_state,
                condition=step.condition,
            )

        else:
            if sig_vector_a is None or sig_vector_b is None or signature_weights is None:
                raise RuntimeError("Execution graph invariant violated: missing signature tensors for measurement step")

            branches = _advance_branches_with_measurement(
                branches=branches,
                state_arena=state_arena,
                gate_system=gate_system,
                measurement=step.measurement,
                n_bits=n_bits,
                measurement_masks=measurement_masks,
                add_state=_add_state,
                sig_vector_a=sig_vector_a,
                sig_vector_b=sig_vector_b,
                signature_scale=signature_scale,
                signature_weights=signature_weights,
            )

        _compact_state_arena(state_arena, branches)

    return _sample_terminal_measurements_from_branches(
        branches=branches,
        state_arena=state_arena,
        terminal_measurements=compiled.terminal_measurements,
        num_shots=num_shots,
        n_qubits=n_qubits,
        n_bits=n_bits,
    )


def measure_all(qubits: QuantumRegister | list[int] | int) -> Circuit:
    """Measure qubits into sequential classical bits starting at 0."""
    if isinstance(qubits, int):
        indices = list(range(qubits))
    elif isinstance(qubits, QuantumRegister):
        indices = list(qubits)
    else:
        indices = list(qubits)
    return Circuit([Measurement(qubit, bit) for bit, qubit in enumerate(indices)])


class BatchedQuantumSystem:
    """Batched state-vector system optimized for MPS execution."""

    state_vectors: Annotated[torch.Tensor, "(batch_size, 2^n_qubits) complex64"]
    bit_registers: Annotated[torch.Tensor, "(batch_size, n_bits) int32"]
    n_qubits: int
    n_bits: int
    batch_size: int
    device: torch.device
    _measurement_masks: dict[int, torch.Tensor]
    _measurement_weights: dict[int, torch.Tensor]
    _gate_tensors: dict[int, torch.Tensor]
    _gate_diagonals: dict[int, torch.Tensor]
    _gate_permutations: dict[int, torch.Tensor]
    _gate_permutation_factors: dict[int, torch.Tensor]
    _diagonal_subindices: dict[tuple[int, ...], torch.Tensor]
    _permutation_source_indices: dict[tuple[tuple[int, ...], int], torch.Tensor]
    _permutation_phase_factors: dict[tuple[tuple[int, ...], int], torch.Tensor]

    def __init__(self, n_qubits: int, n_bits: int, batch_size: int, device: torch.device):
        self.n_qubits = n_qubits
        self.n_bits = n_bits
        self.batch_size = batch_size
        self.device = device

        self._measurement_masks = {}
        self._measurement_weights = {}
        self._gate_tensors = {}
        self._gate_diagonals = {}
        self._gate_permutations = {}
        self._gate_permutation_factors = {}
        self._diagonal_subindices = {}
        self._permutation_source_indices = {}
        self._permutation_phase_factors = {}

        dim = 1 << n_qubits
        self.state_vectors = torch.zeros((batch_size, dim), dtype=torch.complex64, device=device)
        self.state_vectors[:, 0] = 1.0
        self.bit_registers = torch.zeros((batch_size, n_bits), dtype=torch.int32, device=device)

    @torch.inference_mode()
    def apply_gate(self, gate: Gate) -> "BatchedQuantumSystem":
        """Apply a gate to all state vectors."""
        if gate.diagonal is not None:
            return self._apply_diagonal_gate(gate)
        if gate.permutation is not None:
            return self._apply_permutation_gate(gate)

        targets = tuple(gate.targets)
        k = len(targets)

        if k == 1:
            return self._apply_dense_single_qubit_gate(gate, targets[0])
        if k == 2:
            return self._apply_dense_two_qubit_gate(gate, targets)

        gate_nd = self._device_gate_tensor(gate.tensor, n_target_qubits=k)

        state = self.state_vectors.view((self.batch_size,) + (2,) * self.n_qubits)
        axes = tuple(t + 1 for t in targets)
        state = torch.tensordot(state, gate_nd, dims=(axes, list(range(k, 2 * k))))
        state = torch.movedim(state, list(range(-k, 0)), list(axes))

        self.state_vectors = state.reshape(self.batch_size, -1)
        return self

    def _apply_dense_single_qubit_gate(self, gate: Gate, target: int) -> "BatchedQuantumSystem":
        """Apply a dense single-qubit gate via stride-based slicing with CPU scalars."""
        g = gate.tensor.reshape(2, 2)
        g00, g01, g10, g11 = complex(g[0, 0]), complex(g[0, 1]), complex(g[1, 0]), complex(g[1, 1])

        a = 1 << target
        b = 1 << (self.n_qubits - target - 1)

        state = self.state_vectors.view(self.batch_size, a, 2, b)
        s0 = state[:, :, 0, :]
        s1 = state[:, :, 1, :]

        new_0 = g00 * s0 + g01 * s1
        new_1 = g10 * s0 + g11 * s1

        self.state_vectors = torch.stack([new_0, new_1], dim=2).reshape(self.batch_size, -1)
        return self

    def _apply_dense_two_qubit_gate(self, gate: Gate, targets: tuple[int, int]) -> "BatchedQuantumSystem":
        """Apply a dense two-qubit gate via stride-based slicing with CPU scalars."""
        t0, t1 = targets
        g = gate.tensor.reshape(4, 4)

        if t0 > t1:
            t0, t1 = t1, t0
            perm = [0, 2, 1, 3]
            r = [[complex(g[perm[i], perm[j]]) for j in range(4)] for i in range(4)]
        else:
            r = [[complex(g[i, j]) for j in range(4)] for i in range(4)]

        a = 1 << t0
        b = 1 << (t1 - t0 - 1)
        c = 1 << (self.n_qubits - t1 - 1)

        state = self.state_vectors.view(self.batch_size, a, 2, b, 2, c)
        s00 = state[:, :, 0, :, 0, :]
        s01 = state[:, :, 0, :, 1, :]
        s10 = state[:, :, 1, :, 0, :]
        s11 = state[:, :, 1, :, 1, :]

        out_00 = r[0][0]*s00 + r[0][1]*s01 + r[0][2]*s10 + r[0][3]*s11
        out_01 = r[1][0]*s00 + r[1][1]*s01 + r[1][2]*s10 + r[1][3]*s11
        out_10 = r[2][0]*s00 + r[2][1]*s01 + r[2][2]*s10 + r[2][3]*s11
        out_11 = r[3][0]*s00 + r[3][1]*s01 + r[3][2]*s10 + r[3][3]*s11

        result = torch.stack([
            torch.stack([out_00, out_01], dim=3),
            torch.stack([out_10, out_11], dim=3),
        ], dim=2)

        self.state_vectors = result.reshape(self.batch_size, -1)
        return self

    def _apply_diagonal_gate(self, gate: Gate) -> "BatchedQuantumSystem":
        diagonal = gate.diagonal
        assert diagonal is not None

        targets = tuple(gate.targets)
        k = len(targets)

        if k == 1:
            d0, d1 = complex(diagonal[0]), complex(diagonal[1])
            t = targets[0]
            a = 1 << t
            b = 1 << (self.n_qubits - t - 1)
            state = self.state_vectors.view(self.batch_size, a, 2, b)
            self.state_vectors = torch.stack(
                [d0 * state[:, :, 0, :], d1 * state[:, :, 1, :]], dim=2,
            ).reshape(self.batch_size, -1)
            return self

        if k == 2:
            t0, t1 = targets
            if t0 > t1:
                t0, t1 = t1, t0
                swap = [0, 2, 1, 3]
                d = [complex(diagonal[swap[i]]) for i in range(4)]
            else:
                d = [complex(diagonal[i]) for i in range(4)]
            a = 1 << t0
            b = 1 << (t1 - t0 - 1)
            c = 1 << (self.n_qubits - t1 - 1)
            state = self.state_vectors.view(self.batch_size, a, 2, b, 2, c)
            result = torch.stack([
                torch.stack([d[0] * state[:, :, 0, :, 0, :], d[1] * state[:, :, 0, :, 1, :]], dim=3),
                torch.stack([d[2] * state[:, :, 1, :, 0, :], d[3] * state[:, :, 1, :, 1, :]], dim=3),
            ], dim=2)
            self.state_vectors = result.reshape(self.batch_size, -1)
            return self

        diagonal_device = self._device_gate_diagonal(diagonal)
        subindex = self._diagonal_subindex_for_targets(targets)
        factors = diagonal_device[subindex]
        self.state_vectors = self.state_vectors * factors.unsqueeze(0)
        return self

    def _apply_permutation_gate(self, gate: Gate) -> "BatchedQuantumSystem":
        permutation = gate.permutation
        assert permutation is not None

        permutation_device = self._device_gate_permutation(permutation)
        targets = tuple(gate.targets)
        source_indices = self._permutation_source_indices_for_targets(targets, permutation_device)
        state = self.state_vectors[:, source_indices]

        factors = gate.permutation_factors
        if factors is not None:
            factors_device = self._device_gate_permutation_factors(factors)
            phase_factors = self._permutation_phase_factors_for_targets(targets, factors_device)
            state = state * phase_factors.unsqueeze(0)

        self.state_vectors = state
        return self

    @torch.inference_mode()
    def apply_measurement(self, measurement: Measurement) -> "BatchedQuantumSystem":
        """Apply projective measurement to all batch elements."""
        qubit = measurement.qubit
        bit = measurement.bit

        if qubit < 0 or qubit >= self.n_qubits:
            raise ValueError(f"Qubit index {qubit} out of range [0, {self.n_qubits})")
        if bit < 0 or bit >= self.n_bits:
            raise ValueError(f"Classical bit index {bit} out of range [0, {self.n_bits})")

        mask_1 = self._measurement_mask_for_qubit(qubit)
        probs = torch.abs(self.state_vectors) ** 2
        p1 = (probs @ self._measurement_weight_for_qubit(qubit)).clamp(0.0, 1.0)

        outcomes = (torch.rand(self.batch_size, device=self.device) < p1).to(torch.int32)
        self.bit_registers[:, bit] = outcomes

        keep = mask_1.unsqueeze(0) == outcomes.unsqueeze(1).bool()
        self.state_vectors = self.state_vectors * keep

        norms = torch.sqrt((torch.abs(self.state_vectors) ** 2).sum(dim=1, keepdim=True))
        self.state_vectors = self.state_vectors / norms.clamp_min(1e-12)
        return self

    def _measurement_mask_for_qubit(self, qubit: int) -> torch.Tensor:
        mask = self._measurement_masks.get(qubit)
        if mask is not None:
            return mask

        bitpos = self.n_qubits - 1 - qubit
        indices = torch.arange(1 << self.n_qubits, device=self.device)
        mask = ((indices >> bitpos) & 1).bool()
        self._measurement_masks[qubit] = mask
        return mask

    def _measurement_weight_for_qubit(self, qubit: int) -> torch.Tensor:
        weight = self._measurement_weights.get(qubit)
        if weight is not None:
            return weight

        weight = self._measurement_mask_for_qubit(qubit).to(torch.float32)
        self._measurement_weights[qubit] = weight
        return weight

    def _device_gate_tensor(self, tensor: torch.Tensor, *, n_target_qubits: int) -> torch.Tensor:
        key = id(tensor)
        cached = self._gate_tensors.get(key)
        if cached is not None:
            return cached

        moved = tensor if tensor.device == self.device else tensor.to(self.device)
        moved = moved.reshape((2,) * n_target_qubits + (2,) * n_target_qubits)
        self._gate_tensors[key] = moved
        return moved

    def _device_gate_diagonal(self, diagonal: torch.Tensor) -> torch.Tensor:
        key = id(diagonal)
        cached = self._gate_diagonals.get(key)
        if cached is not None:
            return cached

        moved = diagonal if diagonal.device == self.device else diagonal.to(self.device)
        self._gate_diagonals[key] = moved
        return moved

    def _device_gate_permutation(self, permutation: torch.Tensor) -> torch.Tensor:
        key = id(permutation)
        cached = self._gate_permutations.get(key)
        if cached is not None:
            return cached

        permutation_i64 = permutation.to(dtype=torch.int64)
        moved = permutation_i64 if permutation_i64.device == self.device else permutation_i64.to(self.device)
        self._gate_permutations[key] = moved
        return moved

    def _device_gate_permutation_factors(self, factors: torch.Tensor) -> torch.Tensor:
        key = id(factors)
        cached = self._gate_permutation_factors.get(key)
        if cached is not None:
            return cached

        moved = factors if factors.device == self.device else factors.to(self.device)
        self._gate_permutation_factors[key] = moved
        return moved

    def _diagonal_subindex_for_targets(self, targets: tuple[int, ...]) -> torch.Tensor:
        cached = self._diagonal_subindices.get(targets)
        if cached is not None:
            return cached

        indices = torch.arange(1 << self.n_qubits, device=self.device, dtype=torch.int64)
        subindex = torch.zeros_like(indices)
        k = len(targets)

        for out_pos, target in enumerate(targets):
            bitpos = self.n_qubits - 1 - target
            bit = (indices >> bitpos) & 1
            subindex = subindex | (bit << (k - out_pos - 1))

        self._diagonal_subindices[targets] = subindex
        return subindex

    def _permutation_source_indices_for_targets(
        self,
        targets: tuple[int, ...],
        permutation: torch.Tensor,
    ) -> torch.Tensor:
        key = (targets, id(permutation))
        cached = self._permutation_source_indices.get(key)
        if cached is not None:
            return cached

        indices = torch.arange(1 << self.n_qubits, device=self.device, dtype=torch.int64)
        subindex = self._diagonal_subindex_for_targets(targets)
        source_subindex = permutation[subindex]

        target_mask = 0
        for target in targets:
            target_mask |= 1 << (self.n_qubits - 1 - target)
        clear_mask = (1 << self.n_qubits) - 1 - target_mask

        source_indices = indices & clear_mask
        k = len(targets)
        for out_pos, target in enumerate(targets):
            bitpos = self.n_qubits - 1 - target
            bit = (source_subindex >> (k - out_pos - 1)) & 1
            source_indices = source_indices | (bit << bitpos)

        self._permutation_source_indices[key] = source_indices
        return source_indices

    def _permutation_phase_factors_for_targets(
        self,
        targets: tuple[int, ...],
        factors: torch.Tensor,
    ) -> torch.Tensor:
        key = (targets, id(factors))
        cached = self._permutation_phase_factors.get(key)
        if cached is not None:
            return cached

        subindex = self._diagonal_subindex_for_targets(targets)
        phase_factors = factors[subindex]
        self._permutation_phase_factors[key] = phase_factors
        return phase_factors


@torch.inference_mode()
def run_simulation(
    circuit: Circuit,
    num_shots: int,
    *,
    n_qubits: int | None = None,
    n_bits: int | None = None,
    device: torch.device | None = None,
) -> dict[str, int]:
    """Run a quantum circuit simulation and return measurement counts."""
    if n_qubits is None or n_bits is None:
        inferred_qubits, inferred_bits = infer_resources(circuit)
        if n_qubits is None:
            n_qubits = inferred_qubits
        if n_bits is None:
            n_bits = inferred_bits

    if device is None:
        device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    return _run_compiled_simulation(
        circuit=circuit,
        num_shots=num_shots,
        n_qubits=n_qubits,
        n_bits=n_bits,
        device=device,
    )


class QuantumSystem:
    """Single-state wrapper over the batched MPS execution kernels."""

    state_vector: Annotated[torch.Tensor, "(2^n_qubits, 1) complex64"]
    bit_register: Annotated[list[int], "(n_bits) bit string"]
    n_qubits: int
    n_bits: int
    dimensions: int
    device: torch.device

    def __init__(
        self,
        n_qubits: int,
        n_bits: int = 0,
        state_vector: torch.Tensor | None = None,
        *,
        device: torch.device | None = None,
    ):
        if n_qubits <= 0:
            raise ValueError(f"Number of qubits must be positive, got {n_qubits}")
        if n_bits < 0:
            raise ValueError(f"Number of classical bits must be non-negative, got {n_bits}")

        self.n_qubits = n_qubits
        self.n_bits = n_bits
        self.dimensions = 1 << n_qubits
        self.device = (
            device
            if device is not None
            else torch.device(
                "mps"
                if torch.backends.mps.is_available()
                else "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self._system = BatchedQuantumSystem(
            n_qubits=n_qubits,
            n_bits=n_bits,
            batch_size=1,
            device=self.device,
        )

        if state_vector is not None:
            flat = state_vector.reshape(-1)
            if int(flat.shape[0]) != self.dimensions:
                raise ValueError(
                    f"State vector must have {self.dimensions} amplitudes for {n_qubits} qubits, "
                    f"got shape {tuple(state_vector.shape)}"
                )
            self._system.state_vectors[0] = flat.to(dtype=torch.complex64, device=self.device)

        self._sync_views()

    def _sync_views(self) -> None:
        self.state_vector = self._system.state_vectors[0].unsqueeze(1)
        if self.n_bits == 0:
            self.bit_register = []
        else:
            self.bit_register = [int(v) for v in self._system.bit_registers[0].cpu().tolist()]

    def get_distribution(self) -> torch.Tensor:
        return torch.abs(self.state_vector) ** 2

    def get_bits_value(self) -> int:
        result = 0
        for bit in self.bit_register:
            result = (result << 1) | bit
        return result

    def sample(self, num_shots: int) -> list[int]:
        probs = torch.abs(self._system.state_vectors[0]) ** 2
        probs = probs / probs.sum().clamp_min(1e-12)
        values = torch.multinomial(probs, num_shots, replacement=True)
        return [int(v) for v in values.cpu().tolist()]

    @torch.inference_mode()
    def apply_gate(self, gate: Gate) -> "QuantumSystem":
        _ = self._system.apply_gate(gate)
        self._sync_views()
        return self

    @torch.inference_mode()
    def apply_measurement(self, measurement: Measurement) -> "QuantumSystem":
        _ = self._system.apply_measurement(measurement)
        self._sync_views()
        return self

    @torch.inference_mode()
    def apply_one(self, operation: Gate | Measurement | ConditionalGate) -> "QuantumSystem":
        if isinstance(operation, Gate):
            return self.apply_gate(operation)
        if isinstance(operation, Measurement):
            return self.apply_measurement(operation)

        if self.get_bits_value() == operation.condition:
            return self.apply_gate(operation.gate)
        return self

    @torch.inference_mode()
    def apply_circuit(self, circuit: Circuit) -> "QuantumSystem":
        for operation in circuit.operations:
            if isinstance(operation, Circuit):
                _ = self.apply_circuit(operation)
            else:
                _ = self.apply_one(operation)
        return self

    def __repr__(self) -> str:
        return (
            f"QuantumSystem(n_qubits={self.n_qubits}, n_bits={self.n_bits}, "
            f"device='{self.device.type}')"
        )
