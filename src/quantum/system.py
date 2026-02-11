"""Quantum system state management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, cast, override
from collections.abc import Sequence
import os
import torch
import numpy as np
import numpy.typing as npt
from quantum.gates import Gate, Measurement, ConditionalGate, QuantumRegister, Circuit


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
                for t in op.targets:
                    if t > max_qubit:
                        max_qubit = t
            elif isinstance(op, Measurement):
                if op.qubit > max_qubit:
                    max_qubit = op.qubit
                if op.bit > max_bit:
                    max_bit = op.bit
            else:
                # ConditionalGate — walk the inner gate
                for t in op.gate.targets:
                    if t > max_qubit:
                        max_qubit = t

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


def _terminal_measurement_plan(
    circuit: Circuit,
) -> tuple[list[Gate], list[Measurement]] | None:
    """Return (unitary_gates, terminal_measurements) when fast-path eligible.

    Eligible circuits have:
    - no ConditionalGate operations
    - no gates after the first measurement
    """
    flattened: list[Gate | ConditionalGate | Measurement] = []
    _flatten_circuit_operations(circuit.operations, flattened)

    gates: list[Gate] = []
    measurements: list[Measurement] = []
    seen_measurement = False

    for operation in flattened:
        if isinstance(operation, ConditionalGate):
            return None
        if isinstance(operation, Measurement):
            seen_measurement = True
            measurements.append(operation)
            continue

        # Gate
        if seen_measurement:
            return None
        gates.append(operation)

    return gates, measurements


def _dynamic_operation_plan(
    circuit: Circuit,
) -> list[Gate | ConditionalGate | Measurement] | None:
    """Return flattened operations when a dynamic execution path is required."""
    flattened: list[Gate | ConditionalGate | Measurement] = []
    _flatten_circuit_operations(circuit.operations, flattened)

    has_conditional = False
    has_non_terminal_measurement = False
    seen_measurement = False

    for operation in flattened:
        if isinstance(operation, ConditionalGate):
            has_conditional = True
            continue
        if isinstance(operation, Measurement):
            seen_measurement = True
            continue
        if seen_measurement:
            has_non_terminal_measurement = True

    if not has_conditional and not has_non_terminal_measurement:
        return None
    return flattened


@dataclass
class _DynamicBranchState:
    """One branch in dynamic simulation."""

    state_vector: torch.Tensor  # shape: (1, 2^n_qubits), device-resident
    classical_value: int
    shots: int


def _set_classical_bit(value: int, *, bit: int, outcome: int, n_bits: int) -> int:
    """Overwrite one classical register bit in big-endian indexing."""
    if n_bits == 0:
        return value
    shift = n_bits - 1 - bit
    bit_mask = 1 << shift
    return (value & ~bit_mask) | (outcome << shift)


def _state_merge_signature(
    state_vector: torch.Tensor,
    *,
    sig_vector_a: torch.Tensor,
    sig_vector_b: torch.Tensor,
    scale: int,
) -> tuple[int, ...]:
    """Stable compact signature for branch-state merge bucketing."""
    state = state_vector[0]
    projections = torch.stack((
        torch.sum(state * sig_vector_a),
        torch.sum(state * sig_vector_b),
        state[0],
        state[-1],
    ))
    values = torch.view_as_real(projections).reshape(-1).to(torch.float32)
    values_cpu = cast(npt.NDArray[np.float32], values.cpu().numpy())
    quantized = np.rint(values_cpu * scale).astype(np.int64, copy=False)
    return tuple(int(v) for v in quantized)


def _counts_from_dynamic_branches(
    branches: list[_DynamicBranchState],
    *,
    n_bits: int,
    num_shots: int,
) -> dict[str, int]:
    """Convert dynamic branch shot counts to output bitstring histogram."""
    if num_shots == 0:
        return {}
    if n_bits == 0:
        return {"": num_shots}

    counts: dict[str, int] = {}
    for branch in branches:
        if branch.shots <= 0:
            continue
        key = format(branch.classical_value, f"0{n_bits}b")
        counts[key] = counts.get(key, 0) + branch.shots
    return counts


@torch.inference_mode()
def _run_dynamic_branch_simulation(
    *,
    circuit: Circuit,
    num_shots: int,
    n_qubits: int,
    n_bits: int,
    device: torch.device,
) -> dict[str, int] | None:
    """Branch-based dynamic executor for mid-circuit measurement/conditional circuits."""
    if os.environ.get("QUANTUM_DISABLE_DYNAMIC_BRANCH_ENGINE") == "1":
        return None

    operations = _dynamic_operation_plan(circuit)
    if operations is None:
        return None
    if num_shots == 0:
        return {}

    try:
        min_qubits_for_branch = int(os.environ.get("QUANTUM_DYNAMIC_BRANCH_MIN_QUBITS", "3"))
    except ValueError:
        min_qubits_for_branch = 3
    if n_qubits < min_qubits_for_branch:
        return None

    try:
        branch_cap = int(os.environ.get("QUANTUM_DYNAMIC_BRANCH_CAP", "4096"))
    except ValueError:
        branch_cap = 4096
    if branch_cap <= 0:
        branch_cap = 4096

    gate_system = BatchedQuantumSystem(
        n_qubits=n_qubits,
        n_bits=n_bits,
        batch_size=1,
        device=device,
    )
    dim = 1 << n_qubits
    signature_scale = 10 ** 6
    indices = torch.arange(dim, device=device, dtype=torch.float32)
    phase_a = 0.731 * indices + 0.119 * (indices * indices)
    phase_b = 1.213 * indices + 0.071 * (indices * indices)
    sig_vector_a = (torch.cos(phase_a) + 1j * torch.sin(phase_a)).to(torch.complex64)
    sig_vector_b = (torch.cos(phase_b) + 1j * torch.sin(phase_b)).to(torch.complex64)

    initial_state = torch.zeros((1, dim), dtype=torch.complex64, device=device)
    initial_state[0, 0] = 1.0

    branches: list[_DynamicBranchState] = [
        _DynamicBranchState(
            state_vector=initial_state,
            classical_value=0,
            shots=num_shots,
        )
    ]

    measurement_masks: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    for operation in operations:
        if isinstance(operation, Gate):
            for branch in branches:
                gate_system.state_vectors = branch.state_vector
                _ = gate_system.apply_gate(operation)
                branch.state_vector = gate_system.state_vectors
            continue

        if isinstance(operation, ConditionalGate):
            for branch in branches:
                if branch.classical_value != operation.condition:
                    continue
                gate_system.state_vectors = branch.state_vector
                _ = gate_system.apply_gate(operation.gate)
                branch.state_vector = gate_system.state_vectors
            continue

        # Measurement: split branches by sampled outcomes, then merge compatible branches.
        measurement = operation
        mask_entry = measurement_masks.get(measurement.qubit)
        if mask_entry is None:
            mask_1 = gate_system._measurement_mask_for_qubit(measurement.qubit)
            weight_1 = gate_system._measurement_weight_for_qubit(measurement.qubit)
            keep_0 = (~mask_1).to(torch.complex64).unsqueeze(0)
            keep_1 = mask_1.to(torch.complex64).unsqueeze(0)
            measurement_masks[measurement.qubit] = (weight_1, keep_0, keep_1)
        else:
            weight_1, keep_0, keep_1 = mask_entry

        merged_next: dict[tuple[int, tuple[int, ...]], _DynamicBranchState] = {}

        def _add_merged_branch(*, child_state: torch.Tensor, child_classical: int, child_shots: int) -> None:
            if child_shots <= 0:
                return
            signature = _state_merge_signature(
                child_state,
                sig_vector_a=sig_vector_a,
                sig_vector_b=sig_vector_b,
                scale=signature_scale,
            )
            key = (child_classical, signature)
            existing = merged_next.get(key)
            if existing is None:
                merged_next[key] = _DynamicBranchState(
                    state_vector=child_state,
                    classical_value=child_classical,
                    shots=child_shots,
                )
            else:
                existing.shots += child_shots

        for branch in branches:
            shots = branch.shots
            if shots <= 0:
                continue

            probs = torch.abs(branch.state_vector[0]) ** 2
            p1_tensor = (probs @ weight_1).clamp(0.0, 1.0)
            p1 = float(p1_tensor.item())

            # Aggregate sampling by branch avoids per-shot conditional indexing.
            shots_1 = int(np.random.binomial(shots, p1))
            shots_0 = shots - shots_1

            if shots_0 > 0:
                state_0 = branch.state_vector * keep_0
                norm_0 = torch.sqrt((torch.abs(state_0) ** 2).sum()).clamp_min(1e-12)
                state_0 = state_0 / norm_0
                _add_merged_branch(
                    child_state=state_0,
                    child_classical=_set_classical_bit(
                        branch.classical_value,
                        bit=measurement.bit,
                        outcome=0,
                        n_bits=n_bits,
                    ),
                    child_shots=shots_0,
                )

            if shots_1 > 0:
                state_1 = branch.state_vector * keep_1
                norm_1 = torch.sqrt((torch.abs(state_1) ** 2).sum()).clamp_min(1e-12)
                state_1 = state_1 / norm_1
                _add_merged_branch(
                    child_state=state_1,
                    child_classical=_set_classical_bit(
                        branch.classical_value,
                        bit=measurement.bit,
                        outcome=1,
                        n_bits=n_bits,
                    ),
                    child_shots=shots_1,
                )

        branches = list(merged_next.values())
        if len(branches) > branch_cap:
            # Conservative fallback for branch explosion.
            return None

    return _counts_from_dynamic_branches(branches, n_bits=n_bits, num_shots=num_shots)


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

    # Dense histogram is fastest for benchmark-scale bit widths.
    if n_bits <= 20:
        histogram = torch.bincount(register_codes, minlength=1 << n_bits)
        histogram_cpu = cast(npt.NDArray[np.int64], histogram.cpu().numpy())
        for code, count in enumerate(histogram_cpu):
            count_int = int(count)
            if count_int:
                counts[format(code, f"0{n_bits}b")] = count_int
        return counts

    # Fallback avoids allocating a dense 2^n histogram for large classical registers.
    unique_codes, unique_counts = torch.unique(register_codes, return_counts=True, sorted=True)
    unique_codes_cpu = cast(npt.NDArray[np.int64], unique_codes.cpu().numpy())
    unique_counts_cpu = cast(npt.NDArray[np.int64], unique_counts.cpu().numpy())
    for code, count in zip(unique_codes_cpu, unique_counts_cpu, strict=True):
        count_int = int(count)
        if count_int:
            counts[format(int(code), f"0{n_bits}b")] = count_int
    return counts


@torch.inference_mode()
def _run_terminal_measurement_sampling(
    *,
    circuit: Circuit,
    num_shots: int,
    n_qubits: int,
    n_bits: int,
    device: torch.device,
) -> dict[str, int] | None:
    """Fast path for static circuits with terminal measurements only."""
    plan = _terminal_measurement_plan(circuit)
    if plan is None:
        return None

    if num_shots == 0:
        return {}

    gates, measurements = plan
    system = BatchedQuantumSystem(
        n_qubits=n_qubits,
        n_bits=n_bits,
        batch_size=1,
        device=device,
    )

    for gate in gates:
        _ = system.apply_gate(gate)

    probabilities = torch.abs(system.state_vectors[0]) ** 2
    probabilities = probabilities / probabilities.sum().clamp_min(1e-12)
    sampling_device = torch.device("cpu") if device.type == "mps" else probabilities.device
    if probabilities.device != sampling_device:
        probabilities = probabilities.to(sampling_device)

    samples = torch.multinomial(probabilities, num_shots, replacement=True).to(dtype=torch.int64)

    register_codes = torch.zeros(num_shots, dtype=torch.int64, device=sampling_device)
    for measurement in measurements:
        measured_bit = (samples >> (n_qubits - 1 - measurement.qubit)) & 1
        shift = n_bits - 1 - measurement.bit
        bit_mask = 1 << shift
        register_codes = (register_codes & ~bit_mask) | (measured_bit << shift)

    return _counts_from_register_codes(register_codes, n_bits=n_bits, num_shots=num_shots)


def measure_all(qubits: QuantumRegister | list[int] | int) -> Circuit:
    """Measure qubits into sequential classical bits starting at 0.

    measure_all(qr)           # all qubits in register -> bits 0..n-1
    measure_all(qr[0:4])      # sub-register
    measure_all(4)             # qubits 0-3 (shorthand)
    """
    if isinstance(qubits, int):
        indices = list(range(qubits))
    elif isinstance(qubits, QuantumRegister):
        indices = list(qubits)
    else:
        indices = list(qubits)
    return Circuit([Measurement(q, i) for i, q in enumerate(indices)])


class BatchedQuantumSystem:
    """A batched quantum system that processes multiple state vectors in parallel."""

    state_vectors: Annotated[torch.Tensor, "(batch_size, 2^n_qubits) complex64"]
    bit_registers: Annotated[torch.Tensor, "(batch_size, n_bits) int"]
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

        # Initialize all state vectors to |000...0⟩
        # Shape: (batch_size, 2^n_qubits) - each row is a state vector
        dim = 1 << n_qubits
        self.state_vectors = torch.zeros((batch_size, dim), dtype=torch.complex64, device=device)
        self.state_vectors[:, 0] = 1.0

        # Initialize classical bit registers
        self.bit_registers = torch.zeros((batch_size, n_bits), dtype=torch.int32, device=device)

    @torch.inference_mode()
    def apply_gate(self, gate: Gate) -> "BatchedQuantumSystem":
        """Apply a gate to all state vectors via tensor contraction.

        Reshapes the state from (batch, 2^n) to (batch, 2, 2, ..., 2),
        permutes target qubit axes to the end, reshapes to 2D, and does
        a single matmul.  No Kronecker products, no swap matrices.
        O(2^n) per gate instead of O(4^n).
        """
        if gate.diagonal is not None:
            return self._apply_diagonal_gate(gate)
        if gate.permutation is not None:
            return self._apply_permutation_gate(gate)

        targets = gate.targets
        tensor = self._device_gate_tensor(gate.tensor)
        k = len(targets)
        n = self.n_qubits

        # Reshape state: (batch, 2^n) -> (batch, 2, 2, ..., 2)
        state = self.state_vectors.view((self.batch_size,) + (2,) * n)

        # Move target qubit axes to the end: (batch, non_targets..., targets...)
        non_targets = [i for i in range(n) if i not in targets]
        perm = [0] + [i + 1 for i in non_targets] + [t + 1 for t in targets]
        state = state.permute(perm)

        # Flatten to 2D for matmul: (batch * 2^{n-k}, 2^k)
        state = state.reshape(-1, 1 << k)

        # Apply gate: (batch * 2^{n-k}, 2^k) @ (2^k, 2^k)^T
        state = state @ tensor.T

        # Reshape back to (batch, 2, ..., 2) in permuted order, then restore
        state = state.reshape((self.batch_size,) + (2,) * n)
        inv_perm = [0] * (n + 1)
        for new_pos, old_pos in enumerate(perm):
            inv_perm[old_pos] = new_pos
        state = state.permute(inv_perm)

        self.state_vectors = state.reshape(self.batch_size, -1)
        return self

    def _apply_diagonal_gate(self, gate: Gate) -> "BatchedQuantumSystem":
        """Apply a diagonal gate via elementwise multiplication."""
        diagonal = gate.diagonal
        assert diagonal is not None

        diagonal_device = self._device_gate_diagonal(diagonal)
        targets = tuple(gate.targets)
        subindex = self._diagonal_subindex_for_targets(targets)

        # Each basis amplitude is scaled by the gate diagonal entry selected
        # from the target-qubit bit pattern for that basis state.
        factors = diagonal_device[subindex]  # (2^n,)
        self.state_vectors = self.state_vectors * factors.unsqueeze(0)
        return self

    def _apply_permutation_gate(self, gate: Gate) -> "BatchedQuantumSystem":
        """Apply a monomial gate (permutation with optional row factors)."""
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
        """Apply measurement to all state vectors in the batch.

        state_vectors shape: (batch_size, 2^n)
        """
        qubit = measurement.qubit
        bit = measurement.bit

        mask_1 = self._measurement_mask_for_qubit(qubit)

        probs = torch.abs(self.state_vectors) ** 2  # (batch_size, 2^n)
        p1 = (probs @ self._measurement_weight_for_qubit(qubit)).clamp(0.0, 1.0)  # (batch_size,)

        # Sample outcomes for all batches at once
        outcomes = (torch.rand(self.batch_size, device=self.device) < p1).to(torch.int32)  # (batch_size,)

        # Store outcomes in bit registers
        self.bit_registers[:, bit] = outcomes

        # Vectorized projection:
        # keep amplitudes where basis qubit value matches each sample's outcome.
        keep = mask_1.unsqueeze(0) == outcomes.unsqueeze(1).bool()  # (batch_size, 2^n)
        self.state_vectors = self.state_vectors * keep

        norms = torch.sqrt((torch.abs(self.state_vectors) ** 2).sum(dim=1, keepdim=True))
        self.state_vectors = self.state_vectors / norms.clamp_min(1e-12)

        return self

    def _measurement_mask_for_qubit(self, qubit: int) -> torch.Tensor:
        """Basis-state mask where `qubit` is in state |1>."""
        mask = self._measurement_masks.get(qubit)
        if mask is not None:
            return mask

        bitpos = self.n_qubits - 1 - qubit
        indices = torch.arange(1 << self.n_qubits, device=self.device)
        mask = ((indices >> bitpos) & 1).bool()
        self._measurement_masks[qubit] = mask
        return mask

    def _measurement_weight_for_qubit(self, qubit: int) -> torch.Tensor:
        """Float mask for fast p(|1>) computation via matmul."""
        weight = self._measurement_weights.get(qubit)
        if weight is not None:
            return weight

        weight = self._measurement_mask_for_qubit(qubit).to(torch.float32)
        self._measurement_weights[qubit] = weight
        return weight

    def _device_gate_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Cache gate tensors on the active device to avoid per-op transfers."""
        key = id(tensor)
        cached = self._gate_tensors.get(key)
        if cached is not None:
            return cached

        if tensor.device == self.device:
            self._gate_tensors[key] = tensor
            return tensor

        moved = tensor.to(self.device)
        self._gate_tensors[key] = moved
        return moved

    def _device_gate_diagonal(self, diagonal: torch.Tensor) -> torch.Tensor:
        """Cache gate diagonals on device."""
        key = id(diagonal)
        cached = self._gate_diagonals.get(key)
        if cached is not None:
            return cached

        if diagonal.device == self.device:
            self._gate_diagonals[key] = diagonal
            return diagonal

        moved = diagonal.to(self.device)
        self._gate_diagonals[key] = moved
        return moved

    def _device_gate_permutation(self, permutation: torch.Tensor) -> torch.Tensor:
        """Cache gate permutations on device."""
        key = id(permutation)
        cached = self._gate_permutations.get(key)
        if cached is not None:
            return cached

        permutation_i64 = permutation.to(dtype=torch.int64)
        if permutation_i64.device == self.device:
            self._gate_permutations[key] = permutation_i64
            return permutation_i64

        moved = permutation_i64.to(self.device)
        self._gate_permutations[key] = moved
        return moved

    def _device_gate_permutation_factors(self, factors: torch.Tensor) -> torch.Tensor:
        """Cache monomial gate row factors on device."""
        key = id(factors)
        cached = self._gate_permutation_factors.get(key)
        if cached is not None:
            return cached

        if factors.device == self.device:
            self._gate_permutation_factors[key] = factors
            return factors

        moved = factors.to(self.device)
        self._gate_permutation_factors[key] = moved
        return moved

    def _diagonal_subindex_for_targets(self, targets: tuple[int, ...]) -> torch.Tensor:
        """Map basis indices to diagonal-entry indices for target qubits."""
        cached = self._diagonal_subindices.get(targets)
        if cached is not None:
            return cached

        indices = torch.arange(1 << self.n_qubits, device=self.device, dtype=torch.int64)
        subindex = torch.zeros_like(indices)
        k = len(targets)

        # The first target is the most significant bit in the gate basis index.
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
        """Map output basis indices to input basis indices for a target-local permutation."""
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
        """Map local monomial row factors to all global basis indices."""
        key = (targets, id(factors))
        cached = self._permutation_phase_factors.get(key)
        if cached is not None:
            return cached

        subindex = self._diagonal_subindex_for_targets(targets)
        phase_factors = factors[subindex]
        self._permutation_phase_factors[key] = phase_factors
        return phase_factors

    @torch.inference_mode()
    def apply_one(self, operation: Gate | Measurement | ConditionalGate) -> "BatchedQuantumSystem":
        """Apply a single operation to all state vectors."""
        if isinstance(operation, Gate):
            return self.apply_gate(operation)

        if isinstance(operation, Measurement):
            return self.apply_measurement(operation)

        # For conditional gates, check condition for each batch element
        bit_values = self._get_bits_values()  # (batch_size,)
        mask = (bit_values == operation.condition)

        # Only apply gate to state vectors that meet the condition
        if mask.any():
            # This is tricky - we need selective application
            # For simplicity, apply to all and restore non-matching ones
            old_states = self.state_vectors[~mask].clone()
            _ = self.apply_one(operation.gate)
            self.state_vectors[~mask] = old_states

        return self

    @torch.inference_mode()
    def apply_circuit(self, circuit: Circuit) -> "BatchedQuantumSystem":
        """Apply a circuit to all state vectors."""
        for operation in circuit.operations:
            if isinstance(operation, Circuit):
                _ = self.apply_circuit(operation)
            else:
                _ = self.apply_one(operation)
        return self

    def _get_bits_values(self) -> torch.Tensor:
        """Convert bit registers to integer values. Returns (batch_size,) tensor."""
        result = torch.zeros(self.batch_size, dtype=torch.int32, device=self.device)
        for i in range(self.n_bits):
            result = (result << 1) | self.bit_registers[:, i]
        return result

    def _gate_to_qubit(self, gate: torch.Tensor, n_targets: int = 1, offset: int = 0) -> torch.Tensor:
        """Same as QuantumSystem._gate_to_qubit"""
        I = torch.eye(2, dtype=gate.dtype, device=gate.device)
        factors = [*[I for _ in range(offset)], gate, *[I for _ in range(self.n_qubits - n_targets - offset)]]
        full = factors[0]
        for f in factors[1:]:
            full = torch.kron(full, f)
        return full

    def _get_swap_matrix(self, target_1: int, target_2: int) -> torch.Tensor:
        """Same as QuantumSystem._get_swap_matrix"""
        dim = 1 << self.n_qubits
        S = torch.zeros((dim, dim), dtype=torch.complex64, device=self.device)

        b1 = self.n_qubits - 1 - target_1
        b2 = self.n_qubits - 1 - target_2

        i, j = sorted((b1, b2))
        mask = (1 << i) | (1 << j)

        for x in range(dim):
            bi = (x >> i) & 1
            bj = (x >> j) & 1
            y = x if bi == bj else x ^ mask
            S[y, x] = 1

        return S

    def get_results(self) -> dict[str, int]:
        """Collect measurement results from all batches."""
        counts: dict[str, int] = {}

        # Convert bit registers to strings
        bit_registers_cpu = self.bit_registers.cpu().numpy()
        for i in range(self.batch_size):
            row = cast(npt.NDArray[np.int32], bit_registers_cpu[i])
            key = "".join(map(str, row))
            counts[key] = counts.get(key, 0) + 1

        return counts


@torch.inference_mode()
def run_simulation(
    circuit: Circuit,
    num_shots: int,
    *,
    n_qubits: int | None = None,
    n_bits: int | None = None,
    device: torch.device | None = None,
) -> dict[str, int]:
    """Run a quantum circuit simulation multiple times and collect measurement results.

    This uses a batched approach to parallelize operations across all shots on the GPU.

    Args:
        circuit: The circuit to apply
        num_shots: Number of times to run the simulation
        n_qubits: Number of qubits (inferred from circuit if not provided)
        n_bits: Number of classical bits (inferred from circuit if not provided)
        device: Torch device to use (auto-detected if not provided)

    Returns:
        Dictionary mapping bit strings to their counts
    """
    if n_qubits is None or n_bits is None:
        inferred_qubits, inferred_bits = infer_resources(circuit)
        if n_qubits is None:
            n_qubits = inferred_qubits
        if n_bits is None:
            n_bits = inferred_bits

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )

    sampled = _run_terminal_measurement_sampling(
        circuit=circuit,
        num_shots=num_shots,
        n_qubits=n_qubits,
        n_bits=n_bits,
        device=device,
    )
    if sampled is not None:
        return sampled

    dynamic_branch_result = _run_dynamic_branch_simulation(
        circuit=circuit,
        num_shots=num_shots,
        n_qubits=n_qubits,
        n_bits=n_bits,
        device=device,
    )
    if dynamic_branch_result is not None:
        return dynamic_branch_result

    batched_system = BatchedQuantumSystem(
        n_qubits=n_qubits,
        n_bits=n_bits,
        batch_size=num_shots,
        device=device,
    )

    _ = batched_system.apply_circuit(circuit)

    return batched_system.get_results()


class QuantumSystem:
    state_vector: Annotated[torch.Tensor, "(n, 1) complex64 column vector"]
    bit_register: Annotated[list[int], "(n_bits) bit string"]
    n_qubits: int
    n_bits: int
    dimensions: int
    device: torch.device
    ops_done: int

    def __init__(self, n_qubits: int, n_bits: int = 0, state_vector: torch.Tensor | None = None):
        if n_qubits <= 0:
            raise ValueError(f"Number of qubits must be positive, got {n_qubits}")
        if n_bits < 0:
            raise ValueError(f"Number of classical bits must be non-negative, got {n_bits}")

        if state_vector is None:
            # Initialize state vector to |000...0⟩
            state_vector = torch.zeros((2 ** n_qubits, 1), dtype=torch.complex64)
            state_vector[0] = 1.0  # |0⟩ state has amplitude 1 in the first position
        else:
            expected_dim = 1 << n_qubits
            if state_vector.shape != (expected_dim, 1):
                raise ValueError(f"State vector must have shape ({expected_dim}, 1) for {n_qubits} qubits, got {state_vector.shape}")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )
        self.state_vector = state_vector.to(self.device)
        self.bit_register = [0] * n_bits
        self.n_qubits = n_qubits
        self.n_bits = n_bits
        self.dimensions = 2 ** self.n_qubits
        self.ops_done = 0

    def get_distribution(self) -> torch.Tensor:
        return torch.abs(self.state_vector) ** 2

    def get_bits_value(self) -> int:
        result = 0
        for bit in self.bit_register:
            result = (result << 1) | bit
        return result

    def sample(self, num_shots: int) -> list[int]:
        distribution = self.get_distribution().T  # Convert (n, 1) to (1, n) for multinomial
        values = torch.multinomial(distribution, num_shots, replacement=True)
        return [int(x.item()) for x in values[0]]

    @torch.inference_mode()
    def apply_one(self, operation: Gate | Measurement | ConditionalGate) -> "QuantumSystem":
        #print(f"Applying operation {self.ops_done}")
        self.ops_done = self.ops_done + 1

        if isinstance(operation, Gate):
            return self.apply_gate(operation)

        if isinstance(operation, Measurement):
            return self.apply_measurement(operation)

        if self.get_bits_value() == operation.condition:
            return self.apply_one(operation.gate)
        else:
            return self

    @torch.inference_mode()
    def apply_measurement(self, measurement: Measurement) -> "QuantumSystem":
        """Note: this will collapse |ψ⟩ state at that qubit.
        """
        qubit = measurement.qubit
        bit = measurement.bit

        if qubit < 0 or qubit >= self.n_qubits:
            raise ValueError(f"Qubit index {qubit} out of range [0, {self.n_qubits})")
        if bit < 0 or bit >= self.n_bits:
            raise ValueError(f"Classical bit index {bit} out of range [0, {self.n_bits})")

        indices = torch.arange(1 << self.n_qubits, device=self.device)
        bitpos = self.n_qubits - 1 - qubit

        # True on basis states with target qubit set to |1⟩
        mask_1 = ((indices >> bitpos) & 1).bool()
        probs = self.get_distribution().flatten()
        p1 = probs[mask_1].sum()
        outcome = 1 if torch.rand(1).item() < p1 else 0
        self.bit_register[bit] = outcome

        # Build projection operator: project onto |outcome⟩ for the target qubit
        P = torch.tensor([[1 - outcome, 0], [0, outcome]], dtype=torch.complex64, device=self.device)

        # Apply projection using the quantum gate machinery
        P_full = self._gate_to_qubit(P, n_targets=1, offset=qubit)

        # update state vector
        self.state_vector = P_full @ self.state_vector
        norm = torch.sqrt(torch.sum(torch.abs(self.state_vector) ** 2))
        self.state_vector = self.state_vector / norm

        return self

    @torch.inference_mode()
    def apply_gate(self, gate: Gate) -> "QuantumSystem":
        """Apply a quantum gate to the state vector: |ψ⟩ → G |ψ⟩"""
        targets = gate.targets
        tensor = gate.tensor
        n_targets = len(targets)

        # Validate targets
        if n_targets == 0:
            raise ValueError("Must specify at least one target qubit")
        for target in targets:
            if target < 0 or target >= self.n_qubits:
                raise ValueError(f"Target qubit {target} out of range [0, {self.n_qubits})")
        if len(set(targets)) != len(targets):
            raise ValueError(f"Duplicate target qubits not allowed: {targets}")

        # Validate gate dimensions
        expected_dim = 1 << n_targets
        if tensor.shape != (expected_dim, expected_dim):
            raise ValueError(f"Gate matrix must have shape ({expected_dim}, {expected_dim}) for {n_targets} target(s), got {tensor.shape}")

        swaps: list[torch.Tensor] = []
        positions = list(range(self.n_qubits))          # current location of each qubit

        # ---- 1. move the target qubits to the *lowest* positions (0, 1, 2, …) ----
        # This corresponds to the leftmost positions in the Kronecker product
        for i in range(n_targets):
            target = targets[i]                         # original qubit index
            desired_pos = i                              # 0, 1, 2, …
            cur_pos = positions.index(target)

            if cur_pos != desired_pos:
                S = self._get_swap_matrix(cur_pos, desired_pos)
                swaps.append(S)
                # update tracking
                positions[cur_pos], positions[desired_pos] = positions[desired_pos], positions[cur_pos]

        # ---- 2. swap the state vector to the new ordering ----
        for u in swaps:
            self.state_vector = u @ self.state_vector

        # ---- 3. apply the gate on the *left-most* (highest) dimensions ----
        gate_full = self._gate_to_qubit(tensor.to(self.device), n_targets)
        self.state_vector = gate_full @ self.state_vector

        # ---- 4. undo the swaps (they are their own inverse) ----
        for u in reversed(swaps):
            self.state_vector = u @ self.state_vector

        # ---- 5. renormalise (safety) ----
        norm = torch.sqrt(torch.sum(torch.abs(self.state_vector) ** 2))
        assert torch.allclose(norm, torch.tensor(1.0, device=self.device), atol=1e-5), f"Norm drift: {norm}"
        self.state_vector = self.state_vector / norm

        return self

    @torch.inference_mode()
    def apply_circuit(self, circuit: Circuit) -> "QuantumSystem":
        for operation in circuit.operations:
            if isinstance(operation, Circuit):
                _ = self.apply_circuit(operation)
            else:
                _ = self.apply_one(operation)

        return self

    def _gate_to_qubit(self, gate: torch.Tensor, n_targets: int = 1, offset: int = 0) -> torch.Tensor:

        I = torch.eye(2, dtype=gate.dtype, device=gate.device)

        # Build list of local operators for each qubit
        factors = [*[I for _ in range(offset)], gate, *[I for _ in range(self.n_qubits - n_targets - offset)]]
        # Kronecker product left-to-right
        full = factors[0]
        for f in factors[1:]:
            full = torch.kron(full, f)

        return full

    def _get_swap_matrix(self, target_1: int, target_2: int) -> torch.Tensor:

        # 2 ** self.n_qubits with bitshift operator
        dim = 1 << self.n_qubits

        S = torch.zeros((dim, dim), dtype=torch.complex64, device=self.device)
        b1 = self.n_qubits - 1 - target_1
        b2 = self.n_qubits - 1 - target_2

        i, j = sorted((b1, b2))
        mask = (1 << i) | (1 << j)

        for x in range(dim):
            bi = (x >> i) & 1
            bj = (x >> j) & 1

            if bi == bj:
                y = x
            else:
                y = x ^ mask # flip both bits

            S[y, x] = 1

        return S

    @override
    def __repr__(self) -> str:
        """Pretty print the quantum state in basis notation."""
        vec: npt.NDArray[np.complex64] = self.state_vector.cpu().numpy().flatten()
        terms: list[str] = []

        number_of_decimals = 10

        for i in range(len(vec)):
            val = cast(np.complex64, vec[i])
            # Format the basis state |i⟩ as binary (big-endian: qubit 0 on left)
            basis = format(i, f'0{self.n_qubits}b')  # Reverse for big-endian display

            # Format the coefficient
            real = float(np.real(val))
            imag = float(np.imag(val))

            # Skip negligible amplitudes for display purposes
            magnitude = float(np.abs(val))  # pyright: ignore[reportAny]
            if magnitude < 1e-10:
                continue

            if abs(imag) < 1e-10:
                # Pure real
                coef = f"{real:.{number_of_decimals}f}"
            elif abs(real) < 1e-10:
                # Pure imaginary
                coef = f"{imag:.{number_of_decimals}f}i"
            else:
                # Complex
                sign = "+" if imag >= 0 else "-"
                coef = f"({real:.{number_of_decimals}f} {sign} {abs(imag):.{number_of_decimals}f}i)"

            terms.append(f"{coef}|{basis}⟩")

        if not terms:
            return "|ψ⟩ = 0"

        # Join with + signs, handling negative coefficients
        result = "|ψ⟩ = "
        for i, term in enumerate(terms):
            if i == 0:
                result += term
            elif term.startswith("-"):
                result += f" - {term[1:]}"
            else:
                result += f" + {term}"

        # Add classical register display if we have classical bits
        if self.n_bits > 0:
            bit_string = "".join(str(b) for b in self.bit_register)
            result += f"\tClassical register: {bit_string}"

        return result
