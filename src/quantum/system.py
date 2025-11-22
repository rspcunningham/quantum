"""Quantum system state management."""

from __future__ import annotations

from typing import Annotated, cast, override
from collections.abc import Sequence
import torch
import numpy as np
import numpy.typing as npt
from quantum.gates import Gate, Measurement, ConditionalGate

class Circuit:
    operations: list[Gate | ConditionalGate | Measurement | Circuit]

    def __init__(self, operations: Sequence[Gate | ConditionalGate | Measurement | Circuit]):
        self.operations = list(operations)

    def uncomputed(self):
        reversed_operations = self.operations[::-1]
        new_operations: list[Gate | ConditionalGate | Measurement | Circuit] = []
        for op in reversed_operations:
            if isinstance(op, Circuit):
                new_operations.append(op.uncomputed())
            else:
                new_operations.append(op)

        return Circuit(new_operations)

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Circuit):
            return NotImplemented
        return self.operations == other.operations


class QuantumSystem:
    """A batched quantum system that processes multiple state vectors in parallel."""
    state_vectors: Annotated[torch.Tensor, "(batch_size, 2^n_qubits) complex64"]
    bit_registers: Annotated[torch.Tensor, "(batch_size, n_bits) int"]
    n_qubits: int
    n_bits: int
    batch_size: int
    device: torch.device

    def __init__(self, n_qubits: int, n_bits: int, batch_size: int = 100):
        self.n_qubits = n_qubits
        self.n_bits = n_bits
        self.batch_size = batch_size
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )

        # Initialize all state vectors to |000...0⟩
        # Shape: (batch_size, 2^n_qubits) - each row is a state vector
        dim = 1 << n_qubits
        self.state_vectors = torch.zeros((batch_size, dim), dtype=torch.complex64, device=self.device)
        self.state_vectors[:, 0] = 1.0

        # Initialize classical bit registers
        self.bit_registers = torch.zeros((batch_size, n_bits), dtype=torch.int32, device=self.device)

    @torch.inference_mode()
    def apply_gate(self, gate: Gate) -> "QuantumSystem":
        """Apply a gate to all state vectors in the batch.

        state_vectors shape: (batch_size, 2^n)
        gate_full shape: (2^n, 2^n)

        We want: new_state_vectors[i] = gate_full @ state_vectors[i]
        Equivalent: state_vectors @ gate_full.T (but more efficient for the gpu)
        """
        targets = gate.targets
        tensor = gate.tensor.to(self.device)
        n_targets = len(targets)

        # Build the full gate matrix (same logic as QuantumSystem)
        swaps: list[torch.Tensor] = []
        positions = list(range(self.n_qubits))

        for i in range(n_targets):
            target = targets[i]
            desired_pos = i
            cur_pos = positions.index(target)

            if cur_pos != desired_pos:
                S = self._get_swap_matrix(cur_pos, desired_pos)
                swaps.append(S)
                positions[cur_pos], positions[desired_pos] = positions[desired_pos], positions[cur_pos]

        # Apply swaps to all state vectors
        # (batch_size, 2^n) @ (2^n, 2^n)^T = (batch_size, 2^n)
        for u in swaps:
            self.state_vectors = self.state_vectors @ u.T

        # Apply gate to all state vectors
        gate_full = self._gate_to_qubit(tensor, n_targets)
        self.state_vectors = self.state_vectors @ gate_full.T

        # Undo swaps
        for u in reversed(swaps):
            self.state_vectors = self.state_vectors @ u.T

        # Renormalize
        norms = torch.sqrt(torch.sum(torch.abs(self.state_vectors) ** 2, dim=1, keepdim=True))
        self.state_vectors = self.state_vectors / norms

        return self

    @torch.inference_mode()
    def apply_measurement(self, measurement: Measurement) -> "QuantumSystem":
        """Apply measurement to all state vectors in the batch.

        state_vectors shape: (batch_size, 2^n)
        """
        qubit = measurement.qubit
        bit = measurement.bit

        bitpos = self.n_qubits - 1 - qubit

        # Get probabilities for all batches
        indices = torch.arange(1 << self.n_qubits, device=self.device)
        mask_1 = ((indices >> bitpos) & 1).bool()

        probs = torch.abs(self.state_vectors) ** 2  # (batch_size, 2^n)
        p1 = probs[:, mask_1].sum(dim=1)  # (batch_size,)

        # Sample outcomes for all batches at once
        outcomes = (torch.rand(self.batch_size, device=self.device) < p1).int()  # (batch_size,)

        # Store outcomes in bit registers
        self.bit_registers[:, bit] = outcomes

        # Apply projection to each state vector based on its outcome
        # TODO: This loop could potentially be vectorized further
        for i in range(self.batch_size):
            outcome = int(outcomes[i].item())
            P = torch.tensor([[1 - outcome, 0], [0, outcome]], dtype=torch.complex64, device=self.device)
            P_full = self._gate_to_qubit(P, n_targets=1, offset=qubit)

            # P_full @ state_vector[i] is equivalent to state_vector[i] @ P_full.T
            self.state_vectors[i] = self.state_vectors[i] @ P_full.T
            norm = torch.sqrt(torch.sum(torch.abs(self.state_vectors[i]) ** 2))
            self.state_vectors[i] = self.state_vectors[i] / norm

        return self

    @torch.inference_mode()
    def apply_one(self, operation: Gate | Measurement | ConditionalGate) -> "QuantumSystem":
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
    def apply_circuit(self, circuit: Circuit) -> "QuantumSystem":
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
