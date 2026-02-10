"""Quantum system state management."""

from __future__ import annotations

from typing import Annotated, cast, override
from collections.abc import Sequence
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

    def __init__(self, n_qubits: int, n_bits: int, batch_size: int, device: torch.device):
        self.n_qubits = n_qubits
        self.n_bits = n_bits
        self.batch_size = batch_size
        self.device = device

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
        targets = gate.targets
        tensor = gate.tensor.to(self.device)
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

    @torch.inference_mode()
    def apply_measurement(self, measurement: Measurement) -> "BatchedQuantumSystem":
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
