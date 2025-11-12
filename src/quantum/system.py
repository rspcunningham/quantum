"""Quantum system state management."""

from __future__ import annotations

from typing import Annotated, cast, override
import torch
import numpy as np
import numpy.typing as npt
from quantum.gates import Gate

__all__ = ["QuantumSystem", "Circuit", "Measurement"]


class Measurement:
    target: int
    output: int
    def __init__(self, target: int, output: int):
        self.target = target
        self.output = output

class Circuit:
    operations: list[Gate | Measurement | Circuit]

    def __init__(self, operations: list[Gate | Measurement | Circuit]):
        self.operations = operations


class QuantumSystem:
    state_vector: Annotated[torch.Tensor, "(n, 1) complex64 column vector"]
    bit_register: Annotated[list[int], "(n_bits) bit string"]
    n_qubits: int
    n_bits: int
    dimensions: int
    device: torch.device

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

    def get_distribution(self) -> torch.Tensor:
        return torch.abs(self.state_vector) ** 2

    def sample(self, num_shots: int) -> list[int]:
        distribution = self.get_distribution().T  # Convert (n, 1) to (1, n) for multinomial
        values = torch.multinomial(distribution, num_shots, replacement=True)
        return [int(x.item()) for x in values[0]]

    def measure(self, qubit: int, output: int) -> "QuantumSystem":
        """Note: this will collapse |ψ⟩ state at that qubit.

        In big-endian convention, qubit i corresponds to bit position (n_qubits - 1 - i).
        """
        if qubit < 0 or qubit >= self.n_qubits:
            raise ValueError(f"Qubit index {qubit} out of range [0, {self.n_qubits})")
        if output < 0 or output >= self.n_bits:
            raise ValueError(f"Classical bit index {output} out of range [0, {self.n_bits})")

        indices = torch.arange(1 << self.n_qubits, device=self.device)

        # we need this jank because python bit operations are little-endian
        bitpos = self.n_qubits - 1 - qubit
        # True on basis states with target qubit set to |1⟩
        mask_1 = ((indices >> bitpos) & 1).bool()
        probs = self.get_distribution().flatten()
        p1 = probs[mask_1].sum()
        outcome = 1 if torch.rand(1).item() < p1 else 0
        self.bit_register[output] = outcome

        # Build projection operator: project onto |outcome⟩ for the target qubit
        P = torch.tensor([[1 - outcome, 0], [0, outcome]], dtype=torch.complex64, device=self.device)

        # Apply projection using the quantum gate machinery
        P_full = self._gate_to_qubit(P, n_targets=1, offset=qubit)

        # update state vector
        self.state_vector = P_full @ self.state_vector
        norm = torch.sqrt(torch.sum(torch.abs(self.state_vector) ** 2))
        self.state_vector = self.state_vector / norm

        return self

    def apply_quantum_gate(self, gate: torch.Tensor, targets: list[int]) -> "QuantumSystem":
        """Apply a quantum gate to the state vector: |ψ⟩ → G |ψ⟩"""
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
        if gate.shape != (expected_dim, expected_dim):
            raise ValueError(f"Gate matrix must have shape ({expected_dim}, {expected_dim}) for {n_targets} target(s), got {gate.shape}")

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
        gate_full = self._gate_to_qubit(gate.to(self.device), n_targets)
        self.state_vector = gate_full @ self.state_vector

        # ---- 4. undo the swaps (they are their own inverse) ----
        for u in reversed(swaps):
            self.state_vector = u @ self.state_vector

        # ---- 5. renormalise (safety) ----
        norm = torch.sqrt(torch.sum(torch.abs(self.state_vector) ** 2))
        assert torch.allclose(norm, torch.tensor(1.0, device=self.device), atol=1e-5), f"Norm drift: {norm}"
        self.state_vector = self.state_vector / norm

        return self

    def apply_circuit(self, circuit: Circuit) -> "QuantumSystem":
        for operation in circuit.operations:
            if isinstance(operation, Measurement):
                _ = self.measure(operation.target, operation.output)
            elif isinstance(operation, Gate):
                _ = self.apply_quantum_gate(operation.tensor, operation.targets)
            else:
                _ = self.apply_circuit(operation)
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

        i, j = sorted((target_1, target_2))
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
