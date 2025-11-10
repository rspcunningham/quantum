"""Quantum system state management."""

from typing import Annotated, cast, override
import torch
import numpy as np
import numpy.typing as npt
from quantum.gates import Gate

__all__ = ["QuantumSystem", "Circuit"]


class Circuit:
    n_qubits: int
    gates: list[Gate]

    def __init__(self, n_qubits: int, gates: list[Gate]):
        self.n_qubits = n_qubits
        self.gates = gates


class QuantumSystem:
    state_vector: Annotated[torch.Tensor, "(n, 1) complex64 column vector"]
    n_qubits: int
    dimensions: int
    device: torch.device

    def __init__(self, n_qubits: int, state_vector: torch.Tensor | None = None):

        if state_vector is None:
            # Initialize state vector to |000...0⟩
            state_vector = torch.zeros((2 ** n_qubits, 1), dtype=torch.complex64)
            state_vector[0] = 1.0  # |0⟩ state has amplitude 1 in the first position

        self.state_vector = state_vector
        self.n_qubits = n_qubits
        self.dimensions = 2 ** self.n_qubits
        self.device = torch.device("mps")
        self.state_vector = self.state_vector.to(self.device)

    def get_distribution(self) -> torch.Tensor:
        return torch.abs(self.state_vector) ** 2

    def sample(self, num_shots: int) -> list[int]:
        distribution = self.get_distribution().T  # Convert (n, 1) to (1, n) for multinomial
        values = torch.multinomial(distribution, num_shots, replacement=True)
        return [int(x.item()) for x in values[0]]

    def measure(self) -> int:
        """Note: this will collapse |ψ⟩ state."""
        distribution = self.get_distribution().T  # Convert (n, 1) to (1, n) for multinomial
        value = int(torch.multinomial(distribution, 1, replacement=True).item())
        self.state_vector = torch.zeros_like(self.state_vector)
        self.state_vector[value] = 1.0
        return value

    def apply_gate(self, gate: torch.Tensor, targets: list[int]) -> None:
        """Apply a quantum gate to the state vector: |ψ⟩ → G |ψ⟩"""
        n_targets = len(targets)
        swaps: list[torch.Tensor] = []
        positions = list(range(self.n_qubits))          # current location of each qubit

        # ---- 1. move the target qubits to the *highest* positions (n-1, n-2, …) ----
        for i in range(n_targets):
            target = targets[i]                         # original qubit index
            desired_pos = self.n_qubits - 1 - i          # n-1, n-2, …
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
        gate_full = self._gate_to_leftmost_matrix(gate.to(self.device), n_targets)
        self.state_vector = gate_full @ self.state_vector

        # ---- 4. undo the swaps (they are their own inverse) ----
        for u in reversed(swaps):
            self.state_vector = u @ self.state_vector

        # ---- 5. renormalise (safety) ----
        norm = torch.sqrt(torch.sum(torch.abs(self.state_vector) ** 2))
        assert torch.allclose(norm, torch.tensor(1.0, device=self.device), atol=1e-5), f"Norm drift: {norm}"
        self.state_vector = self.state_vector / norm

    def apply_circuit(self, circuit: Circuit):
        for gate in circuit.gates:
            self.apply_gate(gate.tensor, gate.targets)

    def _gate_to_leftmost_matrix(self, gate: torch.Tensor, n_targets: int) -> torch.Tensor:

        I = torch.eye(2, dtype=gate.dtype, device=gate.device)

        # Build list of local operators for each qubit
        factors = [gate, *[I for _ in range(self.n_qubits - n_targets)]]

        # Kronecker product left-to-right
        full = factors[0]
        for f in factors[1:]:
            full = torch.kron(full, f)

        return full

    def _get_swap_matrix(self, target_1: int, target_2: int) -> torch.Tensor:

        # 2 ** self.n_qubits with bitshift operator
        dim: int = 1 << self.n_qubits

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

        for i in range(len(vec)):
            val = cast(np.complex64, vec[i])
            # Format the basis state |i⟩ as binary
            basis = format(i, f'0{self.n_qubits}b')

            # Format the coefficient
            real = float(np.real(val))
            imag = float(np.imag(val))

            # Skip negligible amplitudes for display purposes
            magnitude = float(np.abs(val))  # pyright: ignore[reportAny]
            if magnitude < 1e-10:
                continue

            if abs(imag) < 1e-10:
                # Pure real
                coef = f"{real:.4f}"
            elif abs(real) < 1e-10:
                # Pure imaginary
                coef = f"{imag:.4f}i"
            else:
                # Complex
                sign = "+" if imag >= 0 else "-"
                coef = f"({real:.4f} {sign} {abs(imag):.4f}i)"

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

        return result
