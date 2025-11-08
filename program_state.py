import math
from typing import Callable, Annotated

import torch

class ProgramState:
    state_vector: Annotated[torch.Tensor, "(n, 1) complex64 column vector"]
    n_qbits: int
    dimensions: int
    device: torch.device

    def __init__(self, state_vector: torch.Tensor):
        """
        Initialize the quantum state with a normalized state vector.

        Args:
            state_vector: A normalized (n, 1) complex64 column vector representing the quantum state

        Raises:
            ValueError: If state_vector is not a 2D column vector or not complex64 dtype
        """
        # Type and shape validation
        if state_vector.dim() != 2:
            raise ValueError(f"state_vector must be 2D, got {state_vector.dim()}D")
        if state_vector.shape[1] != 1:
            raise ValueError(f"state_vector must be a column vector (n, 1), got shape {state_vector.shape}")
        if state_vector.dtype != torch.complex64:
            raise ValueError(f"state_vector must be complex64, got {state_vector.dtype}")

        self.state_vector = state_vector
        self.n_qubits = int(math.log2(state_vector.shape[0]))
        self.dimensions = 2 ** self.n_qubits
        self.device = torch.device("mps")
        self.state_vector = self.state_vector.to(self.device)

    @classmethod
    def balanced(cls, n_qubits: int):
        n = 2 ** n_qubits
        vector = torch.ones(n, 1, dtype=torch.complex64) / torch.sqrt(torch.tensor(float(n)))
        return cls(vector)

    def _get_distribution(self):
        return torch.abs(self.state_vector) ** 2

    def sample(self) -> int:
        distribution = self._get_distribution().squeeze()  # Convert (2, 1) to (2,)
        value = torch.multinomial(distribution, 1).item()
        return int(value)

    def apply_transformation(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        atol: float = 1e-6
    ) -> None:
        """Apply a function u -> v and mutate the state in place."""
        u = self.state_vector
        v = fn(u)

        # Safety checks
        if not isinstance(v, torch.Tensor):
            raise TypeError("Transformation must return a torch.Tensor")
        if v.shape != u.shape:
            raise ValueError(f"Shape changed: {u.shape} -> {v.shape}")
        if v.dtype != u.dtype or v.device != u.device:
            raise ValueError("dtype/device changed; keep them consistent")

        self.state_vector = v
