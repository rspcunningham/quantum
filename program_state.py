import math
from typing import Callable

import torch

class ProgramState:
    state_vector: torch.Tensor
    n_qbits: int
    dimensions: int
    device: torch.device

    def __init__(self, state_vector: torch.Tensor):
        """
        Initialize the quantum state with a normalized state vector.

        Args:
            state_vector: A normalized tensor representing the quantum state
        """
        self.state_vector = state_vector
        self.n_qbits = int(math.log2(state_vector.shape[0]))
        self.dimensions = 2 ** self.n_qbits
        self.device = torch.device("mps")
        self.state_vector = self.state_vector.to(self.device)

    @classmethod
    def balanced(cls, n_qbits: int):
        """
        Create a balanced superposition state with equal amplitude for all basis states.

        Args:
            n_qbits: Number of qubits
        """
        n = 2 ** n_qbits
        vector = torch.ones(n, 1) / torch.sqrt(torch.tensor(float(n)))
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

        if not torch.allclose(u.norm(), v.norm(), atol=atol):
            raise ValueError("Transformation did not preserve norm. something is wrong!")

        self.state_vector = v
