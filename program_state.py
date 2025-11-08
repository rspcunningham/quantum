import math
from typing import Callable, Annotated

import torch

class ProgramState:
    state_vector: Annotated[torch.Tensor, "(n, 1) complex64 column vector"]
    n_qbits: int
    dimensions: int
    device: torch.device

    def __init__(self, state_vector: torch.Tensor):
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
        distribution = self._get_distribution().T  # Convert (n, 1) to (1, n) for multinomial
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
        self.state_vector = v
