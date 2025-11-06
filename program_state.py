import math
import torch

class ProgramState:
    state_vector: torch.Tensor
    n_qbits: int
    dimensions: int

    def __init__(self, state_vector: torch.Tensor):
        """
        Initialize the quantum state with a normalized state vector.

        Args:
            state_vector: A normalized tensor representing the quantum state
        """
        self.state_vector = state_vector
        self.n_qbits = int(math.log2(state_vector.shape[0]))
        self.dimensions = 2 ** self.n_qbits

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

    def sample(self):
        distribution = self._get_distribution().squeeze()  # Convert (2, 1) to (2,)
        return torch.multinomial(distribution, 1).item()
