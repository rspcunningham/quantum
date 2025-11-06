from typing import Callable
from program_state import ProgramState
import utils
import torch

# Transformations

def oracle(k: int) -> Callable[[torch.Tensor], torch.Tensor]:
    def _oracle(u: torch.Tensor) -> torch.Tensor:
        U = torch.eye(u.shape[0])
        U[k, k] = -1
        return U @ u

    return _oracle

def diffusion(s: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    def _diffusion(u: torch.Tensor) -> torch.Tensor:
        # let P = ss^T
        P = s @ s.T

        # diffusion matrix D = 2ss^T - I
        D = 2 * P - torch.eye(P.shape[0])

        return D @ u
    return _diffusion

# Program Setup

state = ProgramState.balanced(10)
s = state.state_vector
k = 0

f_oracle = oracle(k)
f_diffusion = diffusion(s)

# Program Execution

utils.display(state)

print("applying oracle")
state.apply_transformation(f_oracle)

utils.display(state)

print("applying diffusion")
state.apply_transformation(f_diffusion)

utils.display(state)

print(f"Sample: {state.state_vector.norm().item()}")
