import math
from re import search
from typing import Callable
from program_state import ProgramState
from telemetry import DistributionPlotter
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

number_of_qubits = 10
target_number = 567

search_space = range(2**number_of_qubits)
print("search space:", search_space)

assert target_number in search_space, f"Target number {target_number} is not in search space {search_space}"

state = ProgramState.balanced(number_of_qubits)
plotter = DistributionPlotter(state._get_distribution())

s = state.state_vector

f_oracle = oracle(target_number)
f_diffusion = diffusion(s)

# Program Execution

iterations = math.floor(math.pi * 0.25 * math.sqrt(2**number_of_qubits))

input("Press Enter to start the search...")

for _ in range(iterations):
    print("applying oracle")
    state.apply_transformation(f_oracle)

    print("applying diffusion")
    state.apply_transformation(f_diffusion)

    plotter.update(state._get_distribution())

    input("Press Enter to continue...")

print(f"Sample: {state.sample()}")
