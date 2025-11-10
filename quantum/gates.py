import torch
from typing import Callable

__all__ = ["I", "H", "X", "Y", "Z", "RZ", "CX", "CZ", "CCX", "Gate"]

# identity gate
I = torch.tensor(
    [[1, 0],
     [0, 1]],
    dtype=torch.complex64)

# hadamard gate
H = (1 / torch.sqrt(torch.tensor(2.0))) * torch.tensor(
    [[1, 1],
     [1, -1]],
    dtype=torch.complex64)

# phase gate
S = torch.tensor(
    [[1, 0],
     [0, 1j]],
    dtype=torch.complex64)

# pi/8 aka T gate
T = torch.tensor(
    [[1, 0],
     [0, torch.exp(torch.tensor(1j * torch.pi / 4))]],
    dtype=torch.complex64)

# pauli-x gate
X = torch.tensor(
    [[0, 1],
     [1, 0]],
    dtype=torch.complex64)

# pauli-y gate
Y = torch.tensor(
    [[0, -1j],
     [1j, 0]],
    dtype=torch.complex64)

# pauli-z gate
Z = torch.tensor(
    [[1, 0],
     [0, -1]],
    dtype=torch.complex64)

# rotate X gate
RX: Callable[[torch.Tensor], torch.Tensor] = lambda theta: torch.tensor(
    [[torch.cos(theta / 2), -1j * torch.sin(theta / 2)],
    [-1j * torch.sin(theta / 2), torch.cos(theta / 2)]],
    dtype=torch.complex64)

# rotate Y gate
RY: Callable[[torch.Tensor], torch.Tensor] = lambda theta: torch.tensor(
    [[torch.cos(theta / 2), -torch.sin(theta / 2)],
    [torch.sin(theta / 2), torch.cos(theta / 2)]],
    dtype=torch.complex64)

# rotate Z gate
# NOTE: theta must be a (1,) tensor
RZ: Callable[[torch.Tensor], torch.Tensor] = lambda theta: torch.tensor(
    [[torch.cos(theta / 2) - 1j * torch.sin(theta / 2), 0],
    [0, torch.cos(theta / 2) + 1j * torch.sin(theta / 2)]],
    dtype=torch.complex64)

# CNOT aka CX
CX = torch.tensor(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 1, 0]],
    dtype=torch.complex64)

CZ = torch.tensor(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, -1]],
    dtype=torch.complex64)

SWAP = torch.tensor(
    [[1, 0, 0, 0],
     [0, 0, 1, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1]],
    dtype=torch.complex64)

# CCNOT aka Toffoli aka CCX
CCX = torch.tensor(
    [[1, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0, 1, 0]],
    dtype=torch.complex64)


class Gate:
    tensor: torch.Tensor
    targets: list[int]

    def __init__(self, tensor: torch.Tensor, targets: list[int]):
        self.tensor = tensor
        self.targets = targets
