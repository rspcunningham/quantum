import torch
from typing import Callable

__all__ = ["I", "H", "X", "Y", "Z", "RZ", "CX", "CZ", "CCX"]

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
RX: Callable[[float], torch.Tensor] = lambda theta: torch.tensor(
    [[torch.cos(torch.tensor(theta / 2)), -1j * torch.sin(torch.tensor(theta / 2))],
    [-1j * torch.sin(torch.tensor(theta / 2)), torch.cos(torch.tensor(theta / 2))]],
    dtype=torch.complex64)

# rotate Y gate
RY: Callable[[float], torch.Tensor] = lambda theta: torch.tensor(
    [[torch.cos(torch.tensor(theta / 2)), -torch.sin(torch.tensor(theta / 2))],
    [torch.sin(torch.tensor(theta / 2)), torch.cos(torch.tensor(theta / 2))]],
    dtype=torch.complex64)

# rotate Z gate
RZ: Callable[[float], torch.Tensor] = lambda theta: torch.tensor(
    [[torch.cos(torch.tensor(theta / 2)) - 1j * torch.sin(torch.tensor(theta / 2)), 0],
    [0, torch.cos(torch.tensor(theta / 2)) + 1j * torch.sin(torch.tensor(theta / 2))]],
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
