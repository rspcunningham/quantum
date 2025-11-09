import torch

__all__ = ["I", "H", "X", "Y", "Z", "RZ", "CNOT"]

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

def RX(theta: float) -> torch.Tensor:
    half_theta = torch.tensor(theta / 2)
    return torch.tensor(
        [[torch.cos(half_theta), -1j * torch.sin(half_theta)],
        [-1j * torch.sin(half_theta), torch.cos(half_theta)]],
        dtype=torch.complex64)

def RY(theta: float) -> torch.Tensor:
    half_theta = torch.tensor(theta / 2)
    return torch.tensor(
        [[torch.cos(half_theta), -torch.sin(half_theta)],
        [torch.sin(half_theta), torch.cos(half_theta)]],
        dtype=torch.complex64)

def RZ(theta: float) -> torch.Tensor:
    half_theta = torch.tensor(theta / 2)
    return torch.tensor(
        [[torch.cos(half_theta) - 1j * torch.sin(half_theta), 0],
        [0, torch.cos(half_theta) + 1j * torch.sin(half_theta)]],
        dtype=torch.complex64)

# CNOT gate (2-qubit gate)
# Basis ordering: |00⟩, |01⟩, |10⟩, |11⟩
# CNOT flips target if control is |1⟩
CNOT = torch.tensor(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 1, 0]],
    dtype=torch.complex64)
