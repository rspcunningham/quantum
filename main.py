from quantum import QuantumSystem, Circuit, Gate, gates
import torch

theta = torch.tensor(torch.pi / 2)

c = Circuit([
        Gate(gates.H, [0]),
        Gate(gates.CX, [0, 1]),
        Gate(gates.RY(theta), [0]),
    ])

c_big = Circuit([
    c, c, c
])


sys = QuantumSystem(2)
print(f"Number of circuits: 0. State: {sys}")

for i in range(10):
    sys = sys.apply_circuit(c)
    print(f"Number of circuits: {i+1}. State: {sys}")
