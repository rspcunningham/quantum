from quantum import QuantumSystem, Circuit, Gate
from quantum.gates import H, CX, RY
import torch

theta = torch.pi / 2

sys = QuantumSystem(2)
print(sys)
sys.apply_gate(H, [0])
print(sys)
sys.apply_gate(CX, [0, 1])
print(sys)
sys.apply_gate(RY(theta), [0])
print(sys)

print("------------------------------------------------------------------------")

sys = QuantumSystem(2)
print(sys)
c = Circuit(
    n_qubits=2,
    gates=[
        Gate(H, [0]),
        Gate(CX, [0, 1]),
        Gate(RY(theta), [0])
    ])

sys.apply_circuit(c)
print(sys)
