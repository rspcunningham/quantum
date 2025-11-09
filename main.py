from quantum import QuantumSystem, gates
import torch

theta = torch.pi / 2

sys = QuantumSystem(2)
print(sys)
sys.apply_gate(gates.H, [0])
print(sys)
sys.apply_gate(gates.CX, [0, 1])
print(sys)
sys.apply_gate(gates.RY(theta), [0])
print(sys)
