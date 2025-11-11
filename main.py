from quantum import QuantumSystem, Circuit, Gate, gates
import torch

from quantum.system import Measurement

theta = torch.tensor(torch.pi / 2)

sys = QuantumSystem(2, 10)
print(sys)

c = Circuit([
    Gate(gates.H, [0]),
    Gate(gates.CX, [0, 1]),
    Gate(gates.RY(theta), [0])
])

sys = sys.apply_circuit(c)
print(sys)

sys = sys.apply_circuit(Circuit([Measurement(0, 0)]))
print(sys)
