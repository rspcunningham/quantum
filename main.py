from quantum import QuantumSystem, Circuit, Gate, gates
import torch

from quantum.system import Measurement

theta = torch.tensor(torch.pi / 2)

c = Circuit([
    Gate(gates.H, [0]),
    Gate(gates.CX, [0, 1]),
    Measurement(0, 0),
    Measurement(1, 1)
])

for i in range(1):
    sys = QuantumSystem(2, 2)
    result = sys.apply_circuit(c)
    print(sys.bit_register)
