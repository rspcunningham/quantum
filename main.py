from system import QuantumSystem
from gates import *

state = QuantumSystem(3)

print(state)

# apply identity gate to state
state.apply_gate(H, [0])
state.apply_gate(H, [2])
state.apply_gate(RZ(15), [2])
state.apply_gate(RZ(0.5), [1])
state.apply_gate(RZ(0.2), [0])

print(state)
print(state.get_distribution())
