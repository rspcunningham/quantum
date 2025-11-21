from quantum import QuantumSystem, Circuit, run_simulation
from quantum.visualization import plot_results
from quantum.gates import H, Measurement, CCX, X, ControlledGateType

# 4-bit search space
# search register = 0, 1, 2, 3
# ancilla = 4

init = Circuit([
    H(0),
    H(1),
    H(2),
    H(3),
    X(4),
    H(4)
])

CCCX = ControlledGateType(CCX)
CCCCX = ControlledGateType(CCCX)
# assume |1111> is the target
oracle = Circuit([
    CCCCX(0, 1, 2, 3, 4)
])

diffuser = Circuit([
    H(0),
    H(1),
    H(2),
    H(3),
    X(0),
    X(1),
    X(2),
    X(3),
    CCCCX(0, 1, 2, 3, 4),
    X(0),
    X(1),
    X(2),
    X(3),
    H(0),
    H(1),
    H(2),
    H(3)
])

measurement = Circuit([
    Measurement(0,0),
    Measurement(1,1),
    Measurement(2,2),
    Measurement(3,3)
])

circuit = Circuit([
    init,
    oracle,
    diffuser,
    oracle,
    diffuser,
    oracle,
    diffuser,
    measurement
])

qs = QuantumSystem(5, 4)
#_ = qs.apply_circuit(circuit)
#print(qs)
#_ = plot_probs(qs)

result = run_simulation(qs, circuit, 100)
_ = plot_results(result)
