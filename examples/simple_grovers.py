from quantum import QuantumSystem, Circuit, run_simulation
from quantum.visualization import plot_results
from quantum.gates import H, Measurement, CCX, X, ControlledGateType


# 3-bit search space
# search register = 0, 1, 2
# ancilla = 3

init = Circuit([
    H(0),
    H(1),
    H(2),
    X(3),
    H(3)
])

CCCX = ControlledGateType(CCX)
# assume |00> is the target
oracle = Circuit([
    X(0),
    X(1),
    X(2),
    CCCX(0, 1, 2, 3),
    X(0),
    X(1),
    X(2)
])

diffuser = Circuit([
    H(0),
    H(1),
    H(2),
    X(0),
    X(1),
    X(2),
    CCCX(0, 1, 2, 3),
    X(0),
    X(1),
    X(2),
    H(0),
    H(1),
    H(2)
])

measurement = Circuit([
    Measurement(0,0),
    Measurement(1,1),
    Measurement(2,2)
])

circuit = Circuit([
    init,
    oracle,
    diffuser,
    oracle,
    diffuser,
    measurement
])

qs = QuantumSystem(4, 3)
#_ = qs.apply_circuit(circuit)
#print(qs)
#_ = plot_probs(qs)

result = run_simulation(qs, circuit, 10)
_ = plot_results(result)
