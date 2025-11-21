from quantum import Circuit, QuantumSystem, run_simulation
from quantum.visualization import plot_results
from quantum.gates import H, CX, Measurement

circuit = Circuit([
    H(0),
    CX(0, 1),
    Measurement(0, 0),
    Measurement(1, 1),
])

qs = QuantumSystem(2, 2)

result = run_simulation(qs, circuit, 1000)
_ = plot_results(result)
