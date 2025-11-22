from quantum import Circuit, QuantumSystem
from quantum.visualization import plot_results
from quantum.gates import H, CX, Measurement

circuit = Circuit([
    H(0),
    CX(0, 1),
    Measurement(0, 0),
    Measurement(1, 1),
])

qs = QuantumSystem(2, 2, 1000)
qs = qs.apply_circuit(circuit)
result = qs.get_results()
_ = plot_results(result)
