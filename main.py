from quantum import QuantumSystem, Circuit, run_simulation
from quantum.visualization import plot_results
from quantum.gates import H, CX, RY, Measurement
import math

qc = Circuit([
    H(0),
    CX(0, 1),
    RY(math.pi/2)(0),
    Measurement(0, 0),
    Measurement(1, 1)
])

num_shots = 1000
initial_system = QuantumSystem(2, 2)
counts = run_simulation(initial_system, qc, num_shots)

print(f"Results from {num_shots} shots:")
for state, count in sorted(counts.items()):
    percentage = (count / num_shots) * 100
    print(f"  {state}: {count:4d} ({percentage:5.1f}%)")
print()

_ = plot_results(counts, title=f"Quantum Circuit Results ({num_shots} shots)")
