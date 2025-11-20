from quantum import QuantumSystem, Circuit, run_simulation
from quantum.gates import H, Measurement, X, ControlledGateType, GateType
from quantum.visualization import plot_results
from hash import get_oracle
import math
import time
import torch

# 4 bit search register
# 4 bit hash intermediate register
# 4 bit hash output register
# 1 bit ancilla

def get_init(len_non_ancilla: int) -> Circuit:
    ops_list = [H(i) for i in range(len_non_ancilla)] + [X(len_non_ancilla)]+ [H(len_non_ancilla)]
    return Circuit(ops_list)

def get_controller(n_controls: int, gate_type: GateType | ControlledGateType) -> GateType | ControlledGateType:
    if n_controls == 0: return gate_type
    return get_controller(n_controls - 1, ControlledGateType(gate_type))

def get_diffuser(len_non_ancilla: int) -> Circuit:
    h_list = [H(i) for i in range(len_non_ancilla)]
    x_list = [X(i) for i in range(len_non_ancilla)]
    controller = get_controller(len_non_ancilla, X)(*range(len_non_ancilla), len_non_ancilla)

    return Circuit(h_list + x_list + [controller] + x_list + h_list)

def get_measure_all(hash_len: int) -> Circuit:
    ops_list = [Measurement(i, i) for i in range(hash_len)]
    return Circuit(ops_list)

hash_len = 4
target_hash = [0, 1, 1, 0]

search_space = 2 ** hash_len
iterations = math.floor(math.pi / 4 * math.sqrt(search_space))

init = get_init(hash_len * 3)
oracle = get_oracle(target_hash)
diffuser = get_diffuser(hash_len * 3)
measurement = get_measure_all(hash_len)

grover_iters = [oracle, diffuser] * iterations

circuit = Circuit([
    init,
    *grover_iters,
    measurement
])

qs = QuantumSystem(hash_len * 3 + 1, hash_len)

print(f"Applying {iterations} iterations")

start_time = time.time()
result = run_simulation(qs, circuit, 1000)
end_time = time.time()

print(f"time: {end_time - start_time}")
_ = plot_results(result)
