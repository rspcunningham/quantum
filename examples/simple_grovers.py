import numpy as np

from quantum import CustomGateType, H, X, compile, measure_all, registers

try:
    from quantum.visualization import plot_results
except ImportError:
    plot_results = None


def _mcx(n_controls: int) -> CustomGateType:
    """Multi-controlled X gate with n_controls control qubits."""
    dim = 1 << (n_controls + 1)
    matrix = np.eye(dim, dtype=np.complex64)
    matrix[dim - 2, dim - 2] = 0
    matrix[dim - 1, dim - 1] = 0
    matrix[dim - 2, dim - 1] = 1
    matrix[dim - 1, dim - 2] = 1
    return CustomGateType(matrix=matrix)


search, ancilla = registers(4, 1)
anc = ancilla[0]

C4X = _mcx(4)

init = H.on(search) + X(anc) + H(anc)
oracle = C4X(*search, anc)
diffuser = H.on(search) + X.on(search) + C4X(*search, anc) + X.on(search) + H.on(search)

circuit = init + (oracle + diffuser) * 3 + measure_all(search)
with compile(circuit) as compiled:
    result = compiled.run(100)
print(result)
if plot_results is not None:
    _ = plot_results(result)
