"""Simple Grover's search: 5 qubits, multi-controlled gates."""

import numpy as np

from quantum import registers, H, X, CustomGateType, measure_all
from benchmarks.cases import BenchmarkCase


def _mcx(n_controls: int) -> CustomGateType:
    """Multi-controlled X gate with n_controls control qubits."""
    dim = 1 << (n_controls + 1)
    matrix = np.eye(dim, dtype=np.complex64)
    matrix[dim - 2, dim - 2] = 0
    matrix[dim - 1, dim - 1] = 0
    matrix[dim - 2, dim - 1] = 1
    matrix[dim - 1, dim - 2] = 1
    return CustomGateType(matrix=matrix)


def simple_grovers() -> BenchmarkCase:
    search, ancilla = registers(4, 1)
    anc = ancilla[0]

    C4X = _mcx(4)

    init = H.on(search) + X(anc) + H(anc)
    oracle = C4X(*search, anc)
    diffuser = H.on(search) + X.on(search) + C4X(*search, anc) + X.on(search) + H.on(search)

    circuit = init + (oracle + diffuser) * 3 + measure_all(search)
    return BenchmarkCase(
        name="simple_grovers",
        circuit=circuit,
        expected={"1111": 0.96},
        tolerance=0.06,
    )
