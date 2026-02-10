"""Simple Grover's search: 5 qubits, multi-controlled gates."""

from quantum import registers, H, X, CCX, ControlledGateType, measure_all
from benchmarks.cases import BenchmarkCase


def simple_grovers() -> BenchmarkCase:
    search, ancilla = registers(4, 1)
    anc = ancilla[0]

    CCCX = ControlledGateType(CCX)
    CCCCX = ControlledGateType(CCCX)

    init = H.on(search) + X(anc) + H(anc)
    oracle = CCCCX(*search, anc)
    diffuser = H.on(search) + X.on(search) + CCCCX(*search, anc) + X.on(search) + H.on(search)

    circuit = init + (oracle + diffuser) * 3 + measure_all(search)
    return BenchmarkCase(
        name="simple_grovers",
        circuit=circuit,
        expected={"1111": 0.96},
        tolerance=0.06,
    )
