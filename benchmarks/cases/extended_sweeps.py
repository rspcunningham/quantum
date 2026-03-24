"""Extended benchmark sweeps for higher qubit counts and deeper circuits."""

from quantum import (
    Circuit,
    QuantumRegister,
    H,
    X,
    CX,
    CCX,
    RZ,
    CP,
    Measurement,
    measure_all,
    registers,
)
from benchmarks.cases import BenchmarkCase
from benchmarks.cases.qft import build_qft, build_inverse_qft


def _ghz_case(n_qubits: int, *, name: str) -> BenchmarkCase:
    qr = QuantumRegister(n_qubits)
    ops = [H(qr[0])] + [CX(qr[0], qr[i]) for i in range(1, n_qubits)]
    circuit = Circuit(ops) + measure_all(qr)
    return BenchmarkCase(
        name=name,
        circuit=circuit,
        expected={"0" * n_qubits: 0.5, "1" * n_qubits: 0.5},
        n_qubits=n_qubits,
        tolerance=0.06,
    )


def _qft_roundtrip_case(n_qubits: int, *, name: str) -> BenchmarkCase:
    qr = QuantumRegister(n_qubits)
    circuit = build_qft(qr, n_qubits) + build_inverse_qft(qr, n_qubits) + measure_all(qr)
    return BenchmarkCase(
        name=name,
        circuit=circuit,
        expected={"0" * n_qubits: 1.0},
        n_qubits=n_qubits,
        tolerance=0.06,
    )


def _phase_ladder(qr: QuantumRegister, n_qubits: int, n_layers: int, *, sign: float) -> Circuit:
    ops = []

    for layer in range(n_layers):
        for q in range(n_qubits):
            angle = sign * (0.029 * (q + 1) * (layer + 1))
            ops.append(RZ(angle)(qr[q]))

        for q in range(n_qubits - 1):
            angle = sign * (0.015 * (layer + 1) * (q + 1))
            ops.append(CP(angle)(qr[q], qr[q + 1]))

        for q in range(n_qubits - 3):
            angle = sign * (0.009 * (layer + 1) * (q + 1))
            ops.append(CP(angle)(qr[q], qr[q + 3]))

    return Circuit(ops)


def _phase_ladder_case(n_qubits: int, n_layers: int, *, name: str) -> BenchmarkCase:
    qr = QuantumRegister(n_qubits)
    forward = _phase_ladder(qr, n_qubits, n_layers, sign=1.0)
    backward = _phase_ladder(qr, n_qubits, n_layers, sign=-1.0)
    circuit = H.on(qr) + forward + backward + H.on(qr) + measure_all(qr)

    return BenchmarkCase(
        name=name,
        circuit=circuit,
        expected={"0" * n_qubits: 1.0},
        n_qubits=n_qubits,
        tolerance=0.06,
    )


def _toffoli_oracle_block(input_reg, work_reg, ancilla) -> Circuit:
    anc = ancilla[0]
    n_input = len(input_reg)
    n_work = len(work_reg)
    ops = []

    for i in range(18):
        a = input_reg[i % n_input]
        b = input_reg[(i + 1) % n_input]
        w = work_reg[i % n_work]
        ops.append(CCX(a, b, w))
        ops.append(CX(w, input_reg[(i + 2) % n_input]))

    for i in range(16):
        w0 = work_reg[i % n_work]
        w1 = work_reg[(i + 1) % n_work]
        tgt = input_reg[(i + 3) % n_input]
        ops.append(CCX(w0, w1, anc))
        ops.append(CCX(anc, tgt, w0))

    return Circuit(ops)


def _toffoli_oracle_case(input_bits: int, work_bits: int, ancilla_bits: int, *, name: str) -> BenchmarkCase:
    input_reg, work_reg, ancilla = registers(input_bits, work_bits, ancilla_bits)
    oracle = _toffoli_oracle_block(input_reg, work_reg, ancilla)

    all_qubits = [*input_reg, *work_reg, *ancilla]
    n_qubits = len(all_qubits)

    circuit = H.on(input_reg) + oracle + oracle.inverse() + H.on(input_reg) + measure_all(all_qubits)
    return BenchmarkCase(
        name=name,
        circuit=circuit,
        expected={"0" * n_qubits: 1.0},
        n_qubits=n_qubits,
        tolerance=0.06,
    )


def _adaptive_feedback_case(n_rounds: int, *, name: str) -> BenchmarkCase:
    qr = QuantumRegister(2)
    ops = []

    for _ in range(n_rounds):
        ops.append(H(qr[0]))
        ops.append(Measurement(qr[0], 0))
        ops.append(X(qr[0]).if_(0))
        ops.append(X(qr[1]).if_(1))
        ops.append(X(qr[1]).if_(1))

    ops.append(Measurement(qr[0], 0))

    return BenchmarkCase(
        name=name,
        circuit=Circuit(ops),
        expected={"1": 1.0},
        n_qubits=2,
        tolerance=0.02,
    )


def ghz_state_16() -> BenchmarkCase:
    return _ghz_case(16, name="ghz_state_16")


def ghz_state_18() -> BenchmarkCase:
    return _ghz_case(18, name="ghz_state_18")


def qft_12() -> BenchmarkCase:
    return _qft_roundtrip_case(12, name="qft_12")


def qft_14() -> BenchmarkCase:
    return _qft_roundtrip_case(14, name="qft_14")


def phase_ladder_13() -> BenchmarkCase:
    return _phase_ladder_case(13, 8, name="phase_ladder_13")


def toffoli_oracle_13() -> BenchmarkCase:
    return _toffoli_oracle_case(8, 4, 1, name="toffoli_oracle_13")


def adaptive_feedback_120() -> BenchmarkCase:
    return _adaptive_feedback_case(120, name="adaptive_feedback_120")
