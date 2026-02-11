"""Generate benchmark circuits as .qasm files using Qiskit.

Run: uv run python benchmarks/generate_circuits.py
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit


OUTPUT_DIR = Path(__file__).parent / "circuits"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(name: str, qc_or_str: QuantumCircuit | str) -> None:
    if isinstance(qc_or_str, str):
        (OUTPUT_DIR / f"{name}.qasm").write_text(qc_or_str)
        return
    try:
        from qiskit.qasm2 import dumps
        qasm_str = dumps(qc_or_str)
    except ImportError:
        qasm_str = qc_or_str.qasm()
    (OUTPUT_DIR / f"{name}.qasm").write_text(qasm_str)


def _random_1q_layer(qc: QuantumCircuit, rng: random.Random) -> None:
    for q in range(qc.num_qubits):
        qc.rz(rng.uniform(0, 2 * math.pi), q)
        qc.ry(rng.uniform(0, 2 * math.pi), q)
        qc.rz(rng.uniform(0, 2 * math.pi), q)


def _cx_brick_layer(qc: QuantumCircuit, offset: int) -> None:
    for i in range(offset, qc.num_qubits - 1, 2):
        qc.cx(i, i + 1)


def _cx_linear_layer(qc: QuantumCircuit) -> None:
    for i in range(qc.num_qubits - 1):
        qc.cx(i, i + 1)


# ---------------------------------------------------------------------------
# STATIC ALGORITHM FAMILIES
# ---------------------------------------------------------------------------

def bernstein_vazirani_circuits():
    for n in [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]:
        rng = random.Random(42 + n)
        n_input = n - 1
        secret = rng.randint(1, (1 << n_input) - 1)
        qc = QuantumCircuit(n, n_input)
        qc.x(n - 1)
        qc.h(range(n))
        for i in range(n_input):
            if (secret >> i) & 1:
                qc.cx(i, n - 1)
        qc.h(range(n_input))
        qc.measure(range(n_input), range(n_input))
        yield f"bernstein_vazirani_{n}", qc


def deutsch_jozsa_circuits():
    for n in [4, 6, 8, 10, 12, 14, 16, 18, 20, 22]:
        n_input = n - 1
        qc = QuantumCircuit(n, n_input)
        qc.x(n - 1)
        qc.h(range(n))
        # Balanced oracle: flip ancilla for half of inputs
        for i in range(n_input):
            qc.cx(i, n - 1)
        qc.h(range(n_input))
        qc.measure(range(n_input), range(n_input))
        yield f"deutsch_jozsa_{n}", qc


def grover_circuits():
    for n in [4, 5, 6, 7, 8, 10, 12]:
        rng = random.Random(100 + n)
        n_work = n - 1
        target = rng.randint(0, (1 << n_work) - 1)
        n_iter = max(1, int(math.pi / 4 * math.sqrt(1 << n_work)))
        n_iter = min(n_iter, 20)
        qc = QuantumCircuit(n, n_work)
        qc.h(range(n_work))
        qc.x(n - 1)
        qc.h(n - 1)
        for _ in range(n_iter):
            # Oracle: mark target state
            for i in range(n_work):
                if not ((target >> i) & 1):
                    qc.x(i)
            # Multi-controlled Z via Toffoli decomposition
            if n_work <= 3:
                if n_work == 1:
                    qc.cx(0, n - 1)
                elif n_work == 2:
                    qc.ccx(0, 1, n - 1)
                else:
                    qc.ccx(0, 1, n - 1)
                    qc.ccx(2, n - 1, 1)
                    qc.ccx(0, 1, n - 1)
                    qc.ccx(2, n - 1, 1)
            else:
                # Decompose MCX: ladder of Toffolis
                _mcx_ladder(qc, list(range(n_work)), n - 1)
            for i in range(n_work):
                if not ((target >> i) & 1):
                    qc.x(i)
            # Diffusion
            qc.h(range(n_work))
            qc.x(range(n_work))
            _mcx_ladder(qc, list(range(n_work)), n - 1)
            qc.x(range(n_work))
            qc.h(range(n_work))
        qc.measure(range(n_work), range(n_work))
        yield f"grover_{n}", qc


def _mcx_ladder(qc: QuantumCircuit, controls: list[int], target: int):
    """Multi-controlled X using Toffoli ladder. Uses controls[-1] as scratch."""
    n = len(controls)
    if n == 1:
        qc.cx(controls[0], target)
        return
    if n == 2:
        qc.ccx(controls[0], controls[1], target)
        return
    # V-chain decomposition
    qc.ccx(controls[0], controls[1], target)
    for i in range(2, n):
        qc.ccx(controls[i], target, controls[i - 1] if i < n else target)
    # Uncompute
    for i in range(n - 1, 1, -1):
        qc.ccx(controls[i], target, controls[i - 1] if i < n else target)
    qc.ccx(controls[0], controls[1], target)


def qpe_circuits():
    for n in [4, 6, 8, 10, 12]:
        n_counting = n - 1
        qc = QuantumCircuit(n, n_counting)
        # Target qubit in eigenstate
        qc.x(n - 1)
        # Apply H to counting qubits
        qc.h(range(n_counting))
        # Controlled rotations
        for i in range(n_counting):
            angle = 2 * math.pi * (1 / 3) * (2 ** i)
            qc.cp(angle, i, n - 1)
        # Inverse QFT on counting register
        for i in range(n_counting // 2):
            qc.swap(i, n_counting - 1 - i)
        for i in range(n_counting):
            for j in range(i):
                qc.cp(-math.pi / (2 ** (i - j)), j, i)
            qc.h(i)
        qc.measure(range(n_counting), range(n_counting))
        yield f"qpe_{n}", qc


def qaoa_maxcut_circuits():
    for n in [4, 6, 8, 10]:
        rng = random.Random(200 + n)
        qc = QuantumCircuit(n, n)
        # Initial superposition
        qc.h(range(n))
        # 2 QAOA rounds
        for _ in range(2):
            gamma = rng.uniform(0.1, math.pi)
            beta = rng.uniform(0.1, math.pi)
            # Random edges for MaxCut
            for i in range(n):
                j = (i + 1) % n
                qc.cx(i, j)
                qc.rz(gamma, j)
                qc.cx(i, j)
            # Mixer
            for i in range(n):
                qc.rx(2 * beta, i)
        qc.measure(range(n), range(n))
        yield f"qaoa_maxcut_{n}", qc


def ghz_circuits():
    for n in [4, 8, 12, 16, 20, 24]:
        qc = QuantumCircuit(n, n)
        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        qc.measure(range(n), range(n))
        yield f"ghz_{n}", qc


def w_state_circuits():
    for n in [3, 4, 6, 8, 10, 12]:
        qc = QuantumCircuit(n, n)
        qc.x(0)
        for i in range(n - 1):
            theta = math.acos(math.sqrt(1 / (n - i)))
            qc.ry(2 * theta, i)
            qc.cx(i, i + 1)
            qc.ry(-2 * theta, i)
        qc.measure(range(n), range(n))
        yield f"w_state_{n}", qc


def graph_state_circuits():
    for n in [4, 8, 12, 16, 20, 24]:
        qc = QuantumCircuit(n, n)
        qc.h(range(n))
        # Ring graph
        for i in range(n):
            qc.cz(i, (i + 1) % n)
        qc.measure(range(n), range(n))
        yield f"graph_state_{n}", qc


def qft_forward_circuits():
    for n in [4, 6, 8, 10, 12]:
        qc = QuantumCircuit(n, n)
        # Prepare an interesting input state
        qc.x(0)
        # QFT
        for i in range(n):
            qc.h(i)
            for j in range(i + 1, n):
                qc.cp(math.pi / (2 ** (j - i)), i, j)
        for i in range(n // 2):
            qc.swap(i, n - 1 - i)
        qc.measure(range(n), range(n))
        yield f"qft_forward_{n}", qc


def simon_circuits():
    for n_half in [2, 3, 4]:
        n = 2 * n_half
        rng = random.Random(300 + n)
        secret = rng.randint(1, (1 << n_half) - 1)
        qc = QuantumCircuit(n, n_half)
        qc.h(range(n_half))
        # Simon oracle
        for i in range(n_half):
            qc.cx(i, n_half + i)
        for i in range(n_half):
            if (secret >> i) & 1:
                qc.cx(0, n_half + i)
        qc.h(range(n_half))
        qc.measure(range(n_half), range(n_half))
        yield f"simon_{n}", qc


# ---------------------------------------------------------------------------
# ROUNDTRIP FAMILIES (forward + inverse → all-zeros)
# ---------------------------------------------------------------------------

def random_universal_circuits():
    for n in [8, 12, 16, 20, 24]:
        rng = random.Random(400 + n)
        qc = QuantumCircuit(n, n)
        depth = max(4, n // 2)
        for _ in range(depth):
            _random_1q_layer(qc, rng)
            _cx_brick_layer(qc, 0)
            _random_1q_layer(qc, rng)
            _cx_brick_layer(qc, 1)
        qc_inv = qc.inverse()
        full = qc.compose(qc_inv)
        full.measure(range(n), range(n))
        yield f"random_universal_rt_{n}", full


def clifford_scrambler_circuits():
    for n in [8, 12, 16, 20, 24]:
        rng = random.Random(500 + n)
        qc = QuantumCircuit(n, n)
        depth = max(4, n // 2)
        gates_1q = [lambda qc, q: qc.h(q), lambda qc, q: qc.s(q),
                     lambda qc, q: qc.sdg(q), lambda qc, q: qc.x(q)]
        for _ in range(depth):
            for q in range(n):
                rng.choice(gates_1q)(qc, q)
            _cx_brick_layer(qc, rng.randint(0, 1))
        qc_inv = qc.inverse()
        full = qc.compose(qc_inv)
        full.measure(range(n), range(n))
        yield f"clifford_scrambler_rt_{n}", full


def brickwork_entangler_circuits():
    for n in [8, 12, 16, 20, 24]:
        rng = random.Random(600 + n)
        qc = QuantumCircuit(n, n)
        depth = max(4, n // 2)
        for _ in range(depth):
            for q in range(n):
                qc.rx(rng.uniform(0, 2 * math.pi), q)
                qc.rz(rng.uniform(0, 2 * math.pi), q)
            _cx_brick_layer(qc, 0)
            for q in range(n):
                qc.rx(rng.uniform(0, 2 * math.pi), q)
                qc.rz(rng.uniform(0, 2 * math.pi), q)
            _cx_brick_layer(qc, 1)
        qc_inv = qc.inverse()
        full = qc.compose(qc_inv)
        full.measure(range(n), range(n))
        yield f"brickwork_entangler_rt_{n}", full


def reversible_mix_circuits():
    for n in [8, 12, 16, 20, 24]:
        rng = random.Random(700 + n)
        qc = QuantumCircuit(n, n)
        depth = max(4, n // 2)
        for d in range(depth):
            _random_1q_layer(qc, rng)
            if d % 2 == 0:
                _cx_linear_layer(qc)
            else:
                _cx_brick_layer(qc, d % 2)
            if d % 3 == 0:
                i, j = rng.sample(range(n), 2)
                qc.swap(i, j)
        qc_inv = qc.inverse()
        full = qc.compose(qc_inv)
        full.measure(range(n), range(n))
        yield f"reversible_mix_rt_{n}", full


def diagonal_mesh_circuits():
    for n in [8, 12, 16, 20, 24]:
        rng = random.Random(800 + n)
        qc = QuantumCircuit(n, n)
        depth = max(4, n // 2)
        for _ in range(depth):
            for q in range(n):
                qc.rz(rng.uniform(0, 2 * math.pi), q)
            for i in range(0, n - 1, 2):
                qc.cx(i, i + 1)
            for q in range(n):
                qc.rz(rng.uniform(0, 2 * math.pi), q)
            for i in range(1, n - 1, 2):
                qc.cx(i, i + 1)
        qc_inv = qc.inverse()
        full = qc.compose(qc_inv)
        full.measure(range(n), range(n))
        yield f"diagonal_mesh_rt_{n}", full


def iqp_roundtrip_circuits():
    for n in [8, 12, 16, 20, 24]:
        rng = random.Random(900 + n)
        qc = QuantumCircuit(n, n)
        # IQP: H, diagonal, H
        qc.h(range(n))
        depth = max(3, n // 3)
        for _ in range(depth):
            for q in range(n):
                qc.rz(rng.uniform(0, 2 * math.pi), q)
            for i in range(0, n - 1, 2):
                angle = rng.uniform(0, 2 * math.pi)
                qc.cx(i, i + 1)
                qc.rz(angle, i + 1)
                qc.cx(i, i + 1)
        qc.h(range(n))
        qc_inv = qc.inverse()
        full = qc.compose(qc_inv)
        full.measure(range(n), range(n))
        yield f"iqp_roundtrip_rt_{n}", full


def hw_efficient_ansatz_circuits():
    for n in [8, 12, 16, 20, 24]:
        rng = random.Random(1000 + n)
        qc = QuantumCircuit(n, n)
        layers = max(3, n // 3)
        for _ in range(layers):
            for q in range(n):
                qc.ry(rng.uniform(0, 2 * math.pi), q)
                qc.rz(rng.uniform(0, 2 * math.pi), q)
            for i in range(n - 1):
                qc.cx(i, i + 1)
        qc_inv = qc.inverse()
        full = qc.compose(qc_inv)
        full.measure(range(n), range(n))
        yield f"hw_efficient_ansatz_rt_{n}", full


def swap_network_circuits():
    for n in [8, 12, 16, 20, 24]:
        rng = random.Random(1100 + n)
        qc = QuantumCircuit(n, n)
        rounds = max(3, n // 3)
        for r in range(rounds):
            _random_1q_layer(qc, rng)
            for i in range(r % 2, n - 1, 2):
                qc.swap(i, i + 1)
        qc_inv = qc.inverse()
        full = qc.compose(qc_inv)
        full.measure(range(n), range(n))
        yield f"swap_network_rt_{n}", full


def qft_roundtrip_circuits():
    for n in [16, 18, 20, 22, 24]:
        qc = QuantumCircuit(n, n)
        # QFT
        for i in range(n):
            qc.h(i)
            for j in range(i + 1, n):
                qc.cp(math.pi / (2 ** (j - i)), i, j)
        for i in range(n // 2):
            qc.swap(i, n - 1 - i)
        qc_inv = qc.inverse()
        full = qc.compose(qc_inv)
        full.measure(range(n), range(n))
        yield f"qft_roundtrip_rt_{n}", full


def parametric_brickwork_circuits():
    for n in [8, 12, 16, 20, 24]:
        rng = random.Random(1200 + n)
        qc = QuantumCircuit(n, n)
        depth = max(4, n // 2)
        for d in range(depth):
            for q in range(n):
                qc.rx(rng.uniform(0, 2 * math.pi), q)
                qc.ry(rng.uniform(0, 2 * math.pi), q)
            _cx_brick_layer(qc, d % 2)
        qc_inv = qc.inverse()
        full = qc.compose(qc_inv)
        full.measure(range(n), range(n))
        yield f"parametric_brickwork_rt_{n}", full


# ---------------------------------------------------------------------------
# DYNAMIC FAMILIES — emit QASM strings directly (Qiskit 2.x dropped c_if)
# ---------------------------------------------------------------------------

def _qasm_header(n_qubits: int, n_bits: int) -> str:
    return (
        'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        f"qreg q[{n_qubits}];\ncreg c[{n_bits}];\n"
    )


def _qasm_cond(creg: str, n_bits: int, bit_index: int, gate_line: str) -> str:
    """Emit QASM if(creg==val) gate; where val has only bit_index set (LE)."""
    val = 1 << bit_index
    return f"if({creg}=={val}) {gate_line};\n"


def teleportation_chain_circuits():
    for n in [3, 5, 7, 9]:
        lines = [_qasm_header(n, n)]
        lines.append("h q[0];\n")
        lines.append("t q[0];\n")
        for i in range(0, n - 2, 2):
            if i + 2 >= n:
                break
            lines.append(f"h q[{i+1}];\n")
            lines.append(f"cx q[{i+1}],q[{i+2}];\n")
            lines.append(f"cx q[{i}],q[{i+1}];\n")
            lines.append(f"h q[{i}];\n")
            lines.append(f"measure q[{i}] -> c[{i}];\n")
            lines.append(f"measure q[{i+1}] -> c[{i+1}];\n")
            lines.append(_qasm_cond("c", n, i + 1, f"x q[{i+2}]"))
            lines.append(_qasm_cond("c", n, i, f"z q[{i+2}]"))
        lines.append(f"tdg q[{n-1}];\n")
        lines.append(f"h q[{n-1}];\n")
        lines.append(f"measure q[{n-1}] -> c[{n-1}];\n")
        yield f"teleportation_chain_{n}", "".join(lines)


def repeat_until_success_circuits():
    configs = [(2, 10), (2, 50), (2, 100), (3, 30)]
    for n_qubits, n_rounds in configs:
        lines = [_qasm_header(n_qubits, n_qubits)]
        for _ in range(n_rounds):
            lines.append("h q[0];\n")
            lines.append("measure q[0] -> c[0];\n")
            lines.append(_qasm_cond("c", n_qubits, 0, "x q[0]"))
            if n_qubits > 2:
                lines.append("cx q[0],q[1];\n")
        lines.append("measure q[0] -> c[0];\n")
        yield f"repeat_until_success_{n_qubits}q_{n_rounds}r", "".join(lines)


def syndrome_extraction_circuits():
    for n_data in [3, 5, 7, 9]:
        n_syndrome = n_data - 1
        n_total = n_data + n_syndrome
        lines = [_qasm_header(n_total, n_total)]
        lines.append("h q[0];\n")
        for i in range(n_data - 1):
            lines.append(f"cx q[{i}],q[{i+1}];\n")
        for _round in range(2):
            for s in range(n_syndrome):
                data_q = s
                syn_q = n_data + s
                bit_idx = n_data + s
                lines.append(f"cx q[{data_q}],q[{syn_q}];\n")
                lines.append(f"cx q[{data_q+1}],q[{syn_q}];\n")
                lines.append(f"measure q[{syn_q}] -> c[{bit_idx}];\n")
                lines.append(_qasm_cond("c", n_total, bit_idx, f"x q[{data_q}]"))
                lines.append(f"measure q[{syn_q}] -> c[{bit_idx}];\n")
                lines.append(_qasm_cond("c", n_total, bit_idx, f"x q[{syn_q}]"))
        for i in range(n_data):
            lines.append(f"measure q[{i}] -> c[{i}];\n")
        yield f"syndrome_extraction_{n_total}", "".join(lines)


def adaptive_scaled_circuits():
    configs = [(3, 80), (4, 60), (6, 40)]
    for n_qubits, n_rounds in configs:
        lines = [_qasm_header(n_qubits, n_qubits)]
        for _ in range(n_rounds):
            lines.append("h q[0];\n")
            lines.append("measure q[0] -> c[0];\n")
            lines.append(_qasm_cond("c", n_qubits, 0, "x q[0]"))
            for q in range(1, n_qubits):
                lines.append(f"cx q[0],q[{q}];\n")
        lines.append("measure q[0] -> c[0];\n")
        yield f"adaptive_scaled_{n_qubits}q_{n_rounds}r", "".join(lines)


def conditional_rotation_circuits():
    for n in [3, 4, 5, 6]:
        rounds = 20
        lines = [_qasm_header(n, n)]
        for _ in range(rounds):
            lines.append("h q[0];\n")
            lines.append("measure q[0] -> c[0];\n")
            lines.append(_qasm_cond("c", n, 0, "x q[0]"))
            for q in range(1, n):
                lines.append(_qasm_cond("c", n, 0, f"x q[{q}]"))
            if n > 2:
                lines.append("cx q[1],q[2];\n")
        lines.append("measure q[0] -> c[0];\n")
        yield f"conditional_rotation_{n}", "".join(lines)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

ALL_FAMILIES = [
    # Static algorithms
    bernstein_vazirani_circuits,
    deutsch_jozsa_circuits,
    grover_circuits,
    qpe_circuits,
    qaoa_maxcut_circuits,
    ghz_circuits,
    w_state_circuits,
    graph_state_circuits,
    qft_forward_circuits,
    simon_circuits,
    # Roundtrip
    random_universal_circuits,
    clifford_scrambler_circuits,
    brickwork_entangler_circuits,
    reversible_mix_circuits,
    diagonal_mesh_circuits,
    iqp_roundtrip_circuits,
    hw_efficient_ansatz_circuits,
    swap_network_circuits,
    qft_roundtrip_circuits,
    parametric_brickwork_circuits,
    # Dynamic
    teleportation_chain_circuits,
    repeat_until_success_circuits,
    syndrome_extraction_circuits,
    adaptive_scaled_circuits,
    conditional_rotation_circuits,
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for family_fn in ALL_FAMILIES:
        for name, qc_or_str in family_fn():
            _save(name, qc_or_str)
            total += 1
            if isinstance(qc_or_str, str):
                print(f"  {name} (dynamic/qasm)")
            else:
                print(f"  {name} ({qc_or_str.num_qubits}q, {qc_or_str.num_clbits}c)")
    print(f"\nGenerated {total} circuit files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
