"""Generate advanced benchmark circuits with non-trivial output distributions.

Focus: complex circuits that produce broad, non-trivial measurement distributions
and stress-test simulators with deep entanglement, non-Clifford gates, and high
qubit counts (up to 30).

Run: uv run python benchmarks/generate_advanced_circuits.py
Then: uv run python benchmarks/generate_expected.py  (to get Aer reference distributions)
"""
from __future__ import annotations

import math
import random
from pathlib import Path

from qiskit import QuantumCircuit


OUTPUT_DIR = Path(__file__).parent / "circuits"


def _save(name: str, qc_or_str) -> None:
    if isinstance(qc_or_str, str):
        (OUTPUT_DIR / f"{name}.qasm").write_text(qc_or_str)
        return
    try:
        from qiskit.qasm2 import dumps
        qasm_str = dumps(qc_or_str)
    except ImportError:
        qasm_str = qc_or_str.qasm()
    (OUTPUT_DIR / f"{name}.qasm").write_text(qasm_str)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_su4_block(qc: QuantumCircuit, q0: int, q1: int, rng: random.Random) -> None:
    """Approximately Haar-random SU(4) via 3 CX + single-qubit ZYZ rotations."""
    for q in (q0, q1):
        qc.rz(rng.uniform(0, 2 * math.pi), q)
        qc.ry(rng.uniform(0, 2 * math.pi), q)
        qc.rz(rng.uniform(0, 2 * math.pi), q)
    qc.cx(q0, q1)
    for q in (q0, q1):
        qc.rz(rng.uniform(0, 2 * math.pi), q)
        qc.ry(rng.uniform(0, 2 * math.pi), q)
    qc.cx(q0, q1)
    for q in (q0, q1):
        qc.rz(rng.uniform(0, 2 * math.pi), q)
        qc.ry(rng.uniform(0, 2 * math.pi), q)
    qc.cx(q0, q1)
    for q in (q0, q1):
        qc.rz(rng.uniform(0, 2 * math.pi), q)


def _random_1q_layer(qc: QuantumCircuit, rng: random.Random) -> None:
    for q in range(qc.num_qubits):
        qc.rz(rng.uniform(0, 2 * math.pi), q)
        qc.ry(rng.uniform(0, 2 * math.pi), q)
        qc.rz(rng.uniform(0, 2 * math.pi), q)


def _cz_brick_layer(qc: QuantumCircuit, offset: int) -> None:
    for i in range(offset, qc.num_qubits - 1, 2):
        qc.cz(i, i + 1)


def _cx_brick_layer(qc: QuantumCircuit, offset: int) -> None:
    for i in range(offset, qc.num_qubits - 1, 2):
        qc.cx(i, i + 1)


# ---------------------------------------------------------------------------
# 1. QUANTUM VOLUME
#    Random SU(4) on shuffled pairs, depth = width.
#    Industry standard. Always broad distributions.
# ---------------------------------------------------------------------------

def quantum_volume_circuits():
    for n in [4, 8, 12, 16, 20, 24, 28]:
        rng = random.Random(2000 + n)
        qc = QuantumCircuit(n, n)
        depth = n
        for _ in range(depth):
            perm = list(range(n))
            rng.shuffle(perm)
            for i in range(0, n - 1, 2):
                _random_su4_block(qc, perm[i], perm[i + 1], rng)
        qc.measure(range(n), range(n))
        yield f"quantum_volume_{n}", qc


# ---------------------------------------------------------------------------
# 2. SUPREMACY-STYLE RANDOM CIRCUITS
#    Google-style: random 1q {sqrt(X), sqrt(Y), T} + CZ brick layers.
#    Gold standard for classical hardness.
# ---------------------------------------------------------------------------

def supremacy_circuits():
    for n in [8, 12, 16, 20, 24, 28, 30]:
        rng = random.Random(3000 + n)
        qc = QuantumCircuit(n, n)
        depth = max(12, n)
        for d in range(depth):
            for q in range(n):
                choice = rng.randint(0, 2)
                if choice == 0:
                    qc.rx(math.pi / 2, q)   # sqrt(X)
                elif choice == 1:
                    qc.ry(math.pi / 2, q)   # sqrt(Y)
                else:
                    qc.t(q)
            _cz_brick_layer(qc, d % 2)
        qc.measure(range(n), range(n))
        yield f"supremacy_{n}", qc


# ---------------------------------------------------------------------------
# 3. IQP FORWARD (no inverse — computationally hard distribution)
#    H → diagonal layers (RZ + CZ) → H → measure.
#    Sampling is #P-hard under plausible conjectures.
# ---------------------------------------------------------------------------

def iqp_forward_circuits():
    for n in [8, 12, 16, 20, 24, 28, 30]:
        rng = random.Random(4000 + n)
        qc = QuantumCircuit(n, n)
        qc.h(range(n))
        depth = max(6, n // 2)
        for _ in range(depth):
            for q in range(n):
                qc.rz(rng.uniform(0, 2 * math.pi), q)
            # Nearest-neighbor CZ (even + odd)
            for i in range(0, n - 1, 2):
                qc.cz(i, i + 1)
            for i in range(1, n - 1, 2):
                qc.cz(i, i + 1)
            # Long-range CZ for richer entanglement
            if n >= 8:
                stride = n // 4
                for i in range(stride):
                    qc.cz(i, i + stride)
        qc.h(range(n))
        qc.measure(range(n), range(n))
        yield f"iqp_forward_{n}", qc


# ---------------------------------------------------------------------------
# 4. TRANSVERSE-FIELD ISING MODEL (Trotter simulation)
#    H = -J Σ ZiZj - h Σ Xi, starting from |+⟩^n.
#    Physically meaningful spin dynamics.
# ---------------------------------------------------------------------------

def ising_trotter_circuits():
    for n in [4, 8, 12, 16, 20, 24, 28]:
        J = 1.0
        h_field = 0.5
        t_total = 2.0
        n_steps = max(4, n // 2)
        dt = t_total / n_steps

        qc = QuantumCircuit(n, n)
        qc.h(range(n))
        for _ in range(n_steps):
            # ZZ interaction (chain)
            for i in range(n - 1):
                qc.cx(i, i + 1)
                qc.rz(2 * J * dt, i + 1)
                qc.cx(i, i + 1)
            # Transverse field
            for i in range(n):
                qc.rx(2 * h_field * dt, i)
        qc.measure(range(n), range(n))
        yield f"ising_trotter_{n}", qc


# ---------------------------------------------------------------------------
# 5. HEISENBERG XXX MODEL (Trotter simulation)
#    H = J Σ (XX + YY + ZZ), starting from Néel state |0101...⟩.
#    Rich spin dynamics with SU(2) symmetry.
# ---------------------------------------------------------------------------

def heisenberg_trotter_circuits():
    for n in [4, 8, 12, 16, 20, 24]:
        J = 1.0
        t_total = 1.5
        n_steps = max(3, n // 4)
        dt = t_total / n_steps

        qc = QuantumCircuit(n, n)
        # Néel state
        for i in range(1, n, 2):
            qc.x(i)
        for _ in range(n_steps):
            for i in range(n - 1):
                # XX interaction: H⊗H · CX · RZ · CX · H⊗H
                qc.h(i)
                qc.h(i + 1)
                qc.cx(i, i + 1)
                qc.rz(2 * J * dt, i + 1)
                qc.cx(i, i + 1)
                qc.h(i)
                qc.h(i + 1)
                # YY interaction: Sdg⊗Sdg · H⊗H · CX · RZ · CX · H⊗H · S⊗S
                qc.sdg(i)
                qc.sdg(i + 1)
                qc.h(i)
                qc.h(i + 1)
                qc.cx(i, i + 1)
                qc.rz(2 * J * dt, i + 1)
                qc.cx(i, i + 1)
                qc.h(i)
                qc.h(i + 1)
                qc.s(i)
                qc.s(i + 1)
                # ZZ interaction: CX · RZ · CX
                qc.cx(i, i + 1)
                qc.rz(2 * J * dt, i + 1)
                qc.cx(i, i + 1)
        qc.measure(range(n), range(n))
        yield f"heisenberg_trotter_{n}", qc


# ---------------------------------------------------------------------------
# 6. VQE HARDWARE-EFFICIENT ANSATZ (forward only)
#    Alternating RY/RZ + linear CX. Random fixed angles = rich output.
# ---------------------------------------------------------------------------

def vqe_ansatz_circuits():
    for n in [8, 12, 16, 20, 24, 28, 30]:
        rng = random.Random(5000 + n)
        qc = QuantumCircuit(n, n)
        layers = max(4, n // 3)
        for _ in range(layers):
            for q in range(n):
                qc.ry(rng.uniform(0, 2 * math.pi), q)
                qc.rz(rng.uniform(0, 2 * math.pi), q)
            for i in range(n - 1):
                qc.cx(i, i + 1)
        # Final rotation
        for q in range(n):
            qc.ry(rng.uniform(0, 2 * math.pi), q)
            qc.rz(rng.uniform(0, 2 * math.pi), q)
        qc.measure(range(n), range(n))
        yield f"vqe_ansatz_{n}", qc


# ---------------------------------------------------------------------------
# 7. RANDOM CLIFFORD+T
#    Non-Clifford from T gates → exponential classical overhead.
# ---------------------------------------------------------------------------

def clifford_t_circuits():
    for n in [8, 12, 16, 20, 24, 28]:
        rng = random.Random(6000 + n)
        qc = QuantumCircuit(n, n)
        depth = max(12, n)
        gate_names = ['h', 's', 't', 'x', 'sdg', 'tdg']
        for _ in range(depth):
            for q in range(n):
                getattr(qc, rng.choice(gate_names))(q)
            qubits = list(range(n))
            rng.shuffle(qubits)
            for i in range(0, n - 1, 2):
                qc.cx(qubits[i], qubits[i + 1])
        qc.measure(range(n), range(n))
        yield f"clifford_t_{n}", qc


# ---------------------------------------------------------------------------
# 8. DEEP QAOA (random 3-regular graphs, 4-6 rounds)
#    Deeper than existing QAOA. Richer combinatorial structure.
# ---------------------------------------------------------------------------

def qaoa_deep_circuits():
    for n in [8, 12, 16, 20, 24]:
        rng = random.Random(9000 + n)
        qc = QuantumCircuit(n, n)

        # Random-ish 3-regular graph: ring + random chords
        edges = [(i, (i + 1) % n) for i in range(n)]
        attempts = 0
        while len(edges) < 3 * n // 2 and attempts < n * 4:
            i, j = sorted(rng.sample(range(n), 2))
            if (i, j) not in edges:
                edges.append((i, j))
            attempts += 1

        qc.h(range(n))
        n_rounds = min(6, max(4, n // 4))
        for _ in range(n_rounds):
            gamma = rng.uniform(0.1, math.pi)
            beta = rng.uniform(0.1, math.pi / 2)
            for i, j in edges:
                qc.cx(i, j)
                qc.rz(gamma, j)
                qc.cx(i, j)
            for i in range(n):
                qc.rx(2 * beta, i)
        qc.measure(range(n), range(n))
        yield f"qaoa_deep_{n}", qc


# ---------------------------------------------------------------------------
# 9. SU(4) CHAIN (bidirectional sweeps)
#    Random 2-qubit unitaries along a linear chain.
#    Builds maximal entanglement progressively.
# ---------------------------------------------------------------------------

def su4_chain_circuits():
    for n in [8, 12, 16, 20, 24, 28, 30]:
        rng = random.Random(10000 + n)
        qc = QuantumCircuit(n, n)
        sweeps = max(3, n // 4)
        for s in range(sweeps):
            if s % 2 == 0:
                for i in range(n - 1):
                    _random_su4_block(qc, i, i + 1, rng)
            else:
                for i in range(n - 2, -1, -1):
                    _random_su4_block(qc, i, i + 1, rng)
        qc.measure(range(n), range(n))
        yield f"su4_chain_{n}", qc


# ---------------------------------------------------------------------------
# 10. QPE LARGE (extend phase estimation to higher qubit counts)
#     Output always concentrated on ~2-8 eigenvalue peaks regardless of n.
#     Structured distribution even at 24+ qubits.
# ---------------------------------------------------------------------------

def qpe_large_circuits():
    for n in [14, 16, 18, 20, 22, 24, 26, 28]:
        n_counting = n - 1
        qc = QuantumCircuit(n, n_counting)
        # Target qubit in eigenstate
        qc.x(n - 1)
        qc.h(range(n_counting))
        # Controlled rotations: eigenvalue = 1/3
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
        yield f"qpe_large_{n}", qc


# ---------------------------------------------------------------------------
# 11. ISING ORDERED (start from |0⟩^n, strong coupling, weak field)
#     Most amplitude stays near initial state even at high qubit counts.
#     Physically: quench dynamics in ordered phase.
# ---------------------------------------------------------------------------

def ising_ordered_circuits():
    for n in [8, 12, 16, 20, 24, 28, 30]:
        J = 2.0        # Strong coupling (keeps state ordered)
        h_field = 0.2  # Weak transverse field
        t_total = 0.5  # Short time (small perturbation)
        n_steps = max(3, n // 4)
        dt = t_total / n_steps

        qc = QuantumCircuit(n, n)
        # Start from |0⟩^n (ground state of ZZ)
        for _ in range(n_steps):
            for i in range(n - 1):
                qc.cx(i, i + 1)
                qc.rz(2 * J * dt, i + 1)
                qc.cx(i, i + 1)
            for i in range(n):
                qc.rx(2 * h_field * dt, i)
        qc.measure(range(n), range(n))
        yield f"ising_ordered_{n}", qc


# ---------------------------------------------------------------------------
# 12. SHALLOW RANDOM (depth O(1), not O(n) — partial scrambling)
#     Deep enough for entanglement, shallow enough for structured output.
#     Tests simulator at moderate entanglement + high qubit count.
# ---------------------------------------------------------------------------

def shallow_random_circuits():
    for n in [12, 16, 20, 24, 28, 30]:
        rng = random.Random(11000 + n)
        qc = QuantumCircuit(n, n)
        depth = 4  # Fixed depth regardless of n
        for d in range(depth):
            for q in range(n):
                qc.ry(rng.uniform(0, 2 * math.pi), q)
                qc.rz(rng.uniform(0, 2 * math.pi), q)
            _cx_brick_layer(qc, d % 2)
        for q in range(n):
            qc.ry(rng.uniform(0, 2 * math.pi), q)
        qc.measure(range(n), range(n))
        yield f"shallow_random_{n}", qc


# ---------------------------------------------------------------------------
# 13. W-STATE LARGE (structured output at any qubit count)
#     Always exactly n outcomes, each with probability 1/n.
#     Even at 30q: each outcome has ~3.3% probability >> 0.001 threshold.
# ---------------------------------------------------------------------------

def w_state_large_circuits():
    for n in [14, 16, 18, 20, 24, 28, 30]:
        qc = QuantumCircuit(n, n)
        qc.x(0)
        for i in range(n - 1):
            theta = math.acos(math.sqrt(1 / (n - i)))
            qc.ry(2 * theta, i)
            qc.cx(i, i + 1)
            qc.ry(-2 * theta, i)
        qc.measure(range(n), range(n))
        yield f"w_state_large_{n}", qc


# ---------------------------------------------------------------------------
# 14. ENTANGLED PERTURBATION (localized excitations spread via entanglement)
#     Moderate angles on subset of qubits + entanglement layers.
#     Output concentrated near |0⟩^n with controllable spread.
# ---------------------------------------------------------------------------

def entangled_perturbation_circuits():
    for n in [12, 16, 20, 24, 28, 30]:
        rng = random.Random(13000 + n)
        qc = QuantumCircuit(n, n)
        # Medium rotations on first half — not too small (trivial), not pi/2 (uniform)
        for q in range(n // 2):
            qc.ry(rng.uniform(0.4, 1.0), q)
        # Entangle via CX chain
        for i in range(n - 1):
            qc.cx(i, i + 1)
        # Second round of rotations on all qubits
        for q in range(n):
            qc.ry(rng.uniform(0.2, 0.6), q)
            qc.rz(rng.uniform(0, math.pi), q)
        # Brick entanglement
        for i in range(0, n - 1, 2):
            qc.cx(i, i + 1)
        for i in range(1, n - 1, 2):
            qc.cx(i, i + 1)
        # Final small rotations
        for q in range(n):
            qc.ry(rng.uniform(0.1, 0.4), q)
        qc.measure(range(n), range(n))
        yield f"entangled_perturbation_{n}", qc


# ---------------------------------------------------------------------------
# ALL FAMILIES
# ---------------------------------------------------------------------------

ALL_FAMILIES = [
    # Deep random (timing benchmarks at high qubit, correctness at ≤12q)
    quantum_volume_circuits,      # 7 circuits (4–28q)
    supremacy_circuits,           # 7 circuits (8–30q)
    iqp_forward_circuits,         # 7 circuits (8–30q)
    clifford_t_circuits,          # 6 circuits (8–28q)
    su4_chain_circuits,           # 7 circuits (8–30q)
    vqe_ansatz_circuits,          # 7 circuits (8–30q)
    # Physical simulation (structured output to ~16q)
    ising_trotter_circuits,       # 7 circuits (4–28q)
    heisenberg_trotter_circuits,  # 6 circuits (4–24q)
    qaoa_deep_circuits,           # 5 circuits (8–24q)
    # Structured output at ALL sizes (correctness tests even at 28-30q)
    qpe_large_circuits,           # 8 circuits (14–28q)
    ising_ordered_circuits,       # 7 circuits (8–30q)
    shallow_random_circuits,      # 6 circuits (12–30q)
    w_state_large_circuits,       # 7 circuits (14–30q)
    entangled_perturbation_circuits,  # 6 circuits (12–30q)
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for family_fn in ALL_FAMILIES:
        for name, qc_or_str in family_fn():
            _save(name, qc_or_str)
            total += 1
            if isinstance(qc_or_str, QuantumCircuit):
                n_gates = qc_or_str.size()
                print(f"  {name} ({qc_or_str.num_qubits}q, {n_gates} gates)")
            else:
                print(f"  {name} (dynamic/qasm)")
    print(f"\nGenerated {total} advanced circuit files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
