# Design: H0 Terminal-Measurement Sampling Fast Path

Date: 2026-02-11
Scope: `src/quantum/system.py` only (no `gates.py` or benchmark case changes)
Status: design only, not implemented yet

## 1. Goal

Add a mathematically equivalent fast path in `run_simulation()` for circuits that:

1. contain no `ConditionalGate`, and
2. have only terminal measurements (after first measurement, no more gates).

For those circuits:

- evolve one state vector (`batch_size=1`) through all unitary gates
- sample `num_shots` outcomes from the final basis distribution
- map sampled basis indices to classical bit registers using measurement operations

This removes unnecessary shot-scaled unitary evolution.

## 2. Behavior To Preserve

The fast path must preserve externally observable behavior of `run_simulation()`:

- same return type: `dict[str, int]`
- same bit-string convention (big-endian classical register order)
- same measurement operation order semantics
- same handling of:
  - repeated measurements of the same qubit
  - repeated writes to the same classical bit (last write wins)
  - unmeasured classical bits (remain `0`)
  - `n_bits=0` (result key is `""`)

## 3. Why It Is Correct

For eligible circuits:

- all quantum gates occur before any measurement
- no classical control depends on measurement outcomes

So one shot is equivalent to:

1. evolve `|0...0>` to final `|psi>`
2. sample one computational basis index `b` from `|psi|^2`
3. derive measured bit values from `b`

Repeating this shot `num_shots` times is equivalent to drawing `num_shots` i.i.d. samples from the same final distribution.

Therefore, evolving `num_shots` independent copies is unnecessary.

## 4. Eligibility Detection

Add a helper that flattens the circuit and returns either:

- `None` (not eligible), or
- a plan containing:
  - ordered list of `Gate` operations before measurement
  - ordered list of terminal `Measurement` operations

Eligibility rejection conditions:

1. any `ConditionalGate` appears
2. any `Gate` appears after first `Measurement`
3. nested `Circuit` content violates (1) or (2)

Notes:

- Circuit flattening must preserve operation order.
- Measurement-only and gate-only circuits are eligible.

## 5. Fast Path Algorithm

Given eligible plan and resolved `(n_qubits, n_bits, device)`:

1. Initialize `BatchedQuantumSystem(..., batch_size=1)`.
2. Apply all planned gates via existing `apply_gate()`.
3. Compute final distribution: `probs = abs(state[0])**2`.
4. Sample basis indices:
   - `samples = torch.multinomial(probs, num_shots, replacement=True)` on device.
5. Build classical register integer per shot (`reg_codes`) in vectorized form:
   - initialize zeros (`int64`, shape `[num_shots]`)
   - for each measurement `(qubit, bit)` in order:
     - extract measured qubit bit from sampled basis index
     - overwrite target classical bit in `reg_codes`
6. Convert `reg_codes` to counts:
   - if `n_bits == 0`: return `{"": num_shots}`
   - else count by `torch.bincount` (or safe fallback) and format keys as fixed-width binary strings.

## 6. Classical Register Encoding

Use integer encoding for classical register state:

- Register bit `0` (leftmost) maps to integer bit position `n_bits - 1`.
- Register bit `n_bits - 1` (rightmost) maps to integer bit position `0`.

Per measurement update:

- measured value: `q = (samples >> (n_qubits - 1 - qubit)) & 1`
- overwrite target bit:
  - `shift = n_bits - 1 - bit`
  - `mask = 1 << shift`
  - `reg_codes = (reg_codes & ~mask) | (q << shift)`

This exactly preserves sequential overwrite semantics.

## 7. Complexity and Expected Effect

Current eligible-path complexity is effectively:

- `O(num_shots * gates * 2^n)` for unitary evolution

New fast path:

- one evolution pass: `O(gates * 2^n)`
- sampling and mapping: `O(num_shots * (1 + measurements))`

From analysis prototype (`docs/06-assessment-hypotheses-2026-02-11.md`):

- 18 static cases @10000: `590.55s -> 3.02s` (directional estimate, ~`195.6x`)

## 8. Edge Cases

1. `num_shots <= 0`
   - keep current behavior (existing code naturally creates empty batch or errors consistently).
   - do not introduce new behavior.

2. `n_bits > 63`
   - integer bit-packing in `int64` may overflow.
   - fallback strategy: construct per-shot bit matrix and count via row hashing/string path.
   - current benchmark scope does not hit this path, but design should guard it.

3. Very large `n_bits` with `bincount` memory pressure (`minlength=2^n_bits`)
   - if `n_bits` is above a threshold (for example `>20`), use `torch.unique(return_counts=True)` instead of dense `bincount`.

## 9. Validation Plan

### 9.1 Correctness

1. Full benchmark correctness (`uv run bench -v`) must remain PASS for all currently passing 22 cases.
2. Explicit edge-case checks (small custom circuits):
   - duplicate qubit measured into multiple classical bits
   - repeated writes to same classical bit
   - subset measurement with larger `n_bits`
   - no-measurement circuit (`n_bits=0`)
3. Distribution-level parity sanity:
   - for representative stochastic circuit(s), compare observed frequencies against expected tolerance (same criteria as benchmark harness).

### 9.2 Performance

1. Expect largest gains at `1000` and `10000` shots on static cases.
2. Dynamic/mid-circuit cases should remain near unchanged.
3. Capture before/after with new benchmark JSONL artifact and per-case deltas.

## 9.3 Pre-Implementation Sanity Checks (Completed)

Prototype vectorized mapping was checked against current `run_simulation()` on deterministic edge cases:

1. repeated writes to same classical bit (`last write wins`)
2. same qubit measured into multiple bits
3. subset measurement with `n_bits` larger than measured-bit count
4. `n_bits=0` no-measurement result key behavior

Observed outputs matched exactly in these checks.

## 10. Implementation Plan (Code-Level)

Primary file: `src/quantum/system.py`

1. Add circuit-analysis helper(s):
   - flatten ops
   - detect eligibility
   - return unitary+measurement plan
2. Add fast-path executor helper:
   - one-state evolution
   - multinomial sample
   - vectorized register mapping and counting
3. Update `run_simulation()`:
   - resolve resources/device as today
   - attempt fast-path for eligible circuit
   - fallback to existing batched execution otherwise

No API changes required.

## 11. Risks and Mitigations

Risk: subtle bit-order or overwrite mismatch.
Mitigation: explicit vectorized overwrite logic plus targeted custom tests.

Risk: `torch` operator support differences on MPS for counting path.
Mitigation: keep fallback counting path and test on MPS during implementation.

Risk: accidental behavior changes for ineligible circuits.
Mitigation: strict eligibility guard and unchanged fallback path.

## 12. Ready-To-Implement Checklist

- Eligibility rules finalized.
- Semantics mapping finalized.
- Counting strategy finalized with fallback.
- Validation protocol defined.

This design is implementation-ready.
