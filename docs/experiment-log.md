# Experiment Log

Each row corresponds to a checkpoint in `docs/progress-data.md` by `idx`. Failed experiments (no checkpoint) are marked with `F` between the relevant indices.

Machine: Apple M1 Max, 32 GB. Backend: PyTorch MPS.

| idx | Commit | What changed | Core-6 @1000 | Verdict |
|---:|---|---|---:|---|
| 0 | `bade1de` | Pre-optimization baseline. Full-matrix Kronecker/swap gate application. | 718.68s | Baseline |
| 1 | `bade1de` | Re-run, no code change. | 721.60s | (noise check) |
| 2 | `86fb8c5` | Replaced Kronecker/swap gate application with tensor contraction. Eliminated `O(4^n)` matrix construction. | 533.13s | Worked (1.35x) |
| 3 | `3df121d` | Tooling cleanup (bench-plot rewrite). Minor env change. | 370.99s | Worked |
| 4 | `587a03f` | Vectorized batched measurement collapse. Removed per-shot projection loop. | 7.79s | Worked (47.6x) |
| 5 | `eb476df` | Cached gate tensors on device. Eliminated repeated CPU→MPS transfers. | 6.53s | Worked (1.19x) |
| 6 | `0c3186d` | Cached measurement weight masks for p(\|1>) matmul. | 6.41s | Worked (1.02x) |
| 7 | `03a0dbb` | Documentation and .gitignore updates. No simulation changes. | 6.35s | (no change) |
| 8 | `9701dc6` | Expanded benchmark schedule to include @10000 shots. First 10k data point. | 6.83s / 68.49s @10k | (infra) |
| 9 | `63d943f` | Added diagonal gate metadata and diagonal fast path. Avoids dense matmul for gates that only scale amplitudes. | 6.88s / 67.12s @10k | Worked (see @10k) |
| 10 | `63d943f` | Added permutation/monomial gate metadata and permutation fast path. | 2.00s / 18.07s @10k | Worked (3.4x / 3.7x) |
| 11 | `63d943f` | Re-run of idx 10. | 1.99s / 17.86s @10k | (noise check) |
| F1 | (reverted) | Replaced global permutation gather with local-axis `index_select`. Regressed on MPS: `real_grovers` 15.67s→22.99s, `toffoli_oracle_13` 10.85s→17.55s @10k. | — | Did not work |
| 12 | `e1fa5b0` | Expanded suite to 24 cases (22 runnable on MPS) with synthetic families. Permutation fast path included. | 2.02s / 18.80s @10k | (infra) |
| 13 | `4e4b2cb` | Terminal-measurement sampling fast path. For static circuits: evolve once, sample many. Removed shot-scaled unitary replay. | 0.20s / 0.28s @10k | Worked (10x / 67x) |
| 14 | `7fbe06b` | MPS sampler offload. Terminal sampling/counting on CPU instead of MPS to avoid backend sampler overhead. | 0.11s / 0.14s @10k | Worked (1.9x / 2.0x) |
| 15 | `c6d7826` | Dynamic branch engine v1. Branch-state execution for conditional/mid-circuit circuits. Replaces mask-and-restore strategy. | 0.085s / 0.085s @10k | Worked (1.3x / 1.7x) |
| 16 | `8fe4ed4` | H2 pass 2: mask multiplication, signature-based merge bucketing. Removed `nonzero`/`allclose` from dynamic path. | 0.090s / 0.093s @10k | Worked |
| 17 | `8fe4ed4` | Re-run with workload-aware dispatch + always-branch-dynamic simplification. | 0.081s / 0.080s @10k | Worked (simplification) |
| F2 | (reverted) | Forced static circuits through branch engine (disabled terminal-sampling fast path). Regressed: median 1.54x slower across 18 static cases @10k. | — | Did not work |
| 18 | `bab05d4` | Compiled edge/node dynamic graph executor. Unified static/dynamic into one graph format (static = zero-node graph). | 0.086s / 0.088s @10k | Worked (architecture) |
| 19 | `a85bf9c` | Replaced permute/matmul dense gate path with `tensordot` contraction. Cleaner code, perf-neutral. `einsum` also tested — slower on MPS, crashes at 15q+. | 0.077s / 0.081s @10k | Worked (simplification) |
| 20 | `4f3e387` | Stride-based slicing for 1q+2q dense gates with CPU scalar dispatch. Eliminates tensordot permutation copies (77% of CPU time) and per-gate MPS transfers (~450μs each). `random_universal_14` 2.4x faster. | 0.074s / 0.076s @10k | Worked (1.04x / 1.06x core-6; 2.4x on parameterized circuits) |
| F3 | (reverted) | Stride-based slicing for 2q diagonal gates. 6D view → 4D non-contiguous slices catastrophically slow on MPS: `phase_ladder_13` 6.6x slower, `qft` 5x slower, `diagonal_mesh_15` 4.9x slower. | — | Did not work |
| 21 | `24ca992` | Replaced diagonal gate device transfers with `torch.where` + CPU scalars. 1q uses cached measurement mask, 2q uses cached subindex. All ops on contiguous tensors, no MPS transfers. `phase_ladder_13` 1.34x, `diagonal_mesh_15` 1.58x faster. | 0.065s / 0.070s @10k | Worked (1.14x / 1.08x core-6) |
| 22 | `662c8be` | Compile-time diagonal gate fusion. Consecutive diagonal gates fused into single full-state diagonal via numpy (avoids MPS interference). Pre-transferred to MPS before main loop (avoids memory fragmentation). `phase_ladder_13` 5.0x, `qft` 3.5x faster. Static suite 43% faster. | 0.047s / 0.050s @10k | Worked (1.39x / 1.40x core-6) |
