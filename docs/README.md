# Docs Guide

These docs are intentionally ordered as a single narrative:

1. `docs/01-baseline-profile.md`
   - Historical pre-optimization profiler snapshot.
2. `docs/02-attempt-history.md`
   - Canonical ledger of every optimization attempt and its measured outcome.
3. `docs/03-findings.md`
   - Stable conclusions derived from completed attempts.
4. `docs/04-roadmap.md`
   - Current forward plan based on findings and latest baseline.
5. `docs/05-experiment-local-permutation.md`
   - Example of a failed experiment postmortem (kept to avoid re-testing known-bad direction).
6. `docs/06-assessment-hypotheses-2026-02-11.md`
   - Current bottleneck assessment, profiler synthesis, external SOTA research notes, and ranked next hypotheses.
7. `docs/07-design-h0-terminal-sampling.md`
   - Implementation-ready design for H0 (terminal-measurement sampling fast path), including correctness invariants and validation plan.

Rules:

- `02-attempt-history.md` is the source of truth for worked vs did-not-work.
- `03-findings.md` should contain only conclusions that are already validated.
- `04-roadmap.md` should contain only next-step hypotheses (not historical logs).
- New failed or ambiguous experiments should get their own `docs/0X-experiment-*.md` record and be summarized in `02-attempt-history.md`.
- `06-assessment-hypotheses-*.md` documents analysis and hypothesis formation before implementation; accepted/rejected outcomes still belong in `02-attempt-history.md`.
- `07-design-*.md` is for implementation-ready technical design prior to code changes.
