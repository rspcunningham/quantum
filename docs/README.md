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

Rules:

- `02-attempt-history.md` is the source of truth for worked vs did-not-work.
- `03-findings.md` should contain only conclusions that are already validated.
- `04-roadmap.md` should contain only next-step hypotheses (not historical logs).
- New failed or ambiguous experiments should get their own `docs/0X-experiment-*.md` record and be summarized in `02-attempt-history.md`.
