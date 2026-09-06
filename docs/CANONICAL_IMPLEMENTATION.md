# Canonical implementation decision

## Decision

For the initial public-release refactor, use the root `plam.py` as the primary reference implementation.

## Evidence

- It is the largest complete PLMA script in the workspace (1,419 lines).
- It contains the PLMA modules `LRAM`, `MFAFM`, and `PLMA`.
- It contains data loading, training, validation, metrics, baseline comparison, cross-subject analysis, ablation analysis, and the executable main workflow.
- `plma_train_improved2.0.py` and `plma_train_improved3.0.py` are byte-identical historical duplicates.
- `plma_scientific_good.py` is a substantially reduced variant and should be retained for comparison, not treated as the canonical entry point until its results are verified against the published paper.
- `plma_train_improved5.0.py` contains additional NGSIM and ablation functionality but also introduces a separate data reader and should remain an experimental branch.

## Caution

This is a code-structure decision, not a claim that `plam.py` is definitively the exact code used for the published result. That must be confirmed by comparing its configuration, dataset, random seed, and reported metrics with the final paper results.
