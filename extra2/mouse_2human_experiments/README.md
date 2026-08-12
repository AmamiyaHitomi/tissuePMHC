# Mouse  Human Uncovered Migration Experiment

This directory only places those parts of the human method that have not been completed in the History Mouse project.
`METHOD_INVENTORY.md`.

## Data and Output

Fixed default input:

```text
data/mousePMHC/mousePMHC_train.csv.gz
data/mousePMHC/mousePMHC_test.csv.gz
```

Fixed default output:

```text
results/mousePMHC_human_method_transfer/
```

Use a separate subdirectories for each entry and save `transfer_contract.json`. History is not overwritten.
`results/mousePMHC_phase*` results.

## All-in-one

Mouse experiments can perform all 12 entrances (overlaying 16 migration options) by one entry in a dependent sequence:

```powershell
$py = "E:/ancd/envs/my_pytorch/python.exe"
& $py extra2/mouse_2human_experiments/run_all_in_one.py --device cuda
```

Default to enable breakpoint run: there are experiments at ZXQ0QZ and `summary_metrics.csv`
Autojump. Each step is recorded in:

```text
results/mousePMHC_human_method_transfer/all_in_one_status.json
```

If the last time it was interrupted in a single experiment, which resulted in a non-empty directory, but not yet `summary_metrics.csv`, check this first
Directory, rerun with ZXQ0QZ; to force rerun of completed items, use both
`--no-resume --overwrite`.

When the repository lacks a real H2 pseudo-file, only ZXQ0QZ and ZXZ are skipped at the main entrance
ZXQ0QZ, the rest of the experiments continue. Repeat the same command after the document is provided.
This document is not started at all, and `--require-h2-pseudo` is added.

It is recommended that all entrance paths and parameters be checked first:

```powershell
& $py extra2/mouse_2human_experiments/run_all_in_one.py --device cuda --dry-run
```

dry-run does not train, does not create result directory. Default stops when a real running error occurs; requires recording errors and continues
Add `--continue-on-error` to the subsequent independent experiment.

## Three corresponding methods missing from the Street Table

The following three methods in the Human List table have not been matched with the results:

1. one-hot logistic regression;
2. Global/H2-Specific MLP drybranch without subsidiary supervision;
3. The tissue/H2 auxiliary subvision global/H2-specific duanch.

One-hot blog response does not use E33 peptide-disjoint olds,
The migration entry of the dual-branch results are not equal to the IDENTICAL-old rule results, so they cannot be written directly
List table. Three methods now apply the frozen E33 identically to frozen paper-disjoint
Many people read trading pool.

All-in-one entrances to three methods:

```powershell
$py = "E:/ancd/envs/my_pytorch/python.exe"
& $py extra2/mouse_2human_experiments/run_strict_three_all_in_one.py --device cuda
```

Default output:

```text
results/mousePMHC_human_method_transfer/strict_missing_three/
```

The entrance press `model × seed × fold` to save checkpoint and can be rerun directly after the interruption.
5 seeds, 3 frozen folds and 25 epochs. Check path and plan first:

```powershell
& $py extra2/mouse_2human_experiments/run_strict_three_all_in_one.py --device cuda --dry-run
```

Bottom entrance ZXQ0QZ also supports only one or more of these operations via `--models`
Two. The strict supplemental entrance is separate from the 12 general migration entrances mentioned above and does not read the combination test.

## Recommended run order

```powershell
$py = "E:/ancd/envs/my_pytorch/python.exe"

& $py extra2/mouse_2human_experiments/run_neural_single_conditioned.py --device cuda
& $py extra2/mouse_2human_experiments/run_tissue_grouping.py --device cuda
& $py extra2/mouse_2human_experiments/run_selective_grouping.py --device cuda
& $py extra2/mouse_2human_experiments/run_adaptive_soft_ensemble.py --device cuda
& $py extra2/mouse_2human_experiments/run_cagrad.py --device cuda
& $py extra2/mouse_2human_experiments/run_mmoe_tuning.py --device cuda
& $py extra2/mouse_2human_experiments/run_dbmtl.py --device cuda
& $py extra2/mouse_2human_experiments/run_auxiliary_soft.py --device cuda
& $py extra2/mouse_2human_experiments/run_mlp_dual_seed_ensemble.py --device cuda
& $py extra2/mouse_2human_experiments/run_tissuepmhc_full.py --device cuda
```

H2 pseudo-equality requires first to provide reliable sources of experiments:

```text
data/processed/h2_pseudo_sequences.csv
```

File format should be consistent with `data/processed/hla_pseudo_sequences.csv` and cover H2-Db, H2-Kb,
H2-Kd, H2-Kk. Run after file exists:

```powershell
& $py extra2/mouse_2human_experiments/run_h2_pseudoseq.py --device cuda
& $py extra2/mouse_2human_experiments/run_h2_hybrid.py --device cuda
```

Codes will stop when the file is missing; no fictitious or unverified H2 sequences may be used.

Any entry may be made first:

```powershell
& $py extra2/mouse_2human_experiments/run_cagrad.py --device cuda --dry-run
```

dry-run only checks parameters and paths, does not train or create a result directory.
