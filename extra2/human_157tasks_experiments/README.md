# Human 157-task migration experiment

This directory migrates experiments from the early human 44-task to 157-task benchmark at the Phase 7 site,
To expand the paper Table 5. Only the migration code is available; formal training has not yet been implemented.

## Data is isolated from output

Fixed input:

- `data/tissuePMHC_phase7_min200/tissuePMHC_phase7_min200_train.csv.gz`
- `data/tissuePMHC_phase7_min200/tissuePMHC_phase7_min200_test.csv.gz`

Fixed default output:

```text
results/tissuePMHC_phase7_min200_migrated_44tasks/
```

The output position remains unchanged until the code is moved. Each experiment uses a separate subdirectories and saves
`migration_contract.json`. The code refuses to read or write the original 44-task result directory.

## All-in-one

A separate entry below will perform 15 experiments corresponding to `--preset table5`, without running extras
E15, E16, E18–E25:

```powershell
$py = "E:/ancd/envs/my_pytorch/python.exe"
& $py extra2/human_157tasks_experiments/run_all_in_one.py --device cuda
```

All parameters and paths can be checked before being formally run:

```powershell
& $py extra2/human_157tasks_experiments/run_all_in_one.py --device cuda --dry-run
```

The main entrance is executed in a series of dependencies, and any experiment is stopped if it fails.
Do not merge, or mix with the output of 44-task. Default enabled repeat: exists
Experiments for ZXQ0QZ and `summary_metrics.csv` will skip; only contract
The uncompleted experiment without the submary will be automatically retested. Use `--no-resume` to close the action.

E4/E4b does not read public pseudo-sequience tables that only cover the early 44-task HLA. The entrance will be based on the current situation.
157-task trade file, generated independently from IPD-IMG/HLA A/B/C protein FASTA in warehouse:

```text
results/tissuePMHC_phase7_min200_migrated_44tasks/_derived_inputs/hla_pseudo_sequences.csv
```

This table is intended for this migration only and does not cover `data/processed/hla_pseudo_sequences.csv`.

The main entrance defaults to skip the calculated high cost of E9 CaGrad and continues with E10-E14:

```powershell
& $py extra2/human_157tasks_experiments/run_all_in_one.py --device cuda
```

E10, E10b and E11 use E9 indicators only for external comparisons and not for training. After skipping, the three
The experiment will still normally train and export its own indicators, but will not generate a comparison of the CAGrad lines.
Restore E9. Add `--include-cagrad` in visible form.

## 15 separate entrances and running order

```powershell
$py = "E:/ancd/envs/my_pytorch/python.exe"

& $py extra2/human_157tasks_experiments/run_e0_traditional_157tasks.py --device cuda
& $py extra2/human_157tasks_experiments/run_e1_e2_e3_neural_157tasks.py --device cuda
& $py extra2/human_157tasks_experiments/run_e4_hla_pseudoseq_157tasks.py --device cuda
& $py extra2/human_157tasks_experiments/run_e4b_hla_hybrid_157tasks.py --device cuda
& $py extra2/human_157tasks_experiments/run_e5_famo_157tasks.py --device cuda
& $py extra2/human_157tasks_experiments/run_e6_task_grouping_157tasks.py --device cuda
& $py extra2/human_157tasks_experiments/run_e7_selective_grouping_157tasks.py --device cuda
& $py extra2/human_157tasks_experiments/run_e8_soft_ensemble_157tasks.py --device cuda
& $py extra2/human_157tasks_experiments/run_e9_cagrad_157tasks.py --device cuda
& $py extra2/human_157tasks_experiments/run_e10_mmoe_157tasks.py --device cuda
& $py extra2/human_157tasks_experiments/run_e10b_mmoe_tuning_157tasks.py --device cuda
& $py extra2/human_157tasks_experiments/run_e11_dbmtl_157tasks.py --device cuda
& $py extra2/human_157tasks_experiments/run_e12_pair_ranking_157tasks.py --device cuda
& $py extra2/human_157tasks_experiments/run_e13_auxiliary_157tasks.py --device cuda
& $py extra2/human_157tasks_experiments/run_e14_auxiliary_soft_157tasks.py --device cuda
```

Each file runs only the corresponding experiment.
It is recommended that a check of the individual entrance be made before it is officially operational:

```powershell
& $py extra2/human_157tasks_experiments/run_e10_mmoe_157tasks.py `
  --device cuda --dry-run
```

## Table 5 Summary

After all planned experiments are completed, run:

```powershell
& $py extra2/human_157tasks_experiments/build_migrated_table5_rows.py
```

Generate:

- `results/tissuePMHC_phase7_min200_migrated_44tasks/table5_migrated_rows.csv`
- `results/tissuePMHC_phase7_min200_migrated_44tasks/table5_migrated_rows.md`

Table 5 Keep at least 10 methods; all more than 10 times, without top-10 cut, without deletion
Negative results. Full range of movements and use of boundaries are shown in `REPORT_LEGACY.md`.
