# Human 157-task-set

Independence entrance:

```powershell
$py = "E:/ancd/envs/my_pytorch/python.exe"
& $py extra2/human_157tasks_experiments/run_selected_factorized_mmoe_157tasks.py --device cuda
```

It has also been the last access:

```powershell
& $py extra2/human_157tasks_experiments/run_all_in_one.py --device cuda
```

If 15 previous entries have been completed, all-in-one will skip the results and run only the new ones
ZXQ0QZ. If you don't want to run it for the time being:

```powershell
& $py extra2/human_157tasks_experiments/run_all_in_one.py --device cuda `
  --skip-experiments e15_selected_factorized_mmoe
```

## Fixed Settings

- Data: 157 of the total number of items -- HLA tasks.
- Structure: Dismantled task-balanced Factorized MMOE from the mouse project.
- Default seed: `20260704 20260705 20260706`.
- Seed: 3-old, Pair-grouped OOF, plus one full-train/fixed-test.
- A total of `3 × (3 + 1) = 12` models are trained by default.
- Final candidate: 3 seeds with an equal probability; while retaining the results of each seed.
- The human test set does not involve structural, hyperparameter or seed selection.

The terminal will print the time spent on each epoch, fold, Seed and the whole experiment separately.

## Break and break.

Enables `--resume` by default. Every completed OOFold and full-train seed is written immediately:

```text
results/tissuePMHC_phase7_min200_migrated_44tasks/
  e15_selected_factorized_mmoe/_resume_shards/
```

Re-execut the same command after training breaks, the complete fragment will skip automatically; only the failed fold or
Full-train seen will run again.

## Official Output

Default directory:

```text
results/tissuePMHC_phase7_min200_migrated_44tasks/
  e15_selected_factorized_mmoe/
```

Main documents:

- `oof_member_predictions.csv`
- `oof_ensemble_predictions.csv`
- `test_member_predictions.csv`
- `test_ensemble_predictions.csv`
- `per_task_metrics.csv`
- `summary_metrics.csv`
- `stability_metrics.csv`
- `training_history.csv`
- `metadata.json`
- `migration_contract.json`

Only orders, paths and planned trainings are checked and no training is initiated:

```powershell
& $py extra2/human_157tasks_experiments/run_selected_factorized_mmoe_157tasks.py `
  --device cuda --dry-run
```
