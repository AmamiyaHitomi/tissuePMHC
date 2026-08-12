# Comparison of complete structure under Issue 9:stric split

This directory addresses item 9 of `remaining_issues_5_9.md`. Codes only read the training set and frozen
No reading test, no model.
Regenerates the olds.

The full test results, matching statistics, component interpretation and papers can be found in:

[ISSUE9_REPORT.md](ISSUE9_REPORT.md)

## Experimental overlay

Human:

1. `human_onehot_logistic_regression`
2. `human_blosum62_random_forest`
3. `human_shared_heads`
4. `human_mlp_dual_branch`:global plain MLP + HLA plain MLP,task-rank fusion
5. `human_auxiliary_dual_branch`:global auxiliary MLP + HLA plain MLP,task-rank fusion
6. ZXQ0QZ: Import Freeze E31 3 seed forecast

Mouse:

1. `mouse_blosum62_random_forest`
2. `mouse_shared_heads`
3. ZXQ0QZ: Default Import Freeze E33 5 seed forecast

Mouse 's `member_per_task_metrics.csv` gives all seed results; pre-assigned
ZXQ0QZ is a single-seed comparison of the paper. ZXQ1QZ gives five seeds
If you need to verify that training is achieved, you can fax ZXQ0QZ but the official paper.
The results of the freeze E33 should be preferred.

## Environment

According to warehouse `AGENTS.md`, use:

```powershell
E:\ancd\envs\my_pytorch\python.exe
```

All commands should be executed from the root directory of the warehouse.

## Verify input first

```powershell
E:\ancd\envs\my_pytorch\python.exe extra\issue9\run_tests.py
E:\ancd\envs\my_pytorch\python.exe extra\issue9\audit_inputs.py
```

The order examines:

- The data is identical to the manifest pair set;
- Only once per pair;
- All three folds exist;
- (b) The amount of the netted digitized digitized + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + % % % % % % + % % % %
- (a) each help-out gold contains all tasks;
- Records of the manifest SHA256.

## Smoke test

Smoke test uses full frozen folds, but only runs one seed, one epoch and the smallest array of models:

```powershell
powershell -ExecutionPolicy Bypass -File extra\issue9\run_all.ps1 -Mode smoke -Device auto
```

## Full run

```powershell
powershell -ExecutionPolicy Bypass -File extra\issue9\run_all.ps1 -Mode full -Device cuda
```

You can also run separately:

```powershell
E:\ancd\envs\my_pytorch\python.exe extra\issue9\run_human.py --device cuda
E:\ancd\envs\my_pytorch\python.exe extra\issue9\run_mouse.py --device cuda
E:\ancd\envs\my_pytorch\python.exe extra\issue9\analyze.py
```

Train to save checkpoints on `model × seed × fold`. Restart the check check with the same command after interruption and restore the full use
Checkpoint, does not silently accept missing or fold files.

The training is conducted only at the end print time, and no separate Timing CSV/JSON is generated.
The epoch, model, old, Seed time-consuming, the cumulative time-consuming of models and the total time-consuming of the whole runner.
And don't save the time.

## Fair comparison constraint

- Humans seems to be fixed to `20260704, 20260705, 20260706`.
- Mouse sees fixed to ZXQ0QZ to `20260708` in five.
- The neuromodels are set by default to 25 epochs, batts size 512, AdamW, study rate 0.001,
  weight decay 0.0001, gradient clipping 1.0.
- The traditional classifier uses its applicable freezing algorithm configuration and does not impose the epoch/optimizer rule on sklearn.
- All models are identical pair pool, task inclusion rule and frozen outer olds.
- Do not adjust the structure or hyperparameter from the poolstact OOF indicator.

## Output

The following are the results for the Human and Mouse tables:

```text
results/issue9_human_strict/
results/issue9_mouse_strict/
```

Each directory contains:

- ZXQ0QZ: a row-level preparation for each model, seed, old;
- `member_oof_predictions.csv.gz`;
- `ensemble_oof_predictions.csv.gz`;
- `member_per_task_metrics.csv`;
- `ensemble_per_task_metrics.csv`;
- `summary_metrics.csv`;
- `metadata.json`.

Unified analysis at `results/issue9_analysis/`:

- `strict_architecture_comparison.csv`;
- `paired_statistics.csv`;
- `per_task_differences.csv`;
- `seed_stability.csv`;
- `worst_group_metrics.csv`;
- `analysis_metadata.json`.

Indicators include AUROC, AUPRC, PairAcc, lf-tie PairAcc, World-task AUROC.
Main PairAcc uses `positive_score > negative_score` strictly; is also included in the main indicator 0 and reported separately
Half-tie version and win/tie/loss.

The pairing statistics include:

- mean/median task difference;
- Hodges–Lehmann difference;
- win/tie/loss;
- task bootstrap 95% interval;
- Wilcoxon signed-rank;
- Perform BH-FDR in each species x metric family.

These tests, which are clearly marked in the metadata as nominal task-level evidence, cannot be interpreted as evidence of an independent, outside force.
