# 人类 157-task Selected Factorized MMoE

独立入口：

```powershell
$py = "E:/ancd/envs/my_pytorch/python.exe"
& $py extra2/human_157tasks_experiments/run_selected_factorized_mmoe_157tasks.py --device cuda
```

它也已经作为最后一项接入：

```powershell
& $py extra2/human_157tasks_experiments/run_all_in_one.py --device cuda
```

如果此前 15 项已经完成，all-in-one 会跳过已有结果，仅运行新增的
`e15_selected_factorized_mmoe`。如暂时不想运行它：

```powershell
& $py extra2/human_157tasks_experiments/run_all_in_one.py --device cuda `
  --skip-experiments e15_selected_factorized_mmoe
```

## 固定设置

- 数据：Phase-7 min200 的全部 157 个 tissue--HLA tasks。
- 结构：从小鼠项目冻结迁移的 task-balanced Factorized MMoE。
- 默认 seed：`20260704 20260705 20260706`。
- 每个 seed：3-fold、pair-grouped OOF，加一次 full-train/fixed-test。
- 默认共训练 `3 × (3 + 1) = 12` 个模型。
- 最终候选：3 个 seed 的等权概率平均；同时保留每个 seed 的结果。
- 人类测试集不参与结构、超参数或 seed 选择。

终端会分别打印每个 epoch、fold、seed 和整个实验的耗时。

## 断点续跑

默认启用 `--resume`。每个完成的 OOF fold 和 full-train seed 都会立即写入：

```text
results/tissuePMHC_phase7_min200_migrated_44tasks/
  e15_selected_factorized_mmoe/_resume_shards/
```

训练中断后重新执行同一命令，完整分片会自动跳过；仅失败所在的 fold 或
full-train seed 会重跑。使用 `--no-resume` 可以强制全部重跑。

## 正式输出

默认目录：

```text
results/tissuePMHC_phase7_min200_migrated_44tasks/
  e15_selected_factorized_mmoe/
```

主要文件：

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

只检查命令、路径和计划训练次数，不开始训练：

```powershell
& $py extra2/human_157tasks_experiments/run_selected_factorized_mmoe_157tasks.py `
  --device cuda --dry-run
```
