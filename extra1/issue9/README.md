# Issue 9：strict split 下的完整架构对照

本目录解决 `remaining_issues_5_9.md` 的第 9 项。代码只读取训练集和已经冻结的
connected-component peptide-disjoint fold manifest，不读取 fixed test，也不会为不同模型
重新生成 folds。

完整实验结果、配对统计、组件解释和论文可用结论见：

[ISSUE9_REPORT_zh.md](ISSUE9_REPORT_zh.md)

## 实验覆盖

Human：

1. `human_onehot_logistic_regression`
2. `human_blosum62_random_forest`
3. `human_shared_heads`
4. `human_mlp_dual_branch`：global plain MLP + HLA plain MLP，task-rank fusion
5. `human_auxiliary_dual_branch`：global auxiliary MLP + HLA plain MLP，task-rank fusion
6. `human_tissuepmhc_net`：导入冻结 E31 三 seed strict 预测

Mouse：

1. `mouse_blosum62_random_forest`
2. `mouse_shared_heads`
3. `mouse_factorized_mmoe`：默认导入冻结 E33 五 seed strict 预测

Mouse 的 `member_per_task_metrics.csv` 给出所有单 seed 结果；预先指定
`20260704` 为论文中的 single-seed 对照。`ensemble_per_task_metrics.csv` 给出五 seed
probability-mean ensemble。若需要验证训练实现，可传 `--retrain-factorized`，但正式论文
比较应优先使用冻结 E33 结果。

## 环境

按照仓库 `AGENTS.md`，使用：

```powershell
E:\ancd\envs\my_pytorch\python.exe
```

所有命令应从仓库根目录执行。

## 先校验输入

```powershell
E:\ancd\envs\my_pytorch\python.exe extra\issue9\run_tests.py
E:\ancd\envs\my_pytorch\python.exe extra\issue9\audit_inputs.py
```

该命令检查：

- 数据与 manifest 的 pair 集完全一致；
- 每个 pair 只出现一次；
- 三个 fold 均存在；
- 每折 fitting/held-out peptide overlap 为 0；
- 每个 held-out fold 包含全部任务；
- 记录 manifest SHA256。

## Smoke test

Smoke test 使用完整 frozen folds，但只运行一个 seed、一个 epoch 和最小模型集合：

```powershell
powershell -ExecutionPolicy Bypass -File extra\issue9\run_all.ps1 -Mode smoke -Device auto
```

## 正式运行

```powershell
powershell -ExecutionPolicy Bypass -File extra\issue9\run_all.ps1 -Mode full -Device cuda
```

也可以分开运行：

```powershell
E:\ancd\envs\my_pytorch\python.exe extra\issue9\run_human.py --device cuda
E:\ancd\envs\my_pytorch\python.exe extra\issue9\run_mouse.py --device cuda
E:\ancd\envs\my_pytorch\python.exe extra\issue9\analyze.py
```

训练按 `model × seed × fold` 保存 checkpoint。中断后用相同命令重启会校验并复用完整
checkpoint，不会静默接受缺行或 fold 不匹配的文件。

训练期间只在终端打印时间，不生成单独的 timing CSV/JSON。终端信息包括每个
epoch、model、fold、seed 的耗时、各模型累计耗时和整个 runner 的总耗时。metadata
也不保存 elapsed time。

## 公平比较约束

- Human seeds 固定为 `20260704, 20260705, 20260706`。
- Mouse seeds 固定为 `20260704` 至 `20260708` 共五个。
- 神经模型默认统一为 25 epochs、batch size 512、AdamW、学习率 0.001、
  weight decay 0.0001、gradient clipping 1.0。
- 传统分类器采用其适用的冻结算法配置，不把 epoch/optimizer 规则强加给 sklearn。
- 所有模型使用完全相同的 pair pool、task inclusion rule 和 frozen outer folds。
- 不从 pooled strict OOF 指标调节架构或超参数。

## 输出

Human 与 mouse 结果目录分别为：

```text
results/issue9_human_strict/
results/issue9_mouse_strict/
```

每个目录包含：

- `checkpoints/`：每个 model、seed、fold 的 row-level prediction；
- `member_oof_predictions.csv.gz`；
- `ensemble_oof_predictions.csv.gz`；
- `member_per_task_metrics.csv`；
- `ensemble_per_task_metrics.csv`；
- `summary_metrics.csv`；
- `metadata.json`。

统一分析位于 `results/issue9_analysis/`：

- `strict_architecture_comparison.csv`；
- `paired_statistics.csv`；
- `per_task_differences.csv`；
- `seed_stability.csv`；
- `worst_group_metrics.csv`；
- `analysis_metadata.json`。

指标包括 AUROC、AUPRC、PairAcc、half-tie PairAcc、worst-task AUROC。
主 PairAcc 严格使用 `positive_score > negative_score`；并列在主指标中计 0，同时另报
half-tie 版本和 win/tie/loss。

配对统计包括：

- mean/median task difference；
- Hodges–Lehmann difference；
- win/tie/loss；
- task bootstrap 95% interval；
- Wilcoxon signed-rank；
- 在每个 species × metric family 内进行 BH-FDR。

这些检验在元数据中明确标记为 nominal task-level inference，不能解释为独立外部队列证据。
