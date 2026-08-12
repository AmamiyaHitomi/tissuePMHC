# Human occurrence-equal 实验

本目录只用于 Human 新 occurrence-equal 数据集。Mouse 代码和结果位于
`extra_mouse_occurence_equal_dataset`，项目根目录下的 `results` 属于旧数据结果，
本套运行器会主动拒绝读取它。

## 固定协议

- 训练集：`data/humanPMHC_occurence_equal_dataset/humanPMHC_train.csv.gz`
- 测试集：`data/humanPMHC_occurence_equal_dataset/humanPMHC_test.csv.gz`
- seeds：`20260704 20260705 20260706`
- 完整任务数：77
- HLA pseudo-sequence：`data/processed/hla_pseudo_sequences_occurrence_equal.csv`
- 新的剩余实验输出：`results/v7_full_rerun`

已经完成并直接复用的新数据结果为 E0、E2、E14a、E29 和 external predictors；
不会从旧数据集重跑或复制结果。E14b 和 E17 需要 E14 的分支级预测，而此前 E14a
运行只保留了融合分数，因此 `auxiliary_soft` 会重新计算这些共享分支，但论文中的
E14a 行仍使用已完成的新数据结果。

## 命令

运行全部缺失项：

```powershell
E:\ancd\envs\my_pytorch\python.exe extra_occurrence_equal_dataset\run_v7_remaining_experiments.py --device cuda
```

断点续跑使用同一命令；带有完整 contract 和 summary 的实验会自动跳过。查看计划而
不运行：

```powershell
E:\ancd\envs\my_pytorch\python.exe extra_occurrence_equal_dataset\run_v7_remaining_experiments.py --dry-run --device cuda
```

单独运行一个方法：

```powershell
E:\ancd\envs\my_pytorch\python.exe extra_occurrence_equal_dataset\run_human_experiment.py hla_pseudoseq --device cuda
```

全部完成后汇总论文来源：

```powershell
E:\ancd\envs\my_pytorch\python.exe extra_occurrence_equal_dataset\aggregate_v7_paper_results.py
```

每个源训练器打印 seed 和逐 epoch 信息；总套件另写
`results/v7_full_rerun/orchestration_timing.csv`，日志保存在 `run_logs`。
