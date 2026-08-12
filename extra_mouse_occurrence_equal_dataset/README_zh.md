# 小鼠 occurrence-equal 实验

本目录固定使用用户提供的 occurrence-equal train/test 文件，不生成或使用 same-label、ordinary cross-validation 或 peptide-disjoint 划分。所有训练采用 seeds `20260704`、`20260705`、`20260706`。

## 基础实验

`run_all_three_seeds.py` 运行 E0、E2、E14a 和 E29。外部预测器仅在固定测试集上评分，包括 MHCflurry 2.2.1 和 NetMHCpan 4.1b。已有结果位于 `results/e0_traditional`、`results/e2_shared_heads`、`results/e14a_auxiliary_dual_branch`、`results/e29_multikernel_cnn` 和 `results/external_predictors`。

## v7 缺失实验补跑

Mouse 专用入口：

```powershell
E:\ancd\envs\my_pytorch\python.exe extra_mouse_occurence_equal_dataset\run_v7_remaining_experiments.py --device cuda
```

该入口固定使用：

- `data/mousePMHC_occurence_equal_dataset/mousePMHC_train.csv.gz`
- `data/mousePMHC_occurence_equal_dataset/mousePMHC_test.csv.gz`
- seeds `20260704 20260705 20260706`
- 新输出目录 `extra_mouse_occurence_equal_dataset/results/v7_full_rerun`

它不会读取或覆盖旧数据集的结果目录。已经在本 occurrence-equal 目录完成的 E0、E2、E14a、E29 和外部预测器结果会复用。H2 pseudo-sequence 两项使用 NetMHCpan 4.1b 版本化资源中的四条 34 位序列；衍生表、源文件哈希和别名映射分别记录在 `data/processed/h2_pseudo_sequences.csv` 与 `h2_pseudo_sequence_provenance.json`。

每个训练日志均保存在 `results/v7_full_rerun/run_logs`，包含每个 epoch、seed 和运行耗时。完整成功运行耗时汇总在 `results/v7_full_rerun/timing_results_complete.csv`。

## 论文结果汇总

```powershell
E:\ancd\envs\my_pytorch\python.exe extra_mouse_occurence_equal_dataset\aggregate_v7_paper_results.py
```

论文唯一结果源位于 `results/v7_full_rerun/paper_results`。数据哈希、seed、每项来源和补跑合同记录在 `paper_results_provenance.json`。
