# Phase 7：人类 tissuePMHC（min_pairs = 200）

本目录是与既有人类项目隔离的复现实验入口。它不修改 `scripts/` 中的原始代码，也不复用原项目的 `data/tissuePMHC/` 或 `results/tissuePMHC_*` 输出。

数据纳入规则固定为 `total_pairs > 200`（即命令行参数 `--min-pairs 200`）。新 benchmark 的数据集标识为 `tissuePMHC_phase7_min200`，所有结果写入 `results/tissuePMHC_phase7_min200_*`。

## 执行顺序

```powershell
python phase7/build_human_dataset_min200.py
python phase7/run_e2_shared_heads.py
python phase7/run_e14_auxiliary_soft_ensemble.py
python phase7/run_e17_seed_ensemble.py
python phase7/run_e26_all_in_one.py
python phase7/run_e29_multikernel_cnn_oof.py
python phase7/run_e29_independent_test.py
python phase7/run_e31_peptide_disjoint_oof.py --device cuda
python phase7/build_protein_disjoint_split.py
python phase7/run_e30_interaction_oof.py --encoder e29_cnn --skip-screen
python phase7/run_e30_interaction_oof.py --encoder e30_interaction `
  --baseline-oof-predictions results/tissuePMHC_phase7_min200_e30_interaction_oof/oof_predictions.csv `
  --baseline-candidate e29_cnn_seed_20260704
```

其中 E2、E14、E17 是一条完整的 min200 标准 split 链路：E14 读取 E2 的指标，E17 读取 E14 的分支预测。E14 的 E8/E13 对比文件是可选项，未生成时只会得到空的对应比较行，不会影响训练或 E17。

E26 会在同一份 min200 数据上重新训练其 OOF E14/E16 候选，并生成 E29 必需的 OOF 基线；它不是复用前面标准 split 的 E14/E17 文件。E29 默认只完成 OOF screen；通过 screen 后，如要生成 Phase 7 test 预测，可额外传入 `--run-test`。不得传入原项目（min500）的 OOF 结果。E30 使用独立的 protein-disjoint development split，先运行其 `e29_cnn` 匹配基线，再运行 `e30_interaction`。

E31 是冻结 E29 的 train-only connected-component peptide-disjoint OOF robustness 实验，不读取 standard test。它实时输出每个 branch/group epoch、每个 fold、每个 seed 与全程耗时，并把终端记录保存到 `results/tissuePMHC_phase7_min200_e31_peptide_disjoint_oof/run.log`；fold/seed/total 汇总另存为 `timing.csv`。正式结果已完成：mean task AUROC/AUPRC 为 `0.76520/0.74522`，worst-10 AUROC 为 `0.63817`，三个 fold 的 peptide/pair overlap 均为 0，总耗时 `1h 30m 26s`。
