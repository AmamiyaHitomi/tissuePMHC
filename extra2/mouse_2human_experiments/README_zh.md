# Mouse ← Human 未覆盖方法迁移实验

本目录只放置人类方法中尚未在历史小鼠项目完成的部分。方法总数和逐项映射见
`METHOD_INVENTORY_zh.md`。

## 数据与输出

固定默认输入：

```text
data/mousePMHC/mousePMHC_train.csv.gz
data/mousePMHC/mousePMHC_test.csv.gz
```

固定默认输出：

```text
results/mousePMHC_human_method_transfer/
```

每个入口使用独立子目录并保存 `transfer_contract.json`。不会覆盖历史
`results/mousePMHC_phase*` 结果。

## All-in-one 入口

小鼠实验可用一个入口按依赖顺序串行执行全部 12 个入口（覆盖 16 种待迁移方法）：

```powershell
$py = "E:/ancd/envs/my_pytorch/python.exe"
& $py extra2/mouse_2human_experiments/run_all_in_one.py --device cuda
```

默认启用断点续跑：同时存在 `transfer_contract.json` 和 `summary_metrics.csv` 的实验会
自动跳过。每一步状态记录在：

```text
results/mousePMHC_human_method_transfer/all_in_one_status.json
```

若上次在单个实验内部中断，导致其目录非空但尚无 `summary_metrics.csv`，请先检查该
目录，再用 `--overwrite` 重跑；若要强制重跑已完成项，则同时使用
`--no-resume --overwrite`。

仓库缺少真实 H2 pseudo-sequence 文件时，总入口只跳过 `h2_pseudoseq` 和
`h2_hybrid`，其余实验继续执行。提供文件后重复运行同一命令即可补跑。若希望在缺少
该文件时完全不启动，添加 `--require-h2-pseudo`。

建议先检查全部入口的路径和参数：

```powershell
& $py extra2/mouse_2human_experiments/run_all_in_one.py --device cuda --dry-run
```

dry-run 不训练、不创建结果目录。默认遇到真实运行错误时停止；需要记录错误并继续
后续独立实验时添加 `--continue-on-error`。

## Strict 表缺失的三种对应方法

Human strict 表中的下列三种方法此前没有对应的 mouse strict 结果：

1. one-hot logistic regression；
2. 无辅助监督的 global/H2-specific MLP dual branch；
3. 带 tissue/H2 auxiliary supervision 的 global/H2-specific dual branch。

旧 mouse E0 的 one-hot logistic regression 不使用 E33 peptide-disjoint folds，普通
迁移入口中的 dual-branch 结果也不等同于 identical-fold strict 结果，因此不能直接写入
strict 表。三种方法现在统一使用冻结的 E33 connected-component peptide-disjoint
manifest，并且只读取 mouse training pool。

三种方法的 all-in-one 入口：

```powershell
$py = "E:/ancd/envs/my_pytorch/python.exe"
& $py extra2/mouse_2human_experiments/run_strict_three_all_in_one.py --device cuda
```

默认输出：

```text
results/mousePMHC_human_method_transfer/strict_missing_three/
```

入口按 `model × seed × fold` 保存 checkpoint，可在中断后直接重新运行。正式配置使用
5 个 seeds、3 个冻结 folds 和 25 epochs。先检查路径和计划：

```powershell
& $py extra2/mouse_2human_experiments/run_strict_three_all_in_one.py --device cuda --dry-run
```

底层入口 `run_mouse_strict_missing_methods.py` 还支持通过 `--models` 只运行其中一项或
两项。该 strict 补充入口与前述 12 项普通迁移总入口相互独立，不读取 fixed test。

## 推荐运行顺序

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

H2 pseudo-sequence 两项需要先提供实验来源可靠的：

```text
data/processed/h2_pseudo_sequences.csv
```

文件格式应与 `data/processed/hla_pseudo_sequences.csv` 一致，并覆盖 H2-Db、H2-Kb、
H2-Kd、H2-Kk。文件存在后运行：

```powershell
& $py extra2/mouse_2human_experiments/run_h2_pseudoseq.py --device cuda
& $py extra2/mouse_2human_experiments/run_h2_hybrid.py --device cuda
```

代码在该文件缺失时会停止；不得使用虚构或未经核验的 H2 序列。

任一入口均可先执行：

```powershell
& $py extra2/mouse_2human_experiments/run_cagrad.py --device cuda --dry-run
```

dry-run 只检查参数和路径，不训练、不创建结果目录。
