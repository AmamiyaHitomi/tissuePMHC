# Human 157-task 迁移实验

本目录将早期人类 44-task 项目的实验实现迁移到 Phase 7 的 157-task benchmark，
用于扩充论文 Table 5。这里只提供迁移代码；正式训练尚未执行。

## 数据与输出隔离

固定输入：

- `data/tissuePMHC_phase7_min200/tissuePMHC_phase7_min200_train.csv.gz`
- `data/tissuePMHC_phase7_min200/tissuePMHC_phase7_min200_test.csv.gz`

固定默认输出：

```text
results/tissuePMHC_phase7_min200_migrated_44tasks/
```

输出位置与代码迁移前保持不变。每个实验使用独立子目录，并保存
`migration_contract.json`。代码会拒绝读取或写入原 44-task 结果目录。

## All-in-one 入口

下面的独立入口固定执行 `--preset table5` 对应的 15 个实验，不会运行额外的
E15、E16、E18–E25：

```powershell
$py = "E:/ancd/envs/my_pytorch/python.exe"
& $py extra2/human_157tasks_experiments/run_all_in_one.py --device cuda
```

正式运行前可检查全部参数和路径：

```powershell
& $py extra2/human_157tasks_experiments/run_all_in_one.py --device cuda --dry-run
```

总入口按依赖顺序串行执行，任一实验失败时停止。各实验结果仍写入原有独立子目录，
不会合并，也不会与 44-task 输出混合。默认启用续跑：存在
`migration_contract.json` 和 `summary_metrics.csv` 的实验会跳过；只有 contract
而没有 summary 的未完成实验会自动重试。使用 `--no-resume` 可关闭该行为。

E4/E4b 不再读取只覆盖早期 44-task HLA 的公共 pseudo-sequence 表。入口会根据当前
157-task train 文件，从仓库内的 IPD-IMGT/HLA A/B/C protein FASTA 生成独立的：

```text
results/tissuePMHC_phase7_min200_migrated_44tasks/_derived_inputs/hla_pseudo_sequences.csv
```

该表只供本次迁移使用，不覆盖 `data/processed/hla_pseudo_sequences.csv`。

总入口默认跳过计算开销很大的 E9 CAGrad，并继续执行 E10–E14：

```powershell
& $py extra2/human_157tasks_experiments/run_all_in_one.py --device cuda
```

E10、E10b 和 E11 只把 E9 指标用于外部比较，不依赖 E9 进行训练。跳过后，这三个
实验仍会正常训练和输出自身指标，只是不生成相对于 CAGrad 的比较行。以后如果需要
恢复 E9，显式添加 `--include-cagrad`。

## 15 个独立入口与运行顺序

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

每个文件只运行对应实验。下游依赖缺失时会直接报错，不会自动重跑上游实验。
建议正式运行前先检查单个入口：

```powershell
& $py extra2/human_157tasks_experiments/run_e10_mmoe_157tasks.py `
  --device cuda --dry-run
```

## Table 5 汇总

所有计划实验完成后运行：

```powershell
& $py extra2/human_157tasks_experiments/build_migrated_table5_rows.py
```

生成：

- `results/tissuePMHC_phase7_min200_migrated_44tasks/table5_migrated_rows.csv`
- `results/tissuePMHC_phase7_min200_migrated_44tasks/table5_migrated_rows.md`

Table 5 至少保留 10 个方法；超过 10 个时全部保留，不做 top-10 截断，也不删除
负结果。完整迁移范围和使用边界见 `REPORT_zh.md`。
