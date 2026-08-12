# Human 44-task 实验向 157-task benchmark 的迁移报告

更新时间：2026-07-28  
状态：**迁移代码已完成，未执行正式训练，未产生或填入新的性能数值。**

## 1. 目标

论文 Table 5 当前只有 Shared heads、Auxiliary dual branch、MLP dual branch ensemble
和 TissuePMHC ensemble 四个方法。为与 Table 4 的方法数量对齐，Table 5 至少需要
10 个方法。本目录迁移早期 44-task 项目的兼容实验实现，在 Phase 7 的 157-task
standard fixed-test benchmark 上重新训练和评估。

所有通过 `n_tasks=157`、数据路径和 migration contract 检查的方法均保留。超过
10 个方法时不按性能截断、不删除负结果，也不只选择表现最好的方法。

## 2. 数据、代码与输出位置

迁移代码位于：

```text
extra2/human_157tasks_experiments/
```

固定数据为：

```text
data/tissuePMHC_phase7_min200/tissuePMHC_phase7_min200_train.csv.gz
data/tissuePMHC_phase7_min200/tissuePMHC_phase7_min200_test.csv.gz
```

默认结果位置保持不变：

```text
results/tissuePMHC_phase7_min200_migrated_44tasks/
```

该目录与原 44-task 的 `results/tissuePMHC_*` 以及既有 Phase 7 的
`results/tissuePMHC_phase7_min200_*` 分开。每个实验写入自己的子目录。

公共适配器 `migrate_44tasks_to_157tasks.py` 负责：

1. 定位项目根目录、原始 `scripts/` 和 Phase 7 数据；
2. 把旧 runner 的 train/test 参数切换为 157-task 数据；
3. 把所有结果路径重定向到独立迁移根目录；
4. 检查并拒绝任何仍指向原 44-task 结果的路径；
5. 写入 `migration_contract.json`，记录 source script、数据、seed、参数和输出；
6. 验证上游依赖只来自本次 157-task 迁移结果。

15 个公开实验文件是彼此独立的运行入口；它们共享上述安全逻辑，但每次只触发
一个实验。

另提供 `run_all_in_one.py` 作为固定的 Table-5 总入口。它与
`migrate_44tasks_to_157tasks.py --preset table5` 选择完全相同的 15 项，按依赖顺序
串行执行，且不允许切换到包含额外实验的 `all` preset。总入口默认跳过已有 contract
和 summary 的完整实验，并自动重试仅有 contract 的中断实验。

E4/E4b 的旧 pseudo-sequence 表只覆盖 44-task HLA，不能直接用于 157-task 数据。
适配器现从仓库内 IPD-IMGT/HLA A/B/C protein FASTA，按既有 NetMHCpan-style 34 位点
构建迁移专用表，写入迁移结果根目录的 `_derived_inputs` 子目录。它不会修改或混用
原 44-task pseudo-sequence 表。

E10、E10b、E11 对 E9 CAGrad 的逐任务指标只用于运行后的外部比较，不参与模型训练，
因此已标记为可选依赖。`run_all_in_one.py` 默认省略 E9 并继续执行 E10–E14；
`--include-cagrad` 可显式恢复 E9。跳过时相关外部比较表不会包含 CAGrad 行，但各
候选模型的训练、逐任务指标和汇总指标不受影响。

## 3. 迁移范围

| 独立入口 | 早期实验 | 方法 |
|---|---|---|
| `run_e0_traditional_157tasks.py` | E0 | five traditional per-task baselines |
| `run_e1_e2_e3_neural_157tasks.py` | E1/E2/E3 | single-task neural、shared heads、conditioned model |
| `run_e4_hla_pseudoseq_157tasks.py` | E4 | HLA pseudo-sequence |
| `run_e4b_hla_hybrid_157tasks.py` | E4b | HLA ID + pseudo-sequence |
| `run_e5_famo_157tasks.py` | E5 | FAMO |
| `run_e6_task_grouping_157tasks.py` | E6 | HLA/tissue grouped hard sharing |
| `run_e7_selective_grouping_157tasks.py` | E7 | selective grouping |
| `run_e8_soft_ensemble_157tasks.py` | E8 | global/HLA soft ensemble |
| `run_e9_cagrad_157tasks.py` | E9 | CAGrad |
| `run_e10_mmoe_157tasks.py` | E10 | MMoE |
| `run_e10b_mmoe_tuning_157tasks.py` | E10b | expanded MMoE configurations |
| `run_e11_dbmtl_157tasks.py` | E11 | DB-MTL |
| `run_e12_pair_ranking_157tasks.py` | E12 | BCE + pair ranking |
| `run_e13_auxiliary_157tasks.py` | E13 | tissue/HLA auxiliary supervision |
| `run_e14_auxiliary_soft_157tasks.py` | E14 | auxiliary global + HLA soft ensemble |

E17、E26 和 E29 已有 Phase 7 的 157-task 结果，因此不重复迁移。E30/E31 使用不同
的 OOF 泛化协议，不作为 standard fixed-test Table 5 迁移项。

## 4. 依赖与运行顺序

推荐顺序为 E0 → E1/E2/E3 → E4 → E4b → E5 → E6 → E7 → E8 → E9 →
E10 → E10b → E11 → E12 → E13 → E14。

主要依赖关系：

- E6、E7、E8 依赖 E1/E2/E3 的逐任务结果；
- E9 依赖 E1/E2/E3、E5、E7 和 E8；
- E10 依赖 E1/E2/E3、E5、E7、E8 和 E9；
- E10b 依赖 E1/E2/E3、E8、E9 和 E10；
- E11 依赖 E1/E2/E3、E8、E9、E10 和 E10b；
- E12 依赖 E1/E2/E3、E8、E10 和 E11；
- E13 依赖 E1/E2/E3、E8、E10 和 E12；
- E14 依赖 E1/E2/E3、E8 和 E13。

下游入口不会自动重跑依赖。依赖文件不存在时程序停止，并报告应先完成的实验。
完整 PowerShell 命令见 `README_zh.md`。

## 5. Table 5 汇总规则

`build_migrated_table5_rows.py` 只读取独立迁移结果根目录。默认至少要求 10 个不同
迁移方法；不足时停止并提示继续实验，超过时输出全部行。汇总结果不得混入原
44-task 指标。

正式论文 Table 5 只能加入满足以下条件的行：

1. `n_tasks=157`；
2. train/test 指向 Phase 7 min200 数据；
3. 存在并通过检查的 `migration_contract.json`；
4. source metrics 来自独立迁移结果目录；
5. 指标和 aggregation 口径在表注中明确说明。

在正式训练和复核完成前，本报告不预填任何迁移模型的 157-task 数值。
