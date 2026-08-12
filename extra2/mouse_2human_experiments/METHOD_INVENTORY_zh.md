# 人类方法与小鼠覆盖清单

统计口径：按可形成独立论文方法行的模型/配置去重，不重复计算 runner 内用于比较的
同一 baseline，也不把同一模型的不同 seed 当作不同方法。

- 人类 Table 5 方法范围：28 种；
- 加入论文中的 MHC-only、MHCflurry、NetMHCpan：31 种；
- 历史小鼠项目已覆盖：15 种；
- 尚未完成并在本目录提供迁移入口：16 种。

| # | 人类方法 | 小鼠状态 | 历史小鼠映射或新入口 |
|---:|---|---|---|
| 1 | One-hot logistic regression | 已完成 | mouse Phase 3 E0 |
| 2 | BLOSUM62 logistic regression | 已完成 | mouse Phase 3 E0 |
| 3 | BLOSUM62 HistGradientBoosting | 已完成 | mouse Phase 3 E0 |
| 4 | BLOSUM62 Extra Trees | 已完成 | mouse Phase 3 E0 |
| 5 | BLOSUM62 random forest | 已完成 | mouse Phase 3 E0 |
| 6 | Neural single-task MLP | 未完成 | `run_neural_single_conditioned.py` |
| 7 | Shared encoder + task heads | 已完成 | mouse Phase 3 E1 |
| 8 | Conditioned tissue + MHC ID | 未完成 | `run_neural_single_conditioned.py` |
| 9 | MHC pseudo-sequence conditioned | 未完成 | `run_h2_pseudoseq.py`；需要真实 H2 pseudo-sequence |
| 10 | MHC ID + pseudo-sequence hybrid | 未完成 | `run_h2_hybrid.py`；需要真实 H2 pseudo-sequence |
| 11 | FAMO dynamic weighting | 已完成 | mouse Phase 3 E5 |
| 12 | MHC-grouped hard sharing | 已完成 | mouse Phase 3 E2 H2-grouped |
| 13 | Tissue-grouped hard sharing | 未完成 | `run_tissue_grouping.py` |
| 14 | Selective global/MHC grouping | 未完成 | `run_selective_grouping.py` |
| 15 | Fixed global/MHC soft average | 已完成 | mouse Phase 4 E11 |
| 16 | Validation-delta clipped ensemble | 未完成 | `run_adaptive_soft_ensemble.py` |
| 17 | Validation-softmax ensemble | 未完成 | `run_adaptive_soft_ensemble.py` |
| 18 | CAGrad shared heads | 未完成 | `run_cagrad.py` |
| 19 | MMoE | 已完成 | mouse Phase 3 E3/E3b |
| 20 | MMoE, 4 experts × width 256 | 未完成 | `run_mmoe_tuning.py` |
| 21 | MMoE, 6 experts × width 128 | 未完成 | `run_mmoe_tuning.py` |
| 22 | DB-MTL | 未完成 | `run_dbmtl.py` |
| 23 | Pair-ranking objective | 已完成 | mouse Phase 5 E17 |
| 24 | Tissue/MHC auxiliary supervision | 已完成 | mouse Phase 4 E12 |
| 25 | Auxiliary-global + plain-MHC dual branch | 未完成 | `run_auxiliary_soft.py` |
| 26 | Auxiliary-global + auxiliary-MHC dual branch | 未完成 | `run_auxiliary_soft.py` |
| 27 | MLP dual-branch row-level seed ensemble | 未完成 | `run_mlp_dual_seed_ensemble.py` |
| 28 | Full TissuePMHC multi-kernel dual branch | 未完成 | `run_tissuepmhc_full.py` |
| 29 | Capacity-matched MHC-only model | 已完成 | `results/issue5_general_pmhc/` |
| 30 | MHCflurry presentation | 已完成 | `results/issue5_general_pmhc/` |
| 31 | NetMHCpan EL | 已完成 | `results/issue5_general_pmhc/` |

## Strict identical-fold 对应关系

Human strict 表有 6 行，原 mouse strict 表只有 BLOSUM62 random forest、shared encoder
和 Factorized MMoE 三行。缺失的严格同折对照不是论文漏写，而是此前未在冻结 E33
peptide-disjoint folds 上运行：

| Human strict 方法 | 原 Mouse strict 状态 | 新入口 |
|---|---|---|
| One-hot logistic regression | 仅旧 E0 普通实验；无 strict 结果 | `run_mouse_strict_missing_methods.py` |
| Dual branch without auxiliary supervision | 无 strict 结果 | `run_mouse_strict_missing_methods.py` |
| Dual branch with auxiliary supervision | 无 strict 结果 | `run_mouse_strict_missing_methods.py` |

三项一次运行使用 `run_strict_three_all_in_one.py`。Mouse Factorized MMoE 是物种内最终
架构，对应 Human strict 表中的最终 TissuePMHC 行仅用于“各物种 selected model”位置，
并非结构完全相同的方法。

“已完成”按方法机制建立语义映射，不声称人类和小鼠 backbone、数据量、seed 数或评估
协议完全一致。例如 mouse FAMO 和 pair ranking 使用 MMoE backbone；它们证明对应机制
已在小鼠项目中测试，因此本目录不重复创建同机制实验。Full TissuePMHC 被列为未完成，
因为 mouse Phase 4 E9 只测试了 shared-head multi-kernel CNN，没有测试人类方法的
auxiliary-global/H2-specific dual-branch rank fusion。
