# tissuePMHC 研究路线图

本文件是当前项目的主路线图，重点说明：

```text
1. 每个代码文件属于哪条实验线。
2. 每个实验在什么模型基础上改进。
3. 每个结果目录对应什么实验意义。
4. 当前后续实验编号如何使用。
```

## 1. 当前主线判断

当前项目有两条线：

```text
E2/E8 性能主线
E4 生物表示线
```

性能主线是：

```text
E2 shared peptide encoder + task-specific heads
→ E6 task grouping
→ E7 selective HLA/global sharing
→ E8 soft ensemble
→ E9 E2 + CAGrad
→ E10/E11/E12/E13 planned extensions
```

生物表示线是：

```text
E3 tissue + HLA ID embedding
→ E4 tissue + HLA pseudo-sequence
→ E4b HLA ID + HLA pseudo-sequence hybrid
```

目前最强模型是：

```text
E8a fixed average soft ensemble
```

它属于 E2/E8 性能主线，不属于 E4 生物表示线。E9-E13 已完成后，当前结论没有改变：E8a 仍是 standard split 下的最佳性能主模型。

## 2. 数据构建代码

| 代码文件 | 输出/用途 | 实验意义 |
|---|---|---|
| `scripts/extract_iedb_human_mhci_ligands.py` | 从 IEDB MHC ligand 原始表提取 human HLA-I 9-mer ligand | 所有后续实验的数据源 |
| `scripts/summarize_tissue_hla_uniprot.py` | 统计 tissue-HLA-UniProt 覆盖情况 | 检查是否有足够 task 和蛋白覆盖 |
| `scripts/build_tissue_specificity_pairs.py` | 构建 tissue specificity 正负样本 pair | 生成二分类任务样本 |
| `scripts/summarize_pair_peptide_lengths.py` | 检查 peptide length 分布 | 确认 9-mer 数据构造合理 |
| `scripts/build_tissuepmhc_dataset.py` | 生成 `tissuePMHC_train/test` | 所有模型共用的 standard split |
| `scripts/build_hla_pseudo_sequences.py` | 生成 HLA pseudo-sequence 表 | E4/E4b 生物表示线输入 |

## 3. 模型代码与编号

| 编号 | 代码文件 | 主线 | 模型基础 | 改进内容/实验意义 |
|---|---|---|---|---|
| E0 | `scripts/run_tissuepmhc_baselines.py` | 传统 baseline | 每个 task 独立训练 | one-hot、BLOSUM62、tree models |
| E1 | `scripts/run_tissuepmhc_neural_baselines_v2.py` | 神经 baseline | 单任务小神经网络 | 检查简单 neural model 是否有效 |
| E2 | `scripts/run_tissuepmhc_neural_baselines_v2.py` | 性能主线 | shared peptide encoder + task-specific heads | 早期最强主 baseline |
| E3 | `scripts/run_tissuepmhc_neural_baselines_v2.py` | 条件表示线 | peptide + tissue ID + HLA ID | 检查显式 tissue/HLA 条件是否有效 |
| E4 | `scripts/run_tissuepmhc_hla_pseudoseq.py` | 生物表示线 | peptide + tissue ID + HLA pseudo-sequence | 检查 HLA 序列信息是否有帮助 |
| E4b | `scripts/run_tissuepmhc_hla_hybrid.py` | 生物表示线 | HLA ID + HLA pseudo-sequence | 检查 ID 表示和生物序列表示是否互补 |
| E5 | `scripts/run_tissuepmhc_famo.py` | E2 性能主线 | E2 task-balanced shared heads | 加入 FAMO task loss weighting |
| E6 | `scripts/run_tissuepmhc_task_grouping.py` | E2 性能主线 | E2 shared heads | 按 HLA 或 tissue 分组共享 encoder |
| E7 | `scripts/run_tissuepmhc_selective_grouping.py` | E2 性能主线 | E2 global branch + E6 HLA branch | validation-based hard selection |
| E8 | `scripts/run_tissuepmhc_soft_ensemble.py` | E2/E8 性能主线 | E2 global branch + E6 HLA branch | soft ensemble 融合 global/HLA score |
| E9 | `scripts/run_tissuepmhc_cagrad.py` | E2 性能主线 | E2 shared heads | 加入 CAGrad 梯度冲突处理 |
| E10 | `scripts/run_tissuepmhc_mmoe.py` | E2/E8 性能主线 | peptide experts + task gates + task heads | MMoE 自动学习 task selective sharing |
| E10b | `scripts/run_tissuepmhc_mmoe_tuning.py` | E2/E8 性能主线 | tuned MMoE configs | 调整 expert 数量、expert 宽度和 gate entropy regularization |
| E11 | `scripts/run_tissuepmhc_dbmtl.py` | E2 性能主线 | E2 shared heads | DB-MTL 动态平衡 task loss 权重 |
| E12 | `scripts/run_tissuepmhc_pair_ranking.py` | E2/E8 性能主线 | E2 shared heads + pair_id | 加入 paired ranking loss，要求正样本分数高于同 pair 负样本 |
| E13 | `scripts/run_tissuepmhc_auxiliary_tasks.py` | E2/E8 性能主线 | E2 shared heads + auxiliary heads | 加入 tissue/HLA 辅助预测任务，帮助 shared encoder 学习任务结构 |

旧版 `scripts/run_tissuepmhc_neural_baselines.py` 是 E1/E2/E3 的第一版 runner，保留用于复现；正式比较优先看 `run_tissuepmhc_neural_baselines_v2.py`。

## 4. 结果目录与含义

| 结果目录 | 对应代码 | 对应实验 | 当前结论 |
|---|---|---|---|
| `results/tissuePMHC_baselines/` | `run_tissuepmhc_baselines.py` | E0 | 传统 baseline best mean AUROC 约 0.7558 |
| `results/tissuePMHC_neural_baselines/` | `run_tissuepmhc_neural_baselines.py` | E1/E2/E3 初版 | 初版结果，只有 1 seed |
| `results/tissuePMHC_neural_baselines_v2/` | `run_tissuepmhc_neural_baselines_v2.py` | E1/E2/E3 | E2 mean AUROC 约 0.7927，是早期最强主 baseline |
| `results/tissuePMHC_hla_pseudoseq/` | `run_tissuepmhc_hla_pseudoseq.py` | E4 | E4 mean AUROC 约 0.7728，未超过 E2 |
| `results/tissuePMHC_hla_hybrid/` | `run_tissuepmhc_hla_hybrid.py` | E4b | E4b mean AUROC 约 0.7824，接近 E3 但弱于 E2 |
| `results/tissuePMHC_famo/` | `run_tissuepmhc_famo.py` | E5 | FAMO-on-E2 mean AUROC 约 0.7758，未超过 E2 |
| `results/tissuePMHC_task_grouping/` | `run_tissuepmhc_task_grouping.py` | E6 | HLA grouping 有局部价值，tissue grouping 明显较弱 |
| `results/tissuePMHC_selective_grouping/` | `run_tissuepmhc_selective_grouping.py` | E7 | hard selection 不够稳定，mean AUROC 约 0.7904 |
| `results/tissuePMHC_soft_ensemble/` | `run_tissuepmhc_soft_ensemble.py` | E8 | E8a mean AUROC 约 0.8050，当前最强 |
| `results/tissuePMHC_cagrad/` | `run_tissuepmhc_cagrad.py` | E9 | E2 + CAGrad 未超过 E8，但可作为梯度冲突分析 |
| `results/tissuePMHC_mmoe/` | `run_tissuepmhc_mmoe.py` | E10 | MMoE 强于 E9/E7/E5，但仍弱于 E8 |
| `results/tissuePMHC_mmoe_tuning/` | `run_tissuepmhc_mmoe_tuning.py` | E10b | tuned MMoE 未超过原始 E10，也未接近 E8 |
| `results/tissuePMHC_dbmtl/` | `run_tissuepmhc_dbmtl.py` | E11 | DB-MTL mean AUROC 约 0.7817，未改善 E2 |
| `results/tissuePMHC_pair_ranking/` | `run_tissuepmhc_pair_ranking.py` | E12 | paired ranking loss mean AUROC 约 0.7807，弱于 E2 |
| `results/tissuePMHC_auxiliary_tasks/` | `run_tissuepmhc_auxiliary_tasks.py` | E13 | E13 mean AUROC 约 0.8023，明显优于 E2/E11/E12，但略弱于 E8 |

## 5. 结果文件说明

| 文件名 | 含义 |
|---|---|
| `per_task_metrics.csv` | 每个 seed、model、task 的详细指标 |
| `summary_metrics.csv` | 按 seed 和 model 聚合后的平均指标 |
| `stability_metrics.csv` | 多 seed 稳定性统计 |
| `metadata.json` | 实验参数、输入输出路径、task mapping |
| `comparison_metrics.csv` | candidate model 相对 baseline 的 per-task 差值 |
| `external_comparison_metrics.csv` | 与外部已有实验结果的比较 |
| `group_summary_metrics.csv` | E6 中每个 HLA/tissue group 的聚合表现 |
| `selection_metrics.csv` | E7 中每个 task 选择 global 还是 HLA branch 的记录 |
| `candidate_metrics.csv` | E7/E8 中 global branch 与 HLA branch 的候选指标 |
| `weight_metrics.csv` | E8 中 HLA/global ensemble 权重 |
| `task_weight_history.csv` | E5 FAMO 中每个 task 的动态权重 |
| `cagrad_weight_history.csv` | E9 CAGrad 中每个 task 的梯度组合权重 |
| `gradient_diagnostics.csv` | E9 CAGrad 中梯度冲突诊断指标 |
| `gate_weight_history.csv` | E10 MMoE 中每个 task 对各 expert 的 gate 权重 |
| `dbmtl_weight_history.csv` | E11 DB-MTL 中每个 task 的动态 loss 权重 |
| `dbmtl_diagnostics.csv` | E11 DB-MTL 中 loss 和动态权重诊断指标 |
| `ranking_diagnostics.csv` | E12 中 BCE loss、ranking loss 和 pair accuracy 的训练诊断 |
| `auxiliary_diagnostics.csv` | E13 中主任务 loss、tissue/HLA auxiliary loss 和 auxiliary accuracy 的训练诊断 |

## 6. 当前模型排序

截至 E13，当前排序为：

```text
E8a fixed soft ensemble
≈ E8b validation-delta clipped ensemble
>
E13 auxiliary tissue/HLA prediction
>
E8c validation softmax ensemble
>
E10 MMoE
≈ E2 sample BCE
>
E10b tuned MMoE
>
E11 DB-MTL
≈ E9 CAGrad
≈ E12 pair ranking
>
E4/E5/E6 等较早分支
```

因此：

```text
当前 standard-split 性能主线可以阶段性收束。
E8a fixed average soft ensemble 继续作为当前最佳性能主模型。
E13 作为最有价值的 representation auxiliary analysis。
E9/E11/E12 作为有解释价值的 negative results。
E4 线保留为 HLA 生物表示分析线，不作为当前性能主线。
```

## 7. 后续实验编号

已完成或已有代码：

```text
E0: traditional single-task baseline
E1: neural single-task baseline
E2: shared peptide encoder + task-specific heads
E3: conditioned tissue + HLA ID
E4: conditioned tissue + HLA pseudo-sequence
E4b: HLA ID + HLA pseudo-sequence hybrid
E5: FAMO on E2
E6: HLA/tissue task grouping on E2
E7: selective HLA/global hard selection
E8: global/HLA soft ensemble
E9: E2 + CAGrad
E10: MMoE selective-sharing model on the E2/E8 line
E10b: tuned MMoE configs
E11: E2 + DB-MTL
E12: E2/E8-line paired ranking loss
E13: auxiliary tissue/HLA prediction
```

当前状态：

```text
E9-E13 已完成。
E13 明显优于 E2/E11/E12，但仍略弱于 E8。
因此当前 standard-split 主线已经可以阶段性收束，后续优先进入 E8 reliability analysis，而不是继续无差别堆叠模型。
```

## 8. 下一步建议

目前最自然的下一步不再是继续跑新模型，而是围绕 E8 做可靠性和泛化边界分析：

```text
1. E8 sample_id 对齐断言。
2. E8 negative control，例如打乱 HLA branch score 后重新 ensemble。
3. peptide-disjoint / protein-disjoint split。
4. E8a/E8b 扩展到更多 seeds。
5. global_score 与 hla_score 的相关性分析。
```

`negative control` 指故意破坏模型中的关键结构，检查性能是否下降。如果打乱 HLA branch 后 E8 性能明显下降，说明原始 E8 的提升确实来自 global/HLA branch 的互补信息。

`peptide-disjoint split` 指 test 中的 peptide 不在 train 中出现，用于测试对新 peptide 的泛化。
`protein-disjoint split` 指 test 中的 source protein 不在 train 中出现，用于测试对新蛋白来源的泛化。

E13 的结论已经明确：

```text
E13 超过 sample BCE baseline。
E13 超过 E11 和 E12。
E13 略强于 E10。
E13 略弱于 E8。
tissue auxiliary accuracy 约 0.30，HLA auxiliary accuracy 约 0.77。
```

因此：

```text
E8 继续作为当前最佳性能主模型。
E13 作为 auxiliary representation analysis。
PLE 可以保留为备选，但优先级低于 E8 reliability analysis。
```

## 9. 报告叙事建议

正式报告可以按以下结构写：

```text
1. 数据集构建
2. 传统单任务 baseline
3. E1/E2/E3 神经 baseline
4. E4/E4b HLA 生物表示线
5. E5 FAMO-on-E2
6. E6 task grouping
7. E7 hard selection
8. E8 soft ensemble
9. E9 CAGrad-on-E2
10. E10 MMoE selective-sharing
11. E10b tuned MMoE
12. E11 DB-MTL
13. E12 paired ranking loss
14. E13 auxiliary tissue/HLA prediction
```

核心故事线：

```text
E2 证明 shared peptide encoder + task-specific heads 是强 baseline。
E4 证明 HLA pseudo-sequence 目前没有超过 HLA ID/E2，但有生物表示价值。
E6 说明 HLA-specific sharing 有局部价值。
E7 说明 hard selection 不够稳定。
E8 说明 global sharing 与 HLA-specific sharing 可以通过 soft ensemble 互补，成为当前最强模型。
E9 用 CAGrad 检查显式梯度冲突处理是否能进一步改善 E2/E8 线。
E10 用 MMoE 检查模型能否自动学习 task selective sharing structure。
E10b 检查更多 experts、更宽 experts 和 gate entropy regularization 能否缩小 E10 与 E8 的差距。
E11 用 DB-MTL 检查动态 task loss balancing 是否能改善 E2 主线中的 negative transfer。
E12 用 paired ranking loss 检查 pair_id 中的正负样本相对排序信号是否能改善 E2/E8 主线。
E13 用 tissue/HLA auxiliary tasks 检查显式任务结构监督是否能改善 shared peptide representation。
```
