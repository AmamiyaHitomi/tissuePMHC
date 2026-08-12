# TissuePMHC Final Phase 无训练分析报告

更新日期：2026-07-17  
状态：01–09 默认分析已完成；peptide-component cluster bootstrap **尚未产生结果**。  
数据边界：本报告比较同一 train pair pool 上的 standard pair-grouped OOF 与 connected-component peptide-disjoint OOF，不把内部 fixed test 与 strict OOF 直接相减。

## 1. 执行范围与产物

`final_phase/` 中的 9 个入口均已实际运行，默认分析总耗时约 93 秒，不包含任何模型训练：

1. PairAcc；
2. standard/strict task-fold size matching audit；
3. parent-UniProt overlap audit；
4. PMID/study/assay/date provenance feasibility audit；
5. task-paired bootstrap、Wilcoxon、Hodges–Lehmann、win/tie/loss 和 BH-FDR；
6. 主表与 Supplementary CSV；
7. standard/strict 可视化；
8. 环境、参数量、时间和文件 hash；
9. dataset card、reproducibility statement 与 ethics/intended-use 草稿。

所有机器生成结果位于 `results/final_phase/`。统一代码入口为：

```powershell
& E:/ancd/envs/my_pytorch/python.exe final_phase/run_all.py
```

注意：`run_all.py` 使用 `--component-bootstrap 0`，不会运行 component bootstrap。

## 2. Standard 与 peptide-disjoint 主结果

| Species | Protocol | Tasks | Accuracy | Mean AUROC | Mean AUPRC | F1 | MCC | Worst group AUROC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Human | Standard OOF | 157 | 0.76051 | 0.82795 | 0.81436 | 0.76319 | 0.52143 | 0.65616 |
| Human | Peptide-disjoint OOF | 157 | 0.69801 | 0.76520 | 0.74521 | 0.70242 | 0.39649 | 0.63817 |
| Mouse | Standard OOF | 24 | 0.77953 | 0.83922 | 0.83158 | 0.77806 | 0.55987 | 0.71012 |
| Mouse | Peptide-disjoint OOF | 24 | 0.68771 | 0.75293 | 0.73222 | 0.68680 | 0.37608 | 0.64808 |

Human AUROC/AUPRC 的 strict − standard gap 为 `-0.06275/-0.06914`；mouse 为 `-0.08629/-0.09936`。两个物种在 unseen-peptide 条件下仍明显高于随机排序，但 standard split 的实体复用会明显抬高性能估计。

源表：`results/final_phase/06_tables/table_4_standard_strict_summary.csv`。

## 3. PairAcc

PairAcc 对每个 `pair_id` 比较正例与匹配负例分数；正例更高记 1，平分记 0.5。

| Species | Standard PairAcc | Strict PairAcc | Strict − standard | Strict worst task | Strict median task |
|---|---:|---:|---:|---:|---:|
| Human | 0.82323 | 0.77026 | -0.05297 | 0.61059 | 0.76471 |
| Mouse | 0.82112 | 0.75962 | -0.06150 | 0.58590 | 0.78645 |

PairAcc 与 AUROC 的方向一致：严格 peptide 隔离降低排序性能，但两个物种仍保留约 0.76–0.77 的成对排序正确率。

源表：`results/final_phase/01_pairacc/pairacc_summary.csv`。

## 4. Matched-standard fold 审计

| Species | Task-folds | Exact held-size matches | Exact match rate | Mean absolute pair difference | Maximum difference | Mean relative difference |
|---|---:|---:|---:|---:|---:|---:|
| Human | 471 | 333 | 70.70% | 0.293 | 1 pair | 0.377% |
| Mouse | 72 | 48 | 66.67% | 0.333 | 1 pair | 0.432% |

standard 与 strict 的每 task、每 fold held-out 规模最大只差 1 pair，整体匹配程度很高。因此已有 standard OOF 是合理的 size-matched 描述性控制；仍需承认两种协议的具体样本组成不同，不能把 gap 解释为纯粹、无任何 fold-composition 影响的因果效应。

Standard OOF 的 held-out unique peptides 中，human 每折约 `69.19%–69.91%`、mouse 约 `71.43%–72.12%` 同时出现在 fitting；strict 三折的 pair overlap 与 peptide overlap 均为 0。

源表：

- `results/final_phase/02_matched_fold_audit/matching_summary.csv`；
- `results/final_phase/02_matched_fold_audit/protocol_overlap_audit.csv`；
- `results/final_phase/02_matched_fold_audit/standard_vs_strict_task_fold_comparison.csv`。

## 5. Parent-protein overlap

Peptide-disjoint 不等于 protein-disjoint。

| Species | Protocol | Held-out unique parent proteins seen in fitting | Held-out rows with seen parent protein |
|---|---|---:|---:|
| Human | Standard | 90.64% | 96.92% |
| Human | Peptide-disjoint | 65.26% | 75.42% |
| Mouse | Standard | 80.69% | 87.36% |
| Mouse | Peptide-disjoint | 14.37% | 13.18% |

Human strict 仍有显著 parent-protein overlap，因此 0.76520 只能解释为 seen-task/unseen-peptide robustness，不能称为 unseen-protein 泛化。Mouse strict 的 protein overlap 较低，但仍非严格 protein-disjoint。

源表：`results/final_phase/03_parent_protein_overlap/protein_overlap_summary.csv`。

## 6. Task-level statistical analysis

默认统计使用固定 seed `20260711`、10,000 次 task-paired bootstrap。BH-FDR 在双物种四项指标的 8 个检验上统一校正。

### 6.1 Human

| Metric | Standard | Strict | Mean delta | Median delta | Hodges–Lehmann delta | Win/tie/loss | Strict 95% CI | Delta 95% CI | FDR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AUROC | 0.82795 | 0.76520 | -0.06275 | -0.04836 | -0.05523 | 12/0/145 | [0.75511, 0.77519] | [-0.07283, -0.05336] | 1.29e-22 |
| AUPRC | 0.81436 | 0.74521 | -0.06914 | -0.05091 | -0.06213 | 10/0/147 | [0.73517, 0.75530] | [-0.07990, -0.05881] | 1.29e-22 |
| Accuracy | 0.76051 | 0.69801 | -0.06250 | -0.04545 | -0.05631 | 17/2/138 | [0.68926, 0.70693] | [-0.07265, -0.05286] | 2.12e-22 |
| MCC | 0.52143 | 0.39649 | -0.12494 | -0.09072 | -0.11247 | 18/0/139 | [0.37892, 0.41440] | [-0.14527, -0.10567] | 2.12e-22 |

### 6.2 Mouse

| Metric | Standard | Strict | Mean delta | Median delta | Hodges–Lehmann delta | Win/tie/loss | Strict 95% CI | Delta 95% CI | FDR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AUROC | 0.83922 | 0.75293 | -0.08629 | -0.08322 | -0.08592 | 2/0/22 | [0.72300, 0.77991] | [-0.11062, -0.06322] | 6.81e-07 |
| AUPRC | 0.83158 | 0.73222 | -0.09936 | -0.10399 | -0.10019 | 2/0/22 | [0.70401, 0.75834] | [-0.12928, -0.07058] | 1.67e-06 |
| Accuracy | 0.77953 | 0.68771 | -0.09182 | -0.08040 | -0.09021 | 1/0/23 | [0.66247, 0.71092] | [-0.12069, -0.06523] | 4.77e-07 |
| MCC | 0.55987 | 0.37608 | -0.18379 | -0.16322 | -0.18019 | 1/0/23 | [0.32572, 0.42242] | [-0.24145, -0.13056] | 4.77e-07 |

结果支持“standard→strict 存在广泛且方向一致的下降”，而不是由少量异常 task 驱动。统计显著性不等于临床或外部有效性；task bootstrap 也不包含重新训练、不同 split seed 或数据来源变化带来的方差。

源表：`results/final_phase/05_statistics/paired_statistical_tests.csv`。

## 7. Peptide-component cluster bootstrap 状态

当前**没有可报告的 peptide-component cluster bootstrap 数值**：

```json
{
  "human": {"status": "not_run"},
  "mouse": {"status": "not_run"},
  "component_bootstrap_repeats": 0
}
```

截至 2026-07-17 19:33:55，`results/final_phase/05_statistics/component_cluster_bootstrap.json` 仍为 `not_run`，仓库内也没有第二份结果文件。不得将 task-bootstrap CI 误写为 component-cluster CI。

正确补跑命令：

```powershell
& E:/ancd/envs/my_pytorch/python.exe final_phase/05_statistical_analysis.py --component-bootstrap 1000
```

补跑后不要再次执行默认 `run_all.py`，否则当前实现会用 `--component-bootstrap 0` 覆盖该 JSON。正式稿若使用 component-cluster CI，应在报告中替换本节状态并注明 repeats 与 seed。

## 8. Provenance feasibility

检查了 human/mouse train 数据以及两份 processed pair 数据。所有文件均没有 PMID、study ID、assay ID、publication/submission date 等候选字段，因此无法仅依赖当前产物重建 study/PMID-disjoint evaluation。

论文应写成：当前处理产物未保留足够 study-level provenance，无法构建可靠 study-disjoint split；这是一项明确的数据局限，而不是用 peptide-disjoint 替代 study-disjoint。

源文件：`results/final_phase/04_provenance_feasibility/provenance_feasibility.json`。

## 9. 可复现性与计算信息

- Python：3.13.11；
- PyTorch：2.10.0+cu126；
- CUDA runtime：12.6；
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU，8,188 MiB；
- NumPy/Pandas/scikit-learn/SciPy：2.4.1/3.0.0/1.8.0/1.17.0；
- Human E29：global auxiliary 160,921 parameters；35 个 HLA plain models 合计 4,672,733；所有分别训练模型合计 4,833,654；
- Mouse E33：68,475 parameters；
- Human E31 strict 总耗时：5,425.718 秒（1h 30m 26s）；
- peak GPU memory：原运行未记录，不能事后精确恢复；
- git commit：当前环境未能解析，manifest 中为 `null`；
- 数据、预测与 split 文件 SHA256 已写入 `results/final_phase/08_reproducibility/file_hashes.csv`。

源文件：`results/final_phase/08_reproducibility/reproducibility_manifest.json`。

## 10. 图、表和文档草稿

### 10.1 图

`results/final_phase/07_figures/` 已生成：

- `01_standard_vs_strict_scatter.png`；
- `02_task_delta_distribution.png`；
- `03_mhc_group_gap.png`；
- `04_parent_protein_overlap.png`；
- `final_phase_figures.pdf`；
- `figure_source_task_comparison.csv`。

### 10.2 表

`results/final_phase/06_tables/` 已生成 benchmark statistics、standard/strict summary、PairAcc、fold matching、overlap audit、protein overlap、统计检验、逐 task metrics 和训练时间等 9 份 CSV。

### 10.3 文档草稿

`results/final_phase/09_dataset_cards/` 已生成：

- `DATASET_CARD.md`；
- `REPRODUCIBILITY_STATEMENT.md`；
- `ETHICS_AND_INTENDED_USE.md`。

## 11. 综合结论

1. Human 和 mouse 在 peptide-disjoint 条件下仍保留 AUROC `0.76520/0.75293` 与 PairAcc `0.77026/0.75962`，说明 seen-task/unseen-peptide 信号仍然可学。
2. Standard→strict 的 AUROC、AUPRC、PairAcc 和 MCC 均显著下降，且 task-level win/loss 广泛一致，说明 standard split 的 peptide 复用会系统性高估实体泛化。
3. task-fold held-out size 最大仅差 1 pair，支持将已有 standard OOF 作为高度匹配的描述性控制。
4. Human strict 仍有 65.26% unique parent-protein overlap，不能外推为 unseen-protein 能力；mouse overlap 较低但不为 0。
5. 当前数据无法恢复 study/PMID provenance，因此不能声称 study-disjoint 或独立来源验证。
6. task-bootstrap 和 Wilcoxon/FDR 已完成；peptide-component cluster bootstrap 尚未完成，不能在正式稿中引用其 CI。
7. 在 strict baselines、外部 pMHC predictors 和核心消融补齐前，不能声称当前主模型在 unseen-peptide 协议下仍优于所有替代方法。

## 12. 仍需补齐

- peptide-component cluster bootstrap；
- human/mouse strict baselines 与外部 pMHC predictors；
- strongest ablation 的 strict multi-seed 复现；
- shuffled tissue/branch negative controls；
- 无需训练即可继续完成的图题、主表排版和论文正文整合；
- 若需要 peak GPU memory，只能进行明确标记为 profiling 的代表性运行，不能伪装成原实验记录。

