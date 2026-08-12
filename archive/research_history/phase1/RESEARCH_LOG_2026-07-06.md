# 2026.7.6 研究日志

## 1. 今日目标

今天的目标是继续 2026.7.5 日志中提出的下一步实验，重点完成并评估：

```text
E4: conditioned model with tissue embedding + HLA pseudo-sequence
E4b: conditioned model with tissue embedding + HLA ID embedding + HLA pseudo-sequence
E5: E2 shared peptide encoder + task-specific heads + FAMO
```

`HLA` 是 Human Leukocyte Antigen，人类白细胞抗原。这里的 HLA allele 例如 `HLA-A*02:01`，表示一个具体 HLA 等位基因。

`pseudo-sequence` 指从 HLA 蛋白序列中抽取的一小段关键氨基酸位点序列，通常选取和 peptide binding 相关的 residue positions。`residue position` 指蛋白序列中的氨基酸编号。

`FAMO` 是 Fast Adaptive Multitask Optimization，一种自适应多任务优化方法。它会动态调整不同 task 的 loss 权重，目标是让不同任务的训练进展更均衡。

## 2. 新增和修改的代码

今天新增了 4 个主要文件或结果表相关脚本。

### 2.1 HLA pseudo-sequence 构建脚本

新增脚本：

```text
scripts/build_hla_pseudo_sequences.py
```

用途：

```text
从 IPD-IMGT/HLA 官方蛋白 FASTA 中抽取当前数据集需要的 HLA class I pseudo-sequence。
```

输入 FASTA：

```text
data/processed/A_prot.fasta
data/processed/B_prot.fasta
data/processed/C_prot.fasta
```

输出：

```text
data/processed/hla_pseudo_sequences.csv
```

生成的 pseudo-sequence 覆盖当前数据集中全部 12 个 HLA allele：

```text
HLA-A*02:01
HLA-A*24:02
HLA-B*07:02
HLA-B*15:01
HLA-B*15:02
HLA-B*27:05
HLA-B*40:01
HLA-B*40:02
HLA-B*51:01
HLA-C*03:04
HLA-C*05:01
HLA-C*12:02
```

每条 pseudo-sequence 长度为 34。

构建方法：

```text
1. 从 IPD-IMGT/HLA protein FASTA 读取 HLA-A、HLA-B、HLA-C 蛋白序列。
2. 去除 N-terminal signal peptide 的前 24 个氨基酸。
3. 按 NetMHCpan-style 34 个 MHC-I binding-site positions 抽取氨基酸。
4. 写出 hla,pseudo_sequence,source 等字段。
```

这里 `signal peptide` 指蛋白质 N 端用于引导分泌或膜定位的一段短序列。IPD-IMGT/HLA FASTA 中包含这段序列，而常见 MHC-I pseudo-sequence position 通常按成熟蛋白编号，所以需要先去掉前 24 个氨基酸。

### 2.2 E4 HLA pseudo-sequence 脚本

新增脚本：

```text
scripts/run_tissuepmhc_hla_pseudoseq.py
```

用途：

```text
比较 E3 HLA ID embedding 与 E4 HLA pseudo-sequence encoder。
```

输出目录：

```text
results/tissuePMHC_hla_pseudoseq/
```

主要输出：

```text
per_task_metrics.csv
summary_metrics.csv
stability_metrics.csv
comparison_metrics.csv
metadata.json
```

### 2.3 E4b hybrid 脚本

新增脚本：

```text
scripts/run_tissuepmhc_hla_hybrid.py
```

用途：

```text
比较 E3、E4 和 E4b。
```

模型定义：

```text
E3:
peptide encoder + tissue embedding + HLA ID embedding

E4:
peptide encoder + tissue embedding + HLA pseudo-sequence encoder

E4b:
peptide encoder + tissue embedding + HLA ID embedding + HLA pseudo-sequence encoder
```

`embedding` 指可训练向量表示。HLA ID embedding 是把 `HLA-A*02:01` 这种类别 ID 变成可训练向量。HLA pseudo-sequence encoder 是把 HLA 的氨基酸 pseudo-sequence 编码成向量。

输出目录：

```text
results/tissuePMHC_hla_hybrid/
```

### 2.4 E5 FAMO 脚本

新增脚本：

```text
scripts/run_tissuepmhc_famo.py
```

用途：

```text
在 E2 shared peptide encoder + task-specific heads 上测试 FAMO-style adaptive task weighting。
```

为了公平比较，该脚本同时跑：

```text
E2_task_balanced
E5_E2_FAMO
```

`task-balanced` 指每个 training step 中，每个 task 抽取相同数量样本。这样 FAMO 与非 FAMO 的比较使用相同 batch 构造方式。

输出目录：

```text
results/tissuePMHC_famo/
```

额外输出：

```text
task_weight_history.csv
```

该文件记录每个 epoch 每个 task 的 FAMO 权重。

## 3. E4 实验结果：HLA pseudo-sequence 替代 HLA ID embedding

E4 的核心问题是：

```text
如果用 HLA pseudo-sequence encoder 替代普通 HLA ID embedding，模型是否变强？
```

实验结果：

| model | mean AUROC | mean AUPRC | mean accuracy | mean MCC | worst-10 mean AUROC |
|---|---:|---:|---:|---:|---:|
| E3 HLA embedding | 0.7833 | 0.7685 | 0.7152 | 0.4350 | 0.6992 |
| E4 HLA pseudo-sequence | 0.7728 | 0.7606 | 0.7042 | 0.4125 | 0.6705 |

`AUROC` 是 Area Under the Receiver Operating Characteristic Curve，用来衡量模型区分正负样本的整体能力。

`AUPRC` 是 Area Under the Precision-Recall Curve，更关注正样本检出能力。

`MCC` 是 Matthews Correlation Coefficient，是二分类任务中比较稳健的综合指标。

E4 相比 E3：

```text
mean AUROC 下降约 0.0105
mean AUPRC 下降约 0.0079
mean MCC 下降约 0.0225
worst-10 mean AUROC 下降约 0.0287
```

按 task 平均：

```text
15 tasks improved
29 tasks degraded
```

E4 提升最大的任务：

| target_tissue | mhc_restriction | AUROC delta |
|---|---:|---:|
| breast | HLA-A*02:01 | +0.0243 |
| ovary | HLA-A*02:01 | +0.0224 |
| lymph node | HLA-A*02:01 | +0.0173 |
| thymus | HLA-A*24:02 | +0.0168 |
| blood | HLA-A*24:02 | +0.0150 |

E4 下降最大的任务：

| target_tissue | mhc_restriction | AUROC delta |
|---|---:|---:|
| uterine cervix | HLA-B*51:01 | -0.0851 |
| blood | HLA-B*15:01 | -0.0673 |
| lymphoid | HLA-C*12:02 | -0.0442 |
| blood | HLA-B*51:01 | -0.0414 |
| lymphoid | HLA-B*40:01 | -0.0368 |

按 HLA 分组，E4 的规律比较明显：

```text
HLA-A*02:01 和 HLA-A*24:02 略有收益。
多数 HLA-B 和 HLA-C 任务下降。
```

结论：

```text
HLA pseudo-sequence 不能直接替代 HLA ID embedding。
```

原因推测：

```text
当前 train/test 是 closed-set HLA setting。
```

`closed-set` 指测试集里的 HLA allele 在训练集中都出现过。在这种情况下，HLA ID embedding 可以直接记住每个 allele 的任务特征，而 pseudo-sequence encoder 需要把 12 个 HLA 压缩到共享序列编码器中，反而形成了信息瓶颈。

## 4. E4b 实验结果：HLA ID embedding + HLA pseudo-sequence

E4b 的核心问题是：

```text
如果不让 pseudo-sequence 替代 HLA ID，而是作为额外生物信息加入，是否会提升 E3？
```

实验结果：

| model | mean AUROC | mean AUPRC | mean accuracy | mean MCC | worst-10 mean AUROC |
|---|---:|---:|---:|---:|---:|
| E3 HLA embedding | 0.7833 | 0.7685 | 0.7152 | 0.4350 | 0.6992 |
| E4b hybrid | 0.7824 | 0.7704 | 0.7133 | 0.4303 | 0.6952 |
| E4 pseudo-sequence only | 0.7728 | 0.7606 | 0.7042 | 0.4125 | 0.6705 |

E4b 相比 E3：

```text
mean AUROC 下降约 0.0009
mean AUPRC 提升约 0.0019
mean MCC 下降约 0.0047
worst-10 mean AUROC 下降约 0.0039
```

E4b 相比纯 E4：

```text
明显恢复了 E4 的性能损失。
```

按 task 平均：

```text
21 tasks improved
23 tasks degraded
```

E4b 提升最大的任务：

| target_tissue | mhc_restriction | AUROC delta vs E3 |
|---|---:|---:|
| blood | HLA-A*24:02 | +0.0482 |
| breast | HLA-A*02:01 | +0.0246 |
| bone | HLA-A*02:01 | +0.0233 |
| lymphoid | HLA-B*51:01 | +0.0190 |
| uterine cervix | HLA-B*07:02 | +0.0164 |

E4b 下降最大的任务：

| target_tissue | mhc_restriction | AUROC delta vs E3 |
|---|---:|---:|
| uterine cervix | HLA-B*51:01 | -0.0481 |
| lymphoid | HLA-B*15:02 | -0.0345 |
| lung | HLA-B*15:01 | -0.0282 |
| lymphoid | HLA-A*24:02 | -0.0223 |
| lymphoid | HLA-B*40:02 | -0.0194 |

按 HLA 分组：

```text
HLA-B*07:02   +0.0071
HLA-A*24:02   +0.0062
HLA-A*02:01   +0.0040
HLA-B*15:02   -0.0345
HLA-B*40:02   -0.0126
HLA-B*15:01   -0.0115
```

结论：

```text
E4b 说明 pseudo-sequence 作为辅助信息有一定价值，尤其对 AUPRC 有轻微帮助。
但 E4b 没有超过 E3，也没有超过 E2。
```

因此 E4b 可以作为后续讨论中的探索性结果，但不应作为当前主模型。

## 5. E5 实验结果：FAMO adaptive task weighting

E5 的核心问题是：

```text
在当前最强的 E2 shared peptide encoder + task-specific heads 上加入 FAMO，是否能缓解 negative transfer？
```

`negative transfer` 指多任务学习中某些任务受到其他任务干扰，性能反而下降。

为了公平比较，E5 脚本使用了 task-balanced batch 构造：

```text
每个 training step 中，每个 task 抽取相同数量样本。
```

比较对象：

```text
E2_task_balanced
E5_E2_FAMO
```

实验结果：

| model | mean AUROC | mean AUPRC | mean accuracy | mean MCC | worst-10 mean AUROC |
|---|---:|---:|---:|---:|---:|
| 原始 E2 shared heads | 0.7927 | 0.7777 | 0.7180 | 0.4404 | 0.7178 |
| E2 task-balanced | 0.7819 | 0.7639 | 0.7131 | 0.4290 | 0.7036 |
| E5 FAMO | 0.7758 | 0.7571 | 0.7077 | 0.4193 | 0.6979 |

E5 FAMO 相比 E2 task-balanced：

```text
mean AUROC 下降约 0.0061
mean AUPRC 下降约 0.0069
mean MCC 下降约 0.0097
worst-10 mean AUROC 下降约 0.0057
```

按 task 平均：

```text
13 tasks improved
31 tasks degraded
```

FAMO 提升最大的任务：

| target_tissue | mhc_restriction | AUROC delta vs E2_task_balanced |
|---|---:|---:|
| lymphoid | HLA-B*15:01 | +0.0288 |
| lymphoid | HLA-B*15:02 | +0.0248 |
| uterine cervix | HLA-B*07:02 | +0.0186 |
| brain | HLA-B*40:02 | +0.0150 |
| lung | HLA-B*15:01 | +0.0125 |

FAMO 下降最大的任务：

| target_tissue | mhc_restriction | AUROC delta vs E2_task_balanced |
|---|---:|---:|
| uterine cervix | HLA-B*51:01 | -0.0381 |
| thymus | HLA-A*02:01 | -0.0330 |
| blood | HLA-C*03:04 | -0.0324 |
| lymphoid | HLA-B*51:01 | -0.0227 |
| lymphoid | HLA-B*40:01 | -0.0220 |

FAMO 最终权重范围较小：

```text
approximately 0.0215 to 0.0271
uniform weight = 1 / 44 = 0.0227
```

这说明 FAMO 没有学出很强的 task 区分。权重最高的一些任务包括：

| task | final mean weight |
|---|---:|
| umbilical cord blood / HLA-A*02:01 | 0.0271 |
| thymus / HLA-A*24:02 | 0.0251 |
| lymphoid / HLA-A*24:02 | 0.0247 |
| lymphoid / HLA-B*51:01 | 0.0244 |
| lung / HLA-B*15:01 | 0.0241 |

权重最低的一些任务包括：

| task | final mean weight |
|---|---:|
| thymus / HLA-A*02:01 | 0.0187 |
| breast / HLA-A*02:01 | 0.0202 |
| uterine cervix / HLA-B*51:01 | 0.0208 |
| lymphoid / HLA-C*03:04 | 0.0212 |
| lung / HLA-A*02:01 | 0.0215 |

结论：

```text
当前 E5 FAMO 版本不成功。
```

更重要的观察是：

```text
E2_task_balanced 本身也明显弱于原始 E2。
```

这说明强制每个 step 对所有 task 等量采样改变了训练分布，可能破坏了原始 E2 的优势。原始 E2 使用自然混合数据分布训练，可能更适合当前数据构造。

## 6. 当前模型排序

结合 2026.7.5 和 2026.7.6 的结果，当前模型排序为：

```text
E2 shared peptide encoder + task-specific heads
>
E3 conditioned tissue + HLA ID embedding
≈ E4b hybrid HLA ID + HLA pseudo-sequence
>
E2 task-balanced
>
E5 FAMO
>
E4 HLA pseudo-sequence only
```

主要性能对照：

| model | mean AUROC | mean AUPRC | mean MCC | worst-10 mean AUROC |
|---|---:|---:|---:|---:|
| E2 shared heads | 0.7927 | 0.7777 | 0.4404 | 0.7178 |
| E3 HLA embedding | 0.7833 | 0.7685 | 0.4350 | 0.6992 |
| E4b hybrid | 0.7824 | 0.7704 | 0.4303 | 0.6952 |
| E2 task-balanced | 0.7819 | 0.7639 | 0.4290 | 0.7036 |
| E5 FAMO | 0.7758 | 0.7571 | 0.4193 | 0.6979 |
| E4 pseudo-sequence only | 0.7728 | 0.7606 | 0.4125 | 0.6705 |

## 7. 今日主要结论

今天最重要的结论是：

```text
E2 仍然是当前最强、最稳定的主 baseline。
```

更具体地说：

1. HLA pseudo-sequence 不能直接替代 HLA ID embedding。
2. HLA ID embedding + HLA pseudo-sequence 的 hybrid model 能明显修复纯 pseudo-sequence 的性能损失，但没有超过 E3。
3. E4b 的 AUPRC 略高于 E3，说明 pseudo-sequence 可能对正样本检出有轻微信息增益。
4. FAMO 当前版本没有提升 E2，反而降低 mean AUROC、mean AUPRC、MCC 和 worst-10 AUROC。
5. task-balanced sampling 本身削弱了 E2，说明当前数据的自然训练分布可能比强制任务均衡更适合。
6. negative transfer 仍然存在，但更可能需要通过 task grouping 或 selective sharing 解决，而不是简单 loss weighting。

## 8. 下一步计划：进入 E6 task grouping

根据今天结果，下一步不建议继续调 E4 或 E5。

推荐进入：

```text
E6: HLA-based and tissue-based task grouping analysis
```

具体实验：

```text
1. Train shared peptide encoder + task-specific heads within each HLA group.
2. Train shared peptide encoder + task-specific heads within each tissue group.
3. Compare grouped training vs original all-task E2.
4. Analyze which tasks benefit from grouping and which tasks need global sharing.
```

为什么 E6 更重要：

```text
E2 已经证明 shared peptide encoder 有效。
但 E2 也出现了 14 个 degraded tasks。
这些 degraded tasks 可能不是 loss weight 问题，而是 task sharing structure 问题。
```

`task sharing structure` 指哪些任务应该共享模型参数、哪些任务不应该共享。当前任务天然分解为：

```text
task = tissue + HLA
```

因此最自然的下一步是分析：

```text
same HLA tasks 是否应该共享？
same tissue tasks 是否应该共享？
哪些 HLA 或 tissue group 容易产生 negative transfer？
```

## 9. 重要文件

HLA pseudo-sequence 构建：

```text
scripts/build_hla_pseudo_sequences.py
data/processed/hla_pseudo_sequences.csv
data/processed/A_prot.fasta
data/processed/B_prot.fasta
data/processed/C_prot.fasta
```

E4 代码和结果：

```text
scripts/run_tissuepmhc_hla_pseudoseq.py
results/tissuePMHC_hla_pseudoseq/per_task_metrics.csv
results/tissuePMHC_hla_pseudoseq/summary_metrics.csv
results/tissuePMHC_hla_pseudoseq/stability_metrics.csv
results/tissuePMHC_hla_pseudoseq/comparison_metrics.csv
results/tissuePMHC_hla_pseudoseq/metadata.json
```

E4b 代码和结果：

```text
scripts/run_tissuepmhc_hla_hybrid.py
results/tissuePMHC_hla_hybrid/per_task_metrics.csv
results/tissuePMHC_hla_hybrid/summary_metrics.csv
results/tissuePMHC_hla_hybrid/stability_metrics.csv
results/tissuePMHC_hla_hybrid/comparison_metrics.csv
results/tissuePMHC_hla_hybrid/metadata.json
```

E5 代码和结果：

```text
scripts/run_tissuepmhc_famo.py
results/tissuePMHC_famo/per_task_metrics.csv
results/tissuePMHC_famo/summary_metrics.csv
results/tissuePMHC_famo/stability_metrics.csv
results/tissuePMHC_famo/comparison_metrics.csv
results/tissuePMHC_famo/task_weight_history.csv
results/tissuePMHC_famo/metadata.json
```

当前主 baseline：

```text
scripts/run_tissuepmhc_neural_baselines_v2.py
results/tissuePMHC_neural_baselines_v2/stability_metrics.csv
```

## 10. E6 task grouping 实验结果补充

完成 E6 后，将 grouped training 与原始 E2 all-task shared heads 进行比较。

整体结果如下：

| model | mean AUROC | mean AUPRC | mean MCC | worst-10 mean AUROC |
|---|---:|---:|---:|---:|
| E2 all-task shared heads | 0.7927 | 0.7777 | 0.4404 | 0.7178 |
| E6 HLA-grouped | 0.7862 | 0.7725 | 0.4316 | 0.7037 |
| E6 tissue-grouped | 0.7387 | 0.7227 | 0.3548 | 0.6699 |

E6 HLA grouping 相比 E2：

```text
mean AUROC 下降约 0.0065
mean AUPRC 下降约 0.0052
mean MCC 下降约 0.0088
```

E6 tissue grouping 相比 E2：

```text
mean AUROC 下降约 0.0541
mean AUPRC 下降约 0.0550
mean MCC 下降约 0.0856
```

因此，E6 没有超过 E2。HLA grouping 只是接近 E2，而 tissue grouping 明显失败。

按 task 统计，HLA grouping 在 44 个 task 中：

```text
18 tasks improved
26 tasks degraded
```

提升最大的 HLA-grouped task：

| target_tissue | mhc_restriction | AUROC delta vs E2 |
|---|---:|---:|
| lung | HLA-A*24:02 | +0.0399 |
| blood | HLA-B*51:01 | +0.0360 |
| lymphoid | HLA-B*51:01 | +0.0331 |
| small intestine | HLA-B*15:01 | +0.0293 |
| uterine cervix | HLA-B*07:02 | +0.0286 |

下降最大的 HLA-grouped task：

| target_tissue | mhc_restriction | AUROC delta vs E2 |
|---|---:|---:|
| uterine cervix | HLA-A*24:02 | -0.0564 |
| lymphoid | HLA-B*40:02 | -0.0540 |
| lymphoid | HLA-C*03:04 | -0.0540 |
| lymphoid | HLA-B*15:02 | -0.0490 |
| lymphoid | HLA-C*05:01 | -0.0419 |

按 HLA group 平均，HLA grouping 中表现相对更好的 group 包括：

```text
HLA-B*51:01   +0.0101 AUROC
HLA-C*12:02   +0.0099 AUROC
HLA-B*07:02   +0.0096 AUROC
HLA-B*27:05   +0.0048 AUROC
HLA-B*15:01   +0.0017 AUROC
```

下降较明显的 group 包括：

```text
HLA-B*15:02   -0.0490 AUROC
HLA-B*40:02   -0.0360 AUROC
HLA-C*03:04   -0.0352 AUROC
HLA-C*05:01   -0.0243 AUROC
```

结论：

```text
HLA grouping 有局部价值，但不能作为全局替代 E2 的主模型。
tissue grouping 不建议继续。
```

原因推测：

```text
同一 HLA 内部共享有时可以减少跨 HLA 的 negative transfer。
但很多 task 仍然需要跨 HLA 的全局 peptide pattern，因此 all-task E2 仍然更强。
tissue grouping 会把 HLA binding pattern 差异很大的 task 强行共享，所以性能明显下降。
```

下一步进入：

```text
E7: validation-based selective HLA/global sharing
```

E7 的核心思想是：

```text
不要一刀切地选择 all-task global sharing 或 HLA grouping。
先在训练集内部切出 validation set。
对每个 task 比较 global E2 branch 与 HLA-grouped branch 的 validation AUROC。
如果 HLA-grouped branch 在 validation 上更好，则该 task 使用 HLA branch；
否则继续使用 global E2 branch。
确定每个 task 的 branch 后，用完整训练集重新训练 global branch 和 HLA branch。
最后在 test set 上评估这个 task-level selective model。
```

这样可以避免直接用 test set 选择模型造成的数据泄漏。
同时，最终 test 评估使用完整训练集重训后的分支，和原始 E2 使用相同训练数据量，比较更公平。

E6 代码和结果：

```text
scripts/run_tissuepmhc_task_grouping.py
results/tissuePMHC_task_grouping/per_task_metrics.csv
results/tissuePMHC_task_grouping/summary_metrics.csv
results/tissuePMHC_task_grouping/stability_metrics.csv
results/tissuePMHC_task_grouping/group_summary_metrics.csv
results/tissuePMHC_task_grouping/comparison_metrics.csv
results/tissuePMHC_task_grouping/metadata.json
```

E7 代码和计划输出：

```text
scripts/run_tissuepmhc_selective_grouping.py
results/tissuePMHC_selective_grouping/per_task_metrics.csv
results/tissuePMHC_selective_grouping/summary_metrics.csv
results/tissuePMHC_selective_grouping/stability_metrics.csv
results/tissuePMHC_selective_grouping/candidate_metrics.csv
results/tissuePMHC_selective_grouping/selection_metrics.csv
results/tissuePMHC_selective_grouping/comparison_metrics.csv
results/tissuePMHC_selective_grouping/metadata.json
```

E7 脚本会在终端直接打印训练耗时：

```text
每个 global branch 的训练用时
每个 HLA branch 的训练用时
每个 seed 的总用时
整个 run 的总用时
```

## 11. E7 validation-based selective HLA/global sharing 结果

E7 的核心问题是：

```text
如果不强制所有 task 使用同一种 sharing structure，
而是在 validation set 上为每个 task 选择 global branch 或 HLA-grouped branch，
是否能超过原始 E2？
```

E7 使用两阶段流程：

```text
1. 从 train 中切出 validation set。
2. 用 train-core 训练 validation global branch 和 validation HLA branch。
3. 对每个 task 用 validation AUROC 选择 branch。
4. 用完整 train set 重新训练 final global branch 和 final HLA branch。
5. 在 test set 上评估每个 task 选中的 final branch。
```

这样既避免了 test leakage，也保证最终 test 评估与 E2 使用相同训练数据量。

整体结果如下：

| model | mean AUROC | mean AUPRC | mean accuracy | mean MCC | worst-10 mean AUROC |
|---|---:|---:|---:|---:|---:|
| E2 all-task shared heads | 0.7927 | 0.7777 | 0.7180 | 0.4404 | 0.7178 |
| E6 HLA-grouped | 0.7862 | 0.7725 | 0.7148 | 0.4316 | 0.7037 |
| E7 selective HLA/global | 0.7904 | 0.7754 | 0.7184 | 0.4405 | 0.7143 |
| E6 tissue-grouped | 0.7387 | 0.7227 | 0.6765 | 0.3548 | 0.6699 |

E7 相比 E2：

```text
mean AUROC 下降约 0.0023
mean AUPRC 下降约 0.0023
mean accuracy 提升约 0.0005
mean MCC 提升约 0.0001
worst-10 mean AUROC 下降约 0.0035
```

因此，E7 明显好于 E6 HLA-grouped，但仍没有超过原始 E2。

E7 的 branch 选择情况：

| seed | global tasks | HLA-grouped tasks |
|---:|---:|---:|
| 20260704 | 25 | 19 |
| 20260705 | 26 | 18 |
| 20260706 | 32 | 12 |

HLA branch 被选择最多的 HLA group：

```text
HLA-A*02:01  13 次
HLA-A*24:02   7 次
HLA-B*15:01   7 次
HLA-B*07:02   4 次
HLA-C*03:04   4 次
HLA-B*51:01   4 次
```

E7 提升最大的 task：

| target_tissue | mhc_restriction | AUROC delta vs E2 |
|---|---:|---:|
| small intestine | HLA-B*15:01 | +0.0446 |
| lung | HLA-A*24:02 | +0.0444 |
| blood | HLA-A*24:02 | +0.0167 |
| blood | HLA-B*40:01 | +0.0141 |
| spleen | HLA-B*15:01 | +0.0133 |

E7 下降最大的 task：

| target_tissue | mhc_restriction | AUROC delta vs E2 |
|---|---:|---:|
| blood | HLA-B*07:02 | -0.0511 |
| lymphoid | HLA-C*03:04 | -0.0332 |
| lymphoid | HLA-B*15:02 | -0.0329 |
| lymph node | HLA-C*03:04 | -0.0250 |
| blood | HLA-C*05:01 | -0.0217 |

按最终被选中的 branch 分组，平均 delta 为：

| selected branch | mean AUROC delta | mean AUPRC delta | mean MCC delta |
|---|---:|---:|---:|
| global | -0.0048 | -0.0068 | -0.0073 |
| HLA-grouped | +0.0019 | +0.0055 | +0.0127 |

这个结果说明：

```text
HLA branch 确实有局部收益。
E7 的 validation-based hard selection 能恢复一部分 E6 损失。
但 hard selection 仍然不够稳定，整体还没有超过 E2。
```

`hard selection` 指每个 task 只能二选一：要么使用 global branch，要么使用 HLA branch。
这种方式的问题是 validation set 上的小波动会直接改变最终 branch 选择。

今天截至 E7 的结论：

```text
E2 仍然是当前最强主 baseline。
E6 说明 HLA grouping 有局部价值，tissue grouping 明显不适合。
E7 说明 task-level selective sharing 有价值，但 hard selection 不够稳定。
```

## 12. 下一步计划：E8 soft ensemble

下一步推荐进入：

```text
E8: validation-weighted soft ensemble of global branch and HLA branch
```

`soft ensemble` 指不再对每个 task 硬选择一个 branch，而是把两个 branch 的 prediction score 按权重融合。
这里 `prediction score` 是模型输出的正类概率，表示 peptide 属于该 tissue-HLA task 的可能性。

E8 的具体实验设计：

```text
1. 继续使用 E7 的 train-core / validation / full-train 两阶段设计。
2. 在 validation set 上分别评估 global branch 和 HLA branch。
3. 对每个 task 根据 validation AUROC 或 AUPRC 计算融合权重。
4. 用完整 train set 训练 final global branch 和 final HLA branch。
5. 在 test set 上输出：
   final_score = w * hla_score + (1 - w) * global_score
6. 比较 E8 soft ensemble 与 E2、E6、E7。
```

可以优先测试三种权重策略：

```text
E8a: fixed average
global_score 和 hla_score 各占 0.5。

E8b: validation-delta clipped weight
如果 HLA validation AUROC 比 global 高，则提高 HLA 权重；
否则降低 HLA 权重，但不让权重变成 0 或 1。

E8c: validation-rank softmax weight
用 validation metric 的 softmax 生成 global/HLA 权重。
softmax 是一种把多个分数转换成非负权重且总和为 1 的函数。
```

预期：

```text
E8 可能比 E7 更稳。
因为即使 validation 误判，soft ensemble 也不会完全丢弃另一个 branch。
```

E7 代码和结果：

```text
scripts/run_tissuepmhc_selective_grouping.py
results/tissuePMHC_selective_grouping/per_task_metrics.csv
results/tissuePMHC_selective_grouping/summary_metrics.csv
results/tissuePMHC_selective_grouping/stability_metrics.csv
results/tissuePMHC_selective_grouping/candidate_metrics.csv
results/tissuePMHC_selective_grouping/selection_metrics.csv
results/tissuePMHC_selective_grouping/comparison_metrics.csv
results/tissuePMHC_selective_grouping/metadata.json
```

## 13. 后续编号校正说明

后续复盘时确认：本日志中 E5、E6、E7 的主线判断是正确的，它们都应归入 E2 shared peptide encoder + task-specific heads 性能主线，而不是 E4 HLA pseudo-sequence 线。

当前正式编号应理解为：

```text
E5: FAMO on E2
E6: HLA/tissue task grouping on E2
E7: selective HLA/global hard selection
E8: global/HLA soft ensemble
E9: E2 + CAGrad
E10: MMoE / PLE selective-sharing model on the E2/E8 line
E11: E2 + DB-MTL
```

其中 `MMoE` 指 Multi-gate Mixture-of-Experts，多门控专家混合模型；`PLE` 指 Progressive Layered Extraction，渐进式分层提取模型。二者都属于 selective-sharing model，也就是让不同 task 自动选择不同共享程度的模型结构。

`DB-MTL` 是一种 dynamic balancing multi-task learning 方法，用来平衡不同 task 的 loss scale 和 gradient magnitude，也就是损失尺度和梯度大小。

因此，本日志第 12 节的 E8 soft ensemble 后续应接正式 E9 CAGrad-on-E2，而不是把 CAGrad 接到 E4 线。
