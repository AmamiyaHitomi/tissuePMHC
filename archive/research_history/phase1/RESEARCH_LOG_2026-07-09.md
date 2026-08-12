# 2026.7.9 研究日志

## 1. 今日目标

今天的目标是把 7.8 之后规划中的 E9-E13 主线补完，并判断当前 `standard split` 是否还需要继续堆叠复杂模型。

`standard split` 指当前固定的 train/test 划分。这里的 test 中 tissue、HLA 和 task 都在 train 中出现过，因此它主要评估 closed-set 场景下的任务内泛化。

今天新增或完成分析的实验是：

```text
E9:  E2 + CAGrad
E10: MMoE selective-sharing
E10b: tuned MMoE configs
E11: E2 + DB-MTL
E12: paired ranking loss
E13: auxiliary tissue/HLA prediction
```

## 2. 关键术语

`CAGrad` 是 Conflict-Averse Gradient Descent，中文可理解为“避免冲突的梯度下降”。它试图缓解不同 task 的梯度方向互相冲突的问题。

`MMoE` 是 Multi-gate Mixture-of-Experts，中文可理解为“多门控专家混合模型”。它让不同 task 用不同 gate 选择多个 expert，从而学习 selective sharing。

`DB-MTL` 是一种 dynamic balancing multi-task learning 方法，目标是动态平衡不同 task 的 loss 或梯度规模。

`paired ranking loss` 指利用正负样本配对关系，让同一个 pair 中正样本分数高于负样本分数。

`auxiliary task` 指辅助任务。E13 中主任务仍然是每个 tissue-HLA task 的正负分类，同时额外预测 tissue label 和 HLA label，希望 shared encoder 学到任务结构。

`AUROC` 是 ROC 曲线下面积，衡量模型区分正负样本的整体排序能力。`AUPRC` 是精确率-召回率曲线下面积，更关注正样本检索质量。`MCC` 是马修斯相关系数，是一个综合考虑四类分类结果的指标。

## 3. 今日新增代码和结果目录

| 实验 | 脚本 | 结果目录 | 实验目的 |
|---|---|---|---|
| E9 | `scripts/run_tissuepmhc_cagrad.py` | `results/tissuePMHC_cagrad/` | 检查梯度冲突处理是否改善 E2 |
| E10 | `scripts/run_tissuepmhc_mmoe.py` | `results/tissuePMHC_mmoe/` | 检查自动 selective sharing 是否接近 E8 |
| E10b | `scripts/run_tissuepmhc_mmoe_tuning.py` | `results/tissuePMHC_mmoe_tuning/` | 调整 expert 数量、宽度和 gate regularization |
| E11 | `scripts/run_tissuepmhc_dbmtl.py` | `results/tissuePMHC_dbmtl/` | 检查动态 task loss balancing |
| E12 | `scripts/run_tissuepmhc_pair_ranking.py` | `results/tissuePMHC_pair_ranking/` | 检查 pair 内排序监督是否有效 |
| E13 | `scripts/run_tissuepmhc_auxiliary_tasks.py` | `results/tissuePMHC_auxiliary_tasks/` | 检查 tissue/HLA auxiliary supervision 是否改善表示 |

所有正式比较都使用 3 个 seed：

```text
20260704
20260705
20260706
```

## 4. 总体结果

今天最重要的总表如下：

| model | mean AUROC | mean AUPRC | mean accuracy | mean MCC | worst-10 mean AUROC | 结论 |
|---|---:|---:|---:|---:|---:|---|
| E8a fixed soft ensemble | 0.8050 | 0.7925 | 0.7313 | 0.4657 | 0.7295 | 当前最强主模型 |
| E13 auxiliary tissue/HLA | 0.8023 | 0.7856 | 0.7292 | 0.4640 | 0.7306 | 明显优于 E2/E11/E12，略弱于 E8 |
| E10 MMoE | 0.7948 | 0.7804 | 0.7210 | 0.4474 | 0.7195 | 略强于 E2/E9/E11/E12，但未接近 E8 |
| E2 sample BCE | 0.7945 | 0.7793 | 0.7197 | 0.4441 | 0.7211 | sample-level E2 baseline |
| E10b 4 experts, width 256 | 0.7928 | 0.7788 | 0.7170 | 0.4393 | 0.7104 | tuned MMoE 未超过 E10 |
| E10b 6 experts, width 128 | 0.7913 | 0.7768 | 0.7178 | 0.4423 | 0.7122 | tuned MMoE 未超过 E10 |
| E11 DB-MTL | 0.7817 | 0.7643 | 0.7144 | 0.4313 | 0.6956 | 未改善 E2 |
| E9 CAGrad | 0.7810 | 0.7649 | 0.7106 | 0.4239 | 0.6979 | 未改善 E2 |
| E12 pair ranking | 0.7807 | 0.7618 | 0.7102 | 0.4227 | 0.7007 | ranking loss 反而变弱 |

## 5. E9 CAGrad 结论

E9 的核心问题是：如果 E2 中存在 task 之间的梯度冲突，CAGrad 是否能改善主线性能。

结果是：

```text
E9 mean AUROC = 0.7810
E9 mean AUPRC = 0.7649
E9 worst-10 mean AUROC = 0.6979
```

它明显低于 E8，也低于 E2 sample BCE。因此当前任务中，显式梯度冲突处理没有带来收益。

这说明性能瓶颈可能不是简单的 task gradient conflict，而更可能是 task-specific structure 如何表达的问题。E8 的成功也支持这一点：global branch 和 HLA-specific branch 的 score 融合比直接改优化器更有效。

## 6. E10 和 E10b MMoE 结论

E10 的目标是让模型自动学习 task selective sharing，而不是人工设计 global branch 和 HLA branch。

E10 结果：

```text
mean AUROC = 0.7948
mean AUPRC = 0.7804
mean accuracy = 0.7210
mean MCC = 0.4474
```

E10 强于 E9、E11、E12，也略强于 E2 sample BCE，但仍明显弱于 E8a。

E10b 测试了更复杂的 MMoE 配置：

```text
e10b_4experts_256 mean AUROC = 0.7928
e10b_6experts_128 mean AUROC = 0.7913
```

调大 expert 数量或宽度没有缩小与 E8 的差距，说明当前数据规模和 task 数量下，MMoE 的自动 gate 学习还不如 E8 的简单 global/HLA soft ensemble 稳定。

## 7. E11 DB-MTL 结论

E11 检查动态 task loss balancing 是否能改善 E2。

结果是：

```text
E11 mean AUROC = 0.7817
E11 mean AUPRC = 0.7643
E11 worst-10 mean AUROC = 0.6956
```

DB-MTL 没有超过 E2，也没有超过 E8。当前结果说明，仅仅动态平衡 task loss 不能解决 tissuePMHC standard split 中的主要性能差距。

## 8. E12 pair ranking 结论

E12 加入 paired ranking loss，希望利用正负样本 pair 的相对排序信息。

结果是：

```text
E12 mean AUROC = 0.7807
E12 mean AUPRC = 0.7618
E12 worst-10 mean AUROC = 0.7007
```

E12 弱于 E2 sample BCE。说明当前 pair_id 中的排序监督并没有提升主任务，可能原因包括：

```text
1. BCE 分类信号已经足够强，ranking loss 反而干扰概率校准。
2. pair 内正负样本差异不一定完全对应 tissue specificity 的可学习边界。
3. ranking loss 权重或 pair sampling 方式可能还不够合适。
```

因此 E12 更适合作为 negative result 写入分析，而不是作为主模型。

## 9. E13 auxiliary tissue/HLA prediction 结论

E13 是今天最有价值的新增实验。

E13 相比 E2 sample BCE 的平均提升：

| 指标 | E2 sample BCE | E13 auxiliary | E13 - E2 |
|---|---:|---:|---:|
| mean AUROC | 0.7945 | 0.8023 | +0.0078 |
| mean AUPRC | 0.7793 | 0.7856 | +0.0063 |
| mean accuracy | 0.7197 | 0.7292 | +0.0095 |
| mean MCC | 0.4441 | 0.4640 | +0.0200 |
| worst-10 mean AUROC | 0.7211 | 0.7306 | +0.0095 |

E13 的提升稳定，3 个 seed 的 mean AUROC 标准差只有 0.0018。

按 task-seed rows 统计，E13 相对 E2 的 AUROC：

```text
86 / 132 improved
46 / 132 decreased
```

E13 相对其他模型：

```text
相对 E10 MMoE: AUROC 平均 +0.0075
相对 E12 pair ranking: AUROC 平均 +0.0216
相对 E8a fixed average: AUROC 平均 -0.0027
```

辅助任务本身的最终训练诊断也比较清楚：

```text
tissue auxiliary accuracy ≈ 0.30
HLA auxiliary accuracy ≈ 0.77
```

这说明 E13 主要学到了比较强的 HLA 结构信息，tissue 结构信息也有但弱很多。因此 E13 的收益更像是来自 HLA-aware representation，而不是完整替代 E8 的 global/HLA ensemble 结构。

## 10. 当前阶段性排序

完成 E9-E13 后，当前 standard split 主线排序可以写成：

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

更简洁地说：

```text
E8 是当前最佳性能主模型。
E13 是最有价值的新增 representation auxiliary result。
E9/E11/E12 是有解释价值的 negative results。
E10 说明自动 selective sharing 有一定效果，但目前不如手工 global/HLA soft ensemble。
```

## 11. 今日核心结论

7.8 之后的 planned roadmap 已经基本跑完。当前最重要的结论是：

```text
在 tissuePMHC standard split 下，最强结构仍然是 E8a fixed average soft ensemble。
```

E13 给出了一个有价值的补充发现：

```text
显式 tissue/HLA auxiliary supervision 可以改善 shared peptide representation，
但还不能替代 E8 的 global branch + HLA-specific branch soft ensemble。
```

因此，当前主线可以阶段性收束。后续不建议继续无差别堆叠复杂多任务方法。更合理的下一步是围绕 E8 做可靠性和泛化边界分析：

```text
1. E8 sample_id 对齐断言和 negative control。
2. peptide-disjoint / protein-disjoint split。
3. 更多 seeds 的 E8a/E8b 稳定性验证。
4. global score 与 HLA score 的相关性分析。
5. 必要时再考虑 PLE，但优先级低于 E8 reliability analysis。
```

## 12. 报告写法建议

正式报告中可以把今天内容写成一个“planned extensions after E8”的结果段：

```text
After identifying E8 as the strongest soft-ensemble model, we evaluated several planned extensions, including gradient conflict handling, dynamic task balancing, automatic expert-based selective sharing, paired ranking supervision, and auxiliary tissue/HLA prediction.
```

中文叙事可以写成：

```text
在确认 E8 soft ensemble 是当前最佳结构后，我们进一步测试了 CAGrad、MMoE、DB-MTL、paired ranking loss 和 tissue/HLA auxiliary prediction。
这些方法中，只有 E13 auxiliary prediction 带来了稳定的额外提升，但仍略弱于 E8。
这说明手工引入的 global/HLA complementary structure 仍是当前 standard split 下最有效的 inductive bias。
```

`inductive bias` 可以理解为模型结构中预先加入的偏好或假设。这里 E8 的 inductive bias 是：peptide 表示既需要跨 HLA 的 global sharing，也需要 HLA-specific sharing。
