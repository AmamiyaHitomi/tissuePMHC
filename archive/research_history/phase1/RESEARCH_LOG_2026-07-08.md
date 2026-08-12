# 2026.7.8 研究日志

## 1. 今日目标

今天的目标是完成并分析：

```text
E8: validation-weighted soft ensemble of global branch and HLA branch
```

E8 延续 E7 的两阶段设计，但不再 hard selection。

`hard selection` 指每个 task 只能二选一：要么使用 global branch，要么使用 HLA branch。
E7 的结果说明这种二选一机制不够稳定。

E8 改为 soft ensemble：

```text
final_score = w_hla * hla_score + (1 - w_hla) * global_score
```

`soft ensemble` 指把多个模型的预测分数按权重融合，而不是只选其中一个模型。
这里的 `prediction score` 是模型输出的正类概率。

## 2. E8 代码

新增脚本：

```text
scripts/run_tissuepmhc_soft_ensemble.py
```

输出目录：

```text
results/tissuePMHC_soft_ensemble/
```

主要输出文件：

```text
per_task_metrics.csv
summary_metrics.csv
stability_metrics.csv
candidate_metrics.csv
weight_metrics.csv
comparison_metrics.csv
metadata.json
```

E8 使用 3 个 seed：

```text
20260704
20260705
20260706
```

训练参数：

```text
device: cuda
epochs: 25
batch_size: 512
learning_rate: 0.001
weight_decay: 0.0001
embedding_dim: 16
hidden_dim: 128
dropout: 0.2
validation_fraction: 0.2
selection_metric: AUROC
```

## 3. E8 实验设计

E8 使用 leakage-safe 两阶段流程：

```text
1. 从 train 中切出 train-core 和 validation。
2. 用 train-core 训练 validation global branch。
3. 用 train-core 内每个 HLA group 训练 validation HLA branch。
4. 在 validation 上得到每个 task 的 global_validation_metric 和 hla_validation_metric。
5. 用完整 train 重新训练 final global branch。
6. 用完整 train 内每个 HLA group 重新训练 final HLA branch。
7. 在 test 上得到 global_score 和 hla_score。
8. 用 validation 决定的权重融合 test score。
```

这样做的目的：

```text
validation 只用于决定融合权重。
test 只用于最终评估。
final branch 使用完整 train 训练，和 E2 使用相同训练数据量。
```

## 4. E8 三种策略

E8 同时测试三种融合策略：

```text
E8a: e8a_fixed_average
E8b: e8b_validation_delta_clipped
E8c: e8c_validation_softmax
```

### 4.1 E8a fixed average

固定平均：

```text
hla_weight = 0.5
global_weight = 0.5
```

也就是：

```text
final_score = 0.5 * hla_score + 0.5 * global_score
```

### 4.2 E8b validation-delta clipped

根据 validation AUROC 差值调整 HLA 权重：

```text
delta = hla_validation_auroc - global_validation_auroc
hla_weight = clip(0.5 + 5.0 * delta, 0.15, 0.85)
```

`clip` 指把数值限制在指定范围内。
这里 HLA 权重不会低于 0.15，也不会高于 0.85。

### 4.3 E8c validation softmax

用 validation AUROC 的 softmax 生成权重。

`softmax` 是一种把多个分数转换成非负权重且总和为 1 的函数。
本实验中 softmax temperature 为：

```text
0.02
```

`temperature` 控制 softmax 的尖锐程度。
temperature 越小，权重越容易接近 0 或 1。

## 5. E8 主要结果

整体结果如下：

| model | mean AUROC | mean AUPRC | mean accuracy | mean MCC | worst-10 mean AUROC |
|---|---:|---:|---:|---:|---:|
| E2 shared heads | 0.7927 | 0.7777 | 0.7180 | 0.4404 | 0.7178 |
| E7 selective HLA/global | 0.7904 | 0.7754 | 0.7184 | 0.4405 | 0.7143 |
| E8a fixed average | 0.8050 | 0.7925 | 0.7313 | 0.4657 | 0.7295 |
| E8b validation-delta clipped | 0.8046 | 0.7916 | 0.7314 | 0.4660 | 0.7304 |
| E8c validation softmax | 0.8020 | 0.7890 | 0.7274 | 0.4583 | 0.7279 |

E8 明确超过 E2 和 E7。

E8a 相比 E2：

```text
mean AUROC  +0.0122
mean AUPRC  +0.0149
mean accuracy +0.0133
mean MCC    +0.0253
```

E8b 相比 E2：

```text
mean AUROC  +0.0119
mean AUPRC  +0.0140
mean accuracy +0.0134
mean MCC    +0.0256
```

E8c 相比 E2：

```text
mean AUROC  +0.0092
mean AUPRC  +0.0114
mean accuracy +0.0094
mean MCC    +0.0179
```

按 task-seed rows 统计，AUROC 提升数量为：

```text
E8a fixed average: 91 / 132 improved
E8b delta clipped: 88 / 132 improved
E8c softmax:       84 / 132 improved
```

`task-seed row` 指一个 task 在一个 seed 下的一条评估记录。
44 个 task 乘 3 个 seed，一共 132 条。

## 6. E8 权重分析

三种策略的 HLA 权重分布：

| strategy | mean HLA weight | std | min | max |
|---|---:|---:|---:|---:|
| E8a fixed average | 0.5000 | 0.0000 | 0.5000 | 0.5000 |
| E8b delta clipped | 0.4576 | 0.1461 | 0.1500 | 0.8500 |
| E8c softmax | 0.4228 | 0.2567 | 0.0073 | 0.9868 |

E8c 权重过于极端，接近 E7 的 hard selection。
这可能解释了为什么 E8c 弱于 E8a 和 E8b。

最重要的观察：

```text
最简单的 E8a fixed average 表现最好。
```

这说明 validation AUROC 对 task-level 权重的估计仍有噪声。
固定平均没有过度相信 validation，因此更稳。

## 7. E8 task-level 结果

E8a 提升最大的 task：

| target_tissue | mhc_restriction | AUROC delta vs E2 |
|---|---:|---:|
| small intestine | HLA-B*15:01 | +0.0541 |
| lung | HLA-A*24:02 | +0.0460 |
| blood | HLA-B*51:01 | +0.0410 |
| lymphoid | HLA-B*51:01 | +0.0369 |
| lymphoid | HLA-B*27:05 | +0.0342 |

E8a 下降最大的 task：

| target_tissue | mhc_restriction | AUROC delta vs E2 |
|---|---:|---:|
| blood | HLA-B*07:02 | -0.0182 |
| uterine cervix | HLA-A*24:02 | -0.0170 |
| lymphoid | HLA-B*15:02 | -0.0147 |
| blood | HLA-C*05:01 | -0.0144 |
| lymphoid | HLA-C*05:01 | -0.0122 |

E8a 不只是提升少数 task。
它在大多数 task-seed rows 上提升，并且 worst-10 mean AUROC 也从 E2 的 0.7178 提升到 0.7295。

这说明 E8 改善了整体性能，也改善了困难 task 的下界。

## 8. 为什么 E8 比 E2 高

E2 是：

```text
shared peptide encoder + task-specific heads
```

E2 的优势是 global sharing。
所有 44 个 tissue-HLA task 共享一个 peptide encoder，因此能学习跨 HLA、跨 tissue 的通用 peptide pattern。

但是 E2 的限制是：

```text
所有 task 共用同一个 peptide representation。
```

这可能让 encoder 更偏向全局平均规律，对 HLA-specific peptide pattern 不够敏感。

E8 增加了 HLA branch：

```text
within-HLA shared peptide encoder + task-specific heads
```

HLA branch 更容易学习某个 HLA allele 内部的 peptide motif。

`motif` 指序列中反复出现、具有功能意义的模式。
在 HLA-I 9-mer peptide 中，某些位置的氨基酸常常对 HLA binding 很关键。
这些关键位置也常被称为 `anchor residues`。

E8 的成功说明：

```text
global branch 和 HLA branch 存在互补信息。
```

global branch 学到跨 HLA 的通用呈递信号。
HLA branch 学到 HLA-specific binding preference。
soft ensemble 把两者合并，因此超过 E2。

## 9. 为什么 E8 比 E7 高

E7 使用 hard selection：

```text
每个 task 只能选 global branch 或 HLA branch。
```

如果 validation set 上的选择有噪声，E7 会完全丢掉另一个 branch 的信息。

E8 使用 soft ensemble：

```text
final_score = w_hla * hla_score + (1 - w_hla) * global_score
```

即使某个 branch 在 validation 上略弱，它仍然能贡献一部分信息。

因此 E8 比 E7 稳定得多。

从结果看：

```text
E7 mean AUROC = 0.7904
E8a mean AUROC = 0.8050
```

这说明问题不在于 HLA branch 没有价值，而在于 E7 使用 HLA branch 的方式太硬。

## 10. E8 逻辑可信性审计

今天对 E8 进行了逻辑可信性分析。

结论：

```text
E8 的逻辑通路是可信的。
E8 在当前 tissuePMHC standard split 下的结果大概率是真实的。
```

### 10.1 没有明显 test leakage

E8 的权重来自 validation，不来自 test。

代码流程是：

```text
train → train-core + validation
validation branch 只在 train-core 上训练
validation metric 只在 validation 上计算
final branch 用完整 train 重新训练
test 只用于最终评估
```

因此 E8 没有直接用 test set 调权重。

特别是 E8a：

```text
hla_weight = 0.5
```

E8a 完全不使用 validation metric 调权重。
它仍然是最强模型。
这进一步支持 E8 的提升来自 global/HLA score 的真实互补，而不是 validation 调参带来的偶然收益。

### 10.2 分支覆盖完整

检查结果：

```text
每个 seed:
validation global: 44 tasks
validation HLA:    44 tasks
test global:       44 tasks
test HLA:          44 tasks
```

最终每个 seed、每个 E8 strategy 都有 44 个 task 结果。

### 10.3 score 对齐基本可信

E8 融合要求：

```text
global_score[i] 和 hla_score[i] 对应同一个 test sample。
```

代码中已经检查：

```text
global_test["y_true"] == hla_test["y_true"]
```

同时，global branch 和 HLA branch 的 test 子集都来自同一个 test_df，并且过滤操作保留原始行顺序。
因此当前 score 对齐基本可信。

但为了论文级严谨，后续建议加入更强断言：

```text
sample_id 顺序也必须完全一致。
```

### 10.4 E8 结果不像实现 bug

如果 E8 提升来自 bug，常见现象可能是：

```text
只有某个 seed 异常高；
只有平均 AUROC 高，但 worst-10 不高；
复杂权重策略高，固定平均不高；
结果集中由少数 task 拉高。
```

实际结果不是这样：

```text
3 个 seed 都稳定高；
worst-10 AUROC 也提升；
最简单的 fixed average 最强；
91/132 task-seed rows 提升。
```

因此，E8 的结果更符合真实 ensemble 增益，而不是明显 bug。

## 11. E8 结果的边界

E8 在当前 benchmark 上可信，但仍需注意：

```text
当前 tissuePMHC split 是 closed-set。
```

`closed-set` 指 test 中的 tissue、HLA 和 task 都在 train 中出现过。

之前检查发现：

```text
test peptide 中约 77.3% 在 train 的其他 task 出现过。
test source protein 中约 95.8% 在 train 中出现过。
```

因此 E8 当前证明的是：

```text
在当前 closed-set tissuePMHC standard split 下，
global sharing + HLA-specific sharing 的 soft ensemble 是最强结构。
```

但它还没有证明：

```text
对完全新 peptide、新 source protein、新 HLA 或外部数据也同样提升。
```

这不是 E8 的逻辑错误，而是当前 benchmark 的泛化边界。

## 12. 当前模型排序

截至 E8，当前模型排序为：

```text
E8a fixed soft ensemble
≈ E8b validation-delta clipped ensemble
>
E8c validation softmax ensemble
>
E2 shared peptide encoder + task-specific heads
>
E7 selective HLA/global
>
E3 conditioned tissue + HLA ID
≈ E4b hybrid HLA ID + HLA pseudo-sequence
>
E6 HLA grouped
>
E5 FAMO
>
E6 tissue grouped
>
E4 HLA pseudo-sequence only
```

当前建议主模型：

```text
E8a fixed average soft ensemble
```

原因：

```text
结构最简单；
mean AUROC 和 mean AUPRC 最高；
提升稳定；
不依赖 validation metric 调权重，因此更少过拟合 validation。
```

E8b 可以作为稳健备选：

```text
MCC 和 worst-10 AUROC 略高于 E8a。
```

## 13. 下一步计划

下一步不建议继续立刻堆复杂模型。
应先做 E8 的可信性确认和泛化边界分析。

推荐下一步：

```text
当日临时建议：E8 validation and stress tests
（后续复盘后，不再把它作为正式 E9；正式 E9 主线见第 15 节。）
```

具体包括：

```text
1. 加入 sample_id 对齐断言。
   确认 global_score 和 hla_score 融合时对应完全相同的 test sample。

2. 做 negative control。
   随机打乱 HLA branch score 后再 ensemble。
   如果性能掉回 E2 附近或更低，说明 E8 提升来自真实互补信息。

3. 做 peptide-disjoint 或 protein-disjoint split。
   测试 E8 在更严格泛化设置下是否仍超过 E2。

4. 增加 seed 数量。
   将 E8a/E8b 从 3 seeds 扩展到 5 或 10 seeds。

5. 分析 global_score 与 hla_score 的相关性。
   如果两者相关但不完全相同，说明 ensemble 的增益来源合理。
```

`negative control` 指故意破坏模型中的某个关键结构，检查性能是否下降。
如果破坏后性能不下降，说明原始提升可能不可信。
如果破坏后性能明显下降，说明原始结构确实有作用。

## 14. 今日结论

今天最重要的结论是：

```text
E8 soft ensemble 明确超过 E2，成为当前最强模型。
```

更具体地说：

```text
E2 证明 multi-task shared peptide encoder 有效。
E6 证明 HLA-specific sharing 有局部价值，但单独使用会损失 global information。
E7 证明 hard selection 不稳定。
E8 证明 global sharing 与 HLA-specific sharing 可以通过 soft ensemble 互补。
```

因此，当前 tissuePMHC 的阶段性核心发现是：

```text
最佳结构不是单一 global sharing，也不是单一 HLA grouping，
而是 global branch + HLA-specific branch 的 soft ensemble。
```

该结果在当前 standard split 下可信。
但在写正式报告或论文时，必须明确说明：

```text
当前结论主要适用于 closed-set tissuePMHC benchmark。
更严格的新 peptide / 新 protein / 外部数据泛化仍需后续验证。
```

## 15. 后续主线校正说明

后续复盘时确认：本项目从 E6 到 E8 的主线应明确归为 E2 线，而不是 E4 线。

E2 线指：

```text
shared peptide encoder + task-specific heads
```

这条线关注的是：

```text
task 之间应该如何共享 peptide encoder；
global sharing、HLA-specific sharing、selective sharing、soft ensemble 哪种更合适。
```

因此，E6、E7、E8 的谱系应理解为：

```text
E2 shared heads
→ E6 HLA/tissue task grouping
→ E7 global branch vs HLA branch hard selection
→ E8 global branch + HLA branch soft ensemble
```

E4 线指：

```text
peptide encoder + tissue embedding + HLA pseudo-sequence encoder
```

这条线关注的是：

```text
HLA pseudo-sequence 这种生物表示是否有价值；
HLA ID、HLA pseudo-sequence、hybrid HLA representation 哪种更合适。
```

由于 E4 在 standard split 上没有超过 E2，所以 E4 暂时不作为后续性能主线。
E4 应保留为 biological representation branch，也就是生物表示分析线。

因此，原 roadmap 中后续优化方法的底座应校正为 E2 线。
同时，由于后续实际新增了 E7 hard selection 和 E8 soft ensemble，
原 roadmap 中更靠后的方法编号需要顺延：

```text
E6: HLA/tissue task grouping on E2
E7: validation-based selective HLA/global sharing
E8: validation-weighted soft ensemble of global branch and HLA branch
E9: E2 + CAGrad
E10: MMoE / PLE selective-sharing model on the E2/E8 shared-head line
E11: E2 + DB-MTL
```

`CAGrad` 指 Conflict-Averse Gradient Descent，用于缓解不同 task 的梯度冲突。
`DB-MTL` 是一种动态多任务学习方法，用于平衡不同 task 的 loss scale 和 gradient magnitude。
`MMoE` 和 `PLE` 都属于 selective-sharing model，用于让不同 task 自动选择不同程度的共享。

因此，后续研究顺序应为：

```text
先沿 E2/E8 性能主线继续完成 E9 CAGrad、E10 MMoE/PLE、E11 DB-MTL 等原 roadmap 内容；
之后再研究 E8 reliability、stress tests、disjoint split 等新 roadmap 扩展内容。
```
