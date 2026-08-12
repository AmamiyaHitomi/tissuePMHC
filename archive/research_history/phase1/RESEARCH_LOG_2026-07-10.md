# RESEARCH_LOG_2026-07-10

## 1. E14: auxiliary soft ensemble

今天完成 E14 full run。E14 的目标是检查 E13 的 tissue/HLA auxiliary supervision 能不能和 E8 的 global/HLA soft ensemble 叠加。

结果目录：

```text
results/tissuePMHC_auxiliary_soft_ensemble/
```

实验设置：

| 项目 | 设置 |
|---|---|
| seeds | 20260704, 20260705, 20260706 |
| tasks | 44 |
| ensemble formula | `0.5 * global_score + 0.5 * hla_score` |
| E14a | global auxiliary branch + HLA plain branch |
| E14b | global auxiliary branch + HLA auxiliary branch |

## 2. 核心结果

| 模型 | Mean AUROC | Mean AUPRC | Mean Accuracy | Mean MCC | Worst-10 Mean AUROC |
|---|---:|---:|---:|---:|---:|
| E14a global auxiliary + HLA plain soft ensemble | 0.8116 | 0.7978 | 0.7372 | 0.4769 | 0.7349 |
| E14b global auxiliary + HLA auxiliary soft ensemble | 0.8093 | 0.7955 | 0.7348 | 0.4735 | 0.7372 |
| E8a fixed soft ensemble | 0.8050 | 0.7925 | 0.7313 | 0.4657 | 0.7295 |
| E13 auxiliary tissue/HLA | 0.8023 | 0.7856 | 0.7292 | 0.4640 | 0.7306 |

E14a 是当前新的最强主结果。

## 3. 与已有模型的对照

E14a vs E8a：

```text
mean delta AUROC = +0.00662
wins/losses over 132 seed-task rows = 85 / 47
median delta AUROC = +0.00645
```

E14a vs E13：

```text
mean delta AUROC = +0.00928
wins/losses over 132 seed-task rows = 97 / 35
median delta AUROC = +0.00765
```

E14b vs E8a：

```text
mean delta AUROC = +0.00433
wins/losses over 132 seed-task rows = 84 / 48
median delta AUROC = +0.00520
```

## 4. 解释

E14 说明 E8 和 E13 的收益可以叠加。最优组合不是两条 branch 都加 auxiliary，而是：

```text
global branch: auxiliary tissue/HLA supervision
HLA branch: plain supervised heads
fusion: fixed average soft ensemble
```

这说明 auxiliary supervision 更适合增强 global shared representation。HLA-specific branch 本来就按 HLA 分组，额外加入 HLA auxiliary supervision 的边际收益较小，甚至可能引入过强的结构约束。

## 5. 任务层面变化

E14a 相比 E8a 提升最大的任务：

| Task | Mean AUROC delta |
|---|---:|
| blood + HLA-B*51:01 | +0.0294 |
| kidney + HLA-A*24:02 | +0.0265 |
| uterine cervix + HLA-A*24:02 | +0.0228 |
| blood + HLA-C*03:04 | +0.0213 |
| blood + HLA-B*27:05 | +0.0206 |

E14a 相比 E8a 下降最大的任务：

| Task | Mean AUROC delta |
|---|---:|
| lymphoid + HLA-B*40:02 | -0.0205 |
| brain + HLA-B*40:02 | -0.0118 |
| thymus + HLA-A*24:02 | -0.0112 |
| lung + HLA-A*02:01 | -0.0105 |
| uterine cervix + HLA-B*51:01 | -0.0104 |

## 6. 当前排序

```text
E14a auxiliary global + HLA plain soft ensemble
>
E14b auxiliary global + auxiliary HLA soft ensemble
>
E8a fixed global/HLA soft ensemble
≈ E8b validation-clipped ensemble
>
E13 auxiliary tissue/HLA prediction
>
E8c validation softmax ensemble
>
E10 MMoE
≈ E2 sample BCE
>
E11 / E9 / E12 等 negative results
```

## 7. 当前结论

在 tissuePMHC standard split 下，当前最佳结构是：

```text
auxiliary-enhanced global sharing + HLA-specific sharing + fixed soft ensemble
```

E14a 应该写入正式报告作为新的主结果。E8a 从“当前最强”降为“此前最强 anchor baseline”。E13 从“未超过 E8 的补充实验”更新为“E14 的关键组成模块”。
