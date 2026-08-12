# Issue 5 实验报告：通用 pMHC predictor 对照

更新日期：2026-07-25  
状态：MHCflurry 2.2.1、NetMHCpan 4.1b、覆盖率审计、逐任务配对统计与增量堆叠分析均已完成。

## 1. 实验目的与结论

Issue 5 检验 TissuePMHC 的预测能力是否主要来自一般 peptide–MHC
binding/presentation propensity，而不是 tissue-conditioned preference。两个冻结的外部对照均不读取
tissue，也未使用本文标签微调：

1. MHCflurry 2.2.1，以 `presentation_score` 为主；
2. NetMHCpan 4.1b，以 `EL_Rank` 为主，`BA_Rank` 作为 binding-only 补充结果。

最终结论如下：

1. 两个外部模型都具有中等预测能力，task-macro AUROC 约为 0.65–0.69，说明一般 pMHC
   结合或呈递倾向确实解释了部分 benchmark 信号。
2. Human 上 MHCflurry presentation 与 NetMHCpan EL 基本相当；前者 AUROC 略高，后者
   PairAcc 略高，差异很小。
3. Mouse 上 NetMHCpan EL 明显优于 MHCflurry presentation，是更合适的小鼠通用 pMHC
   对照。
4. NetMHCpan EL 普遍优于 BA，尤其是 human，因此正文应以 EL 为 NetMHCpan 主结果，
   BA 放入补充或敏感性分析。
5. 两个通用模型均明显低于完整 TissuePMHC；将通用分数加入 TissuePMHC 后，
   AUROC 增量最多约 0.0041，说明完整模型已经吸收了绝大多数可利用的一般 pMHC 信号。

因此，Issue 5 支持以下受限结论：

> General peptide–MHC binding/presentation propensity explains part of the benchmark signal but does not fully account for the performance of the tissue-conditioned model.

## 2. 数据、模型与评价协议

唯一 peptide–MHC 查询数量如下：

| 物种 | 查询数 | MHC allele 数 |
|---|---:|---:|
| Human | 79,759 | 35 |
| Mouse | 6,663 | 4 |

所有分数统一转换为“越高表示越强”的方向。MHCflurry 正式运行时关闭 peptide flank；
NetMHCpan 同一次运行输出 EL 与 BA。所有唯一查询均成功评分：

| 模型与 scoring mode | Human row coverage | Mouse row coverage |
|---|---:|---:|
| MHCflurry presentation | 100% | 100% |
| NetMHCpan EL rank | 100% | 100% |
| NetMHCpan BA rank | 100% | 100% |

评价覆盖以下协议：

- standard fixed test；
- matched standard OOF；
- connected-component peptide-disjoint OOF。

每个 tissue–MHC task 内计算 AUROC、AUPRC 与 PairAcc，再进行 task-macro 汇总。外部模型在
matched standard OOF 与 peptide-disjoint OOF 上使用相同的完整 train row pool，因此其冻结分数的
standalone 指标相同；两个协议的区别体现在与各自 TissuePMHC OOF 预测的配对比较。

## 3. 两个通用模型的主要结果

### 3.1 Standard fixed test

| 物种 | 模型 | Mean AUROC | Mean AUPRC | Mean PairAcc |
|---|---|---:|---:|---:|
| Human | MHCflurry presentation | **0.68511** | **0.67830** | 0.68860 |
| Human | NetMHCpan EL | 0.68332 | 0.66863 | **0.68968** |
| Mouse | MHCflurry presentation | 0.65643 | 0.64357 | 0.67333 |
| Mouse | NetMHCpan EL | **0.68934** | **0.69263** | **0.69167** |

Human 上两者非常接近。MHCflurry presentation 的 AUROC 比 NetMHCpan EL 高 0.00179，
但 NetMHCpan EL 的 PairAcc 高 0.00108，不应把这种量级的差异解释为稳定的模型优势。

Mouse 上 NetMHCpan EL 的优势更清楚：相对 MHCflurry presentation，AUROC、AUPRC 和
PairAcc 分别提高 0.03291、0.04906 和 0.01833。因此 mouse 正文结果应优先使用
NetMHCpan EL。

### 3.2 Train-pool OOF

| 物种 | 模型 | Mean AUROC | Mean AUPRC | Mean PairAcc |
|---|---|---:|---:|---:|
| Human | MHCflurry presentation | **0.68709** | **0.67511** | **0.68975** |
| Human | NetMHCpan EL | 0.68228 | 0.66324 | 0.68751 |
| Mouse | MHCflurry presentation | 0.64735 | 0.62625 | 0.65649 |
| Mouse | NetMHCpan EL | **0.68019** | **0.67845** | **0.68877** |

OOF 结果与 fixed test 的排序一致：human 两个模型接近且 MHCflurry 略高，mouse 则是
NetMHCpan EL 明显更强。

## 4. NetMHCpan EL 与 BA

| 物种与协议 | BA rank AUROC | EL rank AUROC | EL − BA |
|---|---:|---:|---:|
| Human fixed test | 0.65773 | 0.68332 | +0.02559 |
| Human train-pool OOF | 0.65585 | 0.68228 | +0.02643 |
| Mouse fixed test | 0.68474 | 0.68934 | +0.00460 |
| Mouse train-pool OOF | 0.67780 | 0.68019 | +0.00239 |

EL 在两个物种上均不低于 BA，在 human 上优势约为 0.026 AUROC。这与任务更接近
presentation/ranking 而不是纯 binding affinity 的设定一致。NetMHCpan EL 应作为主外部基线，
BA 用于说明 binding-only 信号能够解释的性能下限。

## 5. 与完整 TissuePMHC 的比较

以下比较使用完全相同的可评分 pair。由于外部模型覆盖率为 100%，这里没有因缺失 allele
而改变主模型评估样本。

| 物种 | 协议 | 最强通用模型 | 通用模型 AUROC | TissuePMHC AUROC | 主模型增量 |
|---|---|---|---:|---:|---:|
| Human | Fixed test | MHCflurry presentation | 0.68511 | 0.84478 | +0.15967 |
| Human | Standard OOF | MHCflurry presentation | 0.68709 | 0.82795 | +0.14086 |
| Human | Peptide-disjoint OOF | MHCflurry presentation | 0.68709 | 0.76520 | +0.07811 |
| Mouse | Fixed test | NetMHCpan EL | 0.68934 | 0.85622 | +0.16688 |
| Mouse | Standard OOF | NetMHCpan EL | 0.68019 | 0.83922 | +0.15903 |
| Mouse | Peptide-disjoint OOF | NetMHCpan EL | 0.68019 | 0.75293 | +0.07274 |

在 fixed test 和 standard OOF 中，完整模型相对最强通用模型均提高约 0.14–0.17 AUROC。
在 peptide-disjoint OOF 中差距缩小到约 0.07–0.08，但完整模型的宏平均仍更高。这说明严格实体
分离削弱了 tissue-conditioned 模型的优势，却没有使其退化为单纯的一般 pMHC predictor。

逐任务配对统计中，human 三种协议以及 mouse fixed/standard 上的主模型优势均通过
BH-FDR。Mouse peptide-disjoint OOF 需要谨慎解释：

- 对 MHCflurry presentation，主模型 mean AUROC 增量为 0.10558，Wilcoxon
  `q = 0.000646`；
- 对 NetMHCpan EL，主模型 mean AUROC 增量为 0.07274，task bootstrap 95% interval 为
  `[0.00283, 0.15789]`，但 Wilcoxon `q = 0.68399`，24 个 task 中 11 胜、13 负。

因此不能声称完整模型在 mouse strict 的绝大多数 task 上一致优于 NetMHCpan EL。更准确的解释是：
宏平均差值为正，但 task 间异质性较强，rank-based 配对证据不足。

## 6. 控制通用分数后的增量

交叉拟合 logistic stacker 分别使用“外部分数”和“外部分数 + TissuePMHC 分数”。
主要 AUROC 如下：

| 物种 | 协议 | 外部模型 | External only | External + TissuePMHC |
|---|---|---|---:|---:|
| Human | Fixed test | MHCflurry presentation | 0.68511 | 0.84526 |
| Human | Fixed test | NetMHCpan EL | 0.68332 | 0.84469 |
| Human | Peptide-disjoint OOF | MHCflurry presentation | 0.68673 | 0.76929 |
| Human | Peptide-disjoint OOF | NetMHCpan EL | 0.67404 | 0.76514 |
| Mouse | Fixed test | MHCflurry presentation | 0.65643 | 0.85648 |
| Mouse | Fixed test | NetMHCpan EL | 0.68934 | 0.85639 |
| Mouse | Peptide-disjoint OOF | MHCflurry presentation | 0.64643 | 0.75642 |
| Mouse | Peptide-disjoint OOF | NetMHCpan EL | 0.64195 | 0.75339 |

与 TissuePMHC 单独结果相比，加入外部分数后的最大 AUROC 增量出现在 human
peptide-disjoint OOF 的 MHCflurry stack：`0.76520 → 0.76929`，即 +0.00409。其余主要增量
更小，部分组合几乎不变。这表明外部 predictor 与完整模型并非完全冗余，但其独立贡献有限；
不能把 stack 的微小变化解释为新的模型改进。

## 7. 论文表述建议

正文建议同时报告：

- MHCflurry 2.2.1 presentation score；
- NetMHCpan 4.1b EL rank；
- NetMHCpan BA rank 作为补充 binding-only 结果；
- 100% row/pair/task coverage；
- 与 TissuePMHC 的逐任务配对比较；
- 控制外部分数后的交叉拟合增量。

建议避免写成“通用 predictor 无法预测该任务”。实际结果显示它们具有稳定的中等性能。推荐表述为：

> Both frozen general pMHC predictors achieved moderate discrimination, indicating that generic binding or presentation propensity contributes to the benchmark. However, neither approached the tissue-conditioned model, and adding the external scores to TissuePMHC produced only marginal further gains.

## 8. 局限性

1. 外部 presentation predictor 可能使用过 IEDB 或相关 immunopeptidomics 训练数据。本文的
   peptide-disjoint split 仅相对于内部训练集成立，不能保证相对于外部预训练数据无 peptide
   overlap。
2. 应将 MHCflurry 与 NetMHCpan 描述为 frozen general-signal controls，而不是完全无泄漏的
   公平模型竞争。
3. task-bootstrap 与 Wilcoxon 属于 nominal task-level inference。不同 task 共享 tissue、MHC、
   parent protein 和 peptide component，不能视为相互独立的外部队列。
4. Human 上两个外部模型的差异很小；除非进一步执行预先规定的模型间直接配对检验，否则不应声称
   MHCflurry 显著优于 NetMHCpan。

## 9. 结果文件

- 外部模型汇总：`results/issue5_general_pmhc/external_evaluation/summary_metrics.csv`
- 覆盖率审计：`results/issue5_general_pmhc/external_evaluation/coverage_audit.csv`
- 逐任务结果：`results/issue5_general_pmhc/external_evaluation/per_task_metrics.csv`
- 主模型配对统计：`results/issue5_general_pmhc/external_evaluation/paired_statistics.csv`
- 增量堆叠汇总：`results/issue5_general_pmhc/stack_increment/summary_metrics.csv`
- 堆叠配对统计：`results/issue5_general_pmhc/stack_increment/paired_statistics.csv`
- MHCflurry score cache：`results/issue5_general_pmhc/score_cache/{human,mouse}_mhcflurry.csv.gz`
- NetMHCpan score cache：`results/issue5_general_pmhc/score_cache/{human,mouse}_netmhcpan.csv.gz`
- 版本与 allele 快照：`results/issue5_general_pmhc/raw_outputs/`

