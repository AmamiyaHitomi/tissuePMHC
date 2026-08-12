# 论文剩余问题：通用 pMHC 基线与 strict 架构对照

## 当前状态

论文现有结果已经能够支持以下结论：

- 在 standard pair-disjoint benchmark 中，human TissuePMHC 的 mean task AUROC 为 0.8448；
- 在 matched peptide-disjoint OOF 中，human AUROC 为 0.7652；
- mouse frozen five-seed Factorized MMoE 的 standard fixed-test AUROC 为 0.8562；
- mouse matched peptide-disjoint OOF AUROC 为 0.7529；
- 两个物种在 peptide-disjoint 条件下仍保留高于随机的排序信号，但均存在明显的实体泛化下降。

目前仍有两个需要新增实验才能解决的问题：

1. 第 5 项：缺少通用 peptide–MHC binding/presentation predictor 基线；
2. 第 9 项：缺少相同 strict split 下的架构对照。

---

## 第 5 项：缺少通用结合/呈递模型基线

### 问题是什么

当前模型取得较高 AUROC，但仅凭现有结果无法判断这些性能主要来自：

- 一般 peptide–MHC binding/presentation strength；还是
- TissuePMHC 学到的 tissue-conditioned preference。

虽然正负肽已经匹配相同 MHC restriction 和 parent UniProt protein，但每一对中的两条肽序列不同。正样本仍可能天然具有更强的一般 HLA/H2 结合或呈递能力。因此，MHC 和 parent-protein matching 能减少混杂，却不能完全排除一般 pMHC signal。

### 为什么重要

论文在 Related Work 和 Discussion 中反复强调，本任务不同于一般 binding/presentation prediction。如果没有实际加入通用 predictor，对该区别的支持主要来自任务定义，而不是实验比较。审稿人很可能追问：

> 一个完全不知道 tissue 的通用 pMHC predictor，在同一配对任务上能达到多少 AUROC？

### 最低要求实验

至少选择两个可以冻结运行的通用 predictor。优先覆盖：

1. binding-only score，例如 NetMHCpan BA/affinity；
2. presentation-like score，例如 NetMHCpan EL、MHCflurry presentation 或其他实际可运行的呈递分数。

这些 predictor 不得使用本文标签微调。应记录：

- 工具名称和精确版本；
- 模型或数据文件版本；
- scoring mode；
- allele 支持范围；
- 无法评分的 peptide–MHC 组合；
- 分数方向，例如 percentile rank 越低越好时应转换为统一的“越高越好”方向；
- 运行日期、命令行和环境。

### 建议增加的内部对照

除外部 predictor 外，建议增加一个 capacity-matched HLA-only/H2-only 模型：

\[
s=f(\text{peptide},\text{MHC}),
\]

该模型使用相同 peptide encoder、训练预算和 split，但不能访问 tissue 或 tissue–MHC task identity。它直接检验：

> 在相同数据和训练条件下，只使用 peptide–MHC 信息能够解释多少性能？

现有 shared-task-head、global branch 或 HLA-specific branch 不能完全替代这一对照，因为它们仍在 tissue-conditioned 标签或 tissue–MHC task heads 下训练。

### 评估协议

所有通用分数和 HLA-only/H2-only 模型应在以下协议中分别报告：

- human standard fixed test；
- human matched standard OOF；
- human peptide-disjoint OOF；
- mouse standard fixed test；
- mouse matched standard OOF；
- mouse peptide-disjoint OOF。

每个 tissue–MHC task 内至少计算：

- AUROC；
- AUPRC；
- PairAcc；
- 可评分任务数和覆盖率。

其中：

\[
\mathrm{PairAcc}
=
\frac{1}{N_{\mathrm{pairs}}}
\sum_i
\mathbf{1}
\left[
s(p_i^+)>s(p_i^-)
\right].
\]

建议同时报告：

- task-macro mean/median；
- strongest general baseline 与 TissuePMHC 的逐任务差值；
- win/tie/loss；
- median difference；
- Hodges–Lehmann difference；
- task-bootstrap confidence interval；
- Wilcoxon signed-rank test 和 BH-FDR。

### 推荐的判定逻辑

如果出现以下结果：

- 通用 predictor 明显低于 TissuePMHC；
- HLA-only/H2-only 模型明显低于完整模型；
- 完整模型在控制通用分数后仍有稳定增量；

则可以写：

> General peptide–MHC binding/presentation propensity explains part of the benchmark signal but does not fully account for the performance of the tissue-conditioned model.

如果通用 predictor 或 HLA-only 模型接近完整模型，则不能声称性能主要来自 tissue conditioning。应改写为：

> Much of the observed ranking performance is consistent with general peptide–MHC presentation propensity, while the incremental contribution of tissue conditioning is limited.

### 数据泄漏注意事项

外部 presentation predictor 可能在包含 IEDB 或相关 immunopeptidomics 数据的语料上训练。即使本文使用 peptide-disjoint OOF，也不能保证相对于外部模型训练集的 peptide-disjoint。

因此必须：

- 报告外部模型的训练数据来源；
- 在无法恢复训练实体清单时明确说明潜在预训练重叠；
- 将其描述为“通用信号对照”，而不是完全无泄漏的公平模型竞争。

### 应保存的输出

建议新增独立结果目录，并至少保存：

- unique peptide–MHC score cache；
- row-level predictions；
- per-task metrics；
- PairAcc；
- coverage/missing-allele audit；
- paired statistical comparison；
- tool versions、commands 和 metadata；
- 用于论文的汇总表和图。

### 完成标准

只有同时满足以下条件，才能认为第 5 项解决：

- 至少两个通用 pMHC scoring modes 或 predictors 已实际运行；
- 结果覆盖 standard 和 peptide-disjoint 协议；
- 报告 AUROC、AUPRC、PairAcc 和 coverage；
- 与主模型完成逐任务配对统计；
- 论文根据实际结果限制或加强 tissue-conditioned claim。

---

## 第 9 项：strict 协议下没有架构对照

### 问题是什么

当前 peptide-disjoint OOF 只报告冻结主模型：

- human：frozen TissuePMHC；
- mouse：frozen five-seed Factorized MMoE。

因此 strict 结果能够支持：

> 在 seen-task、unseen-peptide 条件下仍存在可学习信号。

但不能支持：

> TissuePMHC 或 Factorized MMoE 在 strict 条件下优于更简单的架构。

原因是 standard benchmark 中观察到的架构优势可能依赖 peptide overlap；如果基线没有在完全相同的 peptide-disjoint folds 上运行，就无法判断这些优势能否保留。

### Human 最低要求基线

应在现有 human connected-component peptide-disjoint folds 上至少运行：

1. one-hot logistic regression；
2. strongest traditional peptide baseline；
3. shared peptide encoder with task-specific heads；
4. MLP dual-branch baseline；
5. auxiliary dual branch；
6. frozen TissuePMHC。

其中最关键的架构比较是：

- shared heads vs auxiliary branch；
- auxiliary/MLP dual branch vs multi-kernel CNN；
- single branch vs rank fusion；
- single seed vs frozen three-seed ensemble。

如果计算资源有限，最低可接受组合为：

- shared encoder with task heads；
- strongest MLP/auxiliary dual-branch baseline；
- TissuePMHC。

### Mouse 最低要求基线

应在现有 mouse connected-component peptide-disjoint folds 上至少运行：

1. BLOSUM62 random forest；
2. shared encoder；
3. single-seed Factorized MMoE；
4. frozen five-seed Factorized MMoE。

H2-Kk residual adapter 可以作为补充实验，但不是解决第 9 项的最低必要条件。

### 公平比较要求

所有架构必须使用：

- 相同 pair pool；
- 相同 connected-component assignments；
- 相同 outer folds；
- 相同 task inclusion rule；
- 相同评价实现；
- 相同训练 epoch、batch size 和优化器规则，除非模型本身确实需要不同设置；
- 预先固定的 seed 集；
- 不读取 fixed test 或 strict OOF 总体结果进行模型选择。

必须避免：

- 给主模型使用多 seed ensemble、给基线只使用一个不稳定 seed，却直接把差异解释为架构收益；
- 在 pooled OOF predictions 上调整超参数；
- 为不同模型重新生成不一致的 peptide-disjoint folds；
- 只比较总体均值而不报告逐任务差异。

### 建议统计分析

对每个基线与主模型进行 task-paired 比较，至少报告：

- mean task AUROC/AUPRC；
- worst-group AUROC；
- PairAcc；
- single-seed mean 和 standard deviation；
- ensemble 结果；
- mean/median task difference；
- Hodges–Lehmann difference；
- win/tie/loss；
- task-bootstrap interval；
- Wilcoxon signed-rank test；
- 对预先规定的模型–指标 family 进行 BH-FDR。

任务共享 tissue、MHC、parent protein 和 peptide components，因此 task-bootstrap 和 Wilcoxon 结果应继续标记为 nominal task-level inference，不能解释为独立外部队列证据。

### 推荐的判定逻辑

如果 TissuePMHC 在相同 strict folds 上稳定优于 shared-head 和 MLP/auxiliary dual-branch 基线，可以写：

> The architectural advantage observed under the standard benchmark is retained under connected-component peptide-disjoint evaluation.

如果 strict 条件下差异缩小或消失，应写：

> The standard-benchmark architectural advantage does not clearly persist after peptide-identity separation; the strict results support task learnability rather than model superiority.

如果只有部分组件保留收益，应逐项陈述，例如：

- auxiliary supervision 保留收益；
- multi-kernel encoder 收益减弱；
- rank fusion 仅有很小增量；
- ensemble 主要减少随机波动。

### 应保存的输出

建议为 human 和 mouse 分别保存：

- frozen split manifest；
- 每个模型、seed、fold 的 row-level predictions；
- single-seed 与 ensemble per-task metrics；
- pair-level PairAcc；
- strict architecture comparison table；
- paired statistics；
- parameter count、训练时间和显存；
- 完整 metadata 与命令行。

### 完成标准

只有同时满足以下条件，才能认为第 9 项解决：

- 至少一个简单 shared baseline 和一个强 dual-branch/structured baseline 在相同 strict folds 上完成；
- 主模型和基线具有可比较的 seed/ensemble 处理；
- 报告逐任务配对统计及 PairAcc；
- 论文根据结果明确区分“strict task learnability”和“strict architecture superiority”。

---

## 推荐执行顺序

1. 冻结并校验现有 human/mouse component fold manifests；
2. 先运行第 9 项的内部 strict baselines，因为它们不依赖外部软件；
3. 同时整理所有唯一 peptide–MHC 组合，建立外部 predictor score cache；
4. 运行第 5 项的通用 binding/presentation predictors；
5. 将外部分数合并到相同 row-level evaluation frame；
6. 统一生成 AUROC、AUPRC、PairAcc、coverage 和配对统计；
7. 根据结果更新 Results、Discussion 和 Limitations；
8. 最后再决定是否可以加强 tissue-conditioned 和 strict architecture claims。

## 最终交付物

完成两个问题后，论文至少应新增：

- 一张通用 pMHC predictor 对照表；
- 一张 strict architecture comparison 表；
- 一张逐任务差值图或 standard/strict 对比图；
- 外部模型覆盖率与缺失 allele 审计；
- 两组完整的 task-paired statistics；
- 对预训练重叠、任务相关性和适用范围的限制说明。
