# tissuePMHC Phase 1 阶段报告

## 摘要

本报告总结 `tissuePMHC` 项目在传统基线实验 E0 之后完成的新研究。数据集由人源 HLA-I 9-mer peptide ligand 记录构建，共包含 44 个 tissue-HLA 二分类任务。每个任务由一个目标组织和一个 HLA 限制等位基因共同定义。标准 train/test split 包含 96,972 条训练样本和 8,800 条测试样本，每个任务的测试集固定为 100 个正样本和 100 个伪负样本。

本报告中的“新研究”指 E1-E14。研究路线从神经网络单任务和多任务 baseline 开始，逐步扩展到 HLA 生物表示、任务加权、任务分组、选择性共享、soft ensemble、多任务优化、pair ranking、tissue/HLA auxiliary prediction，以及 auxiliary supervision 与 soft ensemble 的组合。当前最强结果是 E14a auxiliary soft ensemble，即带 tissue/HLA auxiliary supervision 的 global shared branch 与 plain HLA-specific branch 的固定平均融合。E14a 达到 mean AUROC 0.8116、mean AUPRC 0.7978、mean accuracy 0.7372、mean MCC 0.4769、worst-10-task mean AUROC 0.7349。

总体结论是：在当前 closed-set tissuePMHC standard split 下，表现最好的结构不是单一 global sharing、单一 HLA grouping，也不是更复杂的通用多任务优化器，而是 auxiliary-enhanced global sharing 与 HLA-specific sharing 的 soft ensemble。该结论是 benchmark 内的探索性模型比较结论，不等价于对全新 peptide、protein、HLA 或外部数据泛化能力的确认。

## 1. 研究背景与任务定义

`tissuePMHC` 项目研究人源 HLA-I peptide 的组织特异性呈递偏好。模型任务是：给定一个 peptide、一个 tissue 和一个 HLA allele，预测该 peptide 是否更可能在该 tissue-HLA 条件下被呈递。

每个任务定义为：

$$
\text{task} = (\text{target tissue}, \text{HLA restriction})
$$

每个任务都是一个二分类问题：

$$
y \in \{0,1\}
$$

其中 \(y=1\) 表示在目标 tissue-HLA 条件下有报告的正样本；\(y=0\) 表示具有相同 HLA 和 parent UniProt、在其他组织有报告但未在目标 tissue-HLA 条件下报告的配对伪负样本。标签 0 不是实验确认的生物学“不呈递”。

当前标准数据划分如下：

| 项目 | 数值 |
|---|---:|
| 任务数 | 44 |
| 训练样本数 | 96,972 |
| 测试样本数 | 8,800 |
| 每个任务测试样本 | 100 正样本 + 100 伪负样本 |
| peptide 长度 | 9 |
| HLA allele 数 | 12 |

需要注意的是，当前 standard split 是 closed-set 设置：测试集中的 tissue、HLA allele 和 tissue-HLA task 都在训练集中出现过；77.69% 的测试行 peptide 和 97.36% 的测试行 parent UniProt 也曾在训练集其他任务中出现。因此，当前结果主要说明模型在该 benchmark 内部的排序与信息共享能力，还不能直接证明对全新 peptide、全新 source protein、全新 HLA 或外部数据同样有效。

## 2. 参照基线：E0

E0 是传统单任务 baseline。每个 tissue-HLA task 单独训练传统机器学习模型，输入特征包括 one-hot peptide encoding 或 BLOSUM62 peptide encoding，模型包括 logistic regression、random forest、extra trees 等。

E0 中平均表现最好的模型是 one-hot logistic regression：

| 模型 | Mean AUROC | Mean AUPRC | Mean Accuracy | Mean MCC |
|---|---:|---:|---:|---:|
| E0 one-hot logistic regression | 0.7558 | 0.7384 | 0.6909 | 0.3841 |

本报告后续的 E1-E14 均属于新的研究内容。E0 只作为传统参照。

## 3. E1-E3：神经网络 baseline 与共享 peptide 表示

第一阶段的新问题是：神经网络模型和多任务共享表示是否可以超过传统 E0 baseline。

本阶段测试了三个方向：

| 实验 | 模型思想 |
|---|---|
| E1 | 每个 tissue-HLA task 单独训练小神经网络 |
| E2 | 所有 task 共享 peptide encoder，每个 task 有自己的 head |
| E3 | peptide encoder 加 tissue ID embedding 和 HLA ID embedding |

E2 是这一阶段最重要的模型。形式上可写为：

$$
\hat{y}_{t}=h_t(f_\theta(p))
$$

其中 \(p\) 是 peptide，\(f_\theta\) 是所有任务共享的 peptide encoder，\(h_t\) 是第 \(t\) 个任务的 task-specific classification head。

3 个 seed 的稳定结果如下：

| 模型 | Mean AUROC | Mean AUPRC | Mean Accuracy | Mean MCC | Worst-10 Mean AUROC |
|---|---:|---:|---:|---:|---:|
| E2 shared peptide encoder + task heads | 0.7927 | 0.7777 | 0.7180 | 0.4404 | 0.7178 |
| E3 best conditioned tissue/HLA ID model | 0.7833 | 0.7685 | 0.7152 | 0.4350 | 0.6992 |

E2 相比 E0 最强传统 baseline 的 mean AUROC 提升为：

$$
0.7927 - 0.7558 = 0.0369
$$

这说明 shared peptide representation 是有效的。E3 也超过传统 baseline，但不如 E2，说明在当前任务中，task-specific head 比单纯把 tissue/HLA 当作条件输入更有效。

## 4. E4-E4b：HLA 生物表示分析

E4 和 E4b 检查 HLA pseudo-sequence 是否能改进模型。项目为当前数据集中的 12 个 HLA allele 构建了 34-residue HLA class I pseudo-sequence。

实验设置如下：

| 实验 | 模型思想 |
|---|---|
| E4 | tissue embedding + HLA pseudo-sequence encoder |
| E4b | tissue embedding + HLA ID embedding + HLA pseudo-sequence encoder |

结果如下：

| 模型 | Mean AUROC | Mean AUPRC | Mean Accuracy | Mean MCC | Worst-10 Mean AUROC |
|---|---:|---:|---:|---:|---:|
| E3 HLA ID embedding | 0.7833 | 0.7685 | 0.7152 | 0.4350 | 0.6992 |
| E4 HLA pseudo-sequence only | 0.7728 | 0.7606 | 0.7042 | 0.4125 | 0.6705 |
| E4b HLA ID + pseudo-sequence hybrid | 0.7824 | 0.7704 | 0.7133 | 0.4303 | 0.6952 |

E4 没有超过 E3，说明 HLA pseudo-sequence 不能直接替代 HLA ID embedding。主要原因可能是当前是 closed-set HLA setting，测试集中的 HLA allele 都已经在训练集中出现过，因此 HLA ID embedding 可以直接学习每个 allele 的特征，而 pseudo-sequence encoder 需要把 12 个 allele 压缩到共享序列编码器中，反而可能形成信息瓶颈。

E4b 相比 E4 明显恢复性能，并且 AUPRC 略高于 E3。这说明 pseudo-sequence 作为辅助生物信息有一定价值，但它还不能成为当前性能主线。

## 5. E5-E7：任务加权、任务分组与 hard selection

E2 虽然强，但存在 negative transfer：部分任务在多任务共享后反而下降。因此下一阶段研究的核心是：任务之间应该如何共享参数。

本阶段包括：

| 实验 | 模型思想 |
|---|---|
| E5 | 在 E2 上加入 FAMO-style adaptive task weighting |
| E6 | 按 HLA 或 tissue 分组训练 shared peptide encoder |
| E7 | 在 validation set 上为每个 task hard-select global branch 或 HLA branch |

主要结果如下：

| 模型 | Mean AUROC | Mean AUPRC | Mean Accuracy | Mean MCC | Worst-10 Mean AUROC |
|---|---:|---:|---:|---:|---:|
| E2 shared heads | 0.7927 | 0.7777 | 0.7180 | 0.4404 | 0.7178 |
| E5 FAMO | 0.7758 | 0.7571 | 0.7077 | 0.4193 | 0.6979 |
| E6 HLA-grouped | 0.7862 | 0.7725 | 0.7148 | 0.4316 | 0.7037 |
| E6 tissue-grouped | 0.7387 | 0.7227 | 0.6765 | 0.3548 | 0.6699 |
| E7 selective HLA/global | 0.7904 | 0.7754 | 0.7184 | 0.4405 | 0.7143 |

E5 没有提升 E2。FAMO 的最终 task weight 接近均匀分布，并且 task-balanced sampling 本身削弱了原始 E2 的训练分布。

E6 表明 HLA grouping 有局部价值，但不能替代全局共享。tissue grouping 明显失败，说明按 tissue 分组会把 HLA binding motif 差异很大的任务强行放在一起。

E7 使用 validation-based hard selection，能恢复一部分 E6 HLA grouping 的损失，但仍没有超过 E2。原因是 hard selection 每个任务只能保留一个 branch，如果 validation 上的选择有噪声，就会完全丢掉另一个 branch 的有用信息。

## 6. E8：Global/HLA Soft Ensemble

E8 把 E7 的 hard selection 改为 soft ensemble。它保留 E7 的 leakage-safe 两阶段流程：先从 train 切出 validation 决定融合策略，再用完整 train 重新训练 final global branch 和 final HLA branch，test 只用于最终评估。

融合公式为：

$$
s_{\text{final}}=w_{\text{HLA}}s_{\text{HLA}}+(1-w_{\text{HLA}})s_{\text{global}}
$$

其中 \(s_{\text{HLA}}\) 是 HLA branch 输出分数，\(s_{\text{global}}\) 是 global branch 输出分数。

E8 测试了三种策略：

| 策略 | 定义 |
|---|---|
| E8a fixed average | \(w_{\text{HLA}}=0.5\) |
| E8b validation-delta clipped | \(w_{\text{HLA}}=\operatorname{clip}(0.5+5\Delta,0.15,0.85)\)，其中 \(\Delta=\text{AUROC}_{\text{HLA,val}}-\text{AUROC}_{\text{global,val}}\) |
| E8c validation softmax | 对 validation AUROC 做 softmax 生成权重 |

结果如下：

| 模型 | Mean AUROC | Mean AUPRC | Mean Accuracy | Mean MCC | Worst-10 Mean AUROC |
|---|---:|---:|---:|---:|---:|
| E2 shared heads | 0.7927 | 0.7777 | 0.7180 | 0.4404 | 0.7178 |
| E7 selective HLA/global | 0.7904 | 0.7754 | 0.7184 | 0.4405 | 0.7143 |
| E8a fixed average | 0.8050 | 0.7925 | 0.7313 | 0.4657 | 0.7295 |
| E8b validation-delta clipped | 0.8046 | 0.7916 | 0.7314 | 0.4660 | 0.7304 |
| E8c validation softmax | 0.8020 | 0.7890 | 0.7274 | 0.4583 | 0.7279 |

E8a 相比 E2 的提升如下：

| 指标 | 提升 |
|---|---:|
| Mean AUROC | +0.0122 |
| Mean AUPRC | +0.0149 |
| Mean accuracy | +0.0133 |
| Mean MCC | +0.0253 |

E8a 在 132 个 task-seed row 中有 91 个 AUROC 提升，并且 worst-10 mean AUROC 从 0.7178 提升到 0.7295。这说明 E8 的收益不是少数任务拉高平均值，而是整体和困难任务下界都得到改善。

最重要的观察是：最简单的固定平均 E8a 反而最好。这说明 global branch 和 HLA branch 确实有互补信息，而 validation-based per-task weighting 仍然有噪声。

## 7. E9-E14：E8 之后的扩展实验

确认 E8 是强结构后，项目继续测试了几类更复杂的多任务学习方法和监督信号，并最终把 E13 的 auxiliary supervision 与 E8 的 soft ensemble 合并为 E14。

| 实验 | 模型思想 |
|---|---|
| E9 | E2 + CAGrad gradient conflict handling |
| E10 | MMoE selective-sharing model |
| E10b | 调整 expert 数量、宽度和 gate regularization 的 MMoE |
| E11 | DB-MTL dynamic task loss balancing |
| E12 | paired ranking loss |
| E13 | 主任务 + tissue/HLA auxiliary prediction |
| E14 | auxiliary global branch + HLA-specific branch soft ensemble |

总体结果如下：

| 模型 | Mean AUROC | Mean AUPRC | Mean Accuracy | Mean MCC | Worst-10 Mean AUROC |
|---|---:|---:|---:|---:|---:|
| E14a global auxiliary + HLA plain soft ensemble | 0.8116 | 0.7978 | 0.7372 | 0.4769 | 0.7349 |
| E14b global auxiliary + HLA auxiliary soft ensemble | 0.8093 | 0.7955 | 0.7348 | 0.4735 | 0.7372 |
| E8a fixed soft ensemble | 0.8050 | 0.7925 | 0.7313 | 0.4657 | 0.7295 |
| E13 auxiliary tissue/HLA | 0.8023 | 0.7856 | 0.7292 | 0.4640 | 0.7306 |
| E10 MMoE | 0.7948 | 0.7804 | 0.7210 | 0.4474 | 0.7195 |
| E2 sample BCE | 0.7945 | 0.7793 | 0.7197 | 0.4441 | 0.7211 |
| E10b 4 experts, width 256 | 0.7928 | 0.7788 | 0.7170 | 0.4393 | 0.7104 |
| E10b 6 experts, width 128 | 0.7913 | 0.7768 | 0.7178 | 0.4423 | 0.7122 |
| E11 DB-MTL | 0.7817 | 0.7643 | 0.7144 | 0.4313 | 0.6956 |
| E9 CAGrad | 0.7810 | 0.7649 | 0.7106 | 0.4239 | 0.6979 |
| E12 pair ranking | 0.7807 | 0.7618 | 0.7102 | 0.4227 | 0.7007 |

E9、E11 和 E12 是有解释价值的 negative results。它们说明，当前主要瓶颈不太可能仅靠梯度冲突处理、动态 loss balancing 或 pair 内排序监督解决。

E10 MMoE 略强于 sample-level E2 baseline，说明自动 selective sharing 有一定作用。但它仍低于更简单的 E8 soft ensemble，且 E10b 的更大 expert 或更宽 expert 没有进一步缩小差距。

E13 是 E8 之后最有价值的新增单模型实验。相比 E2 sample BCE，E13 的 mean AUROC 提升 +0.0078，mean AUPRC 提升 +0.0063，mean accuracy 提升 +0.0095，mean MCC 提升 +0.0200，worst-10 mean AUROC 提升 +0.0095。辅助任务诊断显示 tissue auxiliary accuracy 约为 0.30，HLA auxiliary accuracy 约为 0.77，说明 E13 的收益主要来自 HLA-aware representation。

E14 进一步证明 E8 和 E13 的收益可以叠加。E14a 使用 auxiliary global branch 与 plain HLA-specific branch 做 fixed-average soft ensemble，mean AUROC 达到 0.8116。相比 E8a，E14a 的平均 AUROC 提升 +0.00662，在 132 个 seed-task 点中 85 个提升、47 个下降；相比 E13，E14a 的平均 AUROC 提升 +0.00928，在 132 个点中 97 个提升、35 个下降。E14b 把 auxiliary supervision 也加到 HLA-specific branch，mean AUROC 为 0.8093，低于 E14a，但 worst-10 mean AUROC 更高，为 0.7372。这说明 auxiliary supervision 更适合增强 global shared representation，而 HLA-specific branch 保持 plain supervised heads 更有利于平均性能。

## 8. 算法选择依据与方法定位

本项目选择算法时不是只追求方法新颖性，而是优先考虑算法是否匹配 `tissuePMHC` 的任务结构。具体标准包括：算法必须适合 44 个 tissue-HLA 二分类任务；能够利用任务之间的共享信息；不能把所有任务强行混成一个任务；最好能缓解 negative transfer；能够和当前 baseline 公平比较；实现难度需要适合作为本科科研项目逐步推进。

从这个标准出发，早期候选方向包括 shared peptide encoder + task-specific heads、conditioned model、HLA pseudo-sequence conditioning、FAMO、CAGrad、task grouping、MMoE / PLE、paired ranking loss、auxiliary tasks 和 DB-MTL。E1-E14 的结果表明，这些方向的价值并不相同。

方法定位如下：

| 方向 | 当前判断 | 建议用途 |
|---|---|---|
| auxiliary global branch + HLA-specific branch soft ensemble | 最有效 | 作为当前主线模型 |
| global shared branch + HLA-specific branch soft ensemble | 很有效 | 作为核心 anchor baseline |
| shared peptide encoder + task-specific heads | 稳定有效 | 作为核心 baseline |
| HLA grouping | 有局部价值 | 作为 soft ensemble 中 HLA-specific branch 的来源 |
| auxiliary tissue/HLA prediction | 有补充价值 | 后续用于增强 shared representation |
| conditioned tissue/HLA ID model | 合理但不占优 | 作为对照实验 |
| HLA pseudo-sequence conditioning | 有生物解释价值，但 closed-set 下不占优 | 用于 unseen-HLA 泛化或解释分析 |
| FAMO、DB-MTL | 未提升主线性能 | 作为 dynamic task weighting 的 negative result |
| CAGrad | 未提升主线性能 | 作为 gradient conflict 方向的 negative result |
| MMoE / PLE | 自动 selective sharing 有价值，但不如显式结构稳定 | 作为备选，不作为当前主线 |
| paired ranking loss | 与数据构造逻辑匹配，但指标上未提升 | 作为 negative result 保留 |
| protein language model embeddings | 有后期潜力，工程成本较高 | 放在泛化能力升级阶段 |

因此，本报告的算法选择结论是：当前 standard split 下，最值得保留的主线不是更复杂的通用多任务优化器，而是 `auxiliary-enhanced global sharing + HLA-specific sharing` 的 soft ensemble。FAMO、DB-MTL、CAGrad 和 paired ranking loss 的失败结果也有价值，因为它们说明当前瓶颈更像是任务共享结构设计，而不是单纯的 loss weighting 或 gradient conflict 处理。

## 9. 总体排序与解释

当前阶段性排序为：

```text
E14a auxiliary global + HLA plain soft ensemble
>
E14b auxiliary global + auxiliary HLA soft ensemble
>
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

核心发现可以概括为：

```text
在当前 closed-set tissuePMHC standard split 下，
最强结构是 auxiliary-enhanced global sharing 与 HLA-specific sharing 的 soft ensemble。
```

从模型角度看，global branch 学习跨 tissue、跨 HLA 的通用 peptide pattern；HLA branch 学习同一 HLA allele 内部的 HLA-specific peptide motif。E14a 进一步让 global branch 通过 tissue/HLA auxiliary supervision 学到更强的结构表示，而 HLA branch 保持 plain supervised heads。soft ensemble 同时保留两类信息，因此优于单一 global sharing、hard selection 和未融合的 auxiliary model。

## 10. 可靠性审计与阶段性提升证据

保存结果的指标计算和关键阶段比较经过独立复核。使用同 seed、同 tissue-HLA task 的 AUROC 做配对比较，得到：

| 比较 | Mean AUROC 增益 | 任务胜/负 | 配对任务检验 p 值 |
|---|---:|---:|---:|
| E8a − E2 | +0.01223 | 33 / 11 | 4.18×10⁻⁵ |
| E13 − matched E2 | +0.00779 | 35 / 9 | 2.63×10⁻⁶ |
| E14a − E8a | +0.00662 | 34 / 10 | 2.44×10⁻⁴ |
| E14a − E13 | +0.00928 | 34 / 10 | 4.41×10⁻⁴ |

这些结果表明 E8、E13 与 E14 的提升不是由单个 seed 或少数任务单独驱动，benchmark 内的提升方向具有一致性。但这些 p 值只衡量固定 test 上的任务配对差异，不能消除项目级选择偏差：E1–E14 的研究方向和模型排序反复观察了同一个 standard test。因此，E14a 的数值是真实可复算的，但该 test 已不再是完全未接触的外部确认集。

E8 的 validation 选策略、完整 train 重训、最终 test 评估流程在单次实验内部是 leakage-aware 的；这不应被扩大解释为整个 E1–E14 研究计划没有使用 test 反馈。Phase 1 的所有排名应视为 exploratory benchmark evidence，后续确认必须使用冻结的 outer split 或独立数据。

## 11. 局限性

当前结果需要在 benchmark 边界内理解：

1. 当前 split 对 tissue、HLA 和 task 都是 closed-set。
2. 测试行中 77.69% 的 peptide 曾在训练集其他 task 出现过。
3. 测试行中 97.36% 的 parent UniProt 曾在训练集中出现过。
4. E14 当前证明的是 standard split 下的优势，还没有证明在 peptide-disjoint、protein-disjoint、unseen-HLA 或外部数据上仍然最优。
5. 标签 0 是基于“目标组织未报告”构造的伪负例，不是实验确认的 non-presentation；模型更接近预测组织偏好或证据优先级。
6. 每个测试任务人为保持 1:1 平衡，因此 AUPRC、accuracy、F1 和 MCC 不能直接解释为现实 prevalence 下的部署性能。
7. standard test 在 E1–E14 的研究过程中被反复用于结果比较，存在项目级模型选择偏差。

这些局限不否定当前结论，但限定了结论适用范围。

## 12. 后续建议

当前 standard split 下继续无差别堆复杂模型的收益已经较低。下一步更应该围绕 E14/E8 做可靠性和泛化边界分析：

1. 冻结当前 standard test，不再据此选择模型、融合权重或研究方向。
2. 建立从未查看的 outer confirmation set，并构建 peptide-disjoint、parent-protein-disjoint、unseen-HLA 与 study/assay-disjoint split。
3. 做 negative control，例如打乱 HLA branch score 后再 ensemble。
4. 使用 nested cross-validation 评估完整模型选择流程，而不只是单个最终模型。
5. 使用 task/tissue/protein/pair 层级 cluster bootstrap 报告不确定性。
6. 将标签明确称为 pseudo-negative，并进行 positive-unlabeled 敏感性分析。
7. 分析 global score 与 HLA score 的相关性，并统一保存逐样本预测、环境和数据 hash。

## 13. 结论

E0 之后的新研究显著提升了当前 tissuePMHC standard split 上的预测指标。E2 支持 multi-task shared peptide representation 优于所评估的传统单任务 baseline；E6 和 E7 说明 HLA-specific sharing 有局部价值，但 hard selection 不够稳定；E8 支持 global branch 与 HLA branch 在该 benchmark 上具有互补信息。E9-E13 表明，单纯改变优化器或 loss balancing 不如显式设计 sharing structure 有效；E13 则支持 tissue/HLA auxiliary supervision 作为有价值的表示学习补充。E14a 将 E8 的 complementary soft ensemble 和 E13 的 auxiliary-enhanced global representation 合并，成为 Phase 1 所评估模型中的最强模型。

因此，在当前 44-task、closed-set、平衡采样的 standard split 上，Phase 1 的最佳模型是 E14a auxiliary global + HLA plain fixed-average soft ensemble。该结论不应外推为 unseen peptide、unseen protein、unseen HLA 或外部队列上的已确认优势。

## 14. 参考资料

1. Caruana, R. Multitask Learning  
   https://link.springer.com/article/10.1023/A:1007379606734

2. Ruder, S. An Overview of Multi-Task Learning in Deep Neural Networks  
   https://arxiv.org/abs/1706.05098

3. NetMHCpan: pan-specific MHC class I binding prediction and HLA pseudo-sequence representation  
   https://doi.org/10.1371/journal.pone.0000796

4. NetMHCpan-4.0: improved peptide-MHC class I interaction prediction  
   https://doi.org/10.1093/nar/gkx276

5. FAMO: Fast Adaptive Multitask Optimization  
   https://arxiv.org/abs/2306.03792

6. DB-MTL: Dual-Balancing for Multi-Task Learning  
   https://arxiv.org/abs/2308.12029

7. CAGrad: Conflict-Averse Gradient Descent for Multi-task Learning  
   https://arxiv.org/abs/2110.14048

8. MMoE: Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts  
   https://dl.acm.org/doi/10.1145/3219819.3220007

9. PLE: Progressive Layered Extraction for Multi-Task Learning  
   https://dl.acm.org/doi/10.1145/3383313.3412236

10. RankNet: Learning to Rank using Gradient Descent  
    https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/

11. Dietterich, T. G. Ensemble Methods in Machine Learning  
    https://link.springer.com/chapter/10.1007/3-540-45014-9_1

12. BLOSUM62: Amino acid substitution matrices from protein blocks  
    https://doi.org/10.1073/pnas.89.22.10915

13. ESM: Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences  
    https://doi.org/10.1073/pnas.2016239118

14. ProtTrans: Towards Cracking the Language of Life's Code Through Self-Supervised Deep Learning and High Performance Computing  
    https://doi.org/10.1109/TPAMI.2021.3095381

15. Continued domain-specific pre-training of protein language models for pMHC-I binding prediction  
    https://arxiv.org/abs/2507.13077

16. Gradient-Based Multi-Objective Deep Learning: Algorithms, Theories, Applications, and Beyond  
    https://arxiv.org/abs/2501.10945
