# mousePMHC Phase 3 实验报告

## 1. 项目目标与最终结论

Phase 3 的目标是在小鼠 tissue-H2-I 肽呈递偏好任务上，确定合理的最低样本阈值，并比较单任务学习、完全共享、多专家路由、动态任务加权、任务分组和部分共享等方案。

当前结论如下：

1. 最低任务纳入阈值采用 `min_pairs > 200`。它不是学习曲线预注册稳定规则自动选出的拐点，而是兼顾样本量、任务覆盖和 H2 多样性的工程选择；阈值继续提高到 250 后只剩 20 个任务和 3 种 H2，会直接丢失 H2-Kk。
2. 多任务共享对低资源小鼠任务非常重要。E1 相比最佳传统单任务 E0 的 mean task AUROC 提高约 `0.0543`。
3. E3b 的 task-balanced Factorized MMoE 是稳定的强基线，三 seed mean task AUROC 为 `0.8148 ± 0.0033`。
4. E6 的 TAG-style 硬分组明显失败，说明完全切断组间共享会损失大量正迁移，尤其伤害 H2-Db。
5. E7 在保留 E3b 全局共享的同时，仅为 H2-Kk 增加残差 adapter，取得当前最高的三 seed OOF mean task AUROC：`0.8180 ± 0.0013`。E7 相对 E1 提高 `0.0106`，相对 E3b 提高 `0.0031`。
6. E7 相对 E3b 的提升较小，按三个 seed 差值估计的 95% t 区间为 `[-0.0024, 0.0087]`，尚不能声称 E7 显著优于 E3b。因此最终应将 E7 作为主候选、E3b 作为简化对照，并在冻结方案后进行一次固定测试集评估。

## 2. 数据集与任务定义

数据集为 `mousePMHC`，任务单位是 `target_tissue × H2 restriction`。每个样本是一个配对二分类观测：

- 正例：该 9-mer peptide 在目标 tissue-H2 中被报告；
- 负例：来自相同 UniProt 和相同 H2、在其他组织被报告，但未在目标组织中报告的 peptide；
- 数据限定为小鼠来源、小鼠宿主、MHC-I、H2 restriction、阳性 IEDB 定性测量、具有 parent UniProt、未修饰标准氨基酸和长度为 9 的 peptide。

采用 `min_pairs > 200` 并为每个任务固定保留 100 个 test pairs 后，当前 benchmark 包含：

| 项目 | 数值 |
|---|---:|
| tissue-H2 tasks | 24 |
| tissues | 13 |
| H2 restrictions | 4 |
| train pairs | 6,766 |
| train rows | 13,532 |
| fixed test pairs | 2,400 |
| fixed test rows | 4,800 |
| 每任务 train pairs 范围 | 124–470 |
| 每任务 train pairs 中位数 | 264.5 |

数据元信息见 `data/mousePMHC/mousePMHC_metadata.json`。

## 3. 最低 pairs 阈值选择

学习曲线在每个阈值点纳入该阈值下所有符合条件的任务，并将每个任务下采样到相同 pairs 数；每个点使用 3 个重复 seed。任务覆盖如下：

| 每任务 pairs 数 | 任务数 | tissues | H2 数 |
|---:|---:|---:|---:|
| 100 | 43 | 22 | 4 |
| 150 | 33 | 18 | 4 |
| 200 | 24 | 13 | 4 |
| 250 | 20 | 12 | 3 |
| 300 | 19 | 11 | 3 |
| 350 | 14 | 8 | 3 |
| 400 | 10 | 7 | 3 |

各点的任务集合不同，因此曲线不能被解释为同一批任务上的纯样本量效应。预注册稳定规则在 100–400 范围内没有返回自动推荐值。最终采用 200 的主要理由是：

- 仍保留 24 个任务和全部 4 种 H2；
- 从 200 提高到 250 会丢失 H2-Kk，无法继续研究跨 H2 负迁移；
- 相比 100 或 150，单任务训练样本更充分，且总训练成本仍可接受。

因此，200 是覆盖度与单任务数据量之间的折中，不应表述为严格统计意义上的性能平台点。

## 4. 统一实验协议

- 所有当前 Phase 3 神经网络比较均使用 `min_pairs > 200` 的同一 train benchmark。
- 使用 3-fold pair-grouped OOF，fold split seed 为 `20260711`，避免同一 pair 跨训练折和验证折。
- E3b、E6、E7 以及补充 E1 使用训练 seeds `20260704`、`20260705`、`20260706`。
- 主指标为各任务 AUROC 的宏平均，即 mean task AUROC。
- 同时报告 mean task AUPRC、MCC、最弱任务 AUROC 和 worst-6 task AUROC。
- 所有模型选择、梯度审计和任务分组只使用 train/OOF 数据；截至本报告，fixed test 未被读取。
- 单 seed 筛选的 E2–E5 主要用于淘汰明显不合适的方向；通过筛选的 E3b 才进行三 seed 确认。

## 5. E0 与 E1 基线

### 5.1 E0：传统逐任务模型

E0 比较五个传统候选，每个 tissue-H2 任务独立训练。结果如下：

| E0 候选 | Mean task AUROC | Mean task AUPRC | Mean task MCC |
|---|---:|---:|---:|
| BLOSUM62 Random Forest | **0.7530** | **0.7292** | **0.3894** |
| BLOSUM62 Extra Trees | 0.7509 | 0.7259 | 0.3887 |
| BLOSUM62 HistGradientBoosting | 0.7412 | 0.7180 | 0.3678 |
| One-hot Logistic Regression | 0.7349 | 0.7094 | 0.3472 |
| BLOSUM62 Logistic Regression | 0.7250 | 0.6897 | 0.3446 |

BLOSUM62 Random Forest 是 E0 最佳传统基线。

### 5.2 E1：完全共享 peptide encoder + task heads

E1 使用一个全局共享 peptide encoder 和 24 个任务特异性输出头。三个 seed 的 mean task AUROC 分别为：

- `20260704`: 0.8057
- `20260705`: 0.8057
- `20260706`: 0.8107

三 seed 均值为 `0.8073 ± 0.0029`。E1 明显优于最佳 E0，证明跨任务共享对当前低资源 benchmark 有效。但早期逐 H2 分析显示共享收益不均匀：H2-Db 获益最明显，而 Kb/Kk 存在负迁移风险，因此后续实验集中于选择性共享。

## 6. E2–E7 方法与结果

下表汇总当前 `min_pairs > 200` 路线。单 seed 数值用于方法筛选；三 seed 数值以 `mean ± sample SD` 表示。

| 实验 | 方法 | Seeds | Mean task AUROC | Mean task AUPRC | Mean task MCC | Worst-6 AUROC | 决策 |
|---|---|---:|---:|---:|---:|---:|---|
| E0 | BLOSUM62 Random Forest | 1 | 0.7530 | 0.7292 | 0.3894 | — | 最佳传统基线 |
| E1 | Shared encoder + task heads | 3 | 0.8073 ± 0.0029 | 0.7929 ± 0.0033 | 0.4930 ± 0.0071 | 0.6617 ± 0.0043 | 共享基线 |
| E2 | H2-grouped hard sharing | 1 | 0.8022 | 0.7887 | 0.4932 | — | 低于 E1，停止 |
| E3 | Factorized MMoE-lite | 1 | 0.8132 | 0.8029 | 0.5176 | — | 有潜力，但 Kd 保护线边缘失败 |
| E4 | Factorized PLE-lite | 1 | 0.7830 | 0.7670 | 0.4425 | — | 路由塌缩，停止 |
| E5 control | Task-balanced MMoE | 1 | 0.8164 | 0.8067 | 0.5244 | 0.6804 | 胜出并升级为 E3b |
| E5 | FAMO + MMoE | 1 | 0.7993 | 0.7885 | 0.4886 | — | 显著低于等权 control，停止 |
| E3b | Task-balanced MMoE confirmation | 3 | 0.8148 ± 0.0033 | 0.8043 ± 0.0028 | 0.5176 ± 0.0070 | 0.6771 ± 0.0053 | 稳定强基线 |
| E6 | TAG-style grouped hard sharing | 3 | 0.7403 ± 0.0040 | 0.7172 ± 0.0038 | 0.3640 ± 0.0076 | 0.6144 ± 0.0039 | 明显失败，停止 |
| E7 | E3b + H2-Kk residual adapter | 3 | **0.8180 ± 0.0013** | **0.8065 ± 0.0025** | **0.5218 ± 0.0058** | **0.6816 ± 0.0010** | 当前主候选 |

### 6.1 E2：H2 分组硬共享

E2 为每个 H2 分别建立 peptide encoder，同一 H2 下的任务共享 encoder。E2 mean task AUROC 为 `0.8022`，比同 seed E1 低 `0.0035`。H2-Db 略有改善，但 Kb、Kd、Kk 均下降，说明简单按 H2 完全拆分会损失有用的跨 H2 正迁移。

### 6.2 E3/E3b：Factorized MMoE 与 task-balanced 训练

E3 使用 3 个共享 experts，gate 输入 peptide 表征、tissue embedding 和 H2 embedding，每个任务保留独立 head。初始 E3 单 seed AUROC 为 `0.8132`。

E5 实验中的等权 task-balanced control 达到 `0.8164`，优于原 E3，因此将其命名为 E3b 并补充三个 seed。E3b 三 seed AUROC 为 `0.8148 ± 0.0033`，相对匹配 seed 的 E1 平均提高 `0.0075`；三个 seed 的提升均为正。其 worst-6 AUROC 相对 E1 也平均提高约 `0.0154`。MMoE gate 没有出现 expert collapse，因此 E3b 是可靠的共享主干。

E3b 仍存在残余 H2 异质性。相对 E1 的三 seed H2 组 AUROC 差值为：Db `+0.0199`、Kb `+0.0054`、Kd `-0.0088`、Kk `-0.0120`。其中 Kk 略超出预设的 `-0.010` 保护线，成为 E6/E7 的主要目标。

### 6.3 E4：Factorized PLE-lite

E4 设计 global、tissue 和 H2 三类 route，但 AUROC 仅为 `0.7830`。平均路由权重约为 tissue `0.6508`、H2 `0.2957`、global `0.0535`，部分任务最大路由权重接近 1，表现出明显路由塌缩。该方向停止。

### 6.4 E5：FAMO 动态任务加权

E5 在相同 MMoE 主干和 task-balanced batch 上比较等权 control 与 FAMO。等权 control AUROC 为 `0.8164`，FAMO 为 `0.7993`，差值为 `-0.0170`；24 个任务中仅 4 个改善、20 个下降。动态权重没有缓解当前负迁移，反而破坏了稳定的等权训练，因此 FAMO 停止。

### 6.5 E6：TAG-style affinity 硬分组

E6 在每个 seed-fold 的 fitting 数据上进行一步任务迁移 affinity 估计，再按对称 affinity 形成 4–6 个任务组，每组独立训练共享 encoder。

E6 三 seed AUROC 为 `0.7403 ± 0.0040`，比 E3b 平均低约 `0.0745`，worst-6 AUROC 低约 `0.0626`。分 H2 看，相对 E3b 的 AUROC 差值为：

| H2 | E6 − E3b AUROC | 解释 |
|---|---:|---|
| H2-Db | -0.1205 | 严重损失全局共享带来的正迁移 |
| H2-Kb | -0.0421 | 明显下降 |
| H2-Kd | -0.0333 | 明显下降 |
| H2-Kk | -0.0024 | 基本持平，9 个 task-seed 中 7 个改善 |

E6 表明隔离共享确实能保护部分 Kk 任务，但硬切分同时摧毁了 Db/Kb/Kd 的正迁移。自动分组在各 seed-fold 间也不稳定，因此 E6 不适合作为最终模型，但它支持“保留全局共享，只给冲突组增加小型私有路径”的 E7 设计。

### 6.6 E7：Recon-style H2-Kk residual adapter

E7 保留完整 E3b 全局 MMoE 主干，并在共享 peptide encoder 后只为预声明的 H2-Kk 添加维度为 16 的 residual adapter。每个 seed-fold 仍进行 train-only H2 梯度审计，用于检验机制，但正式架构固定为 H2-Kk adapter，避免按 OOF fold 动态改变候选。

三个 seed 的结果为：

| Seed | E7 AUROC | E3b AUROC | E7 − E3b |
|---:|---:|---:|---:|
| 20260704 | 0.8179 | 0.8164 | +0.0016 |
| 20260705 | 0.8167 | 0.8110 | +0.0057 |
| 20260706 | 0.8193 | 0.8171 | +0.0021 |
| Mean | **0.8180** | 0.8148 | **+0.0031** |

E7 相对 E3b 的 H2 组 AUROC 差值为：

| H2 | E7 − E3b AUROC | 改善的 task-seed 数 |
|---|---:|---:|
| H2-Db | +0.0021 | 20/36 |
| H2-Kb | -0.0020 | 5/12 |
| H2-Kd | +0.0086 | 10/15 |
| H2-Kk | +0.0050 | 6/9 |

E7 不仅改善了目标 H2-Kk，也改善了 Kd 和 Db；Kb 的平均下降仅 `0.0020`。相对 E1，E7 的总体 AUROC 提高 `0.0106`，72 个匹配 task-seed 观测中有 49 个改善。按 E3b 相对 E1 的差值叠加计算，E7 相对 E1 的 H2 组差值约为 Db `+0.0219`、Kb `+0.0034`、Kd `-0.0002`、Kk `-0.0070`，所有 H2 均满足“不低于 E1 超过 0.010”的保护要求。

E7 的 worst-6 AUROC 为 `0.6816 ± 0.0010`，优于 E3b 的 `0.6771 ± 0.0053` 和 E1 的约 `0.6617`。这说明提升没有以牺牲最弱任务为代价。

梯度审计中，H2-Kk 与其他 H2 的平均 cosine 为 `0.0035`，在四个 H2 中最低，但仅 4/9 个 seed-fold 为负；Kd 也有类似波动。因此审计支持 Kk 是最弱共享组，但不构成强而稳定的“梯度持续冲突”证据。E7 的主要证据仍应是匹配 seed OOF 性能，而不是过度解释梯度 cosine。

## 7. 最终模型选择与下一步

当前建议冻结以下选择：

- 主模型：E7，task-balanced Factorized MMoE + H2-Kk residual adapter；
- 简化对照：E3b，task-balanced Factorized MMoE；
- 共享基线：E1；
- 传统基线：E0 BLOSUM62 Random Forest。

E7 满足相对 E1 的总体提升、H2 保护和 worst-6 保护要求，并取得最高平均 OOF AUROC。但 E7 相对 E3b 的绝对增益只有 `0.0031`，三 seed 差值的 95% t 区间跨 0，因此应表述为“当前最佳候选”，而不是“已证明显著优于 E3b”。

后续建议：

1. 不再根据 OOF 结果继续增加 E8 或调 E7 adapter 超参数，避免进一步模型选择偏差。
2. 冻结 E7 架构、超参数、训练 seeds、训练轮数和评价指标。
3. 在完整 train 上分别重训 E7、E3b、E1 和 E0，并对从未读取的 fixed test 做一次最终评估。
4. 最终报告同时给出 macro task AUROC/AUPRC、每 H2 宏平均、worst-6、逐任务指标和 seed 波动。
5. 固定 test 结果若显示 E7 与 E3b 几乎相同，应优先选择结构更简单的 E3b；若 Kk 和 worst-6 的改善复现，则选择 E7。

## 8. 结果解释限制

- pairs 学习曲线各点的任务集合不同，因此不能将曲线变化全部归因于样本数量。
- E2–E5 多数是单 seed 筛选，适合淘汰明显失败方法，不适合精确排序。
- E6 是项目内实现的 TAG-style 一步迁移 affinity 分组，而不是对原论文全部训练流程的逐项复现。
- E7 是 Recon-style 梯度审计与部分共享思想的项目适配，不是原方法的原样复现。
- Phase 3 已多轮使用 OOF 结果进行模型选择，最终泛化结论必须以冻结后的 fixed test 为准。
- 目前所有结论都是 train-only OOF 结论，不能将其描述为独立测试性能。

## 9. 主要代码与结果位置

### 代码

- pairs 学习曲线：`scripts/run_mousepmhc_phase3_pair_learning_curve.py`
- E0：`scripts/run_mousepmhc_phase3_e0_oof.py`
- E1：`scripts/run_mousepmhc_phase3_e1_oof.py`
- E1 补充 seeds：`scripts/run_mousepmhc_phase3_e1_additional_seeds_oof.py`
- E2：`scripts/run_mousepmhc_phase3_e2_h2_grouped_oof.py`
- E3：`scripts/run_mousepmhc_phase3_e3_factorized_mmoe_oof.py`
- E4：`scripts/run_mousepmhc_phase3_e4_factorized_ple_oof.py`
- E5：`scripts/run_mousepmhc_phase3_e5_famo_mmoe_oof.py`
- E3b：`scripts/run_mousepmhc_phase3_e3b_task_balanced_mmoe_oof.py`
- E6：`scripts/run_mousepmhc_phase3_e6_tag_grouped_oof.py`
- E7：`scripts/run_mousepmhc_phase3_e7_recon_h2_adapters_oof.py`

### 结果

- pairs 学习曲线：`results/mousePMHC_phase3_pair_learning_curve/`
- E0：`results/mousePMHC_phase3_e0_oof/`
- E1：`results/mousePMHC_phase3_e1_oof/` 和 `results/mousePMHC_phase3_e1_oof_additional_seeds/`
- E2：`results/mousePMHC_phase3_e2_h2_grouped_oof/`
- E3：`results/mousePMHC_phase3_e3_factorized_mmoe_min200_oof/`
- E4：`results/mousePMHC_phase3_e4_factorized_ple_min200_oof/`
- E5：`results/mousePMHC_phase3_e5_famo_mmoe_min200_oof/`
- E3b：`results/mousePMHC_phase3_e3b_task_balanced_mmoe_min200_oof/`
- E6：`results/mousePMHC_phase3_e6_tag_grouped_min200_oof/`
- E7：`results/mousePMHC_phase3_e7_recon_h2_adapters_min200_oof/`
