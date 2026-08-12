# tissuePMHC Phase 2 阶段报告

更新日期：2026-07-12  
状态：E15–E27 与 E29 已完成；E29 5-seed 为 standard split 最终主结果；预注册确认性扩展已结束，E28 不执行

## 摘要

Phase 2 的目标是在 E14a 的双分支结构上，验证更稳健的分数融合与低成本集成方法能否带来额外收益。E14a 由带 tissue/HLA 辅助监督的全局分支（global auxiliary branch）和按 HLA 分组的普通分支（HLA plain branch）组成；原始融合方式为固定 0.5 概率平均。

本阶段完成了固定融合规则消融（E15）、MC Dropout（E16）、多随机种子预测集成（E17）、validation 选择的全局融合权重（E18）、训练期 checkpoint/snapshot 集成（E19）、SWA（E20）、五项辅助学习/结构探索（E21–E25）、严格 OOF 的 greedy selection 与 stacked generalization（E26–E27），以及 Multi-kernel CNN peptide encoder（E29）。最强且已冻结的 standard-split 结果为预注册 E29 5-seed mean：mean AUROC 0.8373、mean AUPRC 0.8259、worst-10 mean AUROC 0.7670。相对于 E17 5-seed，分别提升 0.0110、0.0121 和 0.0098。

主要结论是：独立随机种子的方差降低与 peptide encoder 的表示归纳偏置可以叠加。task-wise rank fusion 仍是稳健的分支融合规则；MC Dropout、checkpoint/snapshot ensemble、SWA、动态辅助权重和复杂二层融合均未产生同等级收益。E26/E27 表明同构候选无法靠后处理创造新信息；E29 则表明，在保留 E14a 强双分支结构的同时引入多尺度局部卷积 motif 编码，能够在当前 closed-set standard split 上获得稳定、可集成的表示增益。早期 MLP 的 Flatten 编码本身已经保留位置，因此 E29 的新增价值应归因于局部卷积、多尺度感受野和参数共享归纳偏置，而不是“首次保留位置”。

完成全链路审计后，本报告将结论严格限定在当前 benchmark 内：E29 的保存预测与汇总指标可以独立复算，且相对 E17 的提升在任务和随机种子层面均具有一致性；但 standard split 中 77.69% 的测试行 peptide、97.36% 的测试行 parent UniProt 已在训练集其他任务出现，且负例是“在其他组织有记录、在目标组织未记录”的伪负例。因此，本结果不能直接外推为对 unseen peptide、unseen protein、unseen HLA 或真实生物学“不呈递”的证明。

## 1. 实验范围与共同设置

Phase 2 的基础模型是 E14a。每个 task 由 tissue 与 HLA allele 定义；standard split 包含 44 个任务。除 E17 的 5-seed 扩展外，E15、E16、E18、E19、E20 均使用三个训练随机种子：20260704、20260705、20260706。训练配置保持 E14a 的 25 epochs、AdamW、learning rate 0.001、batch size 512。

主要指标为任务宏平均的 AUROC 和 AUPRC；同时报告 worst-10 mean AUROC，避免平均指标掩盖困难任务退化。除特别说明外，表中“±”为三个 seed 间标准差。

所有融合实验均以 `sample_id`、`target_tissue`、`mhc_restriction` 对齐分支预测；仅标签相同不足以保证逐样本对应。涉及随机推理或成对训练的实验统一控制 Python、NumPy、CPU Torch 与 CUDA RNG 状态，避免额外推理或训练路径改变后续模型的随机序列。E18 使用 train 内 validation，E26、E27 与 E29 使用 pair-grouped OOF 完成局部选择；这些规则属于实验设计，关于长期复用 standard test、预注册可信度与跨数据集外推的限制统一留到第 8 节审计。

E14a 作为本阶段起点的结果为：

| 模型 | Mean AUROC | Mean AUPRC | Mean Accuracy | Mean MCC | Worst-10 Mean AUROC |
|---|---:|---:|---:|---:|---:|
| E14a：auxiliary global + plain HLA，固定概率平均 | 0.8116 | 0.7978 | 0.7372 | 0.4769 | 0.7349 |

## 2. Phase 2 完成实验与结果

| 实验 | 核心比较 | 最佳配置 | Mean AUROC | Mean AUPRC | Worst-10 Mean AUROC | 判断 |
|---|---|---|---:|---:|---:|---|
| E15 | 固定融合规则 | task-rank average | 0.8130 ± 0.0012 | 0.7990 ± 0.0008 | 0.7381 ± 0.0023 | 保留 |
| E16 | 单模型 MC Dropout | 20 次预测 + rank fusion | 0.8121 ± 0.0015 | 0.7981 ± 0.0019 | 0.7376 ± 0.0044 | 不优于 E15 |
| E17 | 多 seed 预测平均 | 5-seed + rank fusion | 0.8263 | 0.8139 | 0.7573 | 此前最佳，现为关键对照 |
| E18 | validation 选全局权重 | validation-selected rank weight | 0.8137 ± 0.0016 | 0.7990 ± 0.0020 | 0.7411 ± 0.0035 | 小幅、但不优于 E17 |
| E19 | 训练期集成 | final checkpoint | 0.8122 ± 0.0017 | 0.7980 ± 0.0036 | 0.7385 ± 0.0038 | 不保留 checkpoint/snapshot ensemble |
| E20 | SWA 配对消融 | 两分支均用 final checkpoint | 0.8150 ± 0.0004 | 0.8005 ± 0.0011 | 0.7420 ± 0.0021 | SWA 不保留 |
| E21 | 梯度相似度辅助门控（1-seed screen） | global retrain + 固定 HLA plain | 0.8069 | 0.7932 | 0.7328 | 辅助 gate 近零，停止 |
| E22 | Periodic Nash-MTL（1-seed screen） | 三宏目标 Nash 权重 | 0.8051 | 0.7939 | 0.7304 | 权重稳定但主任务退化，停止 |
| E23 | ForkMerge（1-seed screen） | pair-grouped validation fork/merge | 0.8031 | 0.7895 | 0.7270 | validation 筛选有效，外部泛化未增益 |
| E24 | Auto-Lambda（1-seed screen） | validation primary meta-loss | 0.8056 | 0.7931 | 0.7289 | 权重几乎不变，停止 |
| E25 | HLA-Structured PLE（3-seed） | 2 global + 12 HLA-private experts | 0.7938 ± 0.0030 | 0.7769 ± 0.0033 | 0.7212 ± 0.0034 | gate 健康，但不优于 E10/E14a |
| E26 | 3-fold pair-grouped OOF greedy selection | 仅选 E14 final 3-seed mean | 0.8246 | 0.8116 | 0.7535 | 略高于 E17 3-seed，但不及 E17 5-seed |
| E27 | OOF task-rank Logistic stacking | 固定 L2，C=0.1 | 0.8243 | 0.8109 | 0.7535 | 不优于 E26/E17；候选共线性明显 |
| E29 | Multi-kernel CNN E14a（3-seed） | kernel 2/3/5，位置保持 | 0.8341 | 0.8228 | 0.7634 | 中间确认结果 |
| E29 | Multi-kernel CNN E14a（预注册 5-seed） | 固定新增 seeds 20260707/08；不改权重/成员 | **0.8373** | **0.8259** | **0.7670** | standard split 最终最佳 |

说明：E17 的 5-seed 指将五个独立训练模型的预测在各分支内平均后再做 task-rank fusion。它是一个固定的 5-seed 集成结果，因此稳定性表中只有一个聚合点；不能把其“0 的 seed 间标准差”误解为重复实验已证明零方差。

## 3. E15：固定融合规则消融

E15 在完全对齐的 E14a 分支逐样本预测上比较三种规则：概率平均、logit 平均和 task-wise percentile-rank 平均。排名如下：

| 融合规则 | Mean AUROC | Mean AUPRC | Mean MCC | Worst-10 Mean AUROC |
|---|---:|---:|---:|---:|
| task-rank average | 0.8130 | 0.7990 | 0.4800 | 0.7381 |
| logit average | 0.8118 | 0.7981 | 0.4769 | 0.7337 |
| probability average | 0.8116 | 0.7978 | 0.4769 | 0.7349 |

task-rank average 相对 E14a 的原始概率平均提升约 0.0014 AUROC。它不要求两个分支的概率校准尺度完全一致，只要求各分支在 task 内的排序有信息，因此更适合作为后续集成的默认融合规则。

**决策：** 后续 E16–E20 默认使用 task-rank fusion。

## 4. E16–E20：不确定性与训练轨迹集成

### 4.1 E16：MC Dropout 推理期集成

E16 对同一训练模型在推理期保持 dropout 激活，分别平均 5、10、20 次随机预测，再执行 E15 的 task-rank fusion。

| MC 次数 | Mean AUROC | Mean AUPRC | Worst-10 Mean AUROC |
|---:|---:|---:|---:|
| 5 | 0.8107 | 0.7971 | 0.7360 |
| 10 | 0.8114 | 0.7973 | 0.7361 |
| 20 | 0.8121 | 0.7981 | 0.7376 |

更多 MC draws 有轻微改善，但 20 次预测仍低于 E15 的普通 task-rank fusion。说明当前模型的不确定性近似没有提供足够独立的集成成员。

随机数审计：MC 推理会消耗 dropout mask 的随机数。代码已保存并恢复 Python、NumPy、CPU Torch 与所有 CUDA RNG 状态，确保 MC 推理不会改变随后 HLA 分支的初始化或 DataLoader shuffle。

**决策：** 不作为独立主模型；可在后续真正 OOF 的候选库中作为低优先级互补候选。

### 4.2 E17：多随机种子预测集成

E17 对独立训练的 E14a 模型进行 prediction averaging：先分别平均 global 与 HLA 分支的概率，再进行 task-rank fusion。结果如下：

| 集成规模 | 成员 seed | Mean AUROC | Mean AUPRC | Mean Accuracy | Mean MCC | Worst-10 Mean AUROC |
|---:|---|---:|---:|---:|---:|---:|
| 3-seed | 20260704–20260706 | 0.8243 | 0.8109 | 0.7467 | 0.4936 | 0.7530 |
| 5-seed | 20260704–20260708 | 0.8263 | 0.8139 | 0.7492 | 0.4986 | 0.7573 |

这是 Phase 2 最显著、最稳定的性能来源：从 E14a 的 0.8116 提升至 5-seed 的 0.8263。5-seed 比 3-seed 进一步提高 0.0020 AUROC，且最弱十个任务的 AUROC 从 0.7530 提升至 0.7573。

**决策：** E17 5-seed 曾是 Phase 2 的最强锚点；E29 完成后它降为此前最佳和 seed-ensemble 对照。后续结果仍应与其 0.8263 比较。

### 4.3 E18：validation 选择全局融合权重

E18 在 train 内按 pair_id 分组切出 validation；仅用 validation 选择 global/HLA 的一个全局 rank 权重，最终用完整 train 重训并在 test 评估。该设计避免了用 test 调权重。

| 配置 | Mean AUROC | Mean AUPRC | Mean MCC | Worst-10 Mean AUROC |
|---|---:|---:|---:|---:|
| fixed 0.50 rank average | 0.8130 ± 0.0012 | 0.7990 ± 0.0008 | 0.4800 ± 0.0015 | 0.7381 ± 0.0023 |
| validation-selected global weight | 0.8137 ± 0.0016 | 0.7990 ± 0.0020 | 0.4787 ± 0.0039 | 0.7411 ± 0.0035 |

validation 选权重带来很小的平均 AUROC 增益（约 0.0007），并改善困难任务下界，但没有接近 E17 的多 seed 增益。对 44 个 task 使用单一全局权重也限制了其表达能力。

**决策：** 作为 leakage-safe weighting 对照保留；不取代 E17 5-seed 基线。

### 4.4 E19：训练期 checkpoint 与 snapshot 集成

E19 比较普通训练的 final checkpoint、指定 late epochs 的 checkpoint ensemble，以及 cosine-restart snapshot ensemble。

| 训练期成员 | Mean AUROC | Mean AUPRC | Mean MCC | Worst-10 Mean AUROC |
|---|---:|---:|---:|---:|
| final checkpoint | 0.8122 ± 0.0017 | 0.7980 ± 0.0036 | 0.4720 ± 0.0011 | 0.7385 ± 0.0038 |
| checkpoint ensemble | 0.8096 ± 0.0018 | 0.7951 ± 0.0039 | 0.4670 ± 0.0032 | 0.7387 ± 0.0021 |
| snapshot ensemble | 0.7962 ± 0.0023 | 0.7805 ± 0.0031 | 0.4507 ± 0.0056 | 0.7310 ± 0.0017 |

结果表明相邻训练 checkpoint 的误差不够独立；平均 checkpoint 稀释了 final checkpoint 的有效排序。snapshot ensemble 明显最弱，说明当前 cosine-restart 配置下的周期末模型并不构成互补成员。

**决策：** E19 只保留 ordinary final checkpoint；不将 checkpoint/snapshot ensemble 放入优先候选库。

### 4.5 E20：SWA 配对消融

E20 从相同的训练前缀出发，将普通 AdamW 最终 checkpoint 与 SWA continuation 严格配对比较。普通路径和 SWA 路径在分叉点保存并恢复 Python、NumPy、CPU Torch、CUDA RNG 状态；每一个 late epoch 都检查 RNG 对齐。因此观察到的差异可归因于 SWA 路径，而不是初始化或 batch 顺序差异。

| Global 分支 | HLA 分支 | Mean AUROC | Mean AUPRC | Mean MCC | Worst-10 Mean AUROC |
|---|---|---:|---:|---:|---:|
| final | final | 0.8150 ± 0.0004 | 0.8005 ± 0.0011 | 0.4786 ± 0.0052 | 0.7420 ± 0.0021 |
| SWA | final | 0.8141 ± 0.0003 | 0.7992 ± 0.0018 | 0.4756 ± 0.0016 | 0.7415 ± 0.0029 |
| final | SWA | 0.8121 ± 0.0000 | 0.7972 ± 0.0014 | 0.4765 ± 0.0059 | 0.7398 ± 0.0020 |
| SWA | SWA | 0.8107 ± 0.0007 | 0.7956 ± 0.0019 | 0.4717 ± 0.0020 | 0.7393 ± 0.0029 |

以 final/final 为对照，global-only SWA 的 AUROC 平均下降 0.0010；HLA-only SWA 下降 0.0030；两分支同时 SWA 下降 0.0044。HLA 分支上的退化在三个 seed 中方向一致。

E20 的 final/final 结果高于 E19 final checkpoint，但两者是独立重训，不能将这项差异归因于 SWA。E20 对 SWA 的有效结论只能来自同轮 final/SWA 配对消融。

**决策：** 不使用 SWA；后续采用普通 final checkpoint。SWA 不进入 E21 候选池。

## 5. E21–E25：辅助权重与结构探索

E21–E24 都只重训 E14a 的 global auxiliary branch，并按同一 seed 复用既有 HLA plain 预测后进行 task-rank fusion。E23/E24 采用 train 内 pair-grouped validation，因此其模型只用 80% fitting 数据更新；它们相对完整训练 E14a 的 test 差距不能完全归因于算法本身，但均未达到进入 3-seed 的 screen 条件。E25 是独立的单模型结构实验，直接与 E10 MMoE 和 E14a global branch 比较。

| 实验 | 方法与诊断 | 结果与决策 |
|---|---|---|
| E21 Gradient-Similarity Gating | 平均 tissue/HLA gate 分别为 0.00144/0.00161；约 46% 测量为梯度冲突 | 1-seed 融合 AUROC 0.8069，较同 seed E14 −0.0047；动态门控实质关闭辅助任务，停止 |
| E22 Periodic Nash-MTL | 475 次 Nash 更新均成功；后期三宏目标有效梯度贡献约 35%/32%/32% | 1-seed 融合 AUROC 0.8051，较 E14 −0.0065；平衡优化牺牲主任务泛化，停止 |
| E23 ForkMerge | 5 个周期中后 4 个周期确有 tissue-heavy/HLA-heavy/both-heavy 更新被 validation 选入 | 1-seed 融合 AUROC 0.8031，较 E14 −0.0085；validation 改善量仅 0.00012–0.00082 BCE，未转化为 test 收益，停止 |
| E24 Auto-Lambda | 400 次 meta update；tissue 权重约 0.1000→0.1036，HLA 约 0.1000→0.1007 | 1-seed 融合 AUROC 0.8056，较 E14 −0.0060；元梯度信号过弱，停止 |
| E25 HLA-Structured PLE | 2 个 global experts + 12 个 HLA-private experts；每 task 三路 gate；3-seed 无 gate >0.9 | 3-seed AUROC 0.7938，低于 E10 0.7948 和 E14a global 0.8023；不扩展到 5 seeds，也不进入融合 |

E25 最终 gate 平均熵为 0.858–0.931（理论上限为 log(3)≈1.099），最大 gate 权重范围为 0.802–0.838，说明失败不是 expert collapse。其 HLA-private gate 平均权重为 0.501、0.508、0.566：模型确实使用了 HLA-private 路由，但额外专门化没有稳定优于跨 HLA 共享。

E21–E25 的综合证据表明：在现有 peptide-only 表示和 standard split 上，辅助任务作为固定的轻量正则化有效；进一步动态放大、压缩或结构化分解该辅助信号并未优于 E14a。E26/E27 随后提供了严格 OOF 证据，表明这些方向及同构 E14/MC 候选没有产生额外收益。

## 6. E26–E27：严格 OOF 集成

E26 使用 3-fold pair-grouped OOF。每个 held-out 样本只由未见过该 pair 的模型预测；候选选择全部完成后，才读取由完整 train 重训模型生成的 test 预测。候选池包含 E14 final 与 E16 MC-20 的三个单 seed 及两类 3-seed mean。

| 候选 | OOF Mean AUROC | OOF Worst-10 AUROC | Test Mean AUROC | Test Worst-10 AUROC |
|---|---:|---:|---:|---:|
| E14 final 3-seed mean | 0.8042 | 0.7257 | 0.8246 | 0.7535 |
| E16 MC-20 3-seed mean | 0.8038 | 0.7254 | 0.8239 | 0.7531 |

Greedy selection 在第一步选择 E14 final 3-seed mean；继续加入任何候选都未达到预先设置的 0.0001 mean AUROC 最小增益，因此最终 ensemble 只有一个成员。E26 相对 E17 3-seed 的 test mean AUROC 提升仅 0.0003，在 44 个任务中为 21 胜、22 负、1 平，不能视为稳定的新优势。

E27 将八个候选的 task 内 percentile rank 作为特征，以全部 OOF 标签拟合固定 `C=0.1` 的 L2 Logistic Regression，然后对 test 评估一次。E27 test mean AUROC 为 0.8243、mean AUPRC 为 0.8109、worst-10 mean AUROC 为 0.7535。其系数出现明显正负抵消；E14 final 3-seed mean 与 MC-20 3-seed mean 的 OOF task-rank correlation 为 0.9987，说明二层模型面对的是高度共线的重复信息。

**当时决策：** E26/E27 作为 leakage-safe 负结果保留，E17 5-seed 暂时保持主结果；不通过修改 greedy 阈值、L2 C 或候选子集来追逐已经观察过的 test。该结论直接促成了随后只用 OOF 筛选表示异构成员的 E29。

## 7. E29 结果、5-seed 预注册扩展与 E28 处置

E29 根据 E26/E27 暴露出的表示同质性问题设计。原 E14a peptide encoder 是 `Embedding → Flatten → MLP`；E29 保留 global auxiliary、HLA plain 和 task-rank fusion，只以 kernel size 2、3、5 的轻量一维卷积替换编码器，并保留卷积后的位置信息。该设计比在当前 12-allele closed-set 上优先增加复杂 HLA pseudo-sequence cross-attention 更符合既有证据：E4/E4b 已显示 pseudo-sequence conditioning 未超过强共享模型，而 9-mer 局部 motif 仍存在尚未测试的编码器归纳偏置。

E29 首轮 seed 20260704 的 3-fold pair-grouped OOF 已完成，且未读取 test。CNN 单模型 OOF mean AUROC 为 0.8007，高于匹配 E14 seed 的 0.7915；task-rank correlation 为 0.8727。CNN 与 E14 3-seed mean 等权 rank 融合后，mean AUROC 为 0.8097，相对 0.8042 提升 0.0056；mean AUPRC 从 0.7855 提高到 0.7918；worst-10 mean AUROC 从 0.7257 提高到 0.7314。融合在 44 个任务上为 39 胜、5 负，四项预注册条件全部通过。

E29 随后进入 3-seed 阶段，并以 E14 final 3-seed mean 同时作为 matching baseline 和 fusion baseline。3-seed CNN OOF mean AUROC 为 0.8138，较 E14 的 0.8042 提升 0.0096；相关性为 0.9428。E14/E29 等权 OOF rank 融合为 0.8132，较 E14 提升 0.0090。四项条件再次通过后，脚本才读取 test 并执行 full-train 3-seed 训练。

正式 test 结果如下：

| 模型 | Mean AUROC | Mean AUPRC | Mean Accuracy | Mean MCC | Worst-10 Mean AUROC |
|---|---:|---:|---:|---:|---:|
| E17 5-seed（此前最佳） | 0.8263 | 0.8139 | 0.7492 | 0.4986 | 0.7573 |
| E29 CNN 单 seed 平均表现 | 0.8212 ± 0.0008 | 0.8086 ± 0.0009 | 0.7423 ± 0.0037 | 0.4846 ± 0.0075 | 0.7447 ± 0.0085 |
| E29 CNN 3-seed mean | 0.8341 | 0.8228 | 0.7542 | 0.5084 | 0.7634 |
| E14-3 + E29-3 等权 rank 融合 | 0.8340 | 0.8229 | — | — | 0.7639 |

E29 相对 E17 5-seed 的逐任务 AUROC 为 34 胜、10 负；平均 AUROC 提升 0.0079，任务 bootstrap 95% 区间为 `[0.0041, 0.0116]`。E14/E29 融合略改善 AUPRC 和 worst-10，但 AUROC 比 E29 单独低 0.0002，因此不取代 E29 主结果，也不再基于 test 调融合权重。

在新增 seed 训练前，E29 5-seed 已作为最后一次 standard split 确认性扩展完成预注册。固定新增 seeds 为 20260707/20260708，只复用而不重训既有三个 seed；成员权重固定为等权，禁止删除 seed、改权重或选择 4-seed 子集。进入 test 的三项 OOF 条件为：mean AUROC 增益至少 0.0010、worst-10 AUROC 增益不低于 −0.0010、mean AUPRC 增益不低于 −0.0005；任一条件失败即停在 OOF。

预注册 OOF 门槛全部通过：5-seed 相对 3-seed 的 mean AUROC 为 0.8157 vs 0.8138（+0.0019），mean AUPRC 为 0.7995 vs 0.7971（+0.0024），worst-10 mean AUROC 为 0.7379 vs 0.7349（+0.0030）。因此脚本只执行了一次固定的 5-seed test 评估，正式结果如下。

| 模型 | Mean AUROC | Mean AUPRC | Mean Accuracy | Mean MCC | Worst-10 Mean AUROC |
|---|---:|---:|---:|---:|---:|
| E29 3-seed | 0.8341 | 0.8228 | 0.7542 | 0.5084 | 0.7634 |
| E29 预注册 5-seed | **0.8373** | **0.8259** | **0.7588** | **0.5175** | **0.7670** |
| 5-seed − 3-seed | +0.0032 | +0.0031 | +0.0045 | +0.0091 | +0.0036 |

E29 5-seed 相对 E17 5-seed 的逐任务 AUROC 为 37 胜、7 负，平均提升 0.0110，任务 bootstrap 95% 区间为 `[0.0072, 0.0150]`；AUPRC 平均提升 0.0121（35 胜、9 负）。这说明增益并非只来自增加两个随机种子：CNN 的表示改进在 5-seed 确认性扩展后仍稳定存在。

**最终决策：** E29 预注册 5-seed mean 是 standard split 的冻结主结果。按预注册承诺，不再对该 split 增加成员、修改融合或继续调参。E28 Negative Correlation Learning 未执行，不计入本阶段结论。

## 8. 全链路可靠性审计与结论边界

### 8.1 数据定义、拆分与泄漏边界

数据集中每个 pair 包含一个正例和一个伪负例：二者具有相同的 HLA 与 parent UniProt；正例在目标 tissue-HLA 中被报告，负例在其他组织被报告但未在目标 tissue-HLA 中被报告。该设计控制了部分 protein/HLA 混杂，适合研究相对组织偏好，但标签 0 不是实验确认的“不呈递”。因此，模型输出应解释为 `tissue-HLA-specific presentation preference` 或证据优先级，而不是绝对 presentation probability。

standard split 在每个任务内随机抽取完整 pair，train/test 的 `pair_id` 无重叠，同一 task 内也没有重复 peptide。独立审计得到：

| 检查项 | 结果 |
|---|---:|
| Train rows / test rows | 96,972 / 8,800 |
| Tasks | 44 |
| Train/test `pair_id` overlap | 0 |
| Same-task peptide overlap | 0 |
| Test rows whose peptide appears in training in another task | 6,837 / 8,800（77.69%） |
| Test rows whose parent UniProt appears in training | 8,568 / 8,800（97.36%） |
| Test composition per task | 100 positive + 100 pseudo-negative |

因此当前任务是一个强 closed-set benchmark。跨任务实体重叠不是直接标签泄漏，但会使结果明显高于真正的 peptide-disjoint 或 protein-disjoint 泛化。平衡测试集上的 AUPRC、accuracy、F1 和 MCC 也不应直接解释为现实 prevalence 下的 precision 或临床准确率。

### 8.2 指标与结果文件复核

审计覆盖 244 个结果 CSV，未发现空文件、全空行或无法读取的文件。E29 5-seed 的正式候选包含 8,800 个唯一预测键，无重复、无缺失、覆盖全部 44 个任务。使用保存的逐样本预测独立复算得到：

```text
macro AUROC = 0.83730568
macro AUPRC = 0.82593372
```

与正式汇总文件完全一致。E17 5-seed 与 E29 5-seed 的逐任务比较为 37 胜、7 负，平均 AUROC 增益为 0.01102；任务 bootstrap 95% 区间为 `[0.00717, 0.01495]`。同一五个 seed 的单模型比较也全部为正，增益分别为 0.00959、0.00690、0.00815、0.00726 和 0.01595。这支持“当前 benchmark 内存在稳定 encoder 增益”，而不是单个 seed 偶然获胜。

E17 与 E29 的跨 seed/跨分支融合顺序略有不同。将 E14 改成与 E29 相同的“先逐 seed 做 branch-rank fusion，再跨 seed 平均”后 AUROC 为 0.82589，而正式 E17 为 0.82629，差异仅 0.00039，不能解释 E29 的主要增益。

### 8.3 关键阶段提升的配对证据

使用同 seed、同 tissue-HLA task 的 AUROC 做配对比较，主要提升如下：

| 比较 | Mean AUROC 增益 | 任务胜/负 | 配对任务检验 p 值 |
|---|---:|---:|---:|
| E8a − E2 | +0.01223 | 33 / 11 | 4.18×10⁻⁵ |
| E13 − matched E2 | +0.00779 | 35 / 9 | 2.63×10⁻⁶ |
| E14a − E8a | +0.00662 | 34 / 10 | 2.44×10⁻⁴ |
| E14a − E13 | +0.00928 | 34 / 10 | 4.41×10⁻⁴ |
| E15 rank − probability fusion | +0.00142 | 31 / 13 | 0.00126 |

E8、E13 与 E14 的阶段性提升方向较一致。E15 的收益虽可复现，但效应量很小，应表述为“小幅融合改进”，而不是重大模型突破。E16、E18–E27 的多数结果更适合作为有方法学价值的负结果或边界实验。

### 8.4 Seen/unseen peptide 诊断

按测试 peptide 是否在训练集其他任务出现分层后：

| 子集 | E29 AUROC | E17 AUROC | E29 − E17 |
|---|---:|---:|---:|
| Seen peptide | 0.85328 | 0.83966 | +0.01362 |
| Unseen peptide | 0.74199 | 0.73761 | +0.00438 |

E29 在 unseen-peptide 子集上仍保留一定排序能力，但绝对 AUROC 明显下降，且相对增益缩小。因此“多尺度 CNN 可能增强局部 motif 泛化”是有支持的工作假设，但还不是经过 peptide-disjoint 外部确认的事实。

### 8.5 预注册、test 复用与可复现性边界

E18、E26、E27 与 E29 的脚本级选择分别使用 train 内 validation 或 pair-grouped OOF，E29 也只在 OOF 条件通过后读取对应 test；这些局部流程是合理的。但从项目整体看，E0–E29 的研究方向已经长期观察同一个 standard test，因此不能把该 test 视为完全独立、从未参与研究决策的外部确认集。E29 5-seed 是该 split 的冻结终点，所有已观察的 test 结果均不得用于二次调参、增加成员或选择新结构。

E29 5-seed 的本地文件时间顺序支持“先预注册、后 OOF、再 test”：`preregistration.json` 早于新增 seed 的 OOF 与 test 输出。代码也明确在 OOF gate 通过后才进入 test 阶段。但项目目录缺少可核验的版本历史或外部时间戳，因此这里只能称为本地预注册证据，不能视为不可篡改的第三方预注册。

代码设置了 Python、NumPy、CPU Torch 与 CUDA seed，并保存了充分的逐样本预测；但没有统一启用 deterministic algorithms/cuDNN deterministic 配置，相同 seed 在不同 CUDA/cuDNN 环境下不保证逐位一致。ensemble 稳定性表中的标准差为 0 仅表示它是一个聚合点，不表示统计不确定性为零。

## 9. 模型定位与最终结论

当前推荐的报告主结果为：

```text
E29 5-seed Multi-kernel CNN E14a ensemble（预注册确认性扩展）
mean AUROC = 0.8373
mean AUPRC = 0.8259
worst-10 mean AUROC = 0.7670
```

Phase 2 的核心发现不是更复杂的 optimizer 或训练轨迹权重平均，而是：task-rank fusion 能减弱分支分数尺度差异；独立训练 seed 的预测平均能够降低方差；模型成员的独立性比同一训练轨迹上的 checkpoint、snapshot 或 SWA 平均更重要；在保留 E14a 双分支结构时改变 peptide encoder，可以同时提高单模型强度和成员多样性。相对地，动态辅助任务加权、HLA-private 结构以及高度同质候选上的 OOF greedy selection/stacking 均未产生同等级收益。

按当前证据，推荐层级为：

```text
E29 5-seed Multi-kernel CNN task-rank ensemble
> E29 3-seed Multi-kernel CNN task-rank ensemble
> E17 5-seed task-rank ensemble
> E26 OOF greedy 3-seed mean ≈ E17 3-seed task-rank ensemble ≈ E27 OOF stacking
> E20 final/final 与 E18 validation-selected rank fusion
> E15 task-rank fusion
> E16 MC Dropout-20
> E19 final checkpoint
> checkpoint/snapshot ensemble、SWA、E21–E25 的单模型探索
```

说明：上述层级首先按当前 standard split 的 mean AUROC 排序；E29 3-seed（0.8341）高于 E17 5-seed（0.8263），因此顺序与数值一致。

总结：目前构建了一个在 tissue 与 HLA 维度共享表示的多任务二分类框架，并在当前 44 个 tissue–HLA task 的标准划分上，相对强基线取得了广泛且稳定的逐任务性能提升。

### 当前证据可支持与不可支持的结论

**当前证据可支持：**

> 在当前 44-task、closed-set、平衡采样的 tissuePMHC standard split 上，E29 Multi-kernel CNN E14a 5-seed task-rank ensemble 是已评估模型中的最优模型。其相对 E17 的提升在多数任务和全部五个相同 seed 上方向一致。

**当前证据尚不支持：**

- 对 unseen HLA、严格 peptide-disjoint、protein-disjoint 或外部队列具有同等性能；
- 标签 0 代表生物学上确认的“不呈递”；
- 平衡 benchmark 的 AUPRC/precision 等同于现实 prevalence 下的 precision；
- E29 的收益证明了某一具体生物机制或因果性组织调控机制。

## 10. 关键代码与结果文件

| 内容 | 文件 |
|---|---|
| Phase 2 路线图 | `ROADMAP_PHASE_2_zh.md` |
| E15 | `scripts/run_tissuepmhc_e15_fusion_ablation.py`；`results/tissuePMHC_e15_fusion_ablation/` |
| E16 | `scripts/run_tissuepmhc_e16_mc_dropout_ensemble.py`；`results/tissuePMHC_e16_mc_dropout_ensemble/` |
| E17 | `scripts/run_tissuepmhc_e17_seed_ensemble.py`；`results/tissuePMHC_e17_seed_ensemble/` |
| E18 | `scripts/run_tissuepmhc_e18_global_weight_selection.py`；`results/tissuePMHC_e18_global_weight_selection/` |
| E19 | `scripts/run_tissuepmhc_e19_training_ensemble.py`；`results/tissuePMHC_e19_training_ensemble/` |
| E20 | `scripts/run_tissuepmhc_e20_swa.py`；`results/tissuePMHC_e20_swa/` |
| E21 Gradient-Similarity Gating | `scripts/run_tissuepmhc_e21_gradient_similarity_auxiliary.py`；`results/tissuePMHC_e21_gradient_similarity_auxiliary/` |
| E22 Periodic Nash-MTL | `scripts/run_tissuepmhc_e22_periodic_nash_mtl.py`；`results/tissuePMHC_e22_periodic_nash_mtl/` |
| E23 ForkMerge | `scripts/run_tissuepmhc_e23_forkmerge.py`；`results/tissuePMHC_e23_forkmerge/` |
| E24 Auto-Lambda | `scripts/run_tissuepmhc_e24_auto_lambda.py`；`results/tissuePMHC_e24_auto_lambda/` |
| E25 HLA-Structured PLE | `scripts/run_tissuepmhc_e25_hla_structured_ple.py`；`results/tissuePMHC_e25_hla_structured_ple/` |
| E26 OOF Greedy Selection | `scripts/run_tissuepmhc_e26_all_in_one.py`、`scripts/run_tissuepmhc_e26_greedy_ensemble_selection.py`；`results/tissuePMHC_e26_greedy_ensemble_selection/` |
| E27 Stacked Generalization | `scripts/run_tissuepmhc_e27_stacked_generalization.py`；`results/tissuePMHC_e27_stacked_generalization/` |
| E29 Multi-kernel CNN OOF Screen | `scripts/run_tissuepmhc_e29_multikernel_cnn_oof.py`；1-seed 结果 `results/tissuePMHC_e29_multikernel_cnn/`；3-seed 默认输出 `results/tissuePMHC_e29_multikernel_cnn_3seed/` |
| E29 5-seed 预注册增量扩展 | `E29_5SEED_PREREGISTRATION_zh.md`；`scripts/run_tissuepmhc_e29_incremental_5seed.py`；默认输出 `results/tissuePMHC_e29_multikernel_cnn_5seed/` |

## 11. 参考方法

1. Dietterich, T. G. *Ensemble Methods in Machine Learning* (2000).
2. Gal, Y. & Ghahramani, Z. *Dropout as a Bayesian Approximation* (2016).
3. Lakshminarayanan, B. et al. *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles* (2017).
4. Chen, H. et al. *Training Neural Networks with Multi-branch Architectures* / checkpoint ensemble literature (2017).
5. Huang, G. et al. *Snapshot Ensembles* (2017).
6. Izmailov, P. et al. *Averaging Weights Leads to Wider Optima and Better Generalization* (2018).
7. Caruana, R. et al. *Ensemble Selection from Libraries of Models* (2004).
8. Wolpert, D. H. *Stacked Generalization* (1992).
9. Du, Y. et al. *Adapting Auxiliary Losses Using Gradient Similarity* (2018).
10. Navon, A. et al. *Multi-Task Learning as a Bargaining Game* (2022).
11. Jiang, J. et al. *ForkMerge: Mitigating Negative Transfer in Auxiliary-Task Learning* (2023).
12. Liu, S. et al. *Auto-Lambda: Disentangling Dynamic Task Relationships* (2022).
13. Tang, H. et al. *Progressive Layered Extraction* (2020).
14. Liu, Y. & Yao, X. *Ensemble Learning via Negative Correlation* (1999).
15. Kim, Y. *Convolutional Neural Networks for Sentence Classification* (2014)；E29 借用 multi-kernel 1D CNN 的局部序列模式提取思想，并针对固定 9-mer 保留位置表示。
