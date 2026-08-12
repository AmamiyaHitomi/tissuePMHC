# humanPMHC premium：全模型结果与原因分析

## 1. 结果总览

本次比较全部使用已经完成的单 seed 测试结果；内部模型使用 premium train 训练并在 premium test 评估，MHCflurry 和 NetMHCpan 为不读取 premium train 的冻结外部模型。

| 模型 | 平均 task AUROC | 平均 task AUPRC | PairAcc | 最差 10 个 task 平均 AUROC |
|---|---:|---:|---:|---:|
| E29 multi-kernel CNN | **0.7040** | **0.6929** | **0.7040** | 0.5421 |
| E14a auxiliary dual branch | 0.6984 | 0.6865 | 0.6971 | **0.5557** |
| E2 shared heads | 0.6827 | 0.6720 | 0.6896 | 0.5436 |
| E0 random forest | 0.6779 | 0.6629 | 0.6835 | 0.5446 |
| MHCflurry presentation | 0.6677 | 0.6568 | 0.6781 | 0.5175 |
| NetMHCpan EL rank | 0.6545 | 0.6406 | 0.6659 | 0.4971 |
| NetMHCpan BA rank | 0.6332 | 0.6225 | 0.6496 | 0.4552 |

内部模型仍保持 E0 → E2 → E14a → E29 的平均排名，但阶段增益明显压缩：

| 阶段 | premium AUROC 增益 | 75 个 task 胜/平/负 | task bootstrap 95% 区间 |
|---|---:|---:|---:|
| E2 − E0 RF | +0.0048 | 40 / 0 / 35 | [-0.0066, 0.0161] |
| E14a − E2 | **+0.0157** | **49 / 1 / 25** | **[0.0071, 0.0243]** |
| E29 − E14a | +0.0056 | 42 / 0 / 33 | [-0.0043, 0.0152] |

这里的区间只是在固定单 seed 结果上对 75 个 task 重采样，用于描述 task 间稳定性，不能替代多 seed 训练不确定性。严格来说，E14a 是三个升级中唯一具有清晰、稳定增益的一步；E2 和 E29 的平均增益都小于 0.006，且 task 间方向不一致。

原 44-task 同 seed 的 AUROC 进程为 0.7558 → 0.7918 → 0.8102 → 0.8212，对应增益 +0.0360、+0.0185、+0.0110。premium 为 0.6779 → 0.6827 → 0.6984 → 0.7040，因此不是“升级顺序失效”，而是所有阶段的可利用信号都被压缩，E2 压缩最明显。

## 2. 是模型结构问题，还是数据处理问题？

结论是：**没有发现明显的数据处理 bug；主要原因是更严格且更稀疏的数据划分，加上 premium 数据本身更难、更接近组织特异性问题；模型结构与这种新难度存在明显不匹配。**

### 2.1 未发现明显处理错误

- premium test 的 7,500 行、3,750 个配对、75 个 task 均完整覆盖。
- 每个配对正负平衡，没有 train/test 配对泄漏；序列均为规范 9-mer。
- MHCflurry 和 NetMHCpan 对 32 个 HLA、7,204 个去重 peptide–HLA 查询均为 100% 覆盖。
- 外部模型与内部模型同时下降，因此结果不能用某个内部训练脚本的单点错误解释。
- source molecule 的 UniProt 缺失约 23.3%，但当前 E0/E2/E14a/E29 都没有使用该字段，所以它不是本轮分数下降的直接原因。

### 2.2 数据划分是首要原因之一，但不等于划分“错误”

premium 每个 task 的训练配对中位数只有 137，原 44-task 为 644；premium 的下四分位数为 76，原数据为 562。premium 每个 task 只测试 50 对，原数据为 100 对，因此 task 指标本身也更容易波动。

更关键的是 test 相对 train 的实体重叠明显降低：

| 数据集 | peptide 重叠 | HLA–peptide 重叠 | protein 重叠 |
|---|---:|---:|---:|
| premium | **43.1%** | **28.2%** | 85.2% |
| 原 44-task | 77.3% | 71.3% | 95.8% |
| Phase 7 | 78.6% | 68.3% | 96.0% |

这使 premium 更接近对未见 peptide/HLA–peptide 组合的外推，而原项目更多是在已覆盖实体附近插值。E29 的 PairAcc 从“两个 peptide 都未见”的 0.6786 上升到“两个都见过”的 0.7633；E29 相对 E14a 的优势也主要出现在见过的 peptide 上。这说明 CNN 的新增容量更擅长复用训练中出现过的 motif，而不是解决真正的未见实体外推。

任务训练量四分位数从小到大时，E29 AUROC 为 0.6744、0.6850、0.7126、0.7443；训练量与 E29 task AUROC 的 Spearman 相关为 0.320（p=0.005）。因此数据碎片化对深层模型的限制有直接证据。

### 2.3 数据本身也更难，但目前证据更像“弱标签与任务定义困难”，不是脏数据

premium 的正例是目标组织中被报告的 peptide，负例是在其他组织中被报告、但目标组织中未报告的 peptide。未报告并不等价于生物学上绝对不呈递，仍会受实验覆盖、样本量和检测灵敏度影响。

当 peptide 在其他组织中的呈递计数从 1 增加到 2 和 3+ 时：

- E29 PairAcc：0.7197 → 0.6462 → 0.6484；
- MHCflurry presentation：0.6973 → 0.6220 → 0.5824；
- NetMHCpan EL rank：0.6922 → 0.5698 → 0.5714。

跨组织越普遍的 peptide，正负之间越难分离。这既可能是更真实的生物学模糊性，也可能包含“目标组织未检测到”造成的弱标签噪声。它解释了为什么所有模型、包括冻结外部模型，都在 premium 上下降。

## 3. 各阶段增益为什么小？

### E2

E2 的共享 peptide 编码器确实能跨 task 迁移，但 premium 将训练数据分散到 75 个 tissue–HLA task，每个 task 更小、未见 peptide 更多。共享表示带来的好处被 task 头样本不足和更强的实体外推要求抵消，所以相对 E0 RF 只有 +0.0048。

### E14a

E14a 并不是完全“提升很小”：+0.0157 是唯一稳定的阶段增益。辅助 HLA/tissue 分支提供了比纯 task head 更有结构的归纳偏置，能缓解小 task 的数据稀疏。但它仍主要依赖 peptide 与离散 task 信息，没有直接使用蛋白表达、组织表达、抗原加工或实验来源，因此上限仍受任务信号不足限制。

### E29

E29 的 multi-kernel CNN 对固定 9-mer 增加局部 motif 建模能力，但 E14a 已能学习大量位置特异模式；在每个 HLA/task 数据量很小的条件下，新增 CNN 容量难以稳定估计。E29 相对 E14a 虽平均 +0.0056，但只有 42/75 task 提升，最差 10-task 均值还从 0.5557 降到 0.5421，说明它提高平均值的同时增加了尾部不稳定性。

所以结构问题不是“CNN 写错了”，而是**现有输入特征和 premium 的组织特异性目标之间不匹配**。继续堆 peptide-only 容量的边际收益很可能有限。

## 4. 外部模型为什么低？

最主要的是预测目标不一致：

- NetMHCpan BA 主要学习 peptide–HLA 结合；
- NetMHCpan EL 和 MHCflurry presentation 加入了更接近洗脱/呈递的信号；
- premium 要比较同一 tissue–HLA task 内、且经 HLA/蛋白背景控制后的组织特异性呈递偏好。

外部模型看不到目标 tissue、source protein 表达、蛋白酶体加工、样本与实验上下文，无法回答“这个能呈递的 peptide 为什么更可能出现在组织 A 而不是组织 B”。pair 构造又主动控制了大量一般 pMHC 信号，所以它们只能利用残余的通用结合/呈递倾向。

结果与这一解释一致：MHCflurry presentation AUROC 0.6677，高于其 affinity 0.6355；NetMHCpan EL 0.6545，高于 BA 0.6332。包含呈递训练目标的分数更好，但仍缺少 tissue 条件。

低分不是 HLA 不支持造成的，因为全部 32 个 HLA 和全部测试行均成功预测。相同冻结模型在原 Phase 7 上的 AUROC 分别为 MHCflurry presentation 0.6851、NetMHCpan EL 0.6833、BA 0.6577，在 premium 上全部下降到 0.6677、0.6545、0.6332。这进一步证明 premium 本身去除了更多通用 pMHC 捷径。

## 5. 建议的下一步

1. 保留当前划分作为严格 generalization benchmark，同时额外报告 seen/unseen peptide 和 seen/unseen HLA–peptide 分层指标，不要只报总均值。
2. 若目标是判断结构上限，优先加入真实的 tissue/protein 条件特征，而不是继续加大 peptide-only CNN：例如组织表达、蛋白丰度、抗原加工与实验来源。
3. 对 `other_tissue_presentation_count >= 2` 的样本做独立复核或低权重敏感性分析，判断弱标签对结果的影响。
4. 用固定划分做 3–5 seed 验证 E2 和 E29；当前单 seed 足以说明平均趋势，但不足以断言二者的微小增益可重复。

## 图表文件

- `01_performance_overview`：全部模型、阶段进程、阶段增益稳定性、尾部鲁棒性。
- `02_task_auroc_heatmap`：75 个 task × 7 个主模型的 AUROC 热图。
- `03_data_split_diagnostics`：训练量、跨组织歧义、train/test 实体重叠和 E14a→E29 task 反转。
- `04_external_predictor_analysis`：外部模型目标错配、跨组织退化及原数据对照。
- `premium_results_all_figures.pdf`：以上四张图的合并多页 PDF。
