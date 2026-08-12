# tissuePMHC Phase 7 实验报告：min_pairs=200 同步、模型复验与 E29 最终结果

更新日期：2026-07-17  
状态：Phase 7 核心实验与 E31 peptide-disjoint robustness 已完成；E30 不属于本阶段核心目标，未执行。  
最终模型：E29 multi-kernel CNN，3-seed mean。  
数据定义：每个 tissue-HLA task 的原始 pair 数严格满足 `total_pairs > 200`。

## 1. 实验目标与最终结论

Phase 7 的目标是将人类 tissuePMHC 项目的 `min_pairs` 与小鼠项目同步为 200，在不覆盖原 `min_pairs=500` 代码和结果的前提下，重新构建数据并复验关键模型 E2、E14、E17、E26 与 E29。

最终结论如下：

1. `min_pairs > 200` 将 benchmark 从 44 个 task 扩展到 157 个 task，train pair 从 48,486 增加到 73,798，HLA allele 从 12 个增加到 35 个。
2. E14a 相对 E2 提升 `+0.02663` mean AUROC；E17 的 3-seed rank ensemble 在 E14a 之上进一步提升 `+0.01069`。
3. E26 的 OOF greedy selection 最终只选择 E14 3-seed mean，相对 E17 仅提高 `+0.00007` AUROC，不构成有效的新模型收益。
4. E29 的 OOF CNN 相对匹配 E14 OOF 提升 `+0.01482` AUROC，并通过全部预设筛选条件。
5. E29 在固定测试集上达到 mean AUROC `0.84478`、mean AUPRC `0.83477`、mean MCC `0.55448`、worst-10 mean AUROC `0.68339`，是 Phase 7 的最终主模型。
6. E29 相对 E17 在测试集上提高 `+0.01024` AUROC，在 157 个 task 中 126 个提高、31 个下降；OOF 中观察到的表示增益在测试阶段得到方向一致的确认。
7. 冻结 E29 结构、超参数与三个 seed 后，E31 connected-component peptide-disjoint OOF 取得 mean task AUROC/AUPRC `0.76520/0.74522`。相对同一完整 train pair pool 上的 standard E29 OOF，AUROC/AUPRC 分别下降 `-0.06275/-0.06914`，表明 standard split 的实体重叠会明显抬高 unseen-peptide 泛化估计，但严格划分下仍保留高于随机水平的排序信号。

![Phase 7 模型进展](../results/figures_phase7/01_model_milestones.png)

## 2. 数据构建与 benchmark 扩展

Phase 7 使用与原人类项目相同的原始 pair 数据、标签定义和过滤规则，仅将 task 纳入门槛由 `total_pairs > 500` 改为 `total_pairs > 200`。仍为每个 task 固定抽取 100 个 test pair，每个 pair 含一条正例和一条配对负例。

输入清理阶段共读取 125,649 个 pair，删除 857 个缺少有效 tissue/HLA task 字段的 pair，保留 124,792 个 pair进入正式构建。最终数据如下：

| 项目 | min_pairs=500 | min_pairs=200 | 变化 |
|---|---:|---:|---:|
| Tissue-HLA tasks | 44 | 157 | 3.57 倍 |
| Tissues | 15 | 25 | 1.67 倍 |
| HLA alleles | 12 | 35 | 2.92 倍 |
| Train pairs | 48,486 | 73,798 | 1.52 倍 |
| Test pairs | 4,400 | 15,700 | 3.57 倍 |
| Train rows | 96,972 | 147,596 | 1.52 倍 |
| Test rows | 8,800 | 31,400 | 3.57 倍 |

每个 task 的 test 集均保持 100 个正负配对 pair，因此 task-macro 指标不会被大 task 的样本数直接支配。Phase 7 的 157 个 task 覆盖分布见下图；样本仍高度集中于少数常见 tissue-HLA 组合，低资源 task 虽满足门槛，但训练 pair 数远少于头部 task。

![Phase 7 数据覆盖](../results/figures_phase7/04_dataset_coverage.png)

## 3. 代码隔离与运行链

Phase 7 的入口位于 `phase7/`，数据写入 `data/tissuePMHC_phase7_min200/`，结果写入名称包含 `phase7_min200` 的独立目录。原 `min_pairs=500` 数据和结果没有被覆盖。

核心运行顺序为：

```powershell
& E:/ancd/envs/my_pytorch/python.exe d:/my_python/tissuePMHC/phase7/build_human_dataset_min200.py
& E:/ancd/envs/my_pytorch/python.exe d:/my_python/tissuePMHC/phase7/run_e2_shared_heads.py
& E:/ancd/envs/my_pytorch/python.exe d:/my_python/tissuePMHC/phase7/run_e14_auxiliary_soft_ensemble.py
& E:/ancd/envs/my_pytorch/python.exe d:/my_python/tissuePMHC/phase7/run_e17_seed_ensemble.py
& E:/ancd/envs/my_pytorch/python.exe d:/my_python/tissuePMHC/phase7/run_e26_all_in_one.py
& E:/ancd/envs/my_pytorch/python.exe d:/my_python/tissuePMHC/phase7/run_e29_multikernel_cnn_oof.py --device cuda
& E:/ancd/envs/my_pytorch/python.exe d:/my_python/tissuePMHC/phase7/run_e29_independent_test.py --device cuda
& E:/ancd/envs/my_pytorch/python.exe d:/my_python/tissuePMHC/phase7/run_e31_peptide_disjoint_oof.py --device cuda
```

E17 不训练新网络，而是依赖 E14 的 `branch_predictions.csv` 完成 seed aggregation。E26 在 3-fold OOF 中重新训练匹配候选，为 E29 提供 leakage-safe OOF baseline。E29 先完成 OOF 筛选；通过后，独立测试入口复用冻结的 OOF 决策，仅训练 full-train 3-seed 模型，不重复 OOF。

## 4. Phase 7 核心模型结果

| 实验 | 配置 | Mean AUROC | Mean AUPRC | Accuracy | MCC | Worst-10 AUROC |
|---|---|---:|---:|---:|---:|---:|
| E2 | Shared peptide encoder + task heads，3 seeds 均值 | 0.79723 | 0.78287 | 0.72771 | 0.45984 | 0.62994 |
| E14a | Global auxiliary + HLA plain，3 seeds 均值 | 0.82386 | 0.81172 | 0.75516 | 0.51319 | 0.65006 |
| E17 | E14a 3-seed rank average | 0.83455 | 0.82334 | 0.76551 | 0.53131 | 0.66663 |
| E26 | OOF greedy task-rank selection | 0.83462 | 0.82350 | 0.76465 | 0.52933 | 0.66712 |
| E29 | Multi-kernel CNN 3-seed mean | **0.84478** | **0.83477** | **0.77723** | **0.55448** | **0.68339** |

### 4.1 E2：共享编码器基线

E2 三个 seed 的 mean AUROC 为 `0.79723 ± 0.00143`，mean AUPRC 为 `0.78287 ± 0.00062`。该结果说明扩大到 157 个 task 后，共享 peptide encoder 仍能稳定训练，但最弱 task 的表现有限，worst-10 AUROC 仅为 `0.62994`。

### 4.2 E14：辅助监督与双分支

E14a 的 mean AUROC 为 `0.82386 ± 0.00125`，相对 E2 提升 `+0.02663`。E14b 的 mean AUROC 为 `0.82017 ± 0.00140`；E14a 在 471 个 seed-task 比较中 394 胜、77 负，因此继续选择 global auxiliary + HLA plain。

### 4.3 E17：多 seed 预测集成

E17 对三个 E14a seed 的 global/HLA 分支分别平均，再执行 task-wise rank fusion。最终 mean AUROC 为 `0.83455`，相对 E14a 三 seed 均值提高 `+0.01069`；157 个 task 中 154 个 AUROC 提高，仅 3 个下降。稳定性表中 ensemble 的标准差为零仅表示它是一个固定聚合点，不表示重复实验不存在不确定性。

### 4.4 E26：OOF greedy selection

E26 的候选包括 E14 final 与 E16 MC-dropout 的单 seed 和 3-seed mean。Greedy selection 最终只选择 `e14_final_3seed_mean`；没有第二个候选达到 `0.0001` OOF AUROC 最小增益。其测试 mean AUROC `0.83462` 与 E17 的 `0.83455` 几乎相同，说明高度同构的候选无法仅靠后处理形成新的有效信息。

### 4.5 E29：multi-kernel CNN

E29 使用 kernel size 2/3/5 的 position-preserving Conv1d peptide encoder，并保留 E14 的 global auxiliary 与 HLA plain 双分支。3-fold、3-seed OOF 结果为：

| OOF 模型 | Mean AUROC | Mean AUPRC | Worst-10 AUROC |
|---|---:|---:|---:|
| 匹配 E14 baseline | 0.81313 | 0.79800 | 0.64191 |
| E29 CNN | **0.82795** | **0.81436** | **0.65615** |
| E14 + E29 equal-rank fusion | 0.82508 | 0.81157 | **0.65680** |

E29 CNN 相对 OOF baseline 提高 `+0.01482` AUROC，157 个 task 中 144 个提高、13 个下降；与 baseline 的 task-rank correlation 为 `0.93795`，低于预设上限 0.97。全部四项 OOF gate 均通过。由于 standalone CNN 的 OOF mean AUROC 高于 equal-rank fusion，最终按冻结规则选择 E29 standalone 进入测试。

![E29 OOF 与测试确认](../results/figures_phase7/06_oof_test_confirmation.png)

## 5. E29 独立测试集验证

E29 的 full-train 测试结果如下：

| 模型 | Mean AUROC | Mean AUPRC | Accuracy | MCC | Worst-10 AUROC |
|---|---:|---:|---:|---:|---:|
| E17 3-seed | 0.83455 | 0.82334 | 0.76551 | 0.53131 | 0.66663 |
| E29 3-seed | **0.84478** | **0.83477** | **0.77723** | **0.55448** | **0.68339** |
| E29 − E17 | **+0.01024** | **+0.01142** | **+0.01172** | **+0.02317** | **+0.01677** |

逐 task AUROC 中，E29 有 126 胜、31 负；中位提升为 `+0.00980`。最大增益出现在 blood-HLA-A*11:01（`+0.06375`）、lung-HLA-C*12:03（`+0.05045`）和 blood-HLA-B*51:01（`+0.04900`）。最大下降为 lymphoid-HLA-B*15:02（`-0.02970`）、lymph node-HLA-B*44:02（`-0.02455`）和 lung-HLA-B*27:05（`-0.02175`）。

![E29 与 E17 逐任务比较](../results/figures_phase7/03_e29_vs_e17_task_gain.png)

三个 E29 单 seed 的 mean AUROC 为 `0.83449`、`0.83136` 和 `0.83299`；3-seed mean 达到 `0.84478`。这表明独立初始化的误差具有足够互补性，seed averaging 是 E29 最终性能的重要组成部分。

![E29 seed ensemble](../results/figures_phase7/05_e29_seed_ensemble.png)

## 6. min_pairs=500 与 min_pairs=200 的额外对比

为保持模型配置一致，本节均比较三个相同 seed（20260704/05/06）的 E2、E14a、E17 与 E29。结果分为三类：原 `min500` 的 44 个 task、Phase 7 中与原 benchmark 同名的 44 个 task，以及 Phase 7 全部 157 个 task。

| 模型 | min500：44 tasks | min200：共同 44 tasks | min200：全部 157 tasks | 表观总变化 |
|---|---:|---:|---:|---:|
| E2 | 0.79273 | 0.79358 | 0.79723 | +0.00449 |
| E14a | 0.81158 | 0.81823 | 0.82386 | +0.01228 |
| E17 | 0.82428 | 0.83028 | 0.83455 | +0.01027 |
| E29 | 0.83415 | 0.83709 | 0.84478 | +0.01064 |

![min_pairs=500 与 min_pairs=200](../results/figures_phase7/02_minpairs_500_vs_200.png)

这组差异不能解释为“降低 min_pairs 导致模型性能提高”，原因有两点：

1. **任务组成变化。** 新增 113 个 task；其宏平均难度并不一定高于原 44 个 task。E2 的共同 44-task 变化只有 `+0.00085`，而新增 task 的组成效应贡献约 `+0.00365`，说明 E2 的大部分表观提升来自任务构成。
2. **共同 task 的测试 pair 也被重新抽样。** 数据构建器按 task 顺序连续消耗同一个随机数流；加入新 task 后，后续 task 的抽样状态发生改变。原 44 个 task 在两版中各有 100 个 test pair，但平均仅重合 12.41 个，最少 1 个、最多 23 个，没有任何 task 的 test 集完全相同。因此“共同 44-task”也不是固定测试样本上的严格配对比较。

按 AUROC 分解，表观总变化可写为“共同 task 变化 + 新任务组成变化”：

| 模型 | 共同 44-task 变化 | 新任务组成变化 | 表观总变化 |
|---|---:|---:|---:|
| E2 | +0.00085 | +0.00365 | +0.00449 |
| E14a | +0.00665 | +0.00563 | +0.01228 |
| E17 | +0.00600 | +0.00427 | +0.01027 |
| E29 | +0.00295 | +0.00769 | +0.01064 |

### 6.1 宏平均的任务组成效应是主要原因之一

`min_pairs` 不是模型内部的正则化参数；它只决定哪些 tissue-HLA task 被纳入训练和评估。Phase 7 新增了 113 个 task，占 157-task 宏平均的 `71.97%`。宏平均对每个 task 赋予相同权重，因此新增 task 即使训练 pair 较少，也会共同决定最终分数：

\[
M_{157}=\frac{44}{157}M_{\mathrm{common44}}+\frac{113}{157}M_{\mathrm{new113}}.
\]

新增 113 个 task 在四种模型上都比 Phase 7 中的共同 44 个 task 更容易：

| 模型 | Phase 7 共同 44-task AUROC | 新增 113-task AUROC | 新增 task − 共同 task |
|---|---:|---:|---:|
| E2 | 0.79358 | 0.79865 | +0.00507 |
| E14a | 0.81823 | 0.82605 | +0.00783 |
| E17 | 0.83028 | 0.83621 | +0.00593 |
| E29 | 0.83709 | 0.84778 | +0.01068 |

因此，降低门槛后加入的 task 并没有整体拉低宏平均，反而提高了它。E2 表观总增益的 `81.12%`、E29 的 `72.28%` 可由“Phase 7 全部 task 均值高于共同 44-task 均值”这一组成效应解释。这里的“解释”是算术分解，不是对单个 task 难度来源的因果判断。

### 6.2 pair 数较少不等于分类边界更困难

在 Phase 7 的 157 个 task 中，train pair 数与 AUROC 的 Spearman 相关均很弱且略为负：E2 为 `-0.082`、E14a 为 `-0.107`、E17 为 `-0.103`、E29 为 `-0.109`。只看新增 113 个 task 时，相关系数仍仅为 `-0.088` 至 `-0.114`。这说明当前 benchmark 中，“样本较少”与“AUROC 更低”没有单调关系。

例如，按 train pair 数划分的第二四分位 task（中位数约 208 pairs）在 E29 上的 mean AUROC 为 `0.87375`，反而高于最大样本四分位的 `0.82996`。潜在原因包括：部分新增 tissue-HLA 组合的阳性 peptide motif 更集中、组织条件信号更强，或配对负例构造在这些 task 上形成了更清晰的排序边界。现有结果只能证明这些新增 task 在当前标签与负例协议下更易区分，不能区分生物学可分性与数据构造效应。

### 6.3 更多共享训练数据可能帮助 E14/E17，但证据不是严格因果

从 `min500` 到 `min200`，train rows 从 96,972 增加到 147,596，HLA allele 从 12 个增加到 35 个，tissue 从 15 个增加到 25 个。共同 44 个 task 自身的可用 pair 总量基本不变，但共享 peptide encoder、辅助 tissue/HLA 监督和跨 task 参数会看到更多任务与更多样化的训练样本。

共同 44-task 的表观 AUROC 变化在 E2 上只有 `+0.00085`，在 E14a/E17 上分别为 `+0.00665` 和 `+0.00600`，在 E29 上为 `+0.00295`。这一模式与“辅助监督和多分支结构更能利用额外跨 task 信息”相容；E17 又通过 seed averaging 降低了训练方差。但由于共同 task 的 test pair 已被重新抽样，这些数值不能单独证明跨 task 数据共享造成了提升。

### 6.4 test redraw 足以造成明显的共同-task 波动

共同 44 个 task 的新旧 test 集平均仅重合 `12.41%`。在逐 task 的 `min200 − min500` AUROC 差值中，标准差约为 E2 `0.0351`、E14a `0.0373`、E17 `0.0390`、E29 `0.0326`，远大于共同 task 的平均变化。共同 task 的胜负也并非单向：E2 为 25 胜/19 负，E14a 为 24/20，E17 为 23/21，E29 为 28/16。这说明 test redraw 带来的 task 级抽样波动不可忽略。

测试 pair 重合率与 AUROC 变化的 Spearman 相关接近零（E2/E14a/E17 分别为 `0.040`、`0.036`、`0.011`，E29 为 `-0.104`）。这并不表示 redraw 没有影响，而是表示仅凭重合比例无法预测变化方向；被替换 pair 的具体难度比替换数量更重要。

### 6.5 综合判断

模型得分提高应按以下顺序解释：

1. **新增 task 的宏平均组成效应。** 新增 113 个 task 占最终权重的 71.97%，且在所有模型上平均更容易；这是 E2 和 E29 表观提升的主要组成部分。
2. **额外跨 task 训练信息。** 更多 tissue、HLA 和 peptide 样本可能改善共享表示，E14/E17 在共同 task 上的变化与该机制相容。
3. **共同 task 的 test redraw。** 平均仅 12.41% test pair 重合，使共同-task 对比混入明显抽样差异，无法直接归因于训练集扩展。
4. **模型结构对扩展 benchmark 的适配。** E14 的辅助监督、E17 的 seed averaging 和 E29 的多尺度局部 motif 编码都比 E2 更能利用新增数据；但这些是模型间相对提升的原因，不是 `min_pairs` 本身提高分数的证明。

因此，严谨表述应为：**在重新构建的 `min_pairs > 200` benchmark 上，四个关键模型的 task-macro 分数均高于原 `min_pairs > 500` benchmark；提升主要由新增 task 的难度组成、更多共享训练数据和 test redraw 共同产生，不能解释为降低 `min_pairs` 对模型性能的直接因果增益。**

若要严格隔离 `min_pairs` 的影响，应固定原 44 个 task 的 test pair ID，并分别用 `>500` 与 `>200` 的训练 task pool 训练，然后只在完全相同的 44-task test 集上评估。未来数据构建可改用每个 task 独立的稳定 hash seed，避免新增 task 改变其他 task 的 split。

## 7. E31：人类 peptide-disjoint robustness

### 7.1 冻结协议与 split audit

E31 不读取 Phase 7 standard test，只使用 `tissuePMHC_phase7_min200_train.csv.gz`。模型固定为 E29 multi-kernel CNN 的 global-auxiliary/HLA-plain 双分支 rank fusion，使用 seeds `20260704/05/06`、3 folds、25 epochs；结构、超参数和集成规则均未根据 strict 指标调整。

划分以 pair 为原子：将共享任一 peptide 的 pairs 合并为 connected components，再把完整 component 确定性分配到三个 fold。审计结果如下：

- 147,596 rows、73,798 pairs、59,983 unique peptides；
- 17,036 components，其中 10,923 个包含多个 pair，最大 component 为 755 pairs；
- 三个 fold 均覆盖全部 157 tissue–HLA tasks；
- 每折 held-out 24,598–24,600 pairs，fitting 49,198–49,200 pairs；
- 每 task、每折 held-out 33–2,155 pairs，fitting 67–4,310 pairs；
- 所有 fold 的 global peptide overlap 和 pair overlap 均为 0。

该协议是 seen-task/unseen-peptide OOF；它不是 protein-disjoint、unseen-HLA、study-disjoint 或外部队列验证。

### 7.2 总体结果与 standard OOF gap

| Evaluation | Accuracy | Mean AUROC | Mean AUPRC | F1 | MCC | Worst-10 AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Standard pair-grouped E29 OOF | 0.76051 | 0.82795 | 0.81436 | 0.76319 | 0.52143 | 0.65616 |
| E31 peptide-disjoint E29 OOF | 0.69801 | 0.76520 | 0.74522 | 0.70242 | 0.39649 | 0.63817 |
| Strict − standard | -0.06250 | **-0.06275** | **-0.06914** | -0.06077 | -0.12494 | -0.01799 |

两项 OOF 使用相同的完整 train pair pool、相同模型、seeds 和 fold 数，差别是 E31 将共享 peptide 的完整 component 锁定在同一 fold。strict AUROC 保留 standard OOF 的 `92.42%`，AUPRC 保留 `91.51%`；这说明模型对未见 peptide 仍保留信号，但不能把 standard OOF 的 0.82795 解释为严格 unseen-peptide 性能。

以 157 个 task 为重采样单位、固定 seed `20260711` 进行 10,000 次 nonparametric bootstrap，strict AUROC 的 95% CI 为 `[0.75512, 0.77519]`，AUPRC 为 `[0.73518, 0.75531]`；配对 strict − standard AUROC gap 的 95% CI 为 `[-0.07283, -0.05336]`，AUPRC gap 为 `[-0.07989, -0.05880]`。这些区间刻画 task 间不确定性，不包含重新训练、seed 或 split 选择带来的额外方差。

三个单 seed 的 mean task AUROC 分别为 `0.75014`、`0.75011` 和 `0.75136`，三 seed ensemble 达到 `0.76520`，相对单 seed 均值提高 `+0.01466`，说明 seed averaging 在 strict split 下仍有稳定收益。

### 7.3 Task 与 HLA 异质性

逐 task 的 strict − standard AUROC 中位数为 `-0.04836`，四分位区间为 `[-0.09001, -0.01794]`；157 tasks 中 12 个上升、145 个下降，standard/strict task AUROC Pearson 相关为 `0.69325`。按 HLA locus 描述性汇总：

| HLA locus | Tasks | Standard AUROC | Strict AUROC | ΔAUROC |
|---|---:|---:|---:|---:|
| HLA-A | 55 | 0.85405 | 0.78244 | -0.07161 |
| HLA-B | 79 | 0.82202 | 0.75070 | -0.07132 |
| HLA-C | 23 | 0.78589 | 0.77376 | -0.01212 |

最大下降集中于若干 standard OOF 原本很高的 HLA-A*11:01 与 HLA-B*15:01 tasks，例如 small intestine–HLA-A*11:01（`-0.23113`）、thyroid–HLA-A*11:01（`-0.22724`）和 esophagus–HLA-A*11:01（`-0.22126`）。这些是描述性异质性结果；未做 task-paired bootstrap、置信区间或多重检验前，不解释为 locus 的生物学因果差异。

### 7.4 运行与产物

GPU 总耗时为 `5,425.718 s`（`1h 30m 26s`）。三个 seed 分别耗时 `29m 59s`、`29m 54s` 和 `30m 10s`；单 fold 为 `9m 57s` 至 `10m 8s`。完整 epoch 日志、fold/seed/total timing、split manifest、成员预测、ensemble 预测与逐 task 指标均保存在 `results/tissuePMHC_phase7_min200_e31_peptide_disjoint_oof/`。

## 8. 研究边界与局限

1. Phase 7 的“独立测试”是相对于 E29 的 OOF 模型选择而言；它仍是项目内固定 standard test，不是外部队列。
2. standard split 未强制 peptide-disjoint、protein-disjoint 或 unseen-HLA；结果不能直接解释为对新 peptide、新 parent protein 或新 HLA 的外推能力。
3. 每个 task 的 test 样本仅包含 100 个正负 pair，单 task AUROC 仍存在抽样方差；应优先解释宏平均、worst-10 和广泛的 task 胜负方向。
4. Phase 7 已多次在同一 standard test 上报告模型结果。E29 的结构晋级由 OOF gate 决定，降低了直接 test 调参风险，但整个研究链仍应明确区分开发结论与真正外部验证。
5. E30 的 protein-disjoint OOF 回答的是另一项泛化问题，不是同步 `min_pairs` 所必需的核心实验，因此未纳入 Phase 7 主结果。
6. E31 已回答冻结 E29 在 seen-task/unseen-peptide 条件下的 robustness，并已给出 task-bootstrap 95% CI；但尚未在相同 strict folds 上补齐 traditional/pMHC/shared-encoder baselines、PairAcc、核心消融和包含重新训练方差的敏感性分析，因此不能声称 E29 在 strict 协议下优于所有基线。

## 9. 最终结论与交付物

Phase 7 已完成最初目标：人类项目使用与小鼠项目一致的 `min_pairs=200` 门槛，代码、数据和结果均与原 `min_pairs=500` 项目隔离。E29 multi-kernel CNN 3-seed mean 是最终主模型，其 fixed-test mean AUROC 为 `0.84478`，相对 E17 提升 `+0.01024`，且最弱任务组也同步改善。新增 E31 表明同一冻结模型在严格 peptide-disjoint OOF 下仍达到 `0.76520` AUROC，但相对 standard OOF 下降 `0.06275`；因此论文应同时保留 standard benchmark 主结论，并明确实体重叠造成的 unseen-peptide 泛化边界。

报告配套可视化位于 `results/figures_phase7/`：

- `01_model_milestones.png`：Phase 7 模型进展；
- `02_minpairs_500_vs_200.png`：min_pairs 对比与 benchmark 扩展；
- `03_e29_vs_e17_task_gain.png`：逐 task 增益分布；
- `04_dataset_coverage.png`：157 个 task 的 coverage heatmap；
- `05_e29_seed_ensemble.png`：E29 seed averaging；
- `06_oof_test_confirmation.png`：OOF 筛选与测试确认；
- `tissuepmhc_phase7_figures.pdf`：全部六张图的合并 PDF；
- 对应 CSV：每张图的可复算源数据。

E31 配套产物位于 `results/tissuePMHC_phase7_min200_e31_peptide_disjoint_oof/`：`metadata.json`、`split_audit.json`、`pair_fold_assignments.csv`、`member_oof_predictions.csv`、`ensemble_oof_predictions.csv`、`per_task_metrics.csv`、`summary_metrics.csv`、`timing.csv` 与 `run.log`。

所有图由 `scripts/build_tissuepmhc_phase7_figures.py` 从冻结结果重新生成，不训练模型，也不修改实验输出。
