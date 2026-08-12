# 教授批注定位清单（2026-07-23）

## 定位基准

- 论文源文件：`paper.tex`
- 截图中的正文页码：第 18–23 页及附录第 30 页
- 下列行号对应 2026-07-23 当前工作区版本；后续编辑 `paper.tex` 后行号可能移动。
- “支撑位置”表示论文中已有、可用于解释该批注的定义或结果，不一定是教授落笔的页面。

## 截图 1：第 18 页（RQ1 与 RQ2 开头）

| 教授批注 | 直接位置 | 关联/支撑位置 | 定位说明 |
|---|---:|---:|---|
| “RQ1，这个格式有点问题。” | `paper.tex:424` | `paper.tex:442,507,519,602` | RQ1 小节标题及其与其他 RQ 标题的格式一致性问题。 |
| “这个表太小，增加几列不同的指标，比如 AUPRC、Sensitivity、Specificity、median AUROC……” | `paper.tex:428–438` | `paper.tex:97–110,390–408` | RQ1 的三行、三列表；当前只有 Model、Tasks、Mean AUROC。指标定义集中在 Evaluation Protocol。 |
| “表没有编号” | `paper.tex:428–438` | `paper.tex:572–596,767–787` | 该表只是 `center + tabular`，没有 `table`、`\caption`、`\label`；后两处是已有编号表的写法。 |
| “有提和现有方法的比较吗？” | `paper.tex:426–440` | `paper.tex:384,570–600` | RQ1 这里只比较 one-hot logistic regression 和 shared encoder；更完整的传统/神经基线位于严格折上的架构比较。 |
| “增加几行：建议参考早期报告的 baseline” | `paper.tex:431–435` | `paper.tex:384,581–590,612–616` | 需要扩充的是 RQ1 表格的数据行；当前论文其他位置已有 BLOSUM62 RF、dual branch、MMoE 等候选基线。 |
| “模型名建议去掉 Net 后缀，统一为 TissuePMHC” | 首次定义 `paper.tex:11,20,41`；本页 `paper.tex:444,455,470,478–485,488–500` | 全文所有旧模型名 | 这是全文级命名修改，不只涉及第 18 页。 |
| “增加更多的比较方法，增加更多的行” | `paper.tex:446–458` | `paper.tex:384,570–600` | RQ2 主结果表目前只有 4 个模型；严格架构表有更完整模型清单。 |

### 对应解答与处理结果

1. **“RQ1，这个格式有点问题。”**  
   已修改。RQ1 标题改为问题式标题 **“Is Tissue-specific Preference Learnable?”**，并与 RQ2–RQ5 的研究问题结构保持一致。

2. **“这个表太小，增加几列不同的指标，比如 AUPRC、Sensitivity、Specificity、median AUROC……”**  
   已将原三行小表扩展为正式的 **Table 1**，增加 AUPRC、Worst-10 AUROC 和 Seeds，并将表格字号、行高和 caption 间距统一调整。没有补写 Sensitivity 和 Specificity：早期 44-task 实验没有保存所有模型在统一决策阈值下的混淆矩阵，AUROC/AUPRC 不能反推出这两个指标。Median AUROC 被加入 157-task 主结果表（Table 2）；44-task 历史归档缺少所有 baseline 的可比逐任务中位数，因此未在 Table 1 中虚构该列。

3. **“表没有编号。”**  
   已解决。该表现在使用正式 `table` 环境，具有 `\caption`、`\label` 和正文交叉引用，编号为 **Table 1**。

4. **“有提和现有方法的比较吗？”**  
   已补充。Table 1 现在汇总当前项目中实际完成并可比较的历史 baseline，包括 one-hot logistic regression、HLA pseudo-sequence conditioning、FAMO、pair-ranking、CAGrad、DB-MTL、shared encoder、MMoE、auxiliary supervision、soft dual-branch ensemble 和 auxiliary dual-branch ensemble。NetMHCpan、MHCflurry、BigMHC 等外部方法仍只在 Related Work 中定性比较，因为没有它们在同一 matched benchmark、相同 split 和相同 task inventory 下的 held-out predictions，直接并列数值会形成不公平比较。

5. **“增加几行：建议参考早期报告的 baseline。”**  
   已解决。Table 1 已扩展为 11 个历史 baseline，并在 caption 中说明它们属于原始 44-task standard pair-disjoint benchmark，避免与后续 157-task benchmark 混淆。

6. **“模型名建议去掉 Net 后缀，统一为 TissuePMHC。”**  
   已完成全文修改。标题、摘要、Methods、Results、Discussion、表格、图注和交叉引用中的模型名称均统一为 **TissuePMHC**。

7. **“增加更多的比较方法，增加更多的行。”**  
   已在 RQ1 的历史 baseline 表中增加可验证方法；RQ2 的 157-task Table 2 保留四个具有相同任务清单和评估构造的主要受控模型，以避免把不同任务范围或不同 held-out pools 的结果混入主排名。

## 截图 2：第 19 页（Figure 1、matched OOF 表、RQ3 开头）

| 教授批注 | 直接位置 | 关联/支撑位置 | 定位说明 |
|---|---:|---:|---|
| “图片要复杂化，不能这么简单。” | `paper.tex:460–486` | — | Figure 1 的 PGFPlots 柱状图、图注和标签。 |
| “跟前面的表格对应；增加更多方法；建议 boxplot/violin plot；如果想显示 range” | 数据表 `paper.tex:446–458`；图 `paper.tex:460–486` | 单种子结果 `paper.tex:511–515`；任务级结果来源见 `results/` | 现图用每个方法的单个汇总值画柱状图，没有任务分布、种子分布或误差范围。 |
| “可以按照 tissue/分型进一步细化结果” | `paper.tex:460–486` | `paper.tex:670–704` | 直接批注落在 Figure 1；论文现有 Task Heterogeneity 和 Branch Complementarity 可作为细化入口。 |
| “为什么这个是 baseline？” | `paper.tex:490`；表中名称 `paper.tex:499` | baseline 定义 `paper.tex:384` | 指的是 “matched auxiliary dual-branch baseline / Matched auxiliary branch”。方法部分说明了 auxiliary MLP dual branch 的结构，但此处没有就“为何选它为 matched baseline”单独解释。 |
| “结果在哪里？” | `paper.tex:505` | 详细结果 `paper.tex:519–600`；完整统计表 `paper.tex:767–787` | 第 505 行只再次报告 strict AUROC/AUPRC 并指向 RQ4，完整 strict 结果在 RQ4。 |

### 对应解答与处理结果

1. **“图片要复杂化，不能这么简单。”**  
   已重新设计 Figure 1。新图同时呈现 Mean AUROC、Median AUROC 和 Worst-10 AUROC，使图形既展示总体性能，也展示任务分布中心和低尾表现，而不再只是重复三个汇总值。

2. **“跟前面的表格对应；增加更多方法；建议 boxplot/violin plot；如果想显示 range。”**  
   Figure 1 已与 Table 2 的主要模型和指标对齐。没有强行生成完整 boxplot/violin plot，因为不同历史方法没有保存完全一致的逐任务、逐种子 held-out predictions；强行合并会混合 44-task、157-task、fixed-test 和 OOF 样本池。当前改用能够由现有数据可靠复核的 Mean、Median 和 Worst-10 摘要。

3. **“可以按照 tissue/分型进一步细化结果。”**  
   已在后续 Analysis/RQ5 中新增人类 tissue-level extremes、HLA locus、小鼠 tissue-level extremes 和 H2 restriction 分层表。单任务 tissue 的结果明确标注为描述性结果，避免将极小样本分组解释成稳定效应。

4. **“为什么这个是 baseline？”**  
   正文已补充解释：matched auxiliary dual-branch baseline 与最终 TissuePMHC 共享双分支监督和融合设定，主要差别是 MLP encoder 与 position-preserving multi-kernel encoder，因此它提供了针对编码器替换的受控比较，而不是随意选择的 baseline。

5. **“结果在哪里？”**  
   已删除 RQ1/RQ2 对同一 strict 数值的重复报告，并将 globally unseen-peptide 的完整结果集中到 RQ4。RQ2 仅保留桥接说明，并通过交叉引用指向 RQ4、正式比较表和完整配对统计表。

## 截图 3：第 19 页后半（RQ3 Component Contributions）

| 教授批注 | 直接位置 | 关联/支撑位置 | 定位说明 |
|---|---:|---:|---|
| “引用对应的表格” | `paper.tex:509–515` | 主结果表 `paper.tex:446–458`；matched OOF 表 `paper.tex:492–503`；strict 架构表 `paper.tex:572–596` | RQ3 三段消融结论目前都直接写数值，没有 `Table~\ref{...}`。前两个 RQ2 表也没有 label，暂时无法交叉引用。 |

### 对应解答与处理结果

1. **“引用对应的表格。”**  
   已新增正式的 component evidence 表，并在 RQ3 正文中交叉引用。该表集中呈现 multi-kernel encoder、branch complementarity、fixed rank fusion 和 seed averaging 四类系统级证据。正文同时明确：现有实验支持 standard benchmark 上的系统级贡献，但不能在没有相同 strict folds 隔离消融的情况下声称 strict protocol 下的完整组件排名。

## 截图 4：第 20 页（RQ4 开头与标准/严格结果表）

| 教授批注 | 直接位置 | 关联/支撑位置 | 定位说明 |
|---|---:|---:|---|
| “OOF 啥意思？好像没有必要出现，pair-disjoint？” | `paper.tex:523,528–531` | 三种评估设置 `paper.tex:195–199`；Evaluation Protocol `paper.tex:366–408` | OOF 是 out-of-fold。论文在第 195 行以全称加缩写语境首次介绍，但 RQ4 表格直接使用缩写；pair-disjoint 与 peptide-disjoint 是两种不同拆分约束。 |
| “结果重复？” | `paper.tex:523–534` | `paper.tex:440,490–505` | human strict 0.7652/0.7452 已在 RQ1、RQ2 出现；RQ4 再次出现是为了 matched standard-vs-strict 差值比较。 |
| “小鼠上的完整结果？” | `paper.tex:531` | mouse strict 表 `paper.tex:588–590`；RQ5 `paper.tex:602–664` | RQ4 开头表只列 AUROC；小鼠 AUROC/AUPRC/Worst-6/PairAcc 及标准结果分散在 strict 架构表和 RQ5。 |

### 对应解答与处理结果

1. **“OOF 啥意思？好像没有必要出现，pair-disjoint？”**  
   已在 Methods 和 RQ4 前置定义：OOF 是 **out-of-fold**。standard pair-disjoint OOF 只阻止同一 matched pair ID 同时出现在 fitting 和 held-out 数据中；connected-component peptide-disjoint OOF 则在全局范围隔离 peptide identity。保留 OOF 是为了区分交叉验证折外预测与一次性 fixed test，但首次出现均使用全称。

2. **“结果重复？”**  
   已解决。重复的 0.7652/0.7452 从 RQ1/RQ2 删除，完整 standard-versus-strict 结果和解释集中在 RQ4；其他章节只通过交叉引用指向 RQ4。

3. **“小鼠上的完整结果？”**  
   已补齐。RQ4/RQ5 现在报告小鼠 standard OOF、fixed test 和 peptide-disjoint OOF，并包含 AUROC、AUPRC、Worst-6、PairAcc 等现有可验证指标；同时增加 H2 restriction 和 tissue 分层结果。

## 截图 5：第 20 页（Figure 2 与 RQ4 统计）

| 教授批注 | 直接位置 | 关联/支撑位置 | 定位说明 |
|---|---:|---:|---|
| “对角线用红点？这个是标准操作吗？” | Human `paper.tex:552`；Mouse `paper.tex:555` | 图注 `paper.tex:558` | 源码实际画的是黑色虚线 `y=x` identity line，不是红点；截图中的红色方块是虚线的渲染/显示效果。该线用于区分 strict 高于或低于 standard。 |
| “peptide-disjoint 和 pair-disjoint 两个词有误导；需讨论 unseen in specific task 与 unseen before in any task” | 图轴/图注 `paper.tex:545–558`；解释 `paper.tex:562–566` | 协议定义 `paper.tex:195–199`；限制 `paper.tex:740` | 当前 strict 是全局 peptide entity 不跨折，因此是“该肽在任何任务的训练折中都未出现”，同时任务本身是 seen task；standard pair-disjoint 仅保证 pair ID 不重叠。 |

### 对应解答与处理结果

1. **“对角线用红点？这个是标准操作吗？”**  
   已修改 Figure 2。\(y=x\) identity reference 改为清晰的灰色粗虚线，并在图注明确说明它是参考线而不是实验数据点。该线用于判断 peptide-disjoint AUROC 相对 matched standard OOF AUROC 是上升还是下降。

2. **“peptide-disjoint 和 pair-disjoint 两个词有误导；需讨论 unseen in specific task 与 unseen before in any task。”**  
   已在协议定义、RQ4 正文、坐标轴和图注中统一澄清：strict peptide-disjoint 表示 peptide 在任何任务的 fitting folds 中都未出现，即 **globally unseen peptide**；但被评估的 tissue–MHC task 本身仍是 seen task。它不是只对特定 task 未见，也不是 unseen-task 或 protein-disjoint evaluation。

## 截图 6：第 21 页（RQ5 小鼠主表）

| 教授批注 | 直接位置 | 关联/支撑位置 | 定位说明 |
|---|---:|---:|---|
| “表格太小” | `paper.tex:606–619` | — | 小鼠 train-only OOF 表，目前 5 个方法、3 个性能指标及 Evidence。 |
| “更多的指标、更多的方法” | `paper.tex:608–618` | 完整 strict 方法/指标 `paper.tex:572–596`；指标定义 `paper.tex:390–408` | 可扩充 Accuracy、MCC、PairAcc 等，或补齐其他已完成 mouse baselines；需以已有 held-out predictions 为准。 |
| “按照组织 / 按照分型” | `paper.tex:606–619` | H2 分组结果 `paper.tex:660`；task heterogeneity 分析结构 `paper.tex:670–694` | 教授希望小鼠结果进一步按 tissue 和 H2 restriction 分层展示。当前只在第 660 行给了 H2 group AUROC。 |

### 对应解答与处理结果

1. **“表格太小。”**  
   已统一增加 caption 与表体间距和表格行高；该表保持可读字号，不再使用强制整体缩放。

2. **“更多的指标、更多的方法。”**  
   小鼠 baseline 表已扩展为 BLOSUM62 random forest、shared encoder、Factorized MMoE、H2-Kk residual adapter 和 five-seed Factorized MMoE，并报告 AUROC、AUPRC、Worst-6 和 Evidence。另一个协议汇总表补充 standard OOF、fixed test、peptide-disjoint OOF 以及 PairAcc。仅加入具有现成、同协议 held-out 结果的方法，未补造缺失指标。

3. **“按照组织 / 按照分型。”**  
   已新增 mouse tissue-level extremes 表和 H2-Db/H2-Kb/H2-Kd/H2-Kk 分层表。正文明确这些分组结果是描述性分析，不把任务量很少的分组解释为受控生物学效应。

## 截图 7：第 22 页（Figure 3、OOF 与 fixed-test 表）

| 教授批注 | 直接位置 | 关联/支撑位置 | 定位说明 |
|---|---:|---:|---|
| “为啥有这个东西？” | 对比表 `paper.tex:623–632`；Figure 3 `paper.tex:634–658` | 引入 `paper.tex:621`；解释 `paper.tex:660–664` | 该表/图用于比较 train-only OOF 与冻结模型的一次性 internal fixed test，意图是做标准 pair-disjoint 的内部确认；第 662 行同时承认 fixed test 有显著实体重叠。 |

### 对应解答与处理结果

1. **“为啥有这个东西？”**  
   原 Figure 3 已删除，因为它只重复相邻表格中的三个汇总数值，没有提供分布或不确定性信息。保留并扩展了协议比较表，用来区分 train-only OOF、冻结模型的一次性 fixed test 和 peptide-disjoint OOF；正文同时说明 fixed test 存在显著 train/test entity overlap，因此只能确认 internal standard protocol 下的可学习性，不能证明严格实体泛化或模型迁移。

## 截图 8：第 23 页（HLA locus 描述性表）

| 教授批注 | 直接位置 | 关联/支撑位置 | 定位说明 |
|---|---:|---:|---|
| “表格太小” | `paper.tex:676–690` | 更完整 task 统计 `paper.tex:670–694` | 表只有 HLA-A/B/C 三行，且无编号、caption、label；比较的是 original fixed 与 strict OOF，属于非匹配样本池的描述性比较。 |

### 对应解答与处理结果

1. **“表格太小。”**  
   已改为正式编号表，增加 caption、label、正文引用、caption 间距和表格行高。正文进一步说明 original fixed test 与 strict OOF 使用不同 held-out pools，因此该表的 Difference 只能作描述性比较，不能解释为 peptide-overlap effect；HLA-A/B/C 还包含不同 allele、tissue、训练规模和 peptide-component 结构，也不能解释为受控 locus 效应。

## 截图 9：附录 Table 1（paired statistics）

| 教授批注 | 直接位置 | 关联/支撑位置 | 定位说明 |
|---|---:|---:|---|
| “这个 table 是有编号的，但格式好像有问题。” | `paper.tex:767–787` | 正文引用 `paper.tex:564` | 这是 `tab:paired-stats`，有 caption/label；使用 `\resizebox{0.98\linewidth}{!}` 压缩 8 列，导致字号和可读性问题。表格在附录中因重新编号显示为 Table 1。 |

### 对应解答与处理结果

1. **“这个 table 是有编号的，但格式好像有问题。”**  
   已取消 `resizebox`，将 Human 和 Mouse 拆成两个面板，使用统一的列距、行高和可读字号。表格现在正式编号为 Table 12，并保留 Mean difference、Median difference、Hodges–Lehmann difference、W/T/L、95% CI 和 BH-adjusted Wilcoxon \(p\) 值。拆分面板避免了整张八列统计表被强制缩得过小。

## 截图 10：附录 Figure 4

| 教授批注 | 直接位置 | 关联/支撑位置 | 定位说明 |
|---|---:|---:|---|
| “这个是啥意思？”（上图） | `paper.tex:789–810` | 引导段 `paper.tex:765`；图注 `paper.tex:832` | 上图是按 HLA locus/H2 restriction 汇总的 `peptide-disjoint OOF − matched standard OOF` 平均 AUROC；负值表示严格拆分后下降。 |
| “这个是啥意思？”（下图） | `paper.tex:814–831` | 引导段 `paper.tex:765`；图注 `paper.tex:832` | 下图是每折 held-out 集合中的唯一 parent protein 与 fitting partition 的重叠比例；用来说明 peptide-disjoint 并不等于 protein-disjoint。 |

### 对应解答与处理结果

1. **“这个是啥意思？”（上图）**  
   已重写图前导说明、标题、纵轴和图注。上图表示各 HLA locus/H2 restriction 下的平均泛化差值：
   \[
   \Delta_{\mathrm{AUROC}}
   =
   \mathrm{AUROC}_{\text{peptide-disjoint OOF}}
   -
   \mathrm{AUROC}_{\text{matched standard OOF}}.
   \]
   柱子为负表示在隔离全局 peptide identity 后 AUROC 下降；它量化的是 peptide-separation generalization gap。

2. **“这个是啥意思？”（下图）**  
   已明确为每折 held-out partition 中，唯一 parent protein 同时出现在 fitting partition 的比例。它用于说明：peptide-disjoint 只隔离 peptide identity，并不隔离整个 parent protein，因此 peptide-disjoint 不等于 protein-disjoint。图注还明确 `standard` 指 matched standard OOF control，而不是 original fixed test。

## 跨页问题汇总

1. **表格体系不统一**：RQ1、RQ2 两张表、RQ4 开头表、RQ5 两张表和 HLA locus 表均使用裸 `center + tabular`，没有统一编号、caption 和 label；只有 strict architecture 表与附录统计表采用正式 `table` 环境。
2. **结果重复但叙事目的未显式区分**：0.7652/0.7452 依次出现在 RQ1、RQ2、RQ4；若保留，应分别明确“learnability summary / bridge to strict analysis / matched comparison”，否则合并。
3. **术语需要前置且精确定义**：OOF、standard pair-disjoint、matched standard OOF、connected-component peptide-disjoint、fixed test 在 Results 中密集出现，但清晰定义主要位于 Methods。
4. **图表信息密度不足或展示类型不匹配**：Figure 1 和 Figure 3 只显示汇总柱状值；教授希望看到更多方法、更多指标、分布/range，以及 tissue/MHC 分层。
5. **完整基线其实已存在但分散**：最完整的人/鼠 strict 模型比较位于 `paper.tex:572–600`，Methods 的模型清单位于 `paper.tex:384`；可作为重组主结果表的来源。
6. **命名修改是全文级任务**：旧模型名统一为 `TissuePMHC`，涉及标题、摘要、方法、结果、讨论、图表标签与图注，不能只改截图所在页。

## 跨页问题的最终处理

1. **表格体系不统一**：已将新增和修改的结果表统一为正式 `table`、caption、label 和正文交叉引用；统一增加 caption 间距和行高，并取消会严重缩小字号的 `resizebox`。
2. **结果重复**：strict globally unseen-peptide 结果集中到 RQ4，RQ1/RQ2 只保留必要的交叉引用。
3. **术语定义滞后**：OOF、matched standard OOF、pair-disjoint、connected-component peptide-disjoint、fixed test 和 globally unseen peptide 已在 Methods 与 Results 首次使用处明确解释。
4. **图表信息密度不足**：Figure 1 增加 Mean/Median/Worst-tail 信息；Figure 2 强化 identity reference；原 Figure 3 因重复表格数据而删除；新增 tissue、HLA/H2 和协议分层表。
5. **基线分散**：44-task 历史 baseline、157-task 主模型、小鼠 baseline 和 strict protocol 结果已按研究问题重新组织，不再将不同任务范围和样本池强行混为同一排名。
6. **全文命名**：旧模型名已统一替换为 `TissuePMHC`。

## 当前版本复查结果

- 当前 Overleaf 第二版共 33 页。
- LaTeX Errors：0。
- LaTeX Warnings：0。
- Overfull boxes：0。
- 剩余 Underfull 提示均来自窄列自动换行，不造成越界、遮挡或内容丢失。
