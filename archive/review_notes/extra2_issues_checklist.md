# 论文问题修改清单

> 根据 `pic_issues` 文件夹中的 19 张审阅截图整理。  
> 每项包含审阅原话、归纳的问题以及建议处理方式。完成修改后可将对应的 `[ ]` 改为 `[x]`。

## 一、整体写作与生物学背景

$11. 降低全文的技术化程度，增加通俗解释**

  - 截图：`屏幕截图 2026-07-26 130951.png`
  - 审阅原话：

    > 目前的表述里过于技术化，再使用技术语言的同时，还是要多使用plain language，让来自交叉领域的人也都能理解。
    >
    > 论文整体修改的提示词：
    >
    > 我们计划投一个computational biology或者bioinformatics相关的期刊。目前的论文写作非常技术化，生物学方面的影响和解读偏少。用语在保持技术性的同时，适度修改的让生物领域的人也能明白每一个设置的生物意义。

  - 问题总结：
    - 全文使用了大量计算机和机器学习术语，但对非计算背景读者不够友好。
    - 实验设置的生物学意义及其对组织特异性抗原呈递的影响解释不足。
  - 建议处理：
    - 首次出现专业术语时给出通俗定义。
    - 在每项技术设置后补充“该设置回答什么生物学问题”。
    - 在结果部分同时报告计算结果和生物学解读。

$12. 在 Introduction 中补充组织抗原呈递偏好的分子机制**

  - 截图：`屏幕截图 2026-07-26 131015.png`
  - 审阅原话：

    > Introduction 增加一段解释 不同组织中抗原呈递 preference的分子机制，包括酶切、转运、修剪，呈递等过程参与的基因在不同组织中的差异表达。说明同一个蛋白在不同组织中被呈递的epitope可能是不同的，或者同一个epitope在不同的组织中可能会有不同的呈递效果。因此理解这个问题对肿瘤的治疗很重要。类似这些。似乎可以放在第二段。

  - 问题总结：
    - 引言缺少组织特异性抗原加工与呈递的生物学机制。
    - 尚未充分解释研究该问题对肿瘤免疫治疗的意义。
  - 建议处理：
    - 补充抗原酶切、转运、修剪和呈递相关基因的组织差异表达。
    - 说明相同蛋白或相同 epitope 在不同组织中可能产生不同的呈递结果。
    - 将该背景与肿瘤免疫治疗、靶点选择和脱靶风险联系起来。

## 二、评估协议、数据划分与潜在泄漏

$13. 更清楚地展示 mouse fixed test 中的实体重叠**

  - 截图：`屏幕截图 2026-07-26 131046.png`
  - 审阅原话：

    > 这块不够清楚，看能不能画图或者列表表示。
    >
    > fix的test里面，testing的肽段可能出现在其他的task内部，但不会出现在我们正在看的组织-分型组合子任务中。

  - 问题总结：
    - Mouse benchmark 的 fixed test 虽然按 pair 划分，但测试 peptide 和 parent protein 大量出现在训练数据中。
    - 当前文字未直观说明“任务内未见”与“跨任务已见”的区别。
  - 建议处理：
    - 增加示意图或表格展示 pair、peptide、protein 和 task 层面的重叠。
    - 明确测试 peptide 是否只在当前 tissue–H2 task 中未见，但可出现在其他 task 中。
    - 将该测试准确描述为内部 closed-task confirmation，不将其解释为全局 unseen-peptide 或 unseen-protein 泛化。

$14. 重写 Evaluation Splits and Leakage Audits，并统一命名**

  - 截图：`屏幕截图 2026-07-26 131105.png`
  - 审阅原话：

    > 只有三个吗？标注什么名字
    >
    > 错的吧？是不是不可能出现在该子任务中的其他pair？
    >
    > 这块非常混乱，命名过于模糊，基本无法从名字清楚判断到底如何设计的。
    >
    > 建议增加一个表格，详细解释这么多的setting到底有什么区别和联系：
    >
    > fixed test
    >
    > pair-grouped OOF
    >
    > matched standard 5-seed OOF
    >
    > Frozen 5-seed fixed test
    >
    > Peptide-disjoint 5-seed OOF
    >
    > matched standard OOF
    >
    > peptide disjoint OOF
    >
    > connected-component peptide-disjoint
    >
    > Paired task-level standard-versus-strict statistics
    >
    > 目前的表述和命名过于技术和编码化，需要用更加能够明确目的和辅助理解语言来描述：
    >
    > 比如，
    >
    > standard (testing peptide not seen before in sub-task),
    >
    > peptide-disjoint (testing peptide not seen before in any tasks)
    >
    > 我们计划投一个computational biology或者bioinformatics相关的期刊。这部分的论文写作非常技术化但对目标和意义的表述不够friendly，生物学方面的影响和解读偏少，非硬核计算背景的人可能无法快速的理解。用语在保持技术性的同时，适度修改的让生物领域非硬核计算背景的人也能迅速明白每一个设置的意义。可以考虑加一个表格，把关键的名词，具体描述，目标，以及彼此的区别和联系拆开。

  - 问题总结：
    - 文中声称有三种评估设置，但实际使用的评估名称明显多于三种。
    - 各名称无法直接反映划分单位、实体重叠规则和评估目的。
    - “pair identifier cannot cross the fitting/held-out boundary”等表述可能不准确或容易误解。
  - 建议处理：
    - 重新梳理评估协议的层级，区分核心设置、派生设置和统计比较。
    - 增加协议汇总表，至少列出：划分单位、peptide/protein/task 是否可重叠、是否 OOF、是否 matched、seed 数量和可支持的结论。
    - 统一全文、表格和图注中的命名。

$15. 解释 matched pair-grouped OOF 检查的具体含义**

  - 截图：`屏幕截图 2026-07-26 131133.png`
  - 审阅原话：

    > 这是什么意思？

  - 对应原文：

    > A matched pair-grouped OOF comparison provides an additional check that the result is not unique to the final fixed test.

  - 问题总结：
    - 读者无法判断 matched pair-grouped OOF 控制了哪些因素，以及它为什么能验证 fixed test 结果。
  - 建议处理：
    - 说明 matched 的对象和匹配变量，例如 fold 数、task coverage、样本数量和正负比例。
    - 说明该比较与 final fixed test 的相同点和不同点。
    - 明确该分析能够排除或不能排除哪些替代解释。

$16. 解释 matched 的实现方式以及 mouse 出现在该表中的原因**

  - 截图：`屏幕截图 2026-07-26 131209.png`
  - 审阅原话：

    > 如何实现matched？重构了training data？还是选择了部分testing data？
    >
    > 为什么mouse在这里出现了？

  - 问题总结：
    - Table 8 没有说明 matched standard OOF 是如何构造的。
    - Human 结果小节中突然出现 mouse，章节组织和比较目的不清楚。
  - 建议处理：
    - 明确 matched 是通过重建 fold、调整训练集、筛选测试集还是其他方式实现。
    - 报告具体的匹配条件和匹配后的样本规模。
    - 将 mouse 结果移至 mouse replication 小节，或解释此处跨物种比较的必要性。

$17. 核实 mouse 表格中的 pair-disjoint 定义**

  - 截图：`屏幕截图 2026-07-26 131224.png`
  - 审阅原话：

    > pair-disjoint这个表述可能是错的。
    >
    > 估计是子任务内的peptide-disjoint。
    >
    > 以及和其他所有任务的peptide-disjoint。

  - 问题总结：
    - Table 9、Table 10 使用的 pair-disjoint 可能没有准确反映实际划分规则。
    - 子任务内 peptide-disjoint 与全局 peptide-disjoint 被混淆。
  - 建议处理：
    - 对照数据划分代码核实每张表实际采用的规则。
    - 将“当前 task 内 peptide 未见”和“所有 task 中 peptide 均未见”分别命名。
    - 检查正文、表题、表注和方法部分是否一致。

$18. 解释 fixed test、matched standard OOF、peptide-disjoint OOF 与 train-pool OOF 的关系**

  - 截图：`屏幕截图 2026-07-26 131248.png`
  - 审阅原话：

    > 什么是fixed test？
    >
    > 什么是matched standard OOF
    >
    > 什么是Peptide-disjoint OOF
    >
    > 跟前面的train-pool OOF有啥关系？

  - 问题总结：
    - 主要评估协议在结果表附近没有就地解释。
    - train-pool OOF 与 standard、matched、strict/peptide-disjoint OOF 的关系不明确。
  - 建议处理：
    - 在首次出现处给出一句话定义，并引用统一的协议说明表。
    - 说明每种协议使用的数据池、训练/测试边界和主要评估目标。
    - 避免同一种协议在不同表格中使用不同名称。

## 三、任务、数据和指标定义

$19. 解释 44-task subset 的来源**

  - 截图：`屏幕截图 2026-07-26 131122.png`
  - 审阅原话：

    > 为什么会44 task？哪里定义了44 task？

  - 问题总结：
    - RQ1 突然使用 original 44-task subset，但正文此前没有明确说明其构造方式。
  - 建议处理：
    - 说明每个 task 的定义。
    - 给出 44 个 task 的筛选标准、数据来源和样本量门槛。
    - 解释 44-task subset 与后续 157-task benchmark 的关系。

$110. 定义 Table 4 中的 44-task**

  - 截图：`屏幕截图 2026-07-26 131141.png`
  - 审阅原话：

    > 怎么定义的？

  - 问题总结：
    - Table 4 的表题再次使用 original 44-task standard pair-disjoint benchmark，但仍缺少定义。
  - 建议处理：
    - 在表注中简要说明 44-task 的来源，并引用完整的数据构建章节。
    - 说明该历史基线为何不能直接与 157-task 主实验完全匹配比较。

$111. 解释 pair-grouped 的含义**

  - 截图：`屏幕截图 2026-07-26 131146.png`
  - 审阅原话：

    > pair-grouped 是啥意思？

  - 问题总结：
    - Table 6 中 pair-grouped OOF 没有说明 pair 由哪些实体组成，以及分组如何防止泄漏。
  - 建议处理：
    - 明确 pair 是 peptide–parent protein、tissue–MHC、peptide–MHC，还是其他组合。
    - 说明同一 pair 的所有记录是否始终进入同一个 fold。
    - 说明该规则允许哪些实体跨 fold 重复。

$112. 解释 H2**

  - 截图：`屏幕截图 2026-07-26 131218.png`
  - 审阅原话：

    > H2 是啥意思？

  - 问题总结：
    - RQ5 标题中的 Tissue–H2 对非小鼠免疫遗传学背景的读者不够清楚。
  - 建议处理：
    - 首次出现时说明 H-2 是小鼠主要组织相容性复合体系统。
    - 解释 H2 restriction 与人类 HLA restriction 的对应关系。

$113. 报告 mouse tissue 总数并扩充 tissue-level 结果**

  - 截图：`屏幕截图 2026-07-26 131232.png`
  - 审阅原话：

    > 一共多少个tissue？建议完整报告，建议增加更多的指标，同时放多行数。

  - 问题总结：
    - Table 12 只报告部分高低极端组织，未说明完整组织数量。
    - 每个组织的任务数和指标较少，难以判断稳定性。
  - 建议处理：
    - 报告 mouse benchmark 的 tissue 总数。
    - 在正文或补充材料中提供所有 tissue 的完整结果。
    - 增加任务数、AUROC、AUPRC、PairAcc、置信区间或方差等信息。

$114. 定义 PairAcc 和 MHC-only**

  - 截图：`屏幕截图 2026-07-26 131239.png`
  - 审阅原话：

    > PairAcc是啥意思？
    >
    > MHC-only什么意思？就是把不同tissue的正负样本混到一起？可能有contradicting的标记？

  - 问题总结：
    - PairAcc 的计算对象、公式和 tie policy 不明确。
    - MHC-only 对照的数据构造方式不清楚，可能存在同一 peptide–MHC 在不同 tissue 中标签冲突的问题。
  - 建议处理：
    - 给出 PairAcc 的正式定义、配对规则和并列值处理方式。
    - 列出 MHC-only 模型的输入特征及其排除的 tissue 信息。
    - 说明跨 tissue 合并样本时如何处理重复记录和冲突标签。

$115. 定义 connected-component peptide-disjoint**

  - 截图：`屏幕截图 2026-07-26 131258.png`
  - 审阅原话：

    > 什么是 connected-component peptide-disjoint?

  - 问题总结：
    - 严格架构实验使用了 connected-component peptide-disjoint folds，但没有解释图结构和划分规则。
  - 建议处理：
    - 说明图中的节点和边分别表示什么。
    - 说明如何根据 connected component 分配训练和测试 fold。
    - 解释该规则相比普通 peptide-disjoint 额外防止了哪类信息泄漏。

## 四、实验规模、消融和方法一致性

$116. 增加主结果表的方法数量**

  - 截图：`屏幕截图 2026-07-26 131146.png`
  - 审阅原话：

    > 方法的数量太少，增加更多的row

  - 问题总结：
    - Table 5、Table 6 的模型数量较少，难以全面判断 TissuePMHC 相对现有方法的优势。
  - 建议处理：
    - 增加经典模型、外部 pMHC 工具以及更完整的内部架构对照。
    - 区分“直接匹配比较”和“由于数据设置不同而仅供参考的比较”。

$117. 改进 Figure 2 的视觉质量并增加方法**

  - 截图：`屏幕截图 2026-07-26 131153.png`
  - 审阅原话：

    > 图片不够精致，需要更漂亮。为什么这么少的方法？

  - 问题总结：
    - 箱线图的配色、排版和标签呈现不够适合正式论文。
    - 图中方法较少，且没有充分展示任务级配对差异。
  - 建议处理：
    - 统一论文配色、字体、线宽和标签方向。
    - 增加更多基线方法。
    - 考虑加入任务级散点、配对连线、差值分布或置信区间。

$118. 将 Table 7 明确改造成规范的消融实验**

  - 截图：`屏幕截图 2026-07-26 131202.png`
  - 审阅原话：

    > Ablation test? 具体啥意思？这个表述方式好像不是很合适

  - 问题总结：
    - Table 7 将不同组件证据合并展示，但部分比较使用了不同任务数、seed 或聚合方式。
    - 当前结果不能直接解释为严格的组件消融。
  - 建议处理：
    - 明确该表是 descriptive component evidence 还是正式 ablation study。
    - 若作为消融实验，应保持数据划分、训练流程、seed、任务集合和聚合方法一致。
    - 分别移除或替换 multi-kernel encoder、auxiliary branch、fusion rule 和 seed averaging。

$119. 扩充 mouse 方法和指标，缩减不必要的表格空间**

  - 截图：`屏幕截图 2026-07-26 131224.png`
  - 审阅原话：

    > 方法太少，table太小，多加几个指标，多加几个方法

  - 问题总结：
    - Mouse Table 9、Table 10 的方法和指标数量有限。
    - 页面空间利用率不高，但实验信息仍不充分。
  - 建议处理：
    - 增加与 human 实验对应的方法。
    - 增加 MCC、accuracy、PairAcc、worst-task 和不确定性统计。
    - 调整表格版式，提高信息密度和可比性。

$120. 统一 Human 和 Mouse 严格架构比较中的方法**

  - 截图：`屏幕截图 2026-07-26 131304.png`
  - 审阅原话：

    > 为什么跟上面的方法不对应？

  - 问题总结：
    - Human Table 16 和 Mouse Table 17 使用的候选模型集合不一致。
    - 读者无法判断缺失方法是无法运行、表现较差，还是未完成实验。
  - 建议处理：
    - 尽量使两种物种的严格架构比较使用相同模型集合。
    - 如果某些方法不适用于 mouse，需要明确说明原因。
    - 统一方法名称、排列顺序和指标列。

## 五、建议优先级

### 高优先级：影响实验结论可信度

- [ ] 核实所有数据划分代码与表述，特别是 pair-disjoint、task-level peptide-disjoint 和 global peptide-disjoint。
- [ ] 解释 fixed test 中的 peptide/protein/task 重叠，限定相应结论的适用范围。
- [ ] 定义并统一 fixed test、standard OOF、matched OOF、peptide-disjoint OOF、train-pool OOF 等名称。
- [ ] 说明 matched 数据集或 fold 的具体构造方式。
- [ ] 检查 MHC-only 对照是否存在跨组织冲突标签。

### 中优先级：影响实验完整性

- [ ] 增加其他适用且可复现的基线。
- [ ] 增加严格、同条件的组件消融实验。
- [ ] 扩充 Human 和 Mouse 的方法、指标和不确定性报告。
- [ ] 统一 Human 与 Mouse 架构比较的方法集合。

### 表达与呈现优先级

- [ ] 在 Introduction 中补充组织特异性抗原呈递机制和临床意义。
- [ ] 增加评估协议汇总表和数据划分示意图。
- [ ] 为 44-task、H2、PairAcc、MHC-only、pair-grouped 和 connected-component peptide-disjoint 提供定义。
- [ ] 改进 Figure 2 和相关表格的版式。
- [ ] 全文减少过度技术化表述，并补充生物学解读。
