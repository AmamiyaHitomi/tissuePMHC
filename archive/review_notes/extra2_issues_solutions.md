# 论文问题解决方案

> 对照 `issues_checklist.md` 和当前 `tissuePMHC_latex_v3` 整理。  
> 状态含义：**已基本解决**＝当前正文已有足够内容；**需加强**＝已有内容但仍可能触发相同疑问；**未解决**＝需要新增文字、图表、核查或实验。

## 一、建议先完成的四项修改

1. **增加一张统一的评估协议表**，替代目前分散在正文中的定义。
2. **处理 44-task 来源不明的问题**：能恢复构建规则就完整定义；不能恢复就移到补充材料并明确标为历史探索性结果。
3. **增加数据重叠示意图或审计表**，清楚区分 pair、peptide、protein 和 task 四个层面的隔离。
4. **把 Table 7 定位为“历史组件证据”，不要称作严格消融实验**；真正的消融只能使用相同任务、fold、seed 和训练预算。

---

## 二、逐项解决方案

### 1. 全文过于技术化

- 当前状态：**需加强**。
- 当前版本已经增加了若干限制性说明，但 Methods 和 Results 中仍集中出现大量缩写和协议名称。
- 解决方案：
  1. 每个技术段落按“定义—目的—允许的结论”三句话组织。
  2. 首次出现 OOF、pair-disjoint、matched 和 connected component 时先写通俗解释，再给技术名称。
  3. 避免只写“strict”“standard”，应同时写出其生物学含义：
     - standard：同一个完整 pair 不跨集合，但 peptide 可通过其他 pair/task 再出现；
     - peptide-disjoint：测试 peptide 在任何训练 task 中都未出现；
     - 两者都只测试已经在训练中出现过的 tissue–MHC task。
  4. 在每个 RQ 结尾增加一句“Biological interpretation”或等价的自然段，不只重复指标。

可直接采用的写法：

> In plain terms, the standard split asks whether the model can rank new matched pairs within previously represented tissue--MHC tasks, even when an individual peptide may have been observed elsewhere in the training data. The peptide-disjoint split asks the harder question of whether the same tasks can be predicted for peptide sequences that are absent from all fitting folds.

### 2. Introduction 缺少组织特异性机制

- 当前状态：**部分解决，建议再补一段**。
- 当前 Introduction 已提到 protein abundance、turnover、proteolysis、processing 和 tissue environment，但还可以更明确地描述完整加工链。
- 解决方案：
  - 在 Introduction 第一段后增加一段，明确提到蛋白酶体切割、TAP 转运、ERAP 修剪、MHC 装载以及相关基因的组织差异表达。
  - 不要将模型结果解释为某一个分子机制的直接证据，因为模型没有使用表达量、蛋白组或加工通路特征。
  - paper中最重要的一段（配图：抗原呈递组织特异性偏好的分子机制）

可直接修改后使用的英文草稿：

> Tissue-associated presentation can arise at several stages upstream of peptide--MHC binding. Differences in source-protein abundance and turnover determine which substrates are available, while proteasomal cleavage, TAP-mediated transport, aminopeptidase trimming, and peptide loading determine which fragments reach stable MHC-I complexes. The genes and cell types contributing to these processes vary among tissues, so the same protein may yield different presented peptides in different organs, and the same peptide may have different presentation evidence across tissue contexts. Characterizing these differences may improve tumor-antigen prioritization and help distinguish broadly presented targets from tissue-restricted candidates. Our sequence-only benchmark captures the resulting tissue-associated signal but does not identify its causal molecular source.

### 3. Mouse fixed test 的实体重叠不清楚

- 当前状态：**数值审计已解决，呈现方式需加强**。
- 当前正文已有：
  - pair ID overlap：0；
  - unique peptide overlap：81.47%；
  - unique parent-protein overlap：88.88%；
  - row-level peptide/protein overlap：84.31%/93.04%。
- 解决方案：
  - 将这些内容从长段文字改为一张四行审计表：

| Entity | Train/test overlap rule | Observed overlap | 能否声称 unseen |
|---|---|---:|---|
| Pair ID | 不允许跨集合 | 0% | 可以声称 unseen pair |
| Peptide identity | 允许通过其他 pair/task 重现 | 81.47%（unique） | 不可声称 unseen peptide |
| Parent protein | 允许重现 | 88.88%（unique） | 不可声称 unseen protein |
| Tissue–H2 task | 训练和测试均已出现 | 100% seen tasks | 不可声称 unseen task |

- 同时增加一个小型示意图：同一测试 peptide 可以在 task A 中作为测试样本、在 task B 中作为训练样本，但同一个 pair ID 不能跨集合。

### 4. 评估协议名称混乱

- 当前状态：**正文已有解释，但缺少总表，仍需加强**。
- 建议在 `Problem Formulation and Benchmark` 中加入以下协议表：

| 建议统一名称 | 数据来源 | 划分单位 | 测试 peptide 能否在训练中出现 | Task 状态 | 用途 |
|---|---|---|---|---|---|
| Standard fixed test | 预先冻结的独立 test partition | pair ID | 可以，通过其他 pair/task | seen task | 主 closed-task 结果 |
| Standard pair-grouped OOF | 完整训练 pair pool | pair ID | 可以，通过其他 pair/task | seen task | 训练池内部 OOF |
| Matched standard OOF | 与 strict 完全相同的 pair pool | pair ID；held-out size 与 strict 匹配 | 可以 | seen task | strict 的直接对照 |
| Peptide-disjoint OOF | 与 matched standard 相同的 pair pool | peptide connected component | 不可以在任何训练 task 中出现 | seen task | 全局 unseen-peptide 泛化 |

- 命名规则：
  - “frozen 5-seed”属于模型/集成属性，不属于 split 名称，应放到 Model 或 Seeds 列。
  - “paired task-level standard-versus-strict statistics”属于统计分析，不应与数据划分并列。
  - “strict”可以作为简称，但首次出现必须写完整名称。

### 5. “只有三种设置，但名称更多”的问题

- 当前状态：**需通过层级化命名解决**。
- 解决方案：
  - 明确论文只有三个概念层面的设置：
    1. standard fixed test；
    2. standard OOF；
    3. peptide-disjoint OOF。
  - matched standard OOF 是 standard OOF 的匹配版本。
  - frozen、3-seed/5-seed 是模型评估属性。
  - paired statistics 是分析方法。
  - 不再将这些属性全部称为“evaluation settings”。

建议将原句：

> We distinguish three evaluation settings throughout the paper.

改为：

> We use three split families: a frozen standard test, standard pair-grouped OOF, and globally peptide-disjoint OOF. “Matched” denotes a standard OOF control constructed on the same pair pool and with held-out sizes aligned to the peptide-disjoint folds; seed counts describe model ensembling rather than additional split types.

### 6. Matched pair-grouped OOF 到底做了什么

- 当前状态：**已基本解决，建议增加操作性描述**。
- 当前 Methods 已说明 matched control 使用相同 pair pool，并匹配 fold 数、task coverage 和 task-wise held-out counts。
- 还应增加：
  - 没有筛选 strict 测试样本来制造性能差异；
  - 没有使用测试标签做匹配；
  - 匹配对象是每个 task/fold 的 held-out pair 数；
  - standard 与 strict 的唯一本质变化是 peptide 是否可跨 fold。
- 推荐表述：

> The matched standard control does not modify labels or select examples based on model performance. It reassigns the same complete pair pool by pair ID, using the same number of folds and nearly identical task-by-fold held-out counts as the peptide-component split. The intended contrast therefore changes the peptide-separation constraint while holding the evaluated pair inventory and task coverage fixed.

### 7. Table 8 中 matched 如何实现、为什么出现 mouse

- 当前状态：**matched 已说明；跨物种位置仍可优化**。
- 解决方案：
  - 将 human 和 mouse 的 matched-versus-strict 总表放在 RQ4“Generalization Boundaries”中是合理的，因为该 RQ 回答跨物种复现的同一问题。
  - 在表前明确写：

    > Mouse is included here only to test whether the same peptide-overlap effect is reproduced under the corresponding mouse benchmark; this is not a direct comparison of human and mouse model quality.

  - 在 RQ5 中不重复完整 matched-versus-strict 推断，只引用 RQ4，并集中报告 mouse-specific fixed test、H2 和 tissue 分层结果。

### 8. Pair-disjoint、task-level peptide-disjoint 和 global peptide-disjoint 混淆

- 当前状态：**当前 v3 定义基本正确**。
- 最终统一用法：
  - pair-disjoint：同一个 `pair_id` 不跨 fitting/held-out；
  - task-local peptide-disjoint：peptide 在当前 task 的训练集中未见，但可能出现在其他 task；
  - globally peptide-disjoint：peptide 在所有 task 的 fitting partition 中均未出现；
  - connected-component 是实现 globally peptide-disjoint 的分组算法。
- 全文中若未实际计算 task-local peptide-disjoint，就不要为它单独报告结果，只在概念说明中使用。

### 9. Fixed test、matched OOF、peptide-disjoint OOF 和 train-pool OOF 的关系

- 当前状态：**需用图或表一次讲清楚**。
- 建议使用如下层级：

```text
Original benchmark
├─ Frozen fixed test
│  └─ 只用于主 closed-task confirmation
└─ Complete training pair pool
   ├─ Matched standard pair-grouped OOF
   └─ Connected-component peptide-disjoint OOF
```

- 只有后两个来自相同 pair pool，因此只有二者的差值可用于估计 peptide separation 的影响。
- fixed test 与 strict OOF 的差异只能描述，不能直接归因于 peptide overlap。

### 10. 44-task subset 来源不明

- 当前状态：**仍未真正解决，是当前最明确的缺口之一**。
- 解决方案按优先顺序：
  1. 查找旧 split manifest、task list 和数据过滤脚本，恢复：
     - 44 个 task 的完整列表；
     - 数据版本和下载日期；
     - 最小 pair 数门槛；
     - split seed；
     - 每个 task 的 train/test 数。
  2. 将上述信息放入补充表，并在主文写一句定义。
  3. 如果无法可靠恢复：
     - 将 44-task 结果标为 historical exploratory screen；
     - 移至 Supplementary；
     - 主文 RQ1 改用有完整 provenance 的 157-task shared-head/logistic baseline；
     - 不再把 44-task 结果作为核心证据或与 157-task 结果直接比较。
- 不要只写“original 44-task subset”，这不能回答它为什么有 44 个 task。

### 11. Pair-grouped 是什么意思

- 当前状态：**基本解决，但建议就地定义**。
- 建议定义：

> Pair-grouped means that the two rows belonging to one matched positive–pseudo-negative pair share a persistent pair ID and are always assigned to the same fitting or held-out partition. This prevents direct pair splitting but does not prevent either peptide from reappearing through another pair or task.

- 若代码确实按 `pair_id` 分组，应在补充材料报告断言：
  - 每个 pair 恰好两行；
  - 标签为一正一负；
  - pair 内 tissue、MHC 和 parent UniProt 一致；
  - pair ID 跨 fold overlap 为 0。

### 12. H2 是什么

- 当前状态：**未在首次出现处充分解释**。
- 解决方案：
  - 首次出现改为：

    > the mouse major histocompatibility complex (H-2; written as H2 in the processed task identifiers)

  - RQ5 标题可改为：

    > RQ5: Replication in Mouse Tissue–H2 Tasks

  - 在数据处理部分说明 `H2-Db`、`H2-Kb`、`H2-Kd` 和 `H2-Kk` 是保留的四种 restriction。

### 13. Mouse 一共有多少 tissue，是否应完整报告

- 当前状态：**tissue 总数已解决，完整结果需放补充材料**。
- 当前 benchmark table 已给出 13 tissues、24 tasks、4 H2 restrictions。
- 解决方案：
  - 主文保留高低端描述表，但明确它不是完整结果。
  - Supplementary 增加 13 个 tissue 的完整表，列出：
    - task 数；
    - pair 数；
    - mean/median AUROC；
    - mean AUPRC；
    - PairAcc；
    - 最小/最大 task AUROC；
    - 如可得，task-bootstrap CI。
  - 对只有一个 task 的 tissue 标记为 descriptive，不计算跨 task 方差。

### 14. PairAcc 的定义

- 当前状态：**已基本解决，但有一处需统一**。
- 当前 benchmark section 定义为正例分数高于负例记 1；Experimental Setup 又说明 tie 主分析记 0，并报告 half-credit sensitivity。
- 解决方案：
  - 在公式后直接补充：

    > The primary analysis assigns zero credit to ties; a half-credit tie rule is reported as a sensitivity analysis.

  - 避免其他报告中写成“平分记 0.5”而不注明这是 sensitivity rule。
  - 全文统一 PairAcc 大小写。

### 15. MHC-only 的含义和冲突标签

- 当前状态：**模型输入已经解释，但冲突标签审计仍应显式报告**。
- 当前 v3 已说明 MHC-only 评分形式为 \(s=f(p,m)\)，移除了 tissue identity 和 tissue-specific head。
- 解决方案：
  1. 不要把不同 tissue 的数据简单合并后去重为单一标签；保留原始 row 和 task-specific label。
  2. 模型虽然看不到 tissue，但同一 peptide–MHC 可在不同 tissue row 中拥有不同标签，这正是该负对照的难点。
  3. 增加一个审计表：
     - unique peptide–MHC query 数；
     - 同一 query 是否同时出现正负标签；
     - conflicting query 比例；
     - conflicting rows 比例。
  4. 明确 MHC-only 的作用：检验不使用 tissue identity 时，通用 peptide–MHC 信号能达到什么程度；它不是一个无噪声的绝对 presentation 数据集。

### 16. Connected-component peptide-disjoint 的定义

- 当前状态：**原则已说明，图算法细节仍可加强**。
- 解决方案：
  - 给出精确定义：
    1. 每个 matched pair 是一个节点；
    2. 如果两个 pair 共享任意 peptide sequence，则连接一条边；
    3. 取无向图的 connected components；
    4. 整个 component 分配到同一个 fold；
    5. 因此任何 peptide 及其传递连接的 pair 都不会跨 fitting/held-out。
  - 若当前代码实际采用“peptide 为节点、pair 为边”的二部图实现，应按真实实现调整文字，但最终保证条件相同。
  - 报告 component 数、最大 component、每 fold pair 数、每 task held-out 数和零重叠审计。

### 17. 主结果表方法太少

- 当前状态：**部分解决**。
- 解决方案：
  - 不要为了增加行数而混合不同 task inventory。
  - 主表只放相同 157 tasks、相同 split、具有完整 held-out predictions 的方法。
  - 44-task 历史方法放单独的 historical baseline 表。
  - 外部方法放 general-pMHC control 表。
  - strict architecture 方法放 identical-fold strict 表。
  - 通过分表保证公平，而不是将所有方法塞进一个排行榜。
- 若要补实验，优先级：
  1. 157-task one-hot logistic regression；
  2. 157-task BLOSUM62 random forest；
  3. 157-task shared heads；
  4. 157-task auxiliary dual branch；
  5. TissuePMHC；
  6. 可复现的 external controls。

### 18. Figure 2 不够精致、方法太少

- 当前状态：**原箱线图问题仍需要视觉重做或更换图型**。
- 推荐方案：
  1. 如果各方法有完全相同的 157-task predictions，使用 raincloud/violin + box + task dots。
  2. 如果缺少一致的逐任务结果，不要制作伪可比 boxplot；改用 mean、median、worst-tail 的分组点图。
  3. strict 对比更适合使用 task-wise paired scatter 或差值森林图。
  4. 统一：
     - 色盲友好配色；
     - 水平方法标签；
     - 95% task-bootstrap CI；
     - 相同 y 轴范围；
     - 正文和图中一致的方法顺序。
  5. 图注必须说明红点、箱体、whisker 和 task aggregation 的含义。

### 19. Table 7 是否为 ablation test

- 当前状态：**当前已改名为 component evidence，但仍需防止过度解释**。
- 解决方案：
  - 主文明确写：

    > These comparisons summarize completed historical component evidence and are not a single factorial ablation study because task inventories and aggregation settings differ.

  - Table 7 不使用“Ablation”作为标题。
  - 若要做正式消融，应固定：
    - 157 tasks；
    - 同一 standard 或 strict fold manifest；
    - 同一三个 seeds；
    - 同一 epoch/batch size/optimizer；
    - 同一 ensemble rule；
    - 一次只改变一个组件。
  - 最小正式消融矩阵：

| 模型 | Multi-kernel encoder | Auxiliary loss | Dual branch | Rank fusion | Seed ensemble |
|---|---:|---:|---:|---:|---:|
| Base MLP | 否 | 否 | 否 | 否 | 是 |
| + dual branch | 否 | 否 | 是 | 是 | 是 |
| + auxiliary | 否 | 是 | 是 | 是 | 是 |
| + multi-kernel | 是 | 是 | 是 | 是 | 是 |
| probability fusion | 是 | 是 | 是 | 否 | 是 |

### 20. Mouse 表格方法和指标太少

- 当前状态：**指标已有改善，方法仍有限**。
- 解决方案：
  - standard mouse 表至少包含 BLOSUM62 RF、shared heads、Factorized MMoE 和已完成的 H2-Kk adapter。
  - strict 表只放在 identical strict folds 上完成的方法。
  - 指标保持统一：Mean AUROC、Median AUROC、AUPRC、PairAcc、Worst-6、Worst task。
  - Accuracy/MCC 可放 Supplementary，避免主表过宽。
  - 不要将未在相同 fold 上运行的方法填入 strict 表。

### 21. Human 与 Mouse strict 方法不对应

- 当前状态：**问题客观存在，但可以通过解释与补实验二选一解决**。
- 原因：
  - Human 与 mouse 的最终架构本来就不同；
  - 数据规模、task 数和开发历史也不同；
  - 所以两个 strict 表不是跨物种模型排行榜。
- 最低成本解决方案：
  - 在两个表前明确说明它们是“within-species controls”，不要求方法一一对应。
  - 在 Discussion 中保留“不能识别 species effect”的限制。
- 更强解决方案：
  - 为 mouse 补跑 plain MLP dual branch、auxiliary dual branch；
  - 为 human 补跑与 mouse 对应的 Factorized MMoE；
  - 全部使用各自相同 strict fold manifest。
- 如果计算预算有限，建议采用最低成本方案，不应为了形式对应而加入未完成或不公平的数值。

---

## 三、哪些问题不需要重新训练

以下修改可以直接完成：

- Introduction 生物学机制段；
- plain-language 改写；
- 评估协议总表；
- fixed/OOF/strict 关系图；
- H2、PairAcc、pair-grouped、connected-component 定义；
- mouse overlap 审计表；
- matched 构造说明；
- Table 7 更名及结论降级；
- 解释 human/mouse 方法集合不一致；
- 将完整 tissue 结果移到 Supplementary（前提是现有逐 task 结果足够）。

## 四、哪些问题需要先核查代码或数据

- 44-task 的来源、过滤规则和 split manifest；
- MHC-only 中 peptide–MHC conflicting labels 的数量；
- pair ID、peptide 和 protein 的 fold overlap；
- connected-component 图的真实节点/边实现；
- 每种表格是否混合了 fixed test、standard OOF 或 strict OOF；
- Figure 2 中各方法是否真的具有相同 task inventory 和逐任务预测。

## 五、哪些问题需要新增实验

- 在完全一致的 157-task/fold/seed 条件下完成正式消融；
- 让 human 和 mouse strict architecture 表的方法完全对应；
- 增加尚未完成的 157-task 基线；
- shuffled-tissue-label control；
- protein-disjoint、study-disjoint 或外部 cohort 验证。

这些新增实验不是所有都必须完成。优先级建议为：

1. 44-task provenance 核查；
2. MHC-only conflict audit；
3. 统一 157-task 消融；
4. 跨物种方法补齐。

## 六、建议采用的最终论文结构

1. **Introduction**
   - 生物学机制；
   - tissue–MHC preference 的定义；
   - 临床意义和非因果限制。
2. **Related Work**
   - binding/presentation；
   - context-aware presentation 方法；
   - 与本任务的三点区别。
3. **Benchmark**
   - pair 构造；
   - human/mouse inventory；
   - split 协议总表；
   - overlap audit。
4. **Methods**
   - 模型；
   - MHC-only 和 external controls；
   - 指标、PairAcc 和统计方法。
5. **Results**
   - 统一 task inventory 的主结果；
   - matched standard vs peptide-disjoint；
   - mouse replication；
   - no-tissue/external controls；
   - identical-fold architecture controls。
6. **Supplementary**
   - 44-task 历史结果；
   - 完整 tissue/task 表；
   - fold manifests；
   - conflict/overlap audits；
   - 历史 component evidence。
