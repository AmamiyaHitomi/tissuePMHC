# HumanPMHC Premium 实验 A–E 清单

最后更新：2026-07-31

## 1. 研究范围

本清单只针对以下两个问题，不扩展到普通调参、外部模型融合或无关结构搜索：

1. 当前 E14/E29 的 tissue auxiliary 实际预测 query `target_tissue`，但负例并未在该组织呈递，辅助标签对负例不符合严格的生物学语义。
2. 当前模型主要使用 peptide 表示和 task ID；tissue/HLA 只在末端选择 task head，难以学习组织条件化表示，更不能仅凭 tissue ID 声称学到了组织特异性生物学机制。

## 2. 状态说明

- `[x]`：代码、运行和结果检查均已完成。
- `[c]`：代码已经完成并通过smoke test，尚未正式运行。
- `[-]`：保留为对照或诊断，但根据已有结果不再作为晋级候选。
- `[ ]`：尚未实现或运行。
- `阻塞`：缺少必要的外部数据或映射。

## 3. 统一实验协议

### 已完成的 A 组

- 数据：`data/humanPMHC_premium/humanPMHC_train.csv.gz` 和冻结的 fixed test。
- seeds：`20260704`、`20260705`、`20260706`。
- epochs：25。
- batch size：512。
- global/HLA融合：固定 `0.5/0.5`。
- 随机隔离：global分支和HLA分支训练前分别重置完整随机状态。
- observed-tissue标签：只从当前 premium train 的正例重建，不读取全数据预聚合组织集合。

### 后续 B–E 组

premium test已经被多次观察，B–E不得继续使用它选择结构或超参数。

- 开发数据：只使用 premium train。
- 主开发划分：固定的3-fold pair-grouped OOF。
- 严格检查：补充 peptide-connected-component-disjoint OOF。
- B–E代码入口默认运行 `20260704/05/06` 三seed。
- 如需快速smoke test，可通过 `--seeds 20260704` 临时覆盖；正式结果必须使用三seed。
- 所有 observed-tissue标签必须在每个fold的 fitting部分重新构建。
- 同一比较中的HLA分支、训练参数、epoch和融合方式必须保持一致。

### 统一指标

- mean task AUROC。
- mean task AUPRC。
- mean task PairAcc。
- mean task MCC。
- worst-10 mean AUROC。
- 75个task的胜/平/负。
- `other_tissue_count=1/2/3+`分层PairAcc。
- seen-peptide、unseen-peptide分层。
- 每HLA和每tissue宏平均。
- task配对bootstrap区间。

---

## 4. 实验 A：tissue auxiliary 语义消融

目的：判断错误的负例tissue辅助监督是否是当前模型的主要性能瓶颈。

| 状态 | 编号 | 实验 | 核心改动 | 3-seed AUROC | 3-seed AUPRC | PairAcc | 结论 |
|---|---|---|---|---:|---:|---:|---|
| `[x]` | A0 | 当前auxiliary基线 | 全部样本预测 query `target_tissue`，保留HLA auxiliary | **0.69820** | **0.68596** | **0.70293** | 当前默认基线 |
| `[-]` | A1 | HLA-only | 删除tissue auxiliary | 0.69695 | 0.68399 | 0.70151 | 删除tissue监督没有提升，MCC明显下降 |
| `[-]` | A2 | 正例限定tissue auxiliary | 只有 `label=1` 计算tissue loss | 0.69801 | 0.68589 | 0.70151 | 与A0近似，但没有优于A0 |
| `[-]` | A3 | observed-tissue多标签 | train-only多标签BCE，未观察组织按0处理 | 0.69725 | 0.68389 | 0.70258 | 单seed优势未在3-seed复现 |
| `[-]` | A4 | observed-tissue masked | 只监督观察正组织和query tissue，其余mask | 0.69721 | 0.68517 | 0.70267 | 辅助任务更可学，但没有迁移到主任务 |
| `[-]` | A5 | 其他组织数量 | 预测train-only跨组织数量分桶 `0/1/2+` | 0.69558 | 0.68337 | 0.70178 | 整体最差，count信号接近多数类基线 |

### A组已完成检查

- [x] 18次运行文件完整。
- [x] 每次包含7,500条测试预测和75个task。
- [x] 所有预测有限且位于 `[0,1]`。
- [x] 三个seed下，A0–A5的HLA分支训练轨迹逐值一致。
- [x] A5随机状态混杂已经修复。
- [x] 完成逐seed、逐task、bootstrap及跨组织数量分层比较。

### A组最终结论

1. A0在AUROC、AUPRC、PairAcc和MCC上均为3-seed平均最优。
2. A2与A0非常接近，说明删除负例tissue loss并没有产生明确收益。
3. A3/A4修正了部分辅助标签语义，但没有突破主模型的输入和结构限制。
4. A5的AUROC相对A0下降约 `0.00263`，可以停止。
5. “负例tissue辅助标签错误”不是当前主要性能瓶颈；后续重点转向C组的条件结构。

相关结果：

- `extra_premium/results/experiments/A_experiments_seed_summary.csv`
- `extra_premium/results/experiments/A_experiments_seed_aggregate.csv`

---

## 5. 实验 B：直接测量辅助任务冲突

目的：不只比较最终分数，而是直接判断负例tissue auxiliary是否与主任务产生梯度冲突。

| 状态 | 编号 | 实验 | 必须输出 | 判断标准 |
|---|---|---|---|---|
| `[c]` | B1 | 正负例分开的auxiliary诊断 | 正/负例tissue accuracy、loss；按tissue及other-count分层 | 负例显著更差说明query tissue标签缺乏peptide支持 |
| `[c]` | B2 | 主任务与tissue auxiliary梯度冲突 | 正/负例梯度余弦；冲突比例；other-count分层 | 负例余弦更低或负值更多，才直接支持语义冲突 |
| `[c]` | B3 | tissue标签打乱negative control | 正确标签与fold内打乱标签的OOF差值 | 两者接近说明tissue auxiliary主要充当普通正则化 |

### B组实施要求

- B1/B2使用A0匹配结构。
- B2只统计共享peptide encoder参数的梯度，不能把独立head参数混入余弦。
- 分别计算main BCE、tissue auxiliary和HLA auxiliary梯度。
- 打乱标签必须限制在fitting fold内，不能跨fold交换。
- 即使B2发现冲突，也不能推翻A组“删除该监督未改善最终性能”的结果；两者分别回答机制和效果问题。

---

## 6. 实验 C：让query tissue/HLA进入主模型

目的：让tissue/HLA在peptide表示形成阶段参与计算，而不只是最后选择task head。

A组没有找到优于A0的辅助方案，因此C组默认使用A0 auxiliary配置；必要时把A2作为语义更干净的敏感性对照。

| 状态 | 编号 | 结构 | 作用 |
|---|---|---|---|
| `[c]` | C0 | `peptide → encoder → task-specific head` | OOF结构基线 |
| `[c]` | C1 | peptide embedding与tissue/HLA embedding简单拼接 | 最基础条件化对照 |
| `[c]` | C2 | tissue/HLA FiLM或gating调制peptide表示 | 验证条件信息进入特征提取是否有效 |
| `[c]` | C3 | 条件共享表示 + shared/HLA/tissue/task residual | 主候选；兼顾共享和task局部拟合 |
| `[c]` | C4 | C3移除task residual | 判断独立task成分是否仍然必要 |

### C组固定比较

```text
C0 → C1 → C2 → C3
                 └→ C4消融
```

### C组必须控制

- C0–C4使用完全相同的OOF folds和seed。
- 参数量必须报告；必要时增加parameter-matched对照。
- tissue/HLA embedding不能直接携带held-out统计量。
- C3的task residual需要正则化，避免重新退化成75个互不共享的task head。
- 同一peptide在不同query tissue下的条件表示必须实际不同，并通过单元测试确认。

### C组晋级条件

- AUROC和PairAcc同时优于C0。
- AUPRC不得明显下降。
- unseen-peptide和小样本task不得系统性退化。
- 至少在多数seed和多数task上方向一致。
- 通过D组反事实诊断后，才能解释为使用了tissue条件。

---

## 7. 实验 D：验证模型是否真的使用组织条件

目的：区分“模型输入了tissue ID”和“模型确实依赖tissue条件进行预测”。

只对C0、C2和C3执行；如果C2/C3均未超过C0，则停止D组扩展。

| 状态 | 编号 | 实验 | 必须输出 | 解释 |
|---|---|---|---|---|
| `[c]` | D1 | tissue-swap反事实测试 | 固定peptide/HLA，切换query tissue后的分数矩阵 | 报告组织切换方向是否与observed tissues一致 |
| `[c]` | D2 | validation时打乱tissue输入 | 原始与打乱后的OOF指标差 | 几乎不下降说明模型没有真正使用tissue条件 |
| `[c]` | D3 | 关闭模型分量 | 关闭tissue、HLA、task residual、auxiliary后的指标变化 | 定位最终收益来源 |

### D组注意事项

- C0切换task head也会改变分数，因此“分数变化”本身不是机制证据。
- D1必须结合真实observed-tissue方向、D2 shuffle和D3 ablation共同解释。
- tissue-swap时只能使用推理阶段合法输入。
- D组只能证明组织条件依赖，不能单独证明生物学机制。

---

## 8. 实验 E：processing-first组织机制特征

说明：HPA、UniProt和人工组织映射的获取与审计属于`Pre-E`数据准备，不是
独立的F实验；模型实验仍从E0开始。Pre-E是确定性处理，没有seed。

目的：以三seed最优C4为骨架，先加入能在同蛋白pair内区分两个peptide的
flank/processing信息，再检验parent expression及其与组织加工环境的交互。
所有候选使用相同pair-aware BCE + ranking loss、HLA/tissue residual，且不使用
task residual。默认单seed `20260704`。

| 状态 | 编号 | 实验 | 输入或改动 | 主要问题 |
|---|---|---|---|---|
| `[c]`/数据阻塞 | E0 | matched C4 baseline | C4结构 + E组统一pairwise协议 | 建立严格匹配基线 |
| `[c]`/数据阻塞 | E1 | flank processing | N/C flanks、位置、MHCflurry processing score | peptide加工信息能否直接提高PairAcc |
| `[c]`/数据阻塞 | E2 | parent expression control | query表达、跨组织均值、相对表达和missing mask | 验证同蛋白pair下expression-only应当很弱 |
| `[c]`/数据阻塞 | E3 | tissue-processing interaction | PSMB/TAP/ERAP等表达调制flank表示 | 组织加工环境能否改变peptide排序 |
| `[c]`/数据阻塞 | E4 | full mechanism | E3 + parent expression | 表达是否只在与加工交互后有效 |

E1–E4自动产生匹配负对照：processing按task打乱、expression按完整pair打乱、
以及tissue-machinery swap。负对照只改变held-out输入，不重新训练，不修改标签。

### E组数据审计

- [ ] 确定表达数据版本、下载日期和许可证。
- [ ] 建立premium tissue到表达数据库组织的人工审核映射。
- [ ] 明确使用gene、transcript还是protein层级表达。
- [ ] 审计UniProt到gene/protein的映射覆盖率及一对多关系。
- [ ] 下载并冻结parent protein FASTA，审计canonical/secondary accession。
- [ ] 报告peptide在parent sequence中的exact和unique定位覆盖率。
- [ ] 对无法定位、isoform歧义和多次出现保留missing/ambiguity标志。
- [ ] 使用真实N/C flanks重新计算MHCflurry processing score。
- [ ] 保存缺失比例，并始终输入missing mask。
- [x] `low_proxy`组织表达和加工机器在主训练中作为缺失值；原始proxy值仅作
  推理敏感性对照。
- [x] 逐样本保留exact/synonym、aggregate proxy和low proxy分层，并输出
  `mapping_quality_metrics.csv`。
- [x] `machinery_missing_fraction`显式进入模型。
- [ ] 表达归一化只使用训练允许的信息。
- [ ] 不把UniProt ID直接作为可记忆的类别特征。
- [ ] 对study、样本和组织来源进行泄漏审计。
- [x] 已验证所有正负pair的target tissue、HLA和parent UniProt相同。
- [x] 代码强制检查parent-expression在pair内完全不变。

### E组结论要求

同时满足以下条件，才能支持“加工/表达机制提供了有效信息”：

1. E1/E3/E4在OOF中超过匹配E0；E2单独无效不视为失败。
2. 对应processing/expression/machinery打乱后增益消失或明显减弱。
3. unseen-peptide和peptide-disjoint OOF仍有收益。
4. 收益不只集中在一个组织或少量大task。
5. 缺失值子集和完整映射子集结论方向一致。

---

## 9. 推荐执行顺序

### 当前状态

- [x] A0–A5完成并作出3-seed结论。
- [x] B1–B3三seed正式运行完成。
- [x] C0–C4三seed正式运行完成。
- [x] D1–D3三seed正式运行完成。
- [c] processing-first E0–E4代码完成并通过全流程合成输入smoke test；
  合成结果已删除，真实FASTA/表达数据和人工组织映射仍阻塞正式运行。

### 下一步顺序

1. B1：补充正负例辅助诊断。
2. B2：直接测量共享encoder上的梯度冲突。
3. B3：完成tissue-label shuffle对照。
4. C0：建立严格匹配的train-only OOF结构基线。
5. C1：必要的简单拼接对照。
6. C2：条件调制。
7. C3/C4：条件共享、残差及其消融。
8. 只有C2或C3超过C0时执行D1–D3。
9. 完成UniProt序列、peptide定位、HPA表达及人工组织映射审计。
10. 单seed执行E0→E1→E2→E3→E4；只有候选通过负对照后再做3 seeds。

## 10. 每个实验必须保存的文件

- `oof_predictions.csv`
- `per_task_metrics.csv`
- `summary_metrics.csv`
- `other_count_metrics.csv`
- `seen_unseen_metrics.csv`
- `per_hla_metrics.csv`
- `per_tissue_metrics.csv`
- `training_diagnostics.csv`
- `run_settings.json`
- fold assignments
- 数据文件hash
- 代码版本或脚本hash

## 11. 当前总判断

A组已经表明，单独修正tissue auxiliary标签不足以提高预测效果。当前最合理的研究重心是：

1. 用B组回答辅助任务是否存在可测量的梯度冲突；
2. 用C组解决tissue/HLA没有进入peptide表示的问题；
3. 用D组确认模型是否真正依赖组织条件；
4. 用E组引入flank processing、组织加工机器和parent expression，区分统计
   条件化与生物学机制。
