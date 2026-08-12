# 第三轮审阅新问题清单

> 依据 `pic_issues_3` 中 14 张截图及 `细节总结.txt` 汇总、去重。  
> 本轮工作的前提是：使用新的 occurrence-balanced 数据重新计算全部结果；正、负样本在所有数据集中作为 positive 出现的次数应相同。本轮任务应作为独立于旧软件、旧结果的重建工作处理，不能沿用旧数值。

## P0：数据与实验必须先解决

- [ ] **用新数据重跑全部实验并替换全文结果。** 更新正文数字、表格、图、附录、图注、表注及交叉引用，确保不残留旧数据结果。
- [x] **验证 occurrence balancing。** 已逐 pair 核验 Human/Mouse 的 train、test 和完整数据：正负样本的 positive occurrence 总数与完整频数分布完全相同，mismatch 均为 0；可复核结果见 `extra3/occurrence_balancing_audit.json`。Figure 2 的可视化更新仍在后续图形任务中完成。
- [x] **消除旧数据导致的 bias 表述。** 已验证 occurrence-count shortcut 消失：每个 task 的正负 occurrence 分布相同，count-only AUROC 的最小值、均值和最大值均为 0.5000。论文已改为“该特定捷径被控制”，并保留 IEDB coverage、study/batch、donor、检测深度等残余偏倚限制，避免声称所有 bias 均消失。
- [x] **统一评估协议并核实实际是否使用 ordinary cross-validation。** occurrence-matched 实验仅使用固定 train/test；没有 ordinary 或 peptide-separated cross-validation 的 fold/prediction 产物。v7 的协议表、流程图和方法描述已改为 fixed-test-only，旧交叉验证附录已从主文排除。
- [x] **明确 Human 与 Mouse 的重复运行设置。** 两个物种均使用 seeds 20260704、20260705、20260706；默认汇总为 seed-level metric 的算术均值和样本标准差，只有明确标记 ensemble 时才表示行级预测平均。
- [x] **调查并解释物种间数据规模不对称。** Human 为 77 tasks/25,339 pairs，Mouse 为 11 tasks/2,639 pairs；两者均固定每 task 50 个测试 pairs，测试规模差异来自 task inventory，训练规模差异来自共同筛选和 occurrence matching 后的可用记录覆盖。

## P1：论文整体逻辑与结构

- [ ] **重新设计论文大纲。** 按一般论文叙事顺序重排：数据与任务定义 → 数据平衡与泄漏审计 → 评估协议 → 主结果 → 严格泛化 → 消融/机制分析 → tissue/allele 异质性 → 外部控制与补充结果。
- [ ] **调整全部图表顺序和正文引用。** 图表应在首次被解释的位置出现；消除“正文未提到表格”“结果先出现、定义后出现”和同一结果在正文/附录反复出现的问题。
- [ ] **统一图表编号。** 解决两个 Table 15、两个 Table A2 及 “continued” 表拆分造成的编号和引用混乱；删除、合并后重新编号并更新所有交叉引用。
- [ ] **统一数字精度。** 论文中的性能数值原则上统一保留小数点后 4 位；百分比、计数和统计量另按预先声明的规则统一。

## P1：主文表格

> 已完成：原 Table 1--18 的问题经删除、合并和重构后，对应当前连续编号的 Table 1--13；旧编号不再机械保留。

- [x] **Table 1：删除。** 不再用该表承担 literature survey；必要的相关工作比较改为正文叙述或规范的主实验基线表。
- [x] **Table 2：按新数据更新。** 同步更新正文引用和结论。
- [x] **Table 3：按新数据更新。** 同步更新正文引用和结论。
- [x] **Table 4：按新数据更新并加入 Human。** 当前只给出 Mouse entity-overlap audit，需在同一张表中并列 Human 与 Mouse，并清楚区分 pair、peptide、parent protein 以及 current-/other-combination overlap。
- [x] **Table 5：将两段合并为一张表。** 只保留真正的 evaluation mode，不再收录过于泛化的方法术语、模型描述、重复训练或预测平均等内容；删除未实际使用的 ordinary cross-validation 条目。
- [x] **Table 6：按新数据更新并说明与 Table 4 的关系。** 明确两表分别回答 fixed-test overlap 还是 cross-validation overlap；若信息实质重复，应合并或删去其一。
- [x] **Table 7：重做 label-conflict audit。** 至少报告只有 1 个 label 的 peptide 数、超过 1 个 label 的 peptide 数、其中发生 label conflict 的 peptide 数；冲突比例应以“超过 1 个 label 的 peptide”为直接相关分母，同时保留必要的总量信息。
- [x] **Table 8：扩展至至少 10 种方法。** 方法集合参考现 Table 12，并保证数据划分、任务集合和指标口径可比较。
- [x] **Table 9：删除或证明其必要性。** 当前正文没有引用，也不清楚该两模型比较能支持什么结论；若保留，必须补正文引导、实验目的及与主结果的区别。
- [x] **Table 10：明确是否为 ablation study。** 若是消融实验，应使用相同数据、fold、seed 和指标逐项移除组件；若不是，应改名并将混在一起的两类信息拆成清楚的列。还要说明它与 Table A2 的关系，避免重复证据。
- [x] **Table 12：按新数据更新并增加指标列。** 除 AUROC/AUPRC/Worst-k 外，补充适合的 task-level、稳健性或不确定性指标；核对 Human/Mouse 数据量差异。
- [x] **Table 13：取消单行表。** 优先改成信息量更高的图；若必须保留表格，则增加多个指标和足够的比较对象。
- [x] **Table 14：按新数据更新。** 同时作为整合 Table A4/A5 的位置，统一 tissue-blind/external-control 结果。
- [x] **前一个 Table 15：按新数据更新并扩充方法。** 方法集合与 Table 12 对齐，说明哪些结果可直接比较。
- [x] **后一个 Table 15：删除独立重复表并并入 Table 8。** 合并后避免 Human architecture comparison 重复出现。
- [x] **Table 16：按新数据更新并在正文解释。** 当前 cross-validation 结果未被正文承接；需补充结果解读、与其他评估的关系，并说明各方法为何入选。
- [x] **Table 17：按新数据更新。** 保留每个 tissue 的 task 数量，重新计算全部指标，并验证新数据下原有 bias 是否消失。
- [x] **Table 18：改为极值概览。** 分别列出表现最高和最低的 5 个四位分型（four-digit allele typing），写清排序指标、task 数和必要的稳定性指标。

## P1：主文图

- [x] **Figure 1：只展示 1 个代表性 tissue。** 已改为单一 Human lung 通路，逐层列出 source/cell-state、proteasome、TAP、ERAP、loading-complex 和 surface MHC-I 相关基因；图注说明 lung 的 7-task 选择依据、各层作用及非因果范围。
- [x] **Figure 2：按新数据更新。** 已用完整 occurrence-equal Human/Mouse train+test 逐 tissue 展示 label 0/1 的 total occurrence；13 个 Human tissue 和 4 个 Mouse tissue 的差值均为 0，图注给出求和定义和 log 轴说明。
- [x] **Figure 3：核实 ordinary cross-validation。** Figure 3 仅保留 TissuePMHC 架构，不再出现 ordinary CV；训练内三折只在 Figure 4 和 evaluation-protocol 表中作为调参步骤出现，固定测试是唯一结果协议。
- [x] **Figure 4：重点重设计。** 已重画为上下分离的 training-only 与 untouched-fixed-test 信息流，加入阶段编号、边界、方向箭头和一致的功能色，明确 tuning、contract locking、final fitting 与 reporting 的关系。
- [x] **Figure 5：删除 B 面板，只保留并扩展 A。** 已改为单面板 23-method Human architecture survey，AUROC/AUPRC 及 seed SD 与主结果表完全同源、同任务、同口径。
- [x] **Figure 6：按新数据更新并编码 tissue 信息。** 已展示全部 77 个 Human 与 11 个 Mouse 调参后任务；点颜色仅编码 tissue，navy 横线仅作为 tissue mean 摘要，随机参考线明确为非数据。
- [x] **Figure 7：按新数据更新。** 已使用调参后与调参前的同任务 AUROC 差，正文和图注同步 Human/Mouse 均值、10,000 次 task bootstrap 区间、改进任务数以及 Holm 校正的双侧配对 Wilcoxon 结果。

## P2：附录表图

- [x] **Figure A1：删除。** 旧 ordinary/peptide-separated generalization 图及附录入口均已移除；新稿不报告未在 occurrence-matched 数据上重跑的严格泛化数值。
- [x] **Table A1：删除。** 该表无独立信息价值且无必要正文引用，未在 v7 中保留。
- [x] **Table A2：删除并去重。** 两段重复编号和筛选标准不清的 “selected paired AUROC” 表均已移除；可比较的完整系统证据只在当前主文 Table 8（component-related comparisons）与完整 architecture tables 中报告。
- [x] **Table A3：删除并统一口径。** 主文 architecture tables 已用相同三 seeds 报告 mean $\pm$ sample SD；方法部分已明确默认不跨 seed 平均行级预测，因此不再保留重复且易混淆的 stability/averaging-gain 附表。
- [x] **Table A4 与 A5：已按新数据合并进主文。** 当前 Table 13 在同一 occurrence-matched fixed-test 协议下并列 external-only、TissuePMHC、combined、差值与 pair accuracy，Human/Mouse 使用相同表头和展示顺序。

## P2：说明与一致性检查

- [x] **为每张保留的表回答“这张表说明什么”。** 当前 13 张表和 7 张图均有正文引导；表题/图注区分数据审计、模型结构、evaluation protocol、固定测试性能、异质性和外部控制，并明确适用的数据划分、任务、seeds/确定性计算、指标与结论边界。
- [x] **清理过度泛化的术语表。** 已独立设置 Evaluation Protocol 章节，并将 data partition、training-only model selection、single-seed fitting、fixed-test evaluation、prediction aggregation 与 statistical analysis 分层定义；模型、优化、split 和统计不再混称 evaluation modes。
- [x] **检查所有表图的可比性。** 已加入全稿 comparability contract；species-specific ranking 内统一 fixed test、task inventory 和三 seeds，row-level aggregate 用 $\dagger$ 标记；task/tissue/allele 的两阶段聚合、确定性数据审计、external-only 指标及 Human Worst-10/Mouse Worst-5 的不可直接互换性均已注明。
- [x] **完整检查 Human/Mouse 对称性。** Human/Mouse 基准统计、平衡/泄漏审计、主结果和外部控制已统一定义、指标及展示顺序；Mouse 主结果补齐 Accuracy/MCC，H2 汇总补齐 task 数。23-vs-20 的方法库存、77-vs-11 的任务规模、Worst-k 以及 29 个 HLA typing 仅展示极值而 4 个 H2 restriction 全展示的原因均已明确说明。

## 建议执行顺序

1. 冻结新数据及 occurrence-balancing 验证结果。
2. 确认 evaluation modes、fold 和 run/seed 设计。
3. 重跑全部实验并产出统一结果表。
4. 先完成 Table 4、5、7、8、10、12、14、16–18 的结构重建。
5. 再重绘 Figure 1–7，删除 Figure A1。
6. 合并/删除重复附录表，统一编号与交叉引用。
7. 按新大纲重排正文，最后做数值、术语、图表引用和四位小数的一致性审计。
