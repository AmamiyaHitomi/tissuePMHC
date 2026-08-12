# mousePMHC Phase 6 实验报告：困难 H2 定向提升与停止结论

状态：E25–E28 已完成；E29–E31 不执行；E32 冻结 fixed-test 已完成；E33 peptide-disjoint split audit 已完成、模型训练当前不执行。  
开发锚点：Phase 4 E15，5-seed task-balanced Factorized MMoE probability ensemble。  
开发阶段数据边界：仅使用 `data/mousePMHC/mousePMHC_train.csv.gz`、其 train-only OOF 预测以及由官方小鼠 UniProt 参考蛋白组派生的 train flank 特征。模型与五 seed 等权融合冻结后，E32 才一次性读取 `data/mousePMHC/mousePMHC_test.csv.gz`。

## 1. 目标与结论

Phase 6 的目标是改善 E15 中相对较弱的 H2-Kb 与 H2-Kd，而不损害 H2-Db/H2-Kk。结果如下：

1. E25 排除了 pair 构造错误、task 内 peptide 标签冲突作为 Kb/Kd 低分的主要解释；困难样本同时表现出更高的 seed 分歧。
2. E26 的统一 flank/position processing branch 明确失败：相对匹配 E3b 的 macro AUROC 为 `-0.01110`，四个 H2 均下降，停止。
3. E27 的 Kd rank-8 adapter 有弱的 Kd 局部正信号（`+0.00498`），但总体 AUROC 仅 `+0.00079`，不通过。
4. E28 的独立 Kb/Kd rank-8 adapters 是本阶段最好的新候选，macro AUROC `+0.00221`、AUPRC `+0.00288`，但仍低于预注册的 `+0.0030` AUROC 门槛，bootstrap CI 跨零，且 Kb/Kd 组增益远低于定向门槛。因此不进入五 seed 确认。

因此，Phase 6 结束时的唯一冻结开发阶段胜者仍为 E15：

\[
\mathrm{mean\ task\ AUROC}=0.8392,\quad
\mathrm{mean\ task\ AUPRC}=0.8316,\quad
\mathrm{worst6\ AUROC}=0.7101.
\]

上述数值是开发阶段 train-only OOF。冻结后的一次性 E32 fixed test 取得 mean task AUROC `0.8562`、mean task AUPRC `0.8506`、worst-6 AUROC `0.7245`，确认 E15 的内部 pair-disjoint 泛化；它仍不是 peptide-disjoint、protein-disjoint、unseen-H2 或外部队列性能。

## 2. 统一协议

- 24 个 tissue × H2 task，`min_pairs > 200`；6,766 train pairs、13,532 rows。
- 3-fold pair-grouped OOF，split seed `20260711`。
- E26–E28 使用 seeds `20260704/05/06`，并与同 seed、同 task 的 E3b 比较。
- 主指标为 mean task AUROC；保护指标为 mean task AUPRC、worst-6 AUROC、逐 H2 宏平均和逐 task AUROC。
- E25–E28 开发及停止决策期间 fixed test 没有被读取；E32 仅在 E15 结构、超参数、五个 seed 和等权 probability averaging 全部冻结后读取一次。

三 seed 候选的预注册晋级门槛为：macro AUROC 至少 `+0.0030`，macro AUPRC 不低于 `-0.0010`，worst-6 不低于 `-0.0030`；Db/Kk 均不得下降超过 `0.0050`；Kb 或 Kd 至少一组提高 `0.0120`；Kb/Kd 合计至少 6/9 个 task 改善；task bootstrap AUROC CI 下界不低于 `-0.0010`。

## 3. E25：Kb/Kd 数据与残差审计

E25 不训练模型，只读取 train 与冻结的 E15 OOF 成员预测。全部 6,766 个 pair 均满足：两行、一正一负、同 tissue-H2 task、同 parent UniProt、且正负 peptide 不同。没有无效 pair，也没有同 task 内同一 peptide 同时拥有正负标签。

| H2 | task 数 | E15 mean AUROC | train pairs/task | mean five-seed score std |
|---|---:|---:|---:|---:|
| Db | 12 | 0.8946 | 311.8 | 0.0924 |
| Kb | 4 | 0.7335 | 317.0 | 0.1771 |
| Kd | 5 | 0.7970 | 272.4 | 0.1718 |
| Kk | 3 | 0.8292 | 131.3 | 0.1647 |

在 Kb/Kd 内，E15 错排 pair 的成员分歧高于正确 pair：Kb 为 `0.1931` vs `0.1640`，Kd 为 `0.2017` vs `0.1538`。最困难的三个 task 是 pancreas-Kd（AUROC `0.6666`、140 train pairs）、colon-Kb（`0.6956`、208 pairs）和 liver-Kb（`0.7074`、367 pairs）。这支持“困难样本上表示不稳定”的诊断，但不支持数据错误解释。

## 4. E26：flank/position processing branch

E26 使用官方小鼠 UniProt 参考蛋白组派生 parent-protein 上下游各 10 aa flank、相对位置和 N/C 端距离。13,430/13,532 rows（`99.2462%`）可唯一映射；其余 rows 保留，以缺失 token 和 missing indicator 表示。模型在 E3b 主干的 expert 输出处加入组织条件 processing branch，并把进入主干的 scale 初始化为零。

| 指标 | E26 | matched E3b | 差值 |
|---|---:|---:|---:|
| Mean task AUROC | 0.80374 | 0.81484 | **-0.01110** |
| Mean task AUPRC | — | — | **-0.01252** |
| Task wins/losses | 3 / 21 | — | — |

AUROC task bootstrap 95% CI 为 `[-0.01653, -0.00611]`，完全低于零。分 H2 AUROC 变化为 Db `-0.00520`、Kb `-0.02391`、Kd `-0.01267`、Kk `-0.01504`。此外，processing scale 在不同 fold 中正负翻转，未学到稳定方向。

**决策：** 停止统一 flank/position branch。不得在同一 OOF 上再搜索 flank 长度、branch 宽度、scale 初始化或训练轮数。E26 未通过，故不执行与 adapter 的组合 E29。

## 5. E27：仅 H2-Kd 的零初始化 adapter

E27 在 E3b peptide encoder 后仅对 H2-Kd 增加 rank-8 residual adapter。adapter 上投影零初始化，因此初始模型与 E3b 完全相同；训练继续采用 E3b 的 25 epoch task-balanced 协议。

| 指标 | E27 − matched E3b |
|---|---:|
| Mean task AUROC | +0.00079 |
| Mean task AUPRC | +0.00030 |
| AUROC bootstrap 95% CI | `[-0.00125, +0.00280]` |
| Task wins/losses | 14 / 10 |

分 H2 AUROC 变化为 Db `+0.00050`、Kb `-0.00383`、Kd `+0.00498`、Kk `+0.00117`。Kd 的 pancreas、liver、thymus 改善，但 spleen-Kd 下降。E27 未达到总体 `+0.0030`、Kd `+0.0120` 或 CI 门槛。

**决策：** 作为“Kd 私有容量可能有局部价值”的诊断保留，但不晋级。

## 6. E28：独立 Kb/Kd 零初始化 adapters

E28 以相同协议同时加入独立的 Kb 与 Kd rank-8 adapter；不共享二者私有参数，Db/Kk 不使用 adapter。

| 指标 | E28 | matched E3b | 差值 |
|---|---:|---:|---:|
| Mean task AUROC | 0.81705 | 0.81484 | **+0.00221** |
| Mean task AUPRC | — | — | **+0.00288** |
| Task wins/losses | 14 / 10 | — | — |
| AUROC bootstrap 95% CI | — | — | `[-0.00026, +0.00489]` |

分 H2 AUROC 变化为 Db `+0.00048`、Kb `+0.00520`、Kd `+0.00378`、Kk `+0.00248`。Kb/Kd 的局部改善集中于 colon-Kb（`+0.02107`）、pancreas-Kd（`+0.01230`）、liver-Kd（`+0.00786`）和 skin-Kd（`+0.00763`）；但 skin-Kb（`-0.01017`）和 spleen-Kd（`-0.00907`）下降。

E28 满足 AUPRC、worst-6 和 Db/Kk 保护的方向性要求，但失败于最关键的三项：macro AUROC 小于 `+0.0030`、bootstrap CI 下界低于 `-0.0010` 的稳定性要求、Kb/Kd 任一组都未达到 `+0.0120`。因此它不能被表述为超过 E3b 的已确认结构。

**决策：** E28 停止于三 seed OOF；不执行 E30 条件化集成或 E31 五 seed 确认。

## 7. E32：冻结 E15 的一次性 fixed-test confirmation

E32 在不重复模型选择的前提下，核验 Phase 4 E15 的 OOF gate、模型结构、五个预声明 seed（`20260704–20260708`）和等权 probability mean。随后每个 seed 仅在完整 train 上训练一次，并一次性读取包含 24 个 task、每 task 100 pairs 的 fixed test。metadata 记录 `test_data_read=true`、`model_selection_on_test=false`，数据和冻结 gate 均保存 SHA-256；总训练与推理耗时为 `271.36 s`，参数量为 `68,475`。

| 指标 | Train-only OOF | Frozen fixed test | 描述性差值 |
|---|---:|---:|---:|
| Mean task AUROC | 0.8392 | **0.8562** | +0.0170 |
| Mean task AUPRC | 0.8316 | **0.8506** | +0.0190 |
| Worst-6 AUROC | 0.7101 | **0.7245** | +0.0144 |
| Worst-task AUROC | 0.6666 | **0.6840** | +0.0174 |

fixed test 的 24 个 task 中，18 个 AUROC 高于各自 OOF，6 个下降。描述性分 H2 结果为：

| H2 | Task 数 | OOF AUROC | Fixed-test AUROC | 差值 | Fixed-test AUPRC |
|---|---:|---:|---:|---:|---:|
| H2-Db | 12 | 0.8946 | **0.9152** | +0.0206 | 0.9154 |
| H2-Kb | 4 | 0.7335 | **0.7404** | +0.0069 | 0.7404 |
| H2-Kd | 5 | 0.7970 | **0.8397** | +0.0426 | 0.8283 |
| H2-Kk | 3 | 0.8292 | **0.8024** | -0.0268 | 0.7757 |

提升主要来自 H2-Kd 和 H2-Db，H2-Kb 只有较小改善，H2-Kk 则整体下降。fixed test 最困难的 task 为 colon–Kb（AUROC `0.6840`）、pancreas–Kd（`0.7091`）、skin–Kb（`0.7216`）、colon–Db（`0.7277`）、skin–Kk（`0.7405`）和 liver–Kb（`0.7639`）。因此，E32 支持“小鼠 benchmark 上的任务可学习性得到冻结留出集确认”，但不支持“所有 H2 均同等稳定”。

按完整 `pair_id` 聚类的 500 次诊断性 bootstrap 给出 mean task AUROC 95% interval `[0.8452, 0.8662]`、mean task AUPRC `[0.8415, 0.8645]`、worst-6 AUROC `[0.6909, 0.7484]`。Macro PairAcc 为 `0.8425`，worst-6 PairAcc 为 `0.7117`。正式论文若把 CI 作为主推断，应增加 bootstrap 次数并保存完整分析产物。

### 7.1 Fixed-test 的泛化边界

E32 保证 train/test `pair_id` 零重叠，但不是严格 unseen-peptide 或 unseen-protein 测试：

| 审计项 | 结果 |
|---|---:|
| Train/test pair_id overlap | 0 |
| Test unique peptides also seen in train | 2,453 / 3,011（81.47%） |
| Test rows whose peptide is seen in train | 84.31% |
| Test unique parent UniProt also seen in train | 967 / 1,088（88.88%） |
| Test rows whose parent UniProt is seen in train | 93.04% |

因此 E32 应称为 **frozen internal pair-disjoint fixed-test confirmation**，不能称为 peptide-disjoint、protein-disjoint 或外部独立队列验证。

## 8. E33：peptide-disjoint split audit 与当前停止决定

当前 paired benchmark 的负例依赖“同 H2、同 parent UniProt、但在其他组织报告”的跨组织关系。若采用“发现跨 split peptide 后直接删除相关 pair”的朴素方案，确实会切断大量跨组织配对关系，可能使 benchmark 规模骤降，并使部分低资源 tissue–H2 task 样本不足。因此不采用删除重叠实体来强行制造 peptide-disjoint 子集。

不过，E33 的只读 split audit 检验了更合理的替代方案：把共享 peptide 连接的所有 pair 合并为连通分量，再将完整分量分配到 fold。审计结果表明该方案在当前 train 上**可以**同时保持 pair 与 peptide 不相交，而不删除数据：

| 审计项 | 结果 |
|---|---:|
| 总 pairs / 保留 pairs | 6,766 / 6,766 |
| Unique peptides | 5,041 |
| Pair–peptide connected components | 1,844 |
| Largest component | 50 pairs |
| 每个 held-out fold 的 pairs | 2,255–2,256 |
| 每个 held-out fold 覆盖的 tasks | 24 / 24 |
| 每 task held-out pairs | 41–157 |
| 每 task fitting pairs | 82–314 |
| Peptide overlap / pair overlap | 0 / 0 |

因此，不能把“benchmark 必然骤降或 task 必然消失”写成 component-grouped peptide-disjoint OOF 的事实；它只描述朴素删除方案的风险。当前决定是：**保留已完成的 split feasibility audit，但暂不执行 5-seed × 3-fold 模型训练，不产生 peptide-disjoint 性能结论。**若未来执行，应把它作为预先定义的 robustness protocol，并同时报告 component size、逐 task fold 样本量和与 standard OOF 的差异。这个停止决定不消除 standard fixed test 的 peptide overlap 限制。

## 9. 总结与后续状态

Phase 6 表明，当前低资源 24-task benchmark 中，Kb/Kd 的改进不能仅靠扩大同构 MMoE、加入统一 source-protein flank 信息，或添加小型 H2 私有 adapter 获得。E28 的弱正信号说明完全没有私有表示空间的结论也不成立；但效应不足以抵消重复 OOF 选择带来的乐观偏差。

当前最稳妥的结论是：E15 的收益主要来自稳定的全局共享和独立 seed 的概率平均；在当前数据量下，任何 Kb/Kd 特化结构都尚未展示出可晋级的、广泛稳定的额外价值。E32 进一步确认冻结 E15 在内部 pair-disjoint fixed test 上保持良好性能，但 H2-Kk 的下降以及显著的 train/test peptide、protein overlap 仍限定了外推范围。

下一步不应在相同 OOF 上继续调 adapter rank、learning rate、epoch、H2 目标集合或集成权重。若项目继续，优先级应转向：

1. 将 E32 作为小鼠唯一 frozen fixed-test 主结果，不再根据 test 修改模型；
2. 优先恢复 study/PMID provenance，或建立真正独立来源的外部验证；
3. 若开展严格实体泛化，优先采用已审计可行的 connected-component peptide-disjoint OOF；不得从当前 benchmark 直接删除大量跨组织关系。独立 peptide-/protein-disjoint fixed benchmark 仍需另行设计；
4. 收集或整合更多高质量 Kb/Kd immunopeptidomics 数据后，再重新定义预训练和 adapter 研究。

## 10. 可复现性文件

- E25：`scripts/run_mousepmhc_phase6_e25_kb_kd_audit.py`，`results/mousePMHC_phase6_e25_kb_kd_audit/`
- E26 features：`scripts/prepare_mousepmhc_phase6_e26_flanks.py`，`data/mousePMHC/mousePMHC_train_flank_features.csv.gz`
- E26：`scripts/run_mousepmhc_phase6_e26_flank_mmoe_oof.py`，`results/mousePMHC_phase6_e26_flank_mmoe_oof/`
- E27：`scripts/run_mousepmhc_phase6_e27_kd_adapter_oof.py`，`results/mousePMHC_phase6_e27_kd_adapter_oof/`
- E28：`scripts/run_mousepmhc_phase6_e28_kb_kd_adapters_oof.py`，`results/mousePMHC_phase6_e28_kb_kd_adapters_oof/`
- E32：`scripts/run_mousepmhc_phase6_e32_e15_fixed_test.py`，`results/mousePMHC_phase6_e32_e15_fixed_test/`
- E33 split audit（已完成，未执行模型训练）：`scripts/run_mousepmhc_phase6_e33_peptide_disjoint_oof.py`，`results/mousePMHC_phase6_e33_peptide_disjoint_oof/`
- Phase 6 figures：`scripts/build_mousepmhc_phase6_figures.py`，`results/figures_phase6/`（5 张 PNG、source CSV、合并 PDF 与 metadata；包含按 AUROC 降序排列、E3b 虚线基准和证据类型配色的 milestone 图）
