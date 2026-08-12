# mousePMHC Phase 5 实验报告：算法增益诊断与停止结论

更新日期：2026-07-13  
状态：E16、E17、E19 已完成；E18、E20、E21、E22、E23 不执行；E24 无新候选可确认  
数据边界：仅使用 `data/mousePMHC/mousePMHC_train.csv.gz`；fixed test 未读取

## 1. 目标与结论

Phase 5 检验三个预注册主假设：BCE 与 task-macro AUROC 的排序目标错配、局部稳定梯度冲突，以及 tissue × H2 层级 head 的部分池化价值。所有实验保持 24 tasks、6,766 train pairs、3-fold pair-grouped OOF（split seed `20260711`）和 E3b 的 encoder/experts/gate 训练预算。

结论是：三个主假设在当前 benchmark 上均未产生可晋级的新候选。

1. E16 没有发现任何稳定冲突层，因此不执行 residual adapter、PCGrad、CAGrad 或 RotoGrad。
2. E17 的 matched-pair 与 all-pair ranking 均未通过单 seed 筛选；不补跑其余两个 seed，也不搜索 `lambda`、`tau` 或其他 ranking 变体。
3. E19 的 hierarchical tissue-H2 head 在单 seed 下显著劣于 matched E3b，且出现小样本 liver-H2-Kk 的局部 interaction 主导。
4. E23 的前置条件（E18-H2 或 E19 单 seed 非劣）不成立；E24 没有 Phase 5 新候选可确认。

因此，Phase 5 结束时的开发阶段胜者仍是 Phase 4 冻结的 E15：5-seed task-balanced Factorized MMoE probability ensemble。该结论仍仅限 train-only OOF，不能替代一次性固定测试确认。

## 2. 统一协议与判定

- 数据：13,532 train rows、6,766 pairs、24 个 tissue-H2 tasks、13 tissues、4 H2 restrictions。
- OOF：3 folds，pair-grouped；同一 `pair_id` 不跨 fitting/held-out partition。
- 开发 seeds：`20260704/05/06`；E17/E19 先运行 `20260704`，只有满足单 seed 非劣条件才补齐另外两个 seed。
- 主指标：mean task AUROC；保护指标：mean task AUPRC、worst-6 AUROC、每 H2 宏平均、MCC 和 Brier score。
- 所有 E16/E17/E19 metadata 均记录 `test_data_read=false`。

三 seed候选的正式门槛是：AUROC 至少比 matched E3b 高 `0.0030`，AUPRC 不低于 `-0.0010`，worst-6 不低于 `-0.0030`，任一 H2 不低于 `-0.0100`，至少 14/24 task AUROC 改善，且 task-paired bootstrap AUROC CI 下界大于 `-0.0010`。

## 3. E16：分层主任务梯度审计

E16 对 E3b 的 24 个主任务 BCE 梯度进行审计。每个 seed-fold 在 epoch `1/13/25` 上重复使用每 task 4 个固定 fitting batches；先跨 batches 平均梯度，再计算 peptide embedding、shared encoder、三个 experts 与 gate 的 task/H2/tissue cosine 矩阵。dropout 在审计时关闭，task heads 与任何 auxiliary loss 均排除。

E16 的 OOF 预测与冻结 E3b 完全一致，三 seed mean task AUROC 为 `0.8148 ± 0.0033`，说明诊断没有改变训练模型。

| 层 | 满足稳定冲突层 | 支持 seed 数 | 支持 fold 数 | 结论 |
|---|---:|---:|---:|---|
| peptide embedding | 否 | 0/3 | 0/3 | 不稳定 |
| shared encoder | 否 | 0/3 | 0/3 | 不稳定 |
| expert 0 | 否 | 0/3 | 0/3 | 不稳定 |
| expert 1 | 否 | 1/3 | 1/3 | 仅一次 middle-to-late 支持 |
| expert 2 | 否 | 0/3 | 0/3 | 不稳定 |
| gate | 否 | 0/3 | 0/3 | 不稳定 |

Gate 的负 cosine 强度经常较高，但相邻阶段 task 冲突矩阵相关性均值只有约 `0.103`，最大约 `0.397`，低于 `0.5`。因此其 task-pair 身份随 seed/fold/epoch 改变，不能解释为可操作的稳定结构冲突。虽然中后期存在明显梯度范数失衡，但 E22 需要同时具备稳定方向冲突并先出现 E20/E21 正信号，条件不成立。

**判定：** 跳过 E18、E20、E21、E22；不得依据个别 OOF task 人工指定 H2 adapter。

## 4. E17：task-wise ranking loss

E17 使用与 matched BCE-only E3b 相同的 pair-aware task-balanced batches。每个 task block 含 8 个完整正负 pair（16 rows）：

- E17a：每个原始 `pair_id` 内的一对正负 logit 排序；
- E17b：同一 task block 内所有正负 logit 两两排序；
- 固定 `lambda=0.25`、`tau=1.0`。

单 seed 结果如下。E17b 的 AUROC 比 E17a 高 `0.000827`，小于预注册的 `0.0010` 平局阈值，因此按规则选择计算更快的 E17a 进行筛选。

| 模型（seed 20260704） | Mean task AUROC | Mean task AUPRC | Mean task MCC | Mean task Brier | Worst-6 AUROC |
|---|---:|---:|---:|---:|---:|
| matched E3b | 0.817311 | 0.806068 | 0.523612 | 0.192269 | 0.681798 |
| E17a matched-pair | 0.815397 | 0.802412 | 0.520579 | 0.192304 | 0.678108 |
| E17b all-pair | 0.816224 | 0.805488 | 0.523690 | 0.192435 | 0.678227 |

E17a 相对 matched E3b 的 AUROC `-0.001914`、AUPRC `-0.003656`、worst-6 `-0.003690`；task AUROC 为 9 胜、15 负，AUROC bootstrap 95% CI 为 `[-0.004345, 0.000466]`。它同时违反 AUROC、AUPRC 与 worst-6 的单 seed 保护线。

即使忽略平局规则而考察 E17b，它的 AUROC 仍为 `-0.001088`、worst-6 为 `-0.003571`，同样未通过。两个 ranking loss 都提升 H2-Kk，但以其他 H2 的下降抵消，不能据此再搜索 H2-specific ranking 或超参数。

训练排序损失正常下降，因而失败不是优化崩溃：E17a 从约 `0.683` 降至 `0.049`，E17b 从约 `0.683` 降至 `0.046`。结论是：当前 pair-batch 内的 surrogate ranking 优化并未转化为 held-out task-macro 排序收益，反而略损害 AUPRC、calibration 和困难任务。

**判定：** 停止 E17；不补跑其余两个 seed；不进入 E24。

## 5. E19：hierarchical tissue-H2 head

E19 保留 E3b backbone，仅将 24 个独立 heads 替换为：

\[
z_{t,h}(x)=z_0(x)+z_t^{\mathrm{tissue}}(x)+z_h^{\mathrm{H2}}(x)+z_{t,h}^{\mathrm{int}}(x).
\]

tissue 与 H2 主效应通过参数组均值中心化实现零和约束；rank-4 interaction 由 tissue/H2 双线性因子双重中心化后生成。所有非全局 logit 分量在初始化时为零，interaction 输出投影零初始化；interaction L2 系数 `5e-4` 高于主效应 `1e-4`。候选有 68,413 参数，略少于 E3b 的 68,475 参数。

| 模型（seed 20260704） | Mean task AUROC | Mean task AUPRC | Mean task MCC | Mean task Brier | Worst-6 AUROC |
|---|---:|---:|---:|---:|---:|
| matched E3b | 0.816357 | 0.806668 | 0.524407 | 0.191574 | 0.680400 |
| E19 hierarchical head | 0.807132 | 0.796679 | 0.508397 | 0.198385 | 0.678657 |
| E19 − E3b | **-0.009225** | **-0.009989** | **-0.016010** | **+0.006811** | -0.001743 |

E19 的 AUROC bootstrap 95% CI 为 `[-0.018727, -0.000560]`，AUPRC CI 为 `[-0.019827, -0.000574]`，均完全低于零；24 tasks 中 8 胜、16 负。分 H2 AUROC 变化为 Db `+0.00301`、Kb `-0.00821`、Kd `-0.02418`、Kk `-0.03461`，后两组明显违反正式保护线。

聚合诊断中，interaction/global logit RMS 比约为 `0.286`，并未在总体上主导。然而逐 task 诊断显示小样本 liver-H2-Kk（每 fold 82–83 fitting pairs）的 interaction/global 比在三个 folds 分别为约 `3.57/3.32/2.48`，其 interaction 参数范数也是最大。该 task 的 AUROC `-0.00797`、AUPRC `-0.02930`、Brier `+0.01918`。因此，尽管聚合脚本阈值标记为 mechanism pass，按 roadmap 的“interaction 爆炸或小样本 task 独占参数即机制失败”应将 E19 解释为局部机制失败。

**判定：** 停止 E19；不补跑其余两个 seed；E23 前置条件不成立。

## 6. Phase 5 总结与后续状态

| 实验 | 结果 | 决策 |
|---|---|---|
| E16 gradient audit | 0/6 稳定冲突层 | 跳过 E18/E20/E21/E22 |
| E17 ranking | 单 seed 同时伤害 AUROC、AUPRC 与 worst-6 | 停止 |
| E19 hierarchy | 单 seed AUROC/AUPRC CI 完全为负；Kk/Kd 明显下降 | 停止 |
| E23 hyper-head | E18-H2/E19 均无单 seed 非劣信号 | 不执行 |
| E24 5-seed confirmation | 无新候选 | 不执行 |

Phase 5 给出的负证据同样具有价值：在当前固定 9-mer、24-task、低资源、pair-grouped OOF benchmark 上，task-balanced Factorized MMoE 的收益来自保留充分的全局共享和后续独立 seed 的概率平均；直接把优化目标改为 batch ranking、依据不稳定梯度施加手术，或强行以 tissue-H2 因子分解替代独立 task head，均未带来可泛化收益。

当前冻结开发阶段选择为：

```text
E15: 5-seed task-balanced Factorized MMoE probability ensemble
mean task AUROC = 0.8392
mean task AUPRC = 0.8316
worst-6 AUROC = 0.7101
```

任何 fixed test 评估只能对这一唯一冻结模型及其预先确定的 E15 对照方案进行一次性确认，不能再用 test 选择结构、seed 或融合权重。

## 7. 结果文件

- E16：`results/mousePMHC_phase5_e16_gradient_audit/`
- E17：`results/mousePMHC_phase5_e17_taskwise_ranking_mmoe_oof/`
- E19：`results/mousePMHC_phase5_e19_hierarchical_heads_oof/`
- E16 runner：`scripts/run_mousepmhc_phase5_e16_gradient_audit.py`
- E17 runner：`scripts/run_mousepmhc_phase5_e17_taskwise_ranking_mmoe_oof.py`
- E19 runner：`scripts/run_mousepmhc_phase5_e19_hierarchical_heads_oof.py`
