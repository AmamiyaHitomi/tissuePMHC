# Phase 4 研究报告：当前进度

更新日期：2026-07-13  
状态：E8、E15 已通过；E9、E11、E12 已停止；E10、E13、E14 不执行；fixed test 尚未读取  
冻结数据：`min_pairs > 200`，24 个 tissue-H2 tasks，13 个 tissues，4 个 H2 restrictions

## 1. 研究边界与协议

Phase 4 仅使用 `data/mousePMHC/mousePMHC_train.csv.gz` 进行模型开发。所有已报告结果均为 3-fold pair-grouped、train-only OOF；固定 test 文件没有被 E8、E9、E11、E12 或 E15 读取。

- 训练数据：13,532 rows、6,766 pairs。
- 每 task 训练数据：124–470 pairs，中位数 264.5。
- OOF folds：3，split seed `20260711`。
- 独立训练 seeds：`20260704`、`20260705`、`20260706`。
- 主指标：mean task AUROC；同时报告 AUPRC、worst-6 AUROC、逐 H2 和逐 task 结果。

当前 Phase 3 单模型锚点是 E3b task-balanced Factorized MMoE：3-seed mean task AUROC 为 0.8148。

## 2. E8：E3b 独立 seed prediction ensemble

### 方法

E8 不重训模型，严格复用 E3b 的三个独立 OOF 预测。每个样本均由三个未见过该 held-out pair 的 E3b 模型预测，并进行等权融合。

预先固定两个候选：

1. probability mean：三个 seed 的概率等权平均，作为主候选；
2. task-rank mean：在每个 tissue-H2 task 内将各 seed 分数变换为 percentile rank 后等权平均，作为固定鲁棒性消融。

E8 代码只读取冻结的 E3b OOF 文件，验证每个 seed 对每个 `sample_id` 恰有一条预测；不读取 fixed test，也不选择 seed 或调整权重。

### 结果

| 模型 | Mean task AUROC | Mean task AUPRC | Worst-6 AUROC |
|---|---:|---:|---:|
| E3b 单 seed 指标均值 | 0.8148 | 0.8043 | 0.6771 |
| E8 3-seed probability mean | 0.8350 | 0.8246 | 0.7050 |
| E8 3-seed task-rank mean | 0.8353 | 0.8245 | 0.7068 |

相对 E3b 单 seed 均值，probability mean 的 AUROC 增益为 `+0.0202`，task-rank mean 的增益为 `+0.0205`。两个候选在 24/24 task 上均高于该 task 的三 seed 单模型 AUROC 均值。

按 task 对跨 seed 平均的 AUROC 差异进行 bootstrap：

\[
\Delta \mathrm{AUROC}_{\text{probability}} \in [0.0156, 0.0250],
\]

\[
\Delta \mathrm{AUROC}_{\text{rank}} \in [0.0165, 0.0244].
\]

两个区间均完全大于零。E8 明确通过 Phase 4 的低风险集成晋级门槛，成为后续全局分支锚点。

按 H2 分组，probability mean 相对单 seed 均值的 AUROC 增益分别为：H2-Db `+0.0137`、H2-Kb `+0.0287`、H2-Kd `+0.0199`、H2-Kk `+0.0350`。这表明 ensemble 不仅改善总体平均，也明显改善了样本较少、单模型方差更高的非 Db H2 组。

三个 E3b seed 的逐样本 score 相关性为 0.764–0.776，并非完全冗余；其逐样本预测标准差均值为 0.1065。这解释了独立 seed 平均为何能带来较大而广泛的增益。

## 3. E9：Multi-kernel CNN shared task heads

### 方法

E9 从人类 E29 迁移 multi-kernel CNN peptide encoder。它保留 E1 的共享 task heads、sample-level BCE、AdamW、epochs、folds 和 seeds，仅将 `Embedding -> Flatten/MLP` encoder 替换为位置保留的 Conv1d encoder：kernel sizes `2/3/5`，每 kernel 32 channels，卷积输出裁剪回 9-mer 原长度后拼接并进入 MLP。

E9 使用与 E1 完全对齐的三个已保存 OOF seed 作 matched comparison。E9 参数量为 119,368；E1 对应参数量约为 21,992。因此其失败不能归因于 CNN 参数过少。

### 结果

| 指标 | E1 三 seed 均值 | E9 CNN 三 seed 均值 | E9 − E1 |
|---|---:|---:|---:|
| Mean task AUROC | 0.8073 | 0.7888 | **−0.0186** |
| Mean task AUPRC | 0.7929 | 0.7740 | **−0.0190** |
| Worst-6 AUROC | 0.6617 | 0.6447 | **−0.0170** |
| Worst-task AUROC | 0.6155 | 0.5951 | −0.0204 |

对每个 task 跨 seed 平均后的配对 AUROC 差异，bootstrap 95% CI 为：

\[
\Delta \mathrm{AUROC}_{E9-E1} \in [-0.0240, -0.0130].
\]

仅 2/24 task 在跨 seed 平均后提高，其余 22/24 下降；三个 seed 合计为 9 次 task-level 提升、63 次下降。

| H2 | task 数 | E9 − E1 mean AUROC |
|---|---:|---:|
| H2-Db | 12 | **−0.0280** |
| H2-Kb | 4 | −0.0121 |
| H2-Kd | 5 | −0.0121 |
| H2-Kk | 3 | −0.0001 |

E9 只在 H2-Kk 整体近似持平，且其中 skin–H2-Kk 平均增益约 `+0.0108`；但 H2-Kk 仅三个 task，不能据此将 CNN 作为全局编码器。E9 在 H2-Db 上的系统性下降表明，当前小鼠 24-task、低资源设置下，E1 的 Flatten-MLP 已能有效利用固定 9-mer 的位置模式，而 CNN 的局部 motif 归纳偏置无法稳定转化为增益。

### 决策

E9 不满足 Phase 4 的任何关键晋级条件：AUROC 和 AUPRC 均显著下降、worst-6 下降、H2-Db/Kb/Kd 均超过组别允许下降、正向 task 数不足。因此停止 CNN 主线，**不执行 E10 CNN-E3b**。不通过增加 channels、改 kernel 或重复调参追逐同一 OOF 结果。

## 4. E11：Global/H2 固定软融合

E11 以 E8 3-seed probability mean 为冻结 global branch，并补齐三 seed 的 plain H2-grouped branch。每个 tissue-H2 task 内，两个分支先独立转换为 percentile rank，再固定作 `0.5/0.5` 平均；probability average 为固定消融。没有执行 per-task 权重搜索。

| 方法 | Mean task AUROC | 相对 E8 | Mean task AUPRC | Worst-6 AUROC |
|---|---:|---:|---:|---:|
| E8 global | 0.8350 | — | 0.8246 | 0.7050 |
| H2 branch 3-seed | 0.8174 | −0.0176 | 0.8084 | 0.6687 |
| probability fusion | 0.8344 | −0.0006 | 0.8260 | 0.6975 |
| rank fusion | 0.8351 | +0.0001 | 0.8262 | 0.6986 |

rank fusion 的 AUROC 增量只有 `+0.00006`，task bootstrap 95% CI 为 `[-0.0048, +0.0047]`，而 worst-6 降低 `−0.0065`。它不满足低风险融合门槛，故不保留为最终结构。

H2 branch 与 global branch 的逐样本 score correlation 为 0.8736：分支有少量互补信息，但整体单分支明显较弱，固定 0.5 混合无法抵消这一损失。唯一局部正信号来自 H2-Kd：rank fusion 平均 `+0.0112` AUROC；但其余 H2 组未形成一致获益，且不能据此在相同 OOF 中追加 H2/task 特异权重搜索。

**决策：** E11 停止，不进入 E14。

## 5. E12：H2/tissue auxiliary supervision

E12 在 E1 shared encoder + task heads 上增加 tissue 与 H2 分类头，固定损失为：

\[
L=L_{\mathrm{main}}+0.02L_{\mathrm{tissue}}+0.10L_{\mathrm{H2}}.
\]

| 指标 | E1 三 seed 均值 | E12 三 seed 均值 | E12 − E1 |
|---|---:|---:|---:|
| Mean task AUROC | 0.8073 | 0.7755 | **−0.0318** |
| Mean task AUPRC | 0.7929 | 0.7568 | **−0.0362** |
| Worst-6 AUROC | 0.6617 | 0.6390 | **−0.0227** |

task-level bootstrap 95% CI 为：

\[
\Delta \mathrm{AUROC}_{E12-E1} \in [-0.0425, -0.0214].
\]

仅 4/24 task 在跨 seed 平均后提高；H2-Db 的 12 个 task 在三个 seed 下全部下降，H2-Db 平均变化为 `−0.0545` AUROC。

辅助诊断支持“无效/冲突信号”的解释：tissue auxiliary accuracy 均值为 0.2143，低于最大 tissue 类 spleen 的 0.2205 比例；H2 auxiliary accuracy 为 0.6926，高于 H2-Db 多数类比例 0.5531，但其共享 encoder 梯度与主任务梯度的平均 cosine 为 `−0.0116`。加权辅助梯度的平均 cosine 也为负（`−0.0130`）。辅助目标没有增强主任务排序，反而引入了稳定的负迁移。

**决策：** E12 停止；不执行 E13。由于没有任何 H2 组呈现稳定、明确的相反正向模式，也不执行 H2-only auxiliary 消融。

## 6. E13/E14 的停止依据

- E13 依赖 E12 的 auxiliary signal 通过；E12 显著失败，因此不执行 E13 Auxiliary MMoE。
- E14 依赖有效的 E11 H2 fusion 或 E13 auxiliary global branch；两项前提均不成立，因此不执行 E14。

这不是遗漏实验，而是预先定义的依赖与停止规则的执行：避免在相同 OOF benchmark 上继续堆叠已经证明无效的机制。

## 7. E15：预注册 5-seed 确认

E15 冻结唯一胜出结构为 E3b task-balanced Factorized MMoE，并固定所有成员等权 probability mean。原成员为 seeds `20260704/05/06`，新增成员为预注册的 `20260707/08`；没有删除成员、选择 4-seed 子集或修改权重。

| 模型 | Mean task AUROC | Mean task AUPRC | Worst-6 AUROC |
|---|---:|---:|---:|
| 冻结 3-seed E8 probability mean | 0.8350 | 0.8246 | 0.7050 |
| E15 预注册 5-seed probability mean | **0.8392** | **0.8316** | **0.7101** |
| 5-seed − 3-seed | **+0.0042** | **+0.0069** | **+0.0051** |

预注册 OOF gate 三项均通过：AUROC 增益超过 `+0.0010`，AUPRC 与 worst-6 均未下降。逐 task 上，AUROC 为 19 胜、5 负；task bootstrap 95% CI 为：

\[
\Delta \mathrm{AUROC}_{5-3} \in [0.0016, 0.0066].
\]

四个 H2 组中，H2-Db、H2-Kb、H2-Kd 平均增益分别为 `+0.0040`、`+0.0024`、`+0.0109`；H2-Kk 平均 `−0.0038`，但只含三个 task，且其中 spleen–H2-Kk 仍提高。新增 seed 的单模型 mean AUROC 为 0.8102 和 0.8127，均在原 E3b 单模型分布内；5-seed 收益来自等权方差降低，而不是异常强的新增成员。

E15 的正式 OOF gate 已通过，但截至本报告更新，metadata 记录 `test_data_read=false`。因此当前可确认的最终结论严格限于 train-only OOF。代码已具备受 gate 保护的 fixed-test runner；是否执行该一次性最终确认是后续单独决策。

## 8. Phase 4 总结与冻结状态

当前证据排序为：

\[
\text{E15 5-seed E3b ensemble} > \text{E8 3-seed E3b ensemble} > \text{E3b single model} > \text{E1 shared heads} > \text{E9/E11/E12 negative candidates}.
\]

Phase 4 的核心结论不是更复杂的 encoder、硬 H2 分支或辅助标签会改善小鼠任务；它们均未超过简单的全局 E3b。真正可复现的性能来源是：

1. task-balanced Factorized MMoE 保留跨 tissue/H2 的全局共享，并以轻量 gate 吸收部分异质性；
2. 5 个独立训练 seed 的等权 prediction averaging 显著降低方差；
3. 固定 9-mer、小样本、多 task 的当前设定下，额外结构化约束更容易产生负迁移。

当前 Phase 4 OOF 主结果为：

```text
E15 preregistered 5-seed task-balanced Factorized MMoE probability ensemble
mean task AUROC = 0.8392
mean task AUPRC = 0.8316
worst-6 mean AUROC = 0.7101
```

结论边界：这些数值针对当前 24-task、`min_pairs > 200`、pair-grouped OOF、平衡 pair benchmark。它们尚不是 fixed-test、严格 peptide-disjoint、protein-disjoint、unseen H2 或外部队列的确认性结果。

## 9. 关键文件

- E8 runner：`scripts/run_mousepmhc_phase4_e8_e3b_seed_ensemble_oof.py`
- E8 结果：`results/mousePMHC_phase4_e8_e3b_seed_ensemble_oof/`
- E9 runner：`scripts/run_mousepmhc_phase4_e9_multikernel_cnn_shared_oof.py`
- E9 结果：`results/mousePMHC_phase4_e9_multikernel_cnn_shared_oof/`
- E11 runner：`scripts/run_mousepmhc_phase4_e11_global_h2_soft_fusion_oof.py`
- E11 结果：`results/mousePMHC_phase4_e11_global_h2_soft_fusion_oof/`
- E12 runner：`scripts/run_mousepmhc_phase4_e12_auxiliary_shared_oof.py`
- E12 结果：`results/mousePMHC_phase4_e12_auxiliary_shared_oof/`
- E15 runner：`scripts/run_mousepmhc_phase4_e15_five_seed_confirmation.py`
- E15 结果：`results/mousePMHC_phase4_e15_five_seed_confirmation/`
- Phase 4 路线：`phase4/ROADMAP_PHASE_4_zh.md`
