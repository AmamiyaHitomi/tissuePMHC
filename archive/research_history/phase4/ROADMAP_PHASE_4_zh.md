# Phase 4 Roadmap：小鼠模型迁移、表示增强与冻结确认

更新日期：2026-07-13  
状态：规划完成，尚未执行  
编号范围：延续 Phase 3 的 E7，从 **E8** 开始

## 1. Phase 4 目标

Phase 4 不再无差别增加复杂多任务结构，而是在当前 `min_pairs > 200` benchmark 上，将人类 tissuePMHC 项目中证据最充分、且与小鼠数据特点相容的方法逐项迁移到小鼠项目。

本阶段回答四个问题：

1. E3b 的独立 seed 预测平均能否进一步降低低资源 task 的方差？
2. 人类 E29 的 multi-kernel CNN peptide encoder 能否改善小鼠 9-mer motif 表示？
3. 全局共享与 H2-specific 表示是否能够通过 soft fusion 互补，而不是 hard grouping 二选一？
4. H2/tissue auxiliary supervision 能否增强小鼠共享表示，并与 soft fusion、CNN 和 seed ensemble 叠加？

Phase 4 的目标不是保留全部候选，而是通过严格 train-only OOF 筛选出一个冻结结构，最后只对该结构执行一次预注册的 5-seed 确认。

## 2. 冻结 benchmark 与现有证据

- 数据：`data/mousePMHC/mousePMHC_train.csv.gz`。
- 门槛：构建数据时 `min_pairs > 200`；每 task 固定留出 100 pairs test。
- 当前规模：24 个 tissue-H2 tasks、13 个 tissues、4 个 H2 restrictions。
- 训练规模：6,766 pairs、13,532 rows；每 task 124–470 train pairs，中位数 264.5。
- 固定 test 在 Phase 4 模型开发和 OOF 筛选期间不得读取。
- 固定 OOF：3-fold pair-grouped，split seed `20260711`。
- 初始训练 seeds：`20260704`、`20260705`、`20260706`。

当前锚点：

| 模型 | 方法 | OOF mean task AUROC | 状态 |
|---|---|---:|---|
| E0 | BLOSUM62 Random Forest | 0.7530 | 传统单任务基线 |
| E1 | 全共享 peptide encoder + task heads | 0.8057 | 对应人类 E2 |
| E3b | task-balanced Factorized MMoE | 0.8148（3-seed mean） | Phase 4 单模型主锚点 |

E3b 三个 seed 的 mean task AUROC 为 0.8164、0.8110、0.8171，seed SD 为 0.0033，说明独立 seed ensemble 有明确的方差压缩空间。

Phase 3 已证明：H2 hard grouping、PLE、FAMO 和 TAG hard grouping 均未超过 E3b。因此 Phase 4 不再重复这些方向，也不优先迁移 CAGrad、Nash-MTL、Auto-Lambda、SWA、snapshot ensemble 或复杂 stacking。

## 3. 统一评估协议

### 3.1 OOF 与数据读取

- 所有模型选择只使用 train-only OOF。
- 每个 held-out pair 只能由未见过该 pair 的模型预测。
- 输出必须覆盖每个 seed 的全部训练行，且 `(seed, sample_id)` 不得重复。
- test runner 与 OOF runner 分离；候选通过最终预注册门槛前，代码不得读取 fixed test。
- 所有融合必须使用按 `sample_id`、`target_tissue`、`mhc_restriction` 对齐的逐样本 OOF 预测。

### 3.2 主要指标

- 主指标：mean task AUROC。
- 关键次指标：mean task AUPRC、worst-6 mean AUROC。
- 诊断指标：每 H2 mean AUROC、每 task AUROC、seed SD、参数量和训练时间。
- 每次正式比较同时报告逐 task 配对差值和 task bootstrap 95% CI。

### 3.3 通用晋级门槛

结构候选相对其预声明 matched baseline 必须同时满足：

1. mean task AUROC 提升至少 `+0.0030`；
2. mean task AUPRC 不下降超过 `0.0010`；
3. worst-6 mean AUROC 不下降超过 `0.0050`；
4. 任一 H2 组 mean AUROC 不下降超过 `0.0100`；
5. 至少 14/24 tasks 的 AUROC 方向为正。

对于纯融合或增加 seed 的低风险候选，使用更小但预先固定的门槛：mean task AUROC 至少 `+0.0010`，且 mean AUPRC 与 worst-6 均不下降超过 `0.0010`。

门槛只用于决定是否进入下一阶段，不允许观察 fixed test 后修改。

## 4. E8–E15 正式实验序列

| 编号 | 名称 | 人类项目来源 | 核心比较 | 角色 |
|---|---|---|---|---|
| E8 | E3b 3-seed prediction ensemble | E17 | E3b 单 seed均值 vs 三 seed 逐样本平均 | Phase 4 集成锚点 |
| E9 | Multi-kernel CNN shared heads | E29 encoder | CNN-E1 vs matched E1 | 隔离验证 CNN encoder |
| E10 | Multi-kernel CNN Factorized MMoE | E29 encoder + 小鼠 E3b | CNN-E3b vs matched E3b | 表示增强主候选 |
| E11 | Global/H2 soft rank fusion | E8 | global E3b + H2 branch vs global E3b | 验证软选择性共享 |
| E12 | H2/tissue auxiliary shared model | E13 | auxiliary-E1 vs matched E1 | 隔离验证辅助监督 |
| E13 | Auxiliary Factorized MMoE | E13 + 小鼠 E3b | auxiliary-E3b vs matched E3b | 辅助监督主候选 |
| E14 | Auxiliary global + plain H2 soft ensemble | E14a | 胜出 global branch + plain H2 branch | 机制合并候选 |
| E15 | 冻结胜出结构 5-seed ensemble | E17/E29 5-seed | 5-seed vs 同结构 3-seed | Phase 4 最终确认 |

### E8：E3b 3-seed prediction ensemble

目的：先利用现有 E3b 三个 seed 的 OOF 预测建立最低成本、最强的 Phase 4 锚点。

实现：

- 对三个 seed 的逐样本概率等权平均；不重训、不选 seed、不调权重。
- 同时保留单 seed mean、三 seed probability mean，以及 task-wise percentile-rank mean 三种预声明汇总。
- 默认主候选为 probability mean；rank mean 只作为固定消融，不允许按 test 选择。

决策：若 3-seed ensemble 相对三个单 seed 指标均值达到低风险晋级门槛，则 E8 取代 E3b 单 seed均值，成为后续融合与最终 E15 的 ensemble baseline。

### E9：Multi-kernel CNN shared heads

目的：只改变 peptide encoder，隔离检验人类 E29 的主要表示增益能否迁移到小鼠。

结构：

- 保留 E1 的 task-specific linear heads 和训练协议。
- 将 `Embedding -> Flatten -> MLP` 替换为 position-preserving multi-kernel Conv1d。
- 第一版固定 kernel sizes `2/3/5`、每 kernel 32 channels、dropout 0.2。
- 卷积输出保留位置信息后再进入小型 MLP，不做全局 max pooling。

matched baseline：使用相同 folds、seeds、epochs、batch sampling 和可比参数预算重跑的 E1。

决策：E9 通过通用结构门槛后才执行 E10。若 E9 失败，停止 CNN 主线，不通过扩大 channels、增加 kernel 或反复调参追逐 OOF。

### E10：Multi-kernel CNN Factorized MMoE

目的：验证 CNN motif 表示与 E3b 软选择性共享是否可叠加。

结构：

- 保留 E3b 的 task-balanced batches、3 experts、tissue/H2 conditioned gate 和 task heads。
- 仅将共享 peptide encoder 替换为 E9 已冻结的 CNN 配置。
- 不同时调整 expert 数、expert width、gate entropy 或 loss weighting。

matched baseline：相同 folds/seeds 下的 E3b。

决策：若 E10 通过通用结构门槛，则进入最终候选池；否则保留 E9 的 encoder 结论，但不保留 CNN-E3b 组合。

### E11：Global/H2 soft rank fusion

目的：检验全局共享与 H2-specific 表示是否互补，避免 Phase 3 hard grouping 的信息丢失。

第一阶段不训练新结构，优先复用严格对齐的 OOF：

- global branch：E8，或当时已通过的更强 E10 3-seed预测。
- H2 branch：Phase 3 E2 H2-grouped encoder；如其 OOF seed 数不足，则仅补齐相同 seeds，不改变结构。
- 在每个 tissue-H2 task 内分别转换为 percentile rank，再固定 `0.5/0.5` 平均。
- probability mean 作为消融；不得用 fixed test 选择融合规则。

若复用 E2 的融合失败，可执行一次预声明的轻量 H2 residual branch 版本；不得扩展为大型 per-task experts。

决策：达到低风险融合门槛则保留。若 fixed 0.5 rank fusion 失败，停止该分支，不进行 per-task 权重搜索。

### E12：H2/tissue auxiliary shared model

目的：隔离验证辅助监督本身，而不是一开始将 auxiliary、MMoE 和双分支同时改变。

结构：

- 底座为 matched E1 shared encoder + task heads。
- 增加 H2 auxiliary head 和 tissue auxiliary head。
- 初始固定 `lambda_h2=0.10`、`lambda_tissue=0.02`，反映 H2 motif 信号强于 tissue 信号的先验。
- 主任务仍使用原有 sample-level BCE，不采用 FAMO 或其他动态权重。

必须报告 H2/tissue auxiliary accuracy、辅助梯度与主任务梯度 cosine，以及各 H2 组增益。

决策：E12 通过通用结构门槛后才执行 E13。若总体提升不足但 H2-Db/Kb/Kd/Kk 中存在稳定的相反方向，只允许执行一次 `H2-only auxiliary` 预声明消融。

### E13：Auxiliary Factorized MMoE

目的：将 E12 已确认的辅助信号加入当前最强单模型 E3b。

结构：

- 底座为 E3b，或 E10 已胜出时使用冻结的 CNN-E3b encoder。
- 只加入 E12 胜出的 auxiliary 配置。
- MMoE、gate、sampling 和训练超参数保持不变。

matched baseline：相同 encoder、folds 和 seeds，但 auxiliary loss 权重为 0。

决策：通过通用结构门槛则进入 E14/E15 候选池；否则 auxiliary 只保留为 E12 的机制性结果。

### E14：Auxiliary global + plain H2 soft ensemble

目的：迁移人类 E14a 的核心思想，同时针对小鼠 H2 异质性重构，而不是直接复制旧 5-task E14a。

分支：

- global branch：E13；若 E13 未通过，则使用 E10 或 E8 中当时最强的冻结 global branch。
- H2 branch：plain H2-grouped branch，不加入 auxiliary loss。
- 融合：task-wise percentile-rank fixed average，权重固定 `0.5/0.5`。

E14 只允许组合此前已经单独完成 OOF 的冻结分支，不得在 E14 阶段重新搜索 encoder、辅助权重或 per-task fusion weight。

决策：若 E14 超过最佳单分支并达到低风险融合门槛，则成为 E15 的首选结构；否则 E15 使用最佳单模型结构。

### E15：冻结胜出结构 5-seed ensemble

目的：完成 Phase 4 最后一次确认性扩展，而不是继续模型搜索。

预注册要求：

- 在训练新增 seeds 前冻结结构、超参数、融合规则和成员权重。
- 原三 seeds 固定为 `20260704/05/06`；新增 seeds 固定为 `20260707/08`。
- 所有成员等权，不允许删除 seed、选择 4-seed 子集或根据 test 改权重。
- 先完成新增 seeds 的 train-only OOF；只有预注册 OOF 门槛通过后，才允许进入最终 full-train/fixed-test runner。

5-seed 相对同结构 3-seed 的 OOF 门槛：

1. mean task AUROC 提升至少 `+0.0010`；
2. mean task AUPRC 不下降超过 `0.0005`；
3. worst-6 mean AUROC 不下降超过 `0.0010`。

任何一项失败即停在 OOF，并将 3-seed 结构作为 Phase 4 冻结结果。

## 5. 执行依赖与停止规则

```text
E8：E3b 3-seed ensemble anchor
 ├─ E9：CNN on E1 ──通过──> E10：CNN on E3b
 ├─ E11：global + H2 soft fusion
 └─ E12：auxiliary on E1 ──通过──> E13：auxiliary on E3b/CNN-E3b

E8/E10/E11/E13 中冻结最强单分支与有效互补分支
                         │
                         └─> E14：auxiliary global + plain H2 fusion

最佳冻结结构（E8/E10/E11/E13/E14）
                         └─> E15：5-seed preregistered confirmation
```

停止规则：

- E9 失败则不执行 E10。
- E12 及其唯一 H2-only 消融均失败则不执行 E13。
- E11 固定融合失败则不搜索 per-task 权重。
- E14 不超过最佳单分支则不作为 E15 底座。
- E15 完成后冻结 Phase 4，不继续在同一 benchmark 上增加 seeds、融合成员或结构。

## 6. task 数变化下的解释边界

Phase 4 的正式选择只针对 24-task、`min_pairs > 200` benchmark。模型优先级不得自动外推到其他门槛：

| task 数范围 | 预期策略 |
|---:|---|
| 33–43 | 共享收益可能更大，但低资源和 imbalance 更严重；优先 E1/E3b 类轻量共享与 task-balanced sampling |
| 24 | 当前 Phase 4 正式范围；适合 E3b、CNN、soft fusion、auxiliary 和 seed ensemble |
| 14–20 | 减少 expert/encoder 宽度；H2 branch 的有效样本开始不足 |
| 7–10 | 将树模型与小型共享网络并列；不优先 MMoE、PLE 或复杂双分支 |
| 5 | 共享 task 数过少；旧 5-task 结果仅作历史记录，不用于 24-task 模型筛选 |

如需研究 task 数敏感性，应在 Phase 4 冻结之后建立独立的学习曲线/外部验证阶段；不得用不同门槛上的结果反向选择当前 24-task 模型。

## 7. 计划产物与命名

建议脚本：

- `scripts/run_mousepmhc_phase4_e8_e3b_seed_ensemble_oof.py`
- `scripts/run_mousepmhc_phase4_e9_multikernel_cnn_shared_oof.py`
- `scripts/run_mousepmhc_phase4_e10_multikernel_cnn_mmoe_oof.py`
- `scripts/run_mousepmhc_phase4_e11_global_h2_soft_fusion_oof.py`
- `scripts/run_mousepmhc_phase4_e12_auxiliary_shared_oof.py`
- `scripts/run_mousepmhc_phase4_e13_auxiliary_mmoe_oof.py`
- `scripts/run_mousepmhc_phase4_e14_auxiliary_h2_ensemble_oof.py`
- `scripts/run_mousepmhc_phase4_e15_five_seed_confirmation.py`

建议结果目录统一使用：

```text
results/mousePMHC_phase4_e<编号>_<简短名称>/
```

每个实验至少保存：

- `*_oof_predictions.csv`
- `*_oof_per_task_metrics.csv`
- `*_oof_summary_metrics.csv`
- `*_oof_stability_metrics.csv`
- `*_oof_metadata.json`
- 对应的 gate、auxiliary、fusion 或 selection diagnostics

metadata 必须记录：数据路径、task 数、pair 数、fold seed、训练 seeds、超参数、matched baseline、test 是否读取、晋级门槛及实际判断。

## 8. Phase 4 预期最终报告顺序

最终报告不按“实验编号越大越好”叙述，而按证据链组织：

1. E8：独立 seed 集成是否降低方差；
2. E9/E10：CNN encoder 是否提供独立的 motif 表示增益；
3. E11：global 与 H2 branch 是否互补；
4. E12/E13：辅助监督是否增强共享表示；
5. E14：已确认机制能否无调权地叠加；
6. E15：冻结结构的 5-seed 确认及最终 test 结果；
7. 明确结论只适用于当前 24-task closed-set、平衡 pair benchmark。

Phase 4 的理想终点不是最复杂模型，而是一个在 OOF mean AUROC、困难任务下界、H2 组稳定性和 seed 稳定性上都超过 E3b/E8 锚点，并经过固定 5-seed 协议确认的可解释模型。
