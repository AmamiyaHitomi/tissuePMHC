# Phase 3 Roadmap：小鼠 MHC-I 组织特异性多任务学习

## 当前冻结状态（2026-07-12）

- benchmark：`min_pairs > 200`、每个 task 固定 100 pairs test；共 24 个 tissue-H2 tasks、13 个 tissues、4 个 H2 restrictions。
- train：6,766 pairs（13,532 rows）；每 task 124–470 train pairs，中位数 264.5。
- E0 与 E1 已在这套数据上完成 3-fold pair-grouped train-only OOF；两者均 `test_data_read=false`。
- 旧 `>500`、5-task 的所有 E2+ 结果、脚本假设和性能阈值均为历史记录，不得用于当前比较。
- 当前单 seed 筛选：E2 H2-grouped sharing、E4 Factorized PLE-lite、E5 FAMO 均停止；E5 中的等权 task-balanced MMoE control 成为 E3b，进入 3-seed 确认。

## 已完成基线

| 实验 | 方法 | 当前 OOF mean task AUROC | 角色 |
|---|---|---:|---|
| E0 | BLOSUM62 Random Forest（五个传统候选中最佳） | 0.7530 | 单任务硬基线 |
| E1 | 完全共享 peptide encoder + 24 task heads | 0.8057 | 共享基线 |

E1 相对 E0 的增益具有明显 H2 异质性：H2-Db 的 12 个任务全部提升，平均 `+0.1079` AUROC；H2-Kb、H2-Kk 的组平均增益分别为 `-0.0085`、`-0.0106`。这说明下一阶段的核心问题不是扩大网络，而是按 H2 选择性共享，避免跨 H2 负迁移。

## 统一 OOF 协议

- 开发与候选筛选只读取 `data/mousePMHC/mousePMHC_train.csv.gz`；固定 test 只能由最终冻结的候选读取一次。
- 首轮：3-fold pair-grouped OOF、split seed `20260711`、训练 seed `20260704`。
- 首轮通过后：补 seeds `20260705`、`20260706`，报告 3-seed OOF 均值和标准差。
- 主指标：mean task AUROC；同时报告 AUPRC、worst-6 task AUROC、每个 H2 组 mean AUROC、每 task AUROC、参数量与训练时间。
- 首轮成功：相对 E1，mean task AUROC 至少 `+0.005`；任一 H2 组平均 AUROC 不低于 E1 `0.010` 以上；worst-6 task mean AUROC 不下降超过 `0.010`。
- 不再使用“任一 task 不得下降超过 0.020”作为硬门槛：24 个低资源任务下，单 task OOF 波动过大；仍必须完整报告每 task 差异和 paired bootstrap CI。

## E2–E7：当前正式路线

| 编号 | 方法 | 最小实现与问题 | 首轮决策 |
|---|---|---|---|
| E2 | **H2-grouped hard sharing** | 每个 H2 一个小 peptide encoder；同 H2 tasks 共用 encoder、每 task 保留线性 head。直接检验“跨 H2 共享”是否是负迁移来源。 | 正式首项 |
| E3 | **Factorized MMoE-lite** | 3 个小 shared experts；gate 输入 peptide、tissue、H2；24 个 task heads；记录 gate entropy 与 expert usage。 | E2 后执行 |
| E4 | **Factorized PLE-lite** | global、H2、tissue routes；不设大型 per-task expert，必要时仅保留极小 task residual。 | 仅在 E3 无法保护 Kb/Kk 时执行 |
| E5 | **FAMO** | 在 E3 MMoE backbone 上进行动态任务权重，先 5 epoch 等权 warm-up。 | 已完成单 seed，未优于等权 control，停止 |
| E6 | **TAG-guided grouped training** | 用 train-only 一步任务迁移 affinity 识别应共同训练的 task groups，并与按 H2 的先验分组比较。 | 代码已完成；E3b 在 Kk 组仍有稳定负迁移，因此执行 |
| E7 | **Recon-style partial sharing** | 在每个 train-only seed-fold 审计共享 peptide encoder 的 H2 梯度冲突；为预声明的 H2-Kk 加 residual adapter，保留 E3b 的全局 MMoE 共享（可选 `auto` 审计选择）。 | E6 硬分组失败但 Kk 近乎持平，故作为最终定向候选；代码已完成 |

E3b 使用 E5 内部胜出的等权 task-balanced MMoE control，默认 seeds 为 20260704/05/06；确认通过前不得读取 fixed test。

## 不作为首轮正式候选的方法

- PCGrad、CAGrad、Nash-MTL：24 个 task 的逐任务梯度计算与组合成本高；仅在后续以 4 个 H2 group 梯度做诊断时考虑。
- Auto-Lambda：需要额外 meta-validation；最小 task 仅 124 train pairs，进一步切分 validation 会明显损失训练数据。
- 完整 Cross-Stitch、AdaShare：24 个私有网络或结构搜索在当前样本规模下参数和搜索开销过大。

## 文献依据

- H2 分组和后续自动分组： [TAG, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/e77910ebb93b511588557806310f78f1-Abstract.html)。
- 动态任务权重： [FAMO, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b2fe1ee8d936ac08dd26f2ff58986c8f-Abstract-Conference.html)。
- 冲突层拆分： [Recon, ICLR 2023](https://iclr.cc/virtual/2023/poster/11669)。
- 选择性参数共享： [AdaShare, NeurIPS 2020](https://papers.nips.cc/paper_files/paper/2020/hash/634841a6831464b64c072c8510c7f35c-Abstract.html)。

## 实现与结果位置

- 数据审计：`results/mousePMHC_phase3_data_audit/`。
- E0：`scripts/run_mousepmhc_phase3_e0_oof.py`。
- E1：`scripts/run_mousepmhc_phase3_e1_oof.py`。
- E1 补充 seeds 20260705/06：`scripts/run_mousepmhc_phase3_e1_additional_seeds_oof.py`；输出到 `results/mousePMHC_phase3_e1_oof_additional_seeds/`，不覆盖 seed 20260704。
- E2：`scripts/run_mousepmhc_phase3_e2_h2_grouped_oof.py`。
- E3：`scripts/run_mousepmhc_phase3_e3_factorized_mmoe_oof.py`；正式 min200 输出为 `results/mousePMHC_phase3_e3_factorized_mmoe_min200_oof/`。
- E4：`scripts/run_mousepmhc_phase3_e4_factorized_ple_oof.py`；正式 min200 输出为 `results/mousePMHC_phase3_e4_factorized_ple_min200_oof/`。
- E5：`scripts/run_mousepmhc_phase3_e5_famo_mmoe_oof.py`；正式 min200 输出为 `results/mousePMHC_phase3_e5_famo_mmoe_min200_oof/`。
- E3b：`scripts/run_mousepmhc_phase3_e3b_task_balanced_mmoe_oof.py`；对 E5 中胜出的等权 task-balanced MMoE 进行 3-seed 确认，输出为 `results/mousePMHC_phase3_e3b_task_balanced_mmoe_min200_oof/`。
- `scripts/run_mousepmhc_phase3_e2_structured_ple_oof.py` 是旧 5-task 历史实现，不得直接运行。
