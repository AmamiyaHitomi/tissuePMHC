# Phase 5 Roadmap：小鼠 tissue–H2 多任务模型的算法增益研究

更新日期：2026-07-13  
编号范围：延续 Phase 4，从 **E16** 开始  
当前锚点：E15，5-seed task-balanced Factorized MMoE probability ensemble  
当前 OOF：mean task AUROC `0.8392`，mean task AUPRC `0.8316`，worst-6 AUROC `0.7101`  
数据状态：fixed test 尚未读取；Phase 5 开发期间继续禁止读取

## 1. 结论与设计原则

Phase 3/4 已经给出足够清楚的边界：全局共享很重要，完全隔离会损失正迁移；更大 encoder、固定双分支融合和辅助分类均造成负迁移；而独立 seed ensemble 有显著、广泛的降方差收益。

| 已有证据 | 结果 | Phase 5 含义 |
|---|---:|---|
| E1 相对最佳单任务 RF | AUROC `+0.0543` | 不退回逐 task 建模；必须保留跨 task 全局共享 |
| E3b task-balanced Factorized MMoE | `0.8148 ± 0.0033` | 冻结为所有单模型候选的 matched baseline |
| E6 硬分组 | `0.7403 ± 0.0040` | 禁止切断 global path 或搜索硬 task grouping |
| E7 H2-Kk residual adapter | `0.8180 ± 0.0013` | 小型 residual 私有路径有可行性，但需稳定冲突证据 |
| E5 FAMO | 单 seed `−0.0170` | 不把动态 loss weighting 当作默认升级方向 |
| E9 CNN / E12 auxiliary | 分别 `−0.0186` / `−0.0318` | 不优先扩容；metadata 用于条件化而非辅助预测 |
| E11 固定 global/H2 融合 | worst-6 `−0.0065` | 不使用弱分支的固定平均 |
| E15 5-seed ensemble | 对 E8 AUROC `+0.0042` | 最终要验证 5-seed 后是否仍优于 E15 |

因此正式主假设仅保留三个：

1. **排序目标错配**：BCE 未直接优化 task-macro AUROC；
2. **局部参数冲突**：少数稳定冲突层需要极小的 residual 偏移，但 global path 保留；
3. **层级任务结构未被显式利用**：task 是 `tissue × H2`，head 应做部分池化而非完全独立。

梯度手术和 hypernetwork 不是首批主实验，只在上述诊断或主实验提供正证据后执行。

## 2. 冻结协议与统一判定

### 2.1 数据与比较协议

- 只读取 `data/mousePMHC/mousePMHC_train.csv.gz`：24 tasks、6,766 pairs。
- 固定 3-fold pair-grouped OOF；split seed 为 `20260711`。
- 开发 seeds 固定 `20260704/05/06`；最终确认才新增 `20260707/08`。
- 每个候选都与同 folds、同 seeds、同训练预算重跑的 E3b 配对比较，不以历史均值替代 matched baseline。
- 除算法本身外，embedding、experts、width、dropout、epochs、task-balanced sampling 和 gate entropy 均与 E3b/E15 相同。
- E24 前不读取 fixed test；最终只允许由唯一冻结模型与 E15 对照一次性读取。

### 2.2 报告项目

主指标为 mean task AUROC；关键次指标为 mean task AUPRC、worst-6 AUROC、worst-task AUROC。每个正式结果必须报告：

1. 逐 task 配对差值及 24 task 胜/负数；
2. task-paired bootstrap 95% CI；
3. 每 H2 宏平均变化；
4. 三 seed 均值、SD 和每个 seed 的值；
5. 参数量、训练时间、峰值显存及机制诊断。

### 2.3 晋级与停止规则

三 seed 单模型候选相对 matched E3b 必须同时满足：

1. mean task AUROC `≥ +0.0030`；
2. mean task AUPRC 不下降超过 `0.0010`；
3. worst-6 AUROC 不下降超过 `0.0030`；
4. 任一 H2 组 AUROC 不下降超过 `0.0100`；
5. 至少 14/24 tasks 的三 seed 平均 AUROC 为正；
6. AUROC 增益的 bootstrap 95% CI 下界 `> -0.0010`。

单 seed 筛选只用于决定是否补齐三 seed：AUROC 不得低于 matched E3b `0.0010` 以上，且 AUPRC、worst-6 均不得显著变差。一次失败即停止该预注册配置；不增加宽度、rank、loss 权重或重新挑 seed 追分。

## 3. 重排后的 E16–E24 实验

| 编号 | 实验 | 贴合度 | 执行状态 |
|---|---|---|---|
| E16 | 分层主任务梯度审计 | 必要诊断 | 必做 |
| E17 | Task-wise ranking loss | 很高：pair 数据 + AUROC 主指标 | 主实验 A |
| E18 | Recon-lite residual adapter | 很高：保留 global sharing 的软修正 | 主实验 B，依赖 E16 |
| E19 | Hierarchical tissue–H2 head | 很高：直接利用 task 的交叉层级 | 主实验 C |
| E20 | PCGrad | 中：仅验证稳定方向冲突 | 条件执行，依赖 E16 |
| E21 | CAGrad | 中：PCGrad 的独立替代方案 | 条件执行，依赖 E16 |
| E22 | RotoGrad | 中低：需同时有方向与尺度失衡 | 条件执行，依赖 E16 |
| E23 | FiLM / low-rank hyper-head | 中高：metadata 条件化 | 条件执行，依赖 E18 或 E19 |
| E24 | 冻结胜者 5-seed 确认 | 必要确认 | 最终实验 |

### E16：E3b 分层主任务梯度冲突审计

目的：只在证据支持时使用梯度手术或私有 adapter，避免把小样本梯度噪声误判为冲突。

实现：

- 对 24 个主 task 分别计算梯度；不包含 H2/tissue auxiliary loss。
- 在每个 fold、seed 的训练早/中/末期各取多个固定 fitting batches，累计平均梯度后再计算 cosine，避免单 batch 噪声。
- 分层输出 peptide embedding、shared encoder、每个 expert、gate 的 task-pair cosine、负 cosine 比例、梯度范数及范数比。
- 同时输出 task、H2、tissue 三种聚合冲突矩阵，以及不同 fold/seed/epoch 矩阵的相关性。

稳定冲突层定义：在至少 2/3 folds、2/3 seeds 和至少两个训练阶段中，均位于负 cosine 强度前 25%，且相邻审计阶段的冲突矩阵相关性 `≥ 0.5`。E16 不按 held-out task 选择专属结构，也不参与模型排名。

### E17：E3b + task-wise ranking loss

目的：让训练目标更接近 mean task AUROC，同时不改变 E3b 的推理结构与全局共享。

AUROC 衡量同一 task 内所有正、负样本的排序，因此预注册两个固定消融：

- **E17a：matched-pair ranking**。仅对同一原始 `pair_id` 的正负样本计算排序损失；
- **E17b：task-wise all-pair ranking（主候选）**。在每个 task-balanced batch 内，对同一 task 的全部正负样本两两计算排序损失。

对 task (t)，使用：

\[
L_{\mathrm{rank}}^{(t)}=
\frac{1}{N_t^+N_t^-}
\sum_{i=1}^{N_t^+}\sum_{j=1}^{N_t^-}
\log\left(1+\exp\left[-(s_i^+-s_j^-)/\tau\right]\right).
\]

各 task 等权平均，并与 BCE 相加：

\[
L=L_{\mathrm{BCE}}+\lambda L_{\mathrm{rank}}.
\]

固定 `lambda=0.25`、`tau=1.0`。两种消融只在 fitting split、同 task 内构造，不得跨 fold 或跨 task。先运行同一单 seed 的 E17a/E17b；仅其中较高 AUROC 者进入三 seed。选择规则固定为单 seed mean task AUROC，若差异小于 `0.0010`，选计算更快的 E17a。必须同时报告 AUPRC、MCC、Brier score；AUROC 提升但 AUPRC/MCC 显著恶化不能晋级。

### E18：Recon-lite 稳定冲突层低秩 residual adapter

条件：E16 找到至少一个稳定冲突层。否则明确跳过 E18，不能靠观察 OOF 人工指定 H2-Kk 或其他 task。

目的：在不重复 E6 硬分组错误的前提下，检验 E7 所暗示的局部软修正是否可泛化。

实现：

- 只在 E16 定义的层插入 adapter；global E3b path 永远保留。
- bottleneck rank 固定 `8`，输出层零初始化，residual scale 初始为 `0`，并对 adapter 施加 L2 收缩。
- 先做参数量匹配的 **E18-control shared adapter**；所有 task 共享一个 adapter。
- 再做 **E18-H2 adapter**；四个 H2 各一个 adapter，不创建 24 套 task adapter。

只有 E18-H2 同时优于 E18-control 和 matched E3b，才可将收益解释为 H2 条件化；若只优于 E3b，则结论仅为“额外的受约束 residual 容量有用”。E18-H2 失败不得扩展到 tissue adapter、per-task adapter 或 rank 搜索。

### E19：Hierarchical tissue–H2 head + low-rank interaction

目的：显式表达 task 的 `tissue × H2` 结构，以部分池化替代 24 个完全独立 head。

使用：

\[
z_{t,h}(x)=z_0(x)+z_t^{\mathrm{tissue}}(x)+z_h^{\mathrm{H2}}(x)+z_{t,h}^{\mathrm{int}}(x).
\]

interaction 由 tissue/H2 embedding 的双线性低秩组合生成，rank 固定为 `4`。所有非全局项零初始化并收缩到零；interaction 的正则强于 H2、tissue 主效应。为避免 global、H2、tissue 项相互抵消，实施零均值/中心化约束：

\[
\sum_h z_h^{\mathrm{H2}}=0,\qquad
\sum_t z_t^{\mathrm{tissue}}=0.
\]

需报告各项 logit 分量范数和小样本 task 的参数范数；interaction 爆炸或少样本 task 独占参数即判为机制失败。

### E20：E3b + PCGrad

条件：E16 证实稳定方向冲突；不要求 E18/E19 成功。PCGrad 与 CAGrad 是独立替代方案，不以“PCGrad 不失败”作为 CAGrad 的前置条件。

仅对 E16 定义的稳定冲突层应用 PCGrad；heads 和其余层保持普通梯度。每 step 的 task 投影顺序由 training seed 固定生成。单 seed 筛选通过后补三 seed；记录投影比例、被投影梯度范数和训练时间。失败后不改投影顺序或挑 task 重试。

### E21：E3b + CAGrad

条件：E16 证实稳定方向冲突。预注册 `c=0.2`，只在稳定冲突层组合 task gradients；其余训练协议与 E20 相同。

PCGrad 与 CAGrad 若都通过三 seed 门槛，保留 mean task AUROC 更高者；差异小于 `0.0010` 时保留训练更快、实现更简单者。

### E22：E3b + RotoGrad

条件：E16 同时显示稳定方向冲突、任务梯度范数存在至少 3 倍且跨阶段复现的失衡，并且 E20 或 E21 出现单 seed 正信号但未通过完整门槛。

该实验只检验“方向 + 尺度”联合修正是否必要。若单 seed 未达到相对 matched E3b `+0.0020` AUROC、且关键次指标不劣，则停止；不扩展到 Nash-MTL。Nash-MTL 保留为未来独立阶段的候选，而非本轮自动实验。

### E23：task-conditioned FiLM / low-rank hyper-head

条件：E18-H2 或 E19 至少一个达到单 seed 非劣门槛。

目的：检验 metadata 条件化是否有效，且与 E12 的 metadata 辅助分类严格区分。

- 输入仅为 tissue/H2 embedding；小型网络只生成 FiLM `gamma/beta` 或 low-rank head delta。
- 不生成完整 encoder；总参数增量不超过 E3b 的 15%。
- `gamma=1`、`beta/delta=0` 初始化，因此训练起点等价于 E3b。
- 只选用 E18 或 E19 已显示有效的调制位置；不同时搜索 FiLM、adapter 和 head 三种位置。

### E24：冻结胜者的 5-seed 确认与预注册固定融合

进入 E24 的候选最多两种：E17 的胜出变体，以及 E18–E23 中三 seed 通过统一门槛的最佳 MTL 候选。若没有新候选通过，E15 保持最终模型，Phase 5 结束。

执行：

1. 训练前冻结候选结构、超参数、训练轮数和新增 seeds `20260707/08`；形成各候选的 5-seed 等权 probability ensemble，不得删除成员。
2. 各候选 5-seed ensemble 先直接与 E15 比较；通过门槛才可能成为最终模型。
3. 若两个候选都通过，才额外执行唯一预注册的 `0.5/0.5` task-wise percentile-rank fusion；不使用预测相关性阈值决定是否融合，也不搜索权重。
4. fusion 相对最佳单 ensemble 必须达到：AUROC `≥ +0.0015`，AUPRC、worst-6 均不下降超过 `0.0010`；否则保留最佳单 ensemble。
5. E24 后冻结 Phase 5。固定 test 仅对唯一胜者与冻结 E15 进行一次性确认，不能据 test 再选模型。

## 4. 执行依赖、顺序和算力控制

```text
E16 gradient audit
 ├─> E18 Recon-lite adapter（稳定冲突层存在时）
 ├─> E20 PCGrad（稳定方向冲突时）
 └─> E21 CAGrad（稳定方向冲突时）
       └─> E22 RotoGrad（另有稳定尺度失衡时）

E17 ranking loss ──────────────────────────────┐
E19 hierarchical tissue–H2 head ──正信号──> E23 hyper-head
                                                   └─> E24 5-seed confirmation
```

推荐实际顺序：

1. E16；
2. E17、E19，以及满足 E16 条件时的 E18；
3. E20/E21 仅在稳定主任务梯度冲突出现时做；
4. E22/E23 为条件性延伸；
5. E24。

正式主假设为 E17、E18、E19。梯度方法最多补充一个三 seed 胜出者；进入 E24 的新候选总数不超过两个，以限制同一 OOF benchmark 的多重比较偏差。

以一次 E3b 单 seed 3-fold 成本为 1：E16 约 `0.3–0.6`；E17/E18/E19 各约 `1–1.3`；E20/E21/E22 因逐 task 梯度约 `2–5`；E23 约 `1.1–1.4`；每个 E24 候选新增两 seed 约 `0.67`。所有运行均记录 wall-clock、峰值显存和失败原因。

## 5. 产物与命名

建议脚本：

- `scripts/run_mousepmhc_phase5_e16_gradient_audit.py`
- `scripts/run_mousepmhc_phase5_e17_taskwise_ranking_mmoe_oof.py`
- `scripts/run_mousepmhc_phase5_e18_recon_lora_adapters_oof.py`
- `scripts/run_mousepmhc_phase5_e19_hierarchical_heads_oof.py`
- `scripts/run_mousepmhc_phase5_e20_pcgrad_mmoe_oof.py`
- `scripts/run_mousepmhc_phase5_e21_cagrad_mmoe_oof.py`
- `scripts/run_mousepmhc_phase5_e22_rotograd_mmoe_oof.py`
- `scripts/run_mousepmhc_phase5_e23_task_conditioned_hyperhead_oof.py`
- `scripts/run_mousepmhc_phase5_e24_five_seed_confirmation.py`

结果目录统一为：

```text
results/mousePMHC_phase5_e<编号>_<简短名称>/
```

每个正式实验至少保存 OOF predictions、per-task metrics、summary、stability、metadata、matched baseline 和机制诊断。metadata 必须记录 `test_data_read`、git commit、完整 CLI、环境、fold/seed、参数量、训练时间、晋级门槛和最终判定。

## 6. 文献依据

- Yu et al., *Gradient Surgery for Multi-Task Learning*, NeurIPS 2020（PCGrad）：https://papers.nips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html
- Liu et al., *Conflict-Averse Gradient Descent for Multi-task Learning*, NeurIPS 2021（CAGrad）：https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html
- Javaloy and Valera, *RotoGrad: Gradient Homogenization in Multitask Learning*, ICLR 2022：https://openreview.net/forum?id=T8wHz4rnuGL
- Shi et al., *Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning*, ICLR 2023：https://openreview.net/forum?id=ivwZO-HnzG_
- Navon et al., *Multi-Task Learning as a Bargaining Game*, ICML 2022（Nash-MTL）：https://proceedings.mlr.press/v162/navon22a.html

文献只提供算法候选；Phase 3/4 的本地 OOF 证据优先。所有结论限于当前 24-task、`min_pairs > 200`、pair-grouped OOF 的平衡 benchmark，不能外推为 fixed-test、peptide-disjoint、protein-disjoint、unseen-H2 或外部队列结论。
