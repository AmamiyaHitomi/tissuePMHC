# Phase 6 Roadmap：小鼠 tissue–H2 的困难 H2 定向提升

编号范围：延续 Phase 5，从 **E25** 开始。  
开发锚点：Phase 4 E15，5-seed task-balanced Factorized MMoE probability ensemble。  
当前开发结果：mean task AUROC `0.8392`，mean task AUPRC `0.8316`，worst-6 AUROC `0.7101`。

## 1. 研究问题与边界

E15 的低分 H2 并不等价：H2-Kb 的所有已试基线都较低，说明该组可能信号较弱；H2-Kd 在全局共享 E3b 中相对 E1 出现过下降，提示可能仍有残余负迁移。Phase 6 的目标是以**新的、与组织加工过程相关的输入信息**和**受约束的 H2 私有容量**改善 Kb/Kd，而不是继续在相同 9-mer 特征上搜索通用 MTL 优化器。

本阶段只读取 `data/mousePMHC/mousePMHC_train.csv.gz` 与既有 train-only OOF 文件。固定 test 在整个开发和候选筛选中禁止读取。所有性能结论均为 train-only OOF；不得将其叙述为独立测试性能。

## 2. 已有证据

| H2 | E0 RF | E1 shared | E3b single (3-seed) | E15 5-seed | Phase 6 含义 |
|---|---:|---:|---:|---:|---|
| Db | 0.7471 | 0.8570 | 0.8768 | 0.8946 | 保持全局主干，设保护线 |
| Kb | 0.7069 | 0.6969 | 0.7023 | 0.7335 | 绝对信号弱；优先补充 processing 特征 |
| Kd | 0.7648 | 0.7750 | 0.7662 | 0.7970 | 可能有残余负迁移；检验受约束 adapter |
| Kk | 0.8187 | 0.8100 | 0.7980 | 0.8292 | 不作为本阶段定向对象 |

E15 相对原始 3-seed E3b 单模型均值的集成增益在 Kb/Kd 分别约为 `+0.0311` / `+0.0308`，高于 Db 的 `+0.0177`。因此不再增加普通同构 seed；应寻找能降低 Kb/Kd 系统误差的异构信息。

已停止且不重试的方向：H2 hard grouping、固定 global/H2 soft fusion、CNN encoder、tissue/H2 auxiliary classification、batch ranking loss、PCGrad/CAGrad/Rotograd、层级 interaction head、继续增加 E15 同构 seeds。

## 3. 通用协议与晋级规则

- 任务范围固定为 24 个 `min_pairs > 200` 的 tissue × H2 task；3-fold pair-grouped OOF，split seed `20260711`。
- 首轮候选使用 seeds `20260704/05/06`；仅通过首轮 gate 的唯一结构才允许追加预先指定 seeds `20260707/08`。
- 主要指标：mean task AUROC。保护指标：mean task AUPRC、worst-6 AUROC、各 H2 macro AUROC、每 task AUROC、Brier score。
- 除明确写为诊断的 E25 外，每个新候选必须和同 seed、同 fold 的 E3b 重训对照比较；不得只和 E15 的 5-seed 集成比较。
- 三 seed 晋级 gate：

\[
\Delta\mathrm{AUROC}_{\mathrm{macro}}\ge 0.0030,\quad
\Delta\mathrm{AUPRC}_{\mathrm{macro}}\ge -0.0010,\quad
\Delta\mathrm{worst6}\ge -0.0030.
\]

同时要求：Db、Kk 任一组下降不得超过 `0.0050`；Kb 或 Kd 至少一组提高 `0.0120`；Kb/Kd 合计至少 6/9 task 改善；task-paired bootstrap AUROC 95% CI 下界不低于 `-0.0010`。每个结构只允许一个预先写明的超参数配置；超参数消融只在其对应实验内按预先固定规则做一次选择。

## 4. 实验序列

| 编号 | 候选 | 主要假设 | 结果后的动作 |
|---|---|---|---|
| E25 | Kb/Kd data and residual audit | 低分是否来自标签冲突、数据稀疏、seed 方差或特定 motif/样本 | 仅诊断，不产生模型胜者 |
| E26 | E3b + source-protein flank/position branch | 9-mer 以外的 N/C flank、蛋白位置可补足组织加工信号 | 满足 gate 才进入 E29 |
| E27 | E3b + Kd zero-init low-rank adapter | Kd 可通过小型私有残差修复且不伤害 Db/Kk | 满足 gate 才进入 E29 |
| E28 | E3b + Kb/Kd zero-init low-rank adapters | 同时私有修正两个低分 H2 可超过单 Kd adapter | 与 E27 直接比较，最多保留一个 |
| E29 | E26 与 E27/E28 的冻结组合 | processing 信息与私有容量互补 | 仅在两个父候选均通过时执行 |
| E30 | 固定 H2-conditioned probability ensemble | specialist 仅在 Kb/Kd 以固定权重补充 E15 | 只允许预注册的 0.25 权重；不得 task-wise 调权 |
| E31 | 5-seed confirmation | 验证唯一 3-seed 胜者的降方差收益 | 通过后冻结 Phase 6 |

### E25：数据与残差审计

E25 只读 train 和 E15 OOF 预测，不训练模型、不读 fixed test。必须输出：pair 标签完整性、task/H2 样本和 UniProt 覆盖、同 task peptide 标签冲突、E15 的 AUROC/AUPRC 与成对 margin、五 seed 分歧、按 H2 的氨基酸位置富集，以及 Kb/Kd 困难任务清单。其目的在于排除数据构造问题并决定 E26 的 flank 提取是否可行；不得根据某一个 task 的观察结果建立 task 特异模型。

### E26：flank/position processing branch

从已记录的 parent UniProt 序列中提取目标 peptide 的上下游各 10 aa、N/C 端距离和相对位置。输入分为保持冻结的 E3b 9-mer 主干与新 branch：

\[
h = h_{\mathrm{E3b}} + \alpha\,h_{\mathrm{flank}}(N_{10},C_{10},\mathrm{position},e_{\mathrm{tissue}}),
\]

其中 \(\alpha=0\) 初始化。若 parent 序列无法唯一映射，样本使用缺失 token 和显式 missing indicator；不得删除这些样本。branch 的 width 固定为 32，dropout 固定为 0.2，先冻结 E3b 主干训练 8 epochs，再解冻 encoder/experts 的最后线性层训练 8 epochs。不得额外搜索 flank 长度或训练轮数。

外部 pMHC predictor 分数只能作为探索性诊断，不能直接作为 E26 主结果特征，除非逐条审计其训练数据与当前 IEDB 记录不存在重叠。

### E27/E28：零初始化低秩 H2 adapter

adapter 位于 E3b peptide encoder 后、experts 前：

\[
h'=h+\alpha_{a}U_a\sigma(V_a h),\qquad a\in\{\mathrm{Kd},\mathrm{Kb}\}.
\]

`rank=8`，\(\alpha_a=0\) 初始化，adapter 输出层零初始化，adapter 参数采用主干 10 倍 weight decay。第一阶段冻结 embedding、encoder、experts，只训练 adapter 与 task heads 8 epochs；第二阶段解冻 experts 的最后线性层 8 epochs，并加入 L2-SP 锚定到 E3b 初始权重。E27 只启用 Kd；E28 同时启用 Kb/Kd。它们不是 hard grouping，也不得增加 task 私有 interaction 参数。

### E29/E30/E31：组合、集成与停止

E29 只能组合已经独立通过的 E26 和 E27/E28；结构、训练顺序和超参数等于其父候选的固定并集。E30 只在 E29 或 adapter 候选通过、且其 Kb/Kd OOF score 与 E15 不完全共线时执行。它对 Db/Kk 恒用 E15，对 Kb/Kd 使用固定：

\[
s=0.75s_{\mathrm{E15}}+0.25s_{\mathrm{specialist}}.
\]

任何 E30 的权重、H2 范围或 task 权重均不得由同一 OOF 再选择。E31 对唯一胜者执行五 seed 等权确认；确认后冻结 Phase 6。固定 test 只允许对冻结 E31 胜者和 E15 做一次性确认，不能据 test 重新选结构、seed 或权重。

## 5. 预期与解释

若 Kb 提高 `0.03`、Kd 提高 `0.02`，而其余任务不变，则宏平均预计提高：

\[
\frac{4\times0.03+5\times0.02}{24}=0.00917,
\]

即 E15 的 `0.8392` 可到约 `0.8484`。这是用于量化目标的情景计算，不是成功承诺。若 E25 显示 Kb/Kd 的错误主要是标签冲突或同一 seed 稳定错误，应优先收集更多高质量组织–H2 immunopeptidomics 数据，而不是扩大模型。

## 6. 产物

- E25 runner：`scripts/run_mousepmhc_phase6_e25_kb_kd_audit.py`
- E25 results：`results/mousePMHC_phase6_e25_kb_kd_audit/`
- 后续每个实验：`scripts/run_mousepmhc_phase6_e<编号>_<name>.py`
- 结果目录：`results/mousePMHC_phase6_e<编号>_<name>/`
