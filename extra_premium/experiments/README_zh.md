# Premium tissue auxiliary A0–A5 实验

后续实验 A–E 的状态、设计和执行顺序统一记录在
[`EXPERIMENT_CHECKLIST_A_TO_E_zh.md`](EXPERIMENT_CHECKLIST_A_TO_E_zh.md)。

这些入口均保留原 E14a 结构：一个全局分支、一个按 HLA 训练的 plain
分支，以及固定 `0.5/0.5` 概率融合。实验只改变全局分支的 tissue
auxiliary objective。

| 实验 | 单实验入口 | 全局分支辅助监督 |
|---|---|---|
| A0 | `run_a0_current_auxiliary.py` | HLA + 所有行预测 query `target_tissue` |
| A1 | `run_a1_hla_only_auxiliary.py` | 仅 HLA，不计算 tissue loss |
| A2 | `run_a2_positive_only_tissue_auxiliary.py` | tissue loss 只监督 `label=1` |
| A3 | `run_a3_observed_tissue_multilabel_auxiliary.py` | train-only observed-tissue 多标签 BCE，未观察组织视为 0 |
| A4 | `run_a4_observed_tissue_masked_auxiliary.py` | 只监督观察到的正组织和当前 query tissue，其余组织 mask |
| A5 | `run_a5_other_tissue_count_auxiliary.py` | train-only 其他组织数量三分类：0/1/2+ |

## 三个 seed

所有入口默认运行相同的三个 seed：

```text
20260704 20260705 20260706
```

每次运行都会在训练全局分支之前重置完整随机状态，并在训练 HLA
分支之前再次重置。这保证 A5 新增的 count classifier 不会改变 HLA
分支的初始化、数据 shuffle 或其他随机状态。

可以用 `--seeds` 临时覆盖默认值，例如做单 seed smoke test：

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\experiments\run_a5_other_tissue_count_auxiliary.py --device cuda --epochs 1 --seeds 20260704
```

## All-in-one 入口

下面的命令按 A0 到 A5 的顺序运行全部 18 次训练：

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\experiments\run_all_a_experiments.py --device cuda
```

也可以统一调整训练轮数或 batch size：

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\experiments\run_all_a_experiments.py --device cuda --epochs 25 --batch-size 512
```

## 结果隔离

已有单 seed 结果不会被覆盖。新结果写入：

```text
extra_premium/results/experiments/<experiment_id>/seed_<seed>/
```

每个 seed 目录包含：

- `test_predictions.csv`
- `per_task_metrics.csv`
- `summary_metrics.csv`
- `run_settings.json`
- `training_diagnostics.csv`

每个实验目录额外生成：

- `seed_summary.csv`
- `seed_aggregate.csv`

all-in-one 完成后还会在 `extra_premium/results/experiments/` 生成：

- `A_experiments_seed_summary.csv`
- `A_experiments_seed_aggregate.csv`

A3/A4/A5 不直接使用全数据预聚合的
`reported_tissues_same_hla_uniprot` 或
`other_tissue_presentation_count`，辅助标签只从 premium train 的正例重建。

## B1–B3 train-only OOF诊断

B组不读取premium fixed test，默认运行 `20260704/20260705/20260706`
三个seed，使用固定3-fold pair-grouped OOF：

| 实验 | 入口 | 输出重点 |
|---|---|---|
| B1 | `run_b1_auxiliary_diagnostics.py` | 正负例、tissue及other-count分层辅助准确率/loss |
| B2 | `run_b2_gradient_conflict.py` | 共享peptide encoder上的main–tissue/HLA梯度余弦 |
| B3 | `run_b3_tissue_label_shuffle.py` | 正确tissue标签与fold内打乱标签的匹配OOF比较 |

推荐使用共享入口，B1/B2复用B3的正确标签模型，不重复训练：

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\experiments\run_all_b_experiments.py --device cuda
```

默认结果写入：

```text
extra_premium/results/experiments/B_all_auxiliary_diagnostics/
```

主要文件包括：

- `b1_auxiliary_predictions.csv`
- `b1_auxiliary_metrics.csv`
- `b2_gradient_cosines.csv`
- `b2_gradient_summary.csv`
- `oof_predictions.csv`
- `per_task_metrics.csv`
- `summary_metrics.csv`
- `b3_matched_task_comparison.csv`
- 各分层指标、fold assignments、训练诊断和运行设置

B3的正确标签与打乱标签global模型使用相同初始化和batch顺序，并共享
同一个HLA plain分支。打乱只发生在每个fitting fold内部，不修改主任务
标签。B2在held-out fold上关闭dropout，只审计共享peptide embedding和
encoder参数。

## C0–C4 tissue/HLA条件结构

C组继续使用A0的query-target-tissue和HLA辅助损失，只改变global主分支：

| 实验 | 入口 | Global结构 |
|---|---|---|
| C0 | `run_c0_current_task_head.py` | 当前peptide encoder + task-specific head |
| C1 | `run_c1_late_concatenation.py` | peptide与query tissue/HLA embedding后期拼接 |
| C2 | `run_c2_tissue_hla_film.py` | tissue/HLA FiLM调制 + shared classifier |
| C3 | `run_c3_conditional_task_residual.py` | FiLM + shared/HLA/tissue/task residual |
| C4 | `run_c4_without_task_residual.py` | C3移除task residual |

推荐运行共享入口；每个seed/fold只训练一次HLA plain分支，并供C0–C4
共同使用：

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\experiments\run_all_c_experiments.py --device cuda
```

默认使用 `20260704/20260705/20260706` 三个seed、3-fold
pair-grouped OOF、25 epochs和condition dimension 16。结果写入：

```text
extra_premium/results/experiments/C_all_conditioning_models/
```

主要输出：

- `oof_predictions.csv`
- `per_task_metrics.csv`
- `summary_metrics.csv`
- `matched_c0_comparison.csv`
- `other_count_metrics.csv`
- `seen_unseen_metrics.csv`
- `per_hla_metrics.csv`
- `per_tissue_metrics.csv`
- `training_diagnostics.csv`
- `parameter_counts.csv`
- `fold_assignments.csv`
- `run_settings.json`

所有global模型在初始化前和训练前分别重置随机状态，使batch顺序和dropout
随机流匹配。C3/C4的residual head从零初始化并使用固定L2惩罚。C组不读取
premium fixed test。

## D1–D3 组织条件依赖诊断

D组的 all-in-one 入口只使用 premium train，并对 C0/C2/C3 运行匹配的
3-seed OOF：

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\experiments\run_all_d_experiments.py --device cuda
```

入口同时生成 tissue-swap 分数矩阵、validation tissue shuffle、tissue/HLA/
task residual 推理消融，以及重新训练的 auxiliary-off 对照。结果写入
`extra_premium/results/experiments/D_all_condition_diagnostics/`。

## Pre-E 生物学数据准备

Pre-E不是模型实验，也不占用F编号。它只使用premium train，负责下载或读取
HPA表达数据、审计UniProt→Ensembl映射，并生成人工组织映射模板：

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\experiments\run_all_pre_e_preparation.py
```

人工审核生成的`tissue_mapping_review_template.csv`后，再通过
`--approved-tissue-mapping-csv`运行一次，生成可直接交给E特征预处理器的
`expression.csv.gz`、`protein_mapping.csv`、`tissue_mapping.csv`和
`expression_metadata.json`。Pre-E是确定性数据处理，没有seed；后续E默认使用
单seed `20260704`。耗时只打印到终端，不写入结果文件。

## E0–E4 processing-first 机制实验

新版E组以C4为骨架，并统一使用pair-aware BCE + ranking loss：

| 编号 | 内容 |
|---|---|
| E0 | 匹配C4结构的E组基线 |
| E1 | parent protein真实N/C flanks + MHCflurry processing score |
| E2 | parent-protein query-tissue expression-only负对照 |
| E3 | tissue加工机器表达对flank processing表示进行FiLM调制 |
| E4 | E3加parent expression的完整模型 |

正式运行前需要准备外部特征。第一步从premium train中的parent UniProt下载
蛋白序列和Ensembl映射：

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\experiments\download_e_uniprot_inputs.py
```

然后从Human Protein Atlas官方下载并冻结
`rna_tissue_consensus.tsv.zip`（`https://www.proteinatlas.org/download/tsv/rna_tissue_consensus.tsv.zip`），复制并人工审核
`e_tissue_mapping_template.csv`。准备真实flanks、表达、加工机器表达以及带
flanks的MHCflurry processing score：

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\experiments\prepare_e_mechanism_features.py `
  --protein-fasta extra_premium\external\mechanism\premium_train_parent_uniprot.fasta `
  --uniprot-gene-map data\expression\hpa_v25_1\e_ready\protein_mapping.csv `
  --expression-table data\expression\hpa_v25_1\e_ready\expression.csv.gz `
  --tissue-mapping data\expression\hpa_v25_1\e_ready\tissue_mapping.csv `
  --expression-metadata-json data\expression\hpa_v25_1\e_ready\expression_metadata.json
```

先执行只读审计：

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\experiments\run_all_e_experiments.py `
  --validate-inputs-only
```

审计通过后运行全组。默认只运行单seed `20260704`：

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\experiments\run_all_e_experiments.py --device cuda
```

每个E1–E4模型都会额外生成与自身配对的推理时负对照：processing按task
打乱、expression按完整pair打乱、或将tissue machinery换成另一组织。
连续外部特征只用fitting fold拟合标准化参数。E组不读取premium test，
默认结果写入`extra_premium/results/experiments/E_all_processing_mechanisms/`。

组织映射不确定性采用保守协议：`low_proxy`（blood、bone和umbilical cord
blood）的query/relative expression及tissue machinery在主训练中作为缺失值，
但保留不依赖目标组织的cross-tissue mean expression。原始proxy值只用于
`low_proxy_values_enabled`推理敏感性对照，不参与主训练。结果额外写出
`mapping_quality_metrics.csv`，分别报告exact/synonym、aggregate proxy、
low proxy和non-low-proxy子集。`machinery_missing_fraction`会显式进入模型，
用于区分真实低表达与组织映射不可用。

B–E 的耗时只打印在终端，不写入 `run_settings.json` 或 CSV 结果。
