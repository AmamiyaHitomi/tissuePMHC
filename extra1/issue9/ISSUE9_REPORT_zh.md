# 第 9 项完成报告：strict 协议下的架构对照

完成日期：2026-07-23  
报告整理日期：2026-07-25  
状态：已完成

## 1. 问题与目标

原有 connected-component peptide-disjoint OOF 仅包含冻结主模型：

- human：三 seed TissuePMHC；
- mouse：五 seed Factorized MMoE。

这些结果能够证明在 seen-task、unseen-peptide 条件下仍存在可学习信号，但不能证明主模型在相同 strict 条件下优于简单架构。第 9 项的目标是在完全相同的 peptide-component folds 上补齐架构对照，并区分：

1. strict task learnability；
2. strict architecture superiority；
3. 单模型架构收益与多 seed ensemble 收益。

## 2. 实验协议

### 2.1 Frozen splits

所有模型直接读取既有 fold manifest，不重新生成 folds。

| 物种 | Pairs | Components | Tasks | Outer folds | Peptide overlap | Pair overlap |
|---|---:|---:|---:|---:|---:|---:|
| Human | 73,798 | 17,036 | 157 | 3 | 0 | 0 |
| Mouse | 6,766 | 1,844 | 24 | 3 | 0 | 0 |

Human manifest SHA256：

```text
e0fccb2adcf809d055b35284279e6027ec31c8f590cad0d88726f29954ba4c29
```

Mouse manifest SHA256：

```text
32e7f865a1e6ebc0f0a7749e086ceb1a6ea5074f593fac742afe850300a73eb1
```

### 2.2 Models

Human：

1. one-hot logistic regression；
2. BLOSUM62 random forest；
3. shared peptide encoder with task-specific heads；
4. plain MLP dual branch；
5. auxiliary MLP dual branch；
6. frozen TissuePMHC。

Human 神经模型采用 seeds：

```text
20260704, 20260705, 20260706
```

Mouse：

1. BLOSUM62 random forest；
2. shared peptide encoder with task-specific heads；
3. frozen Factorized MMoE。

Mouse 模型采用 seeds：

```text
20260704, 20260705, 20260706, 20260707, 20260708
```

### 2.3 Common neural training settings

- epochs：25；
- batch size：512；
- optimizer：AdamW；
- learning rate：0.001；
- weight decay：0.0001；
- dropout：0.2；
- gradient-norm cap：1.0。

传统分类器使用各自冻结的算法设置，不强制套用神经网络 optimizer/epoch 规则。

### 2.4 Evaluation and statistics

逐 task 计算：

- AUROC；
- AUPRC；
- PairAcc；
- half-tie PairAcc；
- worst-task AUROC；
- human worst-10 mean AUROC；
- mouse worst-6 mean AUROC。

主 PairAcc 严格使用：

```text
positive_score > negative_score
```

统计分析包括：

- mean/median task difference；
- Hodges--Lehmann difference；
- win/tie/loss；
- 10,000-replicate task-bootstrap interval；
- two-sided Wilcoxon signed-rank test；
- species × metric family 内的 BH-FDR。

统计单位是 tissue--MHC task。由于任务之间可能共享 tissue、MHC、parent protein 和 peptide component，这些检验属于 nominal task-level inference，不能解释为独立外部队列证据。

## 3. Human strict architecture comparison

### 3.1 Ensemble performance

| Model | Mean AUROC | Median AUROC | Mean AUPRC | PairAcc | Worst-10 AUROC | Worst task |
|---|---:|---:|---:|---:|---:|---:|
| One-hot logistic regression | 0.7127 | 0.7069 | 0.6906 | 0.7157 | 0.5873 | 0.5533 |
| BLOSUM62 random forest | 0.7212 | 0.7208 | 0.7007 | 0.7263 | 0.5895 | 0.5634 |
| Shared heads | 0.7561 | 0.7565 | 0.7340 | 0.7631 | 0.6248 | 0.6003 |
| Plain MLP dual branch | 0.7555 | 0.7577 | 0.7339 | 0.7575 | 0.6163 | 0.5905 |
| Auxiliary dual branch | 0.7642 | 0.7660 | 0.7433 | 0.7686 | 0.6283 | 0.6034 |
| TissuePMHC | **0.7652** | 0.7606 | **0.7452** | **0.7700** | **0.6382** | **0.6142** |

### 3.2 TissuePMHC versus prespecified baselines

#### Versus shared heads

- AUROC difference：+0.0091；
- median difference：+0.0095；
- Hodges--Lehmann difference：+0.0091；
- win/tie/loss：118/0/39；
- 95% task-bootstrap interval：[0.0067, 0.0116]；
- BH-FDR：\(q=2.27\times10^{-11}\)。

PairAcc difference 为 +0.0069，95% interval 为 [0.0029, 0.0110]，BH-FDR 为 \(q=0.00163\)。

结论：TissuePMHC 在 strict folds 上稳定优于简单 shared-head 架构。

#### Versus plain MLP dual branch

- AUROC difference：+0.0097；
- median difference：+0.0081；
- Hodges--Lehmann difference：+0.0093；
- win/tie/loss：122/0/35；
- 95% interval：[0.0076, 0.0119]；
- BH-FDR：\(q=1.36\times10^{-14}\)。

结论：完整模型优于不含 auxiliary supervision 的普通 MLP dual branch。

#### Versus traditional peptide baselines

相对 BLOSUM62 random forest：

- AUROC difference：+0.0440；
- win/tie/loss：149/0/8；
- 95% interval：[0.0392, 0.0489]；
- BH-FDR：\(q=3.43\times10^{-26}\)。

相对 one-hot logistic regression：

- AUROC difference：+0.0525；
- win/tie/loss：157/0/0；
- 95% interval：[0.0479, 0.0571]；
- BH-FDR：\(q=8.14\times10^{-27}\)。

结论：human strict 信号不能由简单传统 peptide baseline 充分解释。

#### Versus auxiliary dual branch

- AUROC difference：+0.0010；
- median difference：-0.0008；
- Hodges--Lehmann difference：+0.0007；
- win/tie/loss：76/1/80；
- 95% interval：[-0.0008, 0.0029]；
- BH-FDR：\(q=0.442\)。

AUPRC difference 为 +0.0019，\(q=0.105\)；PairAcc difference 为 +0.0014，\(q=0.344\)。

结论：TissuePMHC 与最强 auxiliary dual-branch control 在 strict 条件下统计上相当，不能声称 multi-kernel 主模型稳定优于该强基线。

### 3.3 Human component interpretation

Auxiliary dual branch 相对 plain MLP dual branch：

- AUROC difference：+0.0087；
- win/tie/loss：144/0/13；
- 95% interval：[0.0074, 0.0100]；
- BH-FDR：\(q=3.02\times10^{-23}\)。

Auxiliary dual branch 相对 shared heads：

- AUROC difference：+0.0081；
- win/tie/loss：123/0/34；
- 95% interval：[0.0058, 0.0103]；
- BH-FDR：\(q=3.30\times10^{-11}\)。

Plain MLP dual branch 相对 shared heads：

- AUROC difference：-0.0006；
- BH-FDR：\(q=0.906\)；
- PairAcc difference：-0.0056。

因此：

1. auxiliary supervision 是 human strict 条件下最明确、最稳定的组件收益；
2. plain dual/rank-fusion structure 本身没有产生 AUROC 优势；
3. 将 rank-fused auxiliary MLP encoder 替换为 multi-kernel CNN 后，增量收益不明确。

### 3.4 Human seed stability and ensemble gain

| Model | Single-seed mean AUROC | Seed SD | Ensemble AUROC | Ensemble gain |
|---|---:|---:|---:|---:|
| One-hot logistic regression | 0.7127 | 0.0000 | 0.7127 | 0.0000 |
| BLOSUM62 random forest | 0.7193 | 0.0006 | 0.7212 | +0.0019 |
| Shared heads | 0.7313 | 0.0043 | 0.7561 | +0.0248 |
| Plain MLP dual branch | 0.7394 | 0.0043 | 0.7555 | +0.0161 |
| Auxiliary dual branch | 0.7503 | 0.0038 | 0.7642 | +0.0139 |
| TissuePMHC | 0.7505 | 0.0007 | 0.7652 | +0.0147 |

Ensemble 对所有 human 神经模型均有明确增益，其中 shared heads 的 ensemble 增益最大。

## 4. Mouse strict architecture comparison

### 4.1 Ensemble performance

| Model | Mean AUROC | Median AUROC | Mean AUPRC | PairAcc | Worst-6 AUROC | Worst task |
|---|---:|---:|---:|---:|---:|---:|
| BLOSUM62 random forest | 0.7507 | 0.7768 | 0.7295 | 0.7573 | 0.6418 | 0.5819 |
| Shared heads | **0.7538** | 0.7786 | 0.7321 | 0.7567 | 0.6430 | 0.5477 |
| Factorized MMoE | 0.7529 | **0.7822** | **0.7322** | **0.7596** | **0.6481** | **0.5859** |

三种 ensemble 的总体表现接近。

### 4.2 Factorized MMoE versus shared heads

- AUROC difference：-0.0008；
- median difference：-0.0018；
- Hodges--Lehmann difference：-0.0017；
- win/tie/loss：10/0/14；
- 95% interval：[-0.0062, 0.0049]；
- BH-FDR：\(q=0.509\)。

AUPRC difference 为 +0.0001，\(q=0.705\)；PairAcc difference 为 +0.0030，\(q=0.661\)。

结论：Factorized MMoE 不优于 shared heads。

### 4.3 Factorized MMoE versus BLOSUM62 random forest

- AUROC difference：+0.0023；
- median difference：+0.0039；
- Hodges--Lehmann difference：+0.0038；
- win/tie/loss：14/0/10；
- 95% interval：[-0.0052, 0.0091]；
- BH-FDR：\(q=0.509\)。

AUPRC difference 为 +0.0027，\(q=0.705\)；PairAcc difference 为 +0.0024，\(q=0.661\)。

结论：Factorized MMoE 不显著优于传统 BLOSUM62 random forest。

### 4.4 Mouse seed stability and ensemble gain

| Model | Single-seed mean AUROC | Seed SD | Ensemble AUROC | Ensemble gain |
|---|---:|---:|---:|---:|
| BLOSUM62 random forest | 0.7491 | 0.0012 | 0.7507 | +0.0016 |
| Shared heads | 0.7327 | 0.0031 | 0.7538 | +0.0211 |
| Factorized MMoE | 0.7211 | 0.0027 | 0.7529 | +0.0318 |

Factorized MMoE 的单 seed 平均 AUROC 低于 shared heads 和 random forest。最终 ensemble 竞争力主要来自 +0.0318 的五 seed averaging 增益，而不是一致的单模型架构优势。

## 5. 第 9 项最终判定

| Claim | 判定 |
|---|---|
| Human strict task learnability | 成立 |
| Mouse strict task learnability | 成立 |
| Human 主模型优于传统与简单 shared baselines | 成立 |
| Human auxiliary-supervision benefit | 成立 |
| Human multi-kernel encoder 优于最强 auxiliary control | 不成立 |
| Human plain dual/rank-fusion 独立收益 | 不成立 |
| Mouse Factorized MMoE architecture superiority | 不成立 |
| Mouse ensemble benefit | 成立且明显 |
| Universal strict architecture superiority | 不成立 |

第 9 项的实验与统计完成标准已经满足。最终结论属于：

> strict task learnability 成立，但 universal strict architecture superiority 不成立；human 中保留了明确的 auxiliary-supervision benefit，而 mouse 的最终性能主要受益于多 seed ensemble。

## 6. 论文可用表述

### Results

> Under connected-component peptide-disjoint evaluation, all evaluated model families retained substantial ranking signal. In the human benchmark, TissuePMHC outperformed the shared-head, plain dual-branch, and traditional peptide baselines, but performed comparably to the auxiliary dual-branch model. The auxiliary objective, rather than the multi-kernel encoder or rank-fusion structure alone, accounted for the clearest retained architectural gain. In the mouse benchmark, the Factorized MMoE, shared-head model, and BLOSUM62 random forest achieved comparable five-seed ensemble performance.

### Discussion

> The standard-benchmark architectural advantage only partially persisted after peptide-identity separation. The strict results support task learnability and a reproducible benefit from auxiliary supervision in the human benchmark, but do not establish universal superiority of the full TissuePMHC or Factorized MMoE architectures. For the mouse model, most of the final performance gain was attributable to seed ensembling rather than consistently stronger single-model performance.

## 7. 输出与可追溯性

Human 正式输出：

```text
results/issue9_human_strict/
```

Mouse 正式输出：

```text
results/issue9_mouse_strict/
```

统一分析：

```text
results/issue9_analysis/
```

核心文件：

- `strict_architecture_comparison.csv`；
- `paired_statistics.csv`；
- `per_task_differences.csv`；
- `seed_stability.csv`；
- `worst_group_metrics.csv`；
- `analysis_metadata.json`。

论文已根据本报告的实际结果更新：

```text
paper.tex
```

## 8. 保留限制

1. strict folds 只隔离 exact peptide identity，不隔离 parent protein、study、platform 或相似 motif；
2. 所有任务均为训练中出现过的 tissue--MHC tasks；
3. task-specific heads 不支持 unseen tissue、unseen allele 或 unseen tissue--MHC combination；
4. task-bootstrap 与 Wilcoxon 未建模任务之间的 crossed dependencies；
5. 结果不能解释为 protein-disjoint、external-cohort 或 clinical generalization。

