# Completion report 9: contrast of structure under the strict protocol

Completion date: 2026-07-23
Report collated: 2026-07-25
Status: Completed

## 1. Issues and objectives

The original negotiated-component peptide-disjoint OOF only contains the freezing master model:

- # Human: Three seen TissePMTC;
- Mouse: Five seen Factorized MMOE.

These results can prove that there are still learning signs in the seenn-task, unseen-peptide conditions, but they cannot prove that the main model is superior to a simple structure under the same conditions of strict. The objective of item 9 is to complete the structure comparison on exactly the same peptide-component folds and to distinguish between:

1. strict task learnability;
2. strict architecture superiority;
3. Single model structure gains versus more seed Esmble benefits.

## 2. Experimental agreements

### 2.1 Frozen splits

All models read the existing old manifest directly, without regenerating the olds.

Pairs species  Tasks
|---|---:|---:|---:|---:|---:|---:|
| Human | 73,798 | 17,036 | 157 | 3 | 0 | 0 |
| Mouse | 6,766 | 1,844 | 24 | 3 | 0 | 0 |

Human manifest SHA256:

```text
e0fccb2adcf809d055b35284279e6027ec31c8f590cad0d88726f29954ba4c29
```

Mouse manifest SHA256:

```text
32e7f865a1e6ebc0f0a7749e086ceb1a6ea5074f593fac742afe850300a73eb1
```

### 2.2 Models

Human:

1. one-hot logistic regression;
2. BLOSUM62 random forest;
3. shared peptide encoder with task-specific heads;
4. plain MLP dual branch;
5. auxiliary MLP dual branch;
6. frozen TissuePMHC.

Human neuromodels use seeds:

```text
20260704, 20260705, 20260706
```

Mouse:

1. BLOSUM62 random forest;
2. shared peptide encoder with task-specific heads;
3. frozen Factorized MMoE.

Mouse models use seeds:

```text
20260704, 20260705, 20260706, 20260707, 20260708
```

### 2.3 Common neural training settings

- epochs:25;
- batch size:512;
- optimizer:AdamW;
- learning rate:0.001;
- weight decay:0.0001;
- dropout:0.2;
- gradient-norm cap:1.0.

Traditional taxonomy uses their own freezing algorithm settings and does not impose the neuronet optimizer/epoch rule.

### 2.4 Evaluation and statistics

Calculate by task:

- AUROC;
- AUPRC;
- PairAcc;
- half-tie PairAcc;
- worst-task AUROC;
- human worst-10 mean AUROC;
- mouse worst-6 mean AUROC.

Master PairAcc Strictly used:

```text
positive_score > negative_score
```

The statistical analysis includes:

- mean/median task difference;
- Hodges--Lehmann difference;
- win/tie/loss;
- 10,000-replicate task-bootstrap interval;
- two-sided Wilcoxon signed-rank test;
- BH-FDR in species x metric family.

The statistical unit is the tissue-MHC task. Because of the possibility of sharing the tissue, MHC, parent protein and peptide company between missions, these tests are of nominal task-level evidence and cannot be interpreted as evidence from an independent foreign force.

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

- AUROC difference:+0.0091;
- median difference:+0.0095;
- Hodges--Lehmann difference:+0.0091;
- win/tie/loss:118/0/39;
- 95% task-bootstrap interval:[0.0067, 0.0116];
- BH-FDR:\(q=2.27\times10^{-11}\).

PairAcc differenceence is +0.0069, 95% interval is [0.0029, 0.0110] and BH-FDR is \(q=0.00163\.

Conclusion: TisuePMHC stabilizes on strictolds better than a simple share-head structure.

#### Versus plain MLP dual branch

- AUROC difference:+0.0097;
- median difference:+0.0081;
- Hodges--Lehmann difference:+0.0093;
- win/tie/loss:122/0/35;
- 95% interval:[0.0076, 0.0119];
- BH-FDR:\(q=1.36\times10^{-14}\).

Conclusion: The complete model is better than the normal MLP dual branch that does not contain the auxiliary subvision.

#### Versus traditional peptide baselines

Compare BLOSUM62 grandom forest:

- AUROC difference:+0.0440;
- win/tie/loss:149/0/8;
- 95% interval:[0.0392, 0.0489];
- BH-FDR:\(q=3.43\times10^{-26}\).

The next post is part of the post.

- AUROC difference:+0.0525;
- win/tie/loss:157/0/0;
- 95% interval:[0.0479, 0.0571];
- BH-FDR:\(q=8.14\times10^{-27}\).

Conclusion: The human target signal cannot be fully explained by a simple tradition of peptide baseline.

#### Versus auxiliary dual branch

- AUROC difference:+0.0010;
- median difference:-0.0008;
- Hodges--Lehmann difference:+0.0007;
- win/tie/loss:76/1/80;
- 95% interval:[-0.0008, 0.0029];
- BH-FDR:\(q=0.442\).

AUPRC differenceence is +0.0019, ZXQ0QZ; PairAcc difference is +0.0014, ZXQ1QZ.

Conclusion: TisuePMMHC is statistically comparable to the strongest auxiliary dual-branch control under the conditions of strict and cannot claim that the multi-kernel master model is stable above that strong baseline.

### 3.3 Human component interpretation

The #MapDownDownBranch #AuxiliaryDualBranch

- AUROC difference:+0.0087;
- win/tie/loss:144/0/13;
- 95% interval:[0.0074, 0.0100];
- BH-FDR:\(q=3.02\times10^{-23}\).

Auxiliary dul brach relative to shared heads:

- AUROC difference:+0.0081;
- win/tie/loss:123/0/34;
- 95% interval:[0.0058, 0.0103];
- BH-FDR:\(q=3.30\times10^{-11}\).

The Plain MLP dul brach relative to share head:

- AUROC difference:-0.0006;
- BH-FDR:\(q=0.906\);
- PairAcc difference:-0.0056.

Therefore:

1. Auxiliary subvision is the most explicit and stable component gain under human condition;
2. The plain duan/rank-fundion stucure does not produce AUROC advantages;
3. The incremental gain is unclear when the rank-fused auxiliary MLP encoder is replaced by the multi-kernel CNN.

### 3.4 Human seed stability and ensemble gain

| Model | Single-seed mean AUROC | Seed SD | Ensemble AUROC | Ensemble gain |
|---|---:|---:|---:|---:|
| One-hot logistic regression | 0.7127 | 0.0000 | 0.7127 | 0.0000 |
| BLOSUM62 random forest | 0.7193 | 0.0006 | 0.7212 | +0.0019 |
| Shared heads | 0.7313 | 0.0043 | 0.7561 | +0.0248 |
| Plain MLP dual branch | 0.7394 | 0.0043 | 0.7555 | +0.0161 |
| Auxiliary dual branch | 0.7503 | 0.0038 | 0.7642 | +0.0139 |
| TissuePMHC | 0.7505 | 0.0007 | 0.7652 | +0.0147 |

Ensemble has a clear gain for all human neuro models, with the largest gains of the Esmble.

## 4. Mouse strict architecture comparison

### 4.1 Ensemble performance

| Model | Mean AUROC | Median AUROC | Mean AUPRC | PairAcc | Worst-6 AUROC | Worst task |
|---|---:|---:|---:|---:|---:|---:|
| BLOSUM62 random forest | 0.7507 | 0.7768 | 0.7295 | 0.7573 | 0.6418 | 0.5819 |
| Shared heads | **0.7538** | 0.7786 | 0.7321 | 0.7567 | 0.6430 | 0.5477 |
| Factorized MMoE | 0.7529 | **0.7822** | **0.7322** | **0.7596** | **0.6481** | **0.5859** |

Three esmbles are on the same level.

### 4.2 Factorized MMoE versus shared heads

- AUROC difference:-0.0008;
- median difference:-0.0018;
- Hodges--Lehmann difference:-0.0017;
- win/tie/loss:10/0/14;
- 95% interval:[-0.0062, 0.0049];
- BH-FDR:\(q=0.509\).

AUPRC differenceence is +0.0001, ZXQ0QZ; PairAcc difference is +0030, ZXQ1QZ.

Conclusion: Factorized MMOE is no better than share heads.

### 4.3 Factorized MMoE versus BLOSUM62 random forest

- AUROC difference:+0.0023;
- median difference:+0.0039;
- Hodges--Lehmann difference:+0.0038;
- win/tie/loss:14/0/10;
- 95% interval:[-0.0052, 0.0091];
- BH-FDR:\(q=0.509\).

AUPRC differenceence is +0027, ZXQ0QZ; PairAcc difference is +0024, ZXQ1QZ.

Conclusion: Factorized MMOE does not significantly superior to traditional BLOSUM62 grandom forest.

### 4.4 Mouse seed stability and ensemble gain

| Model | Single-seed mean AUROC | Seed SD | Ensemble AUROC | Ensemble gain |
|---|---:|---:|---:|---:|
| BLOSUM62 random forest | 0.7491 | 0.0012 | 0.7507 | +0.0016 |
| Shared heads | 0.7327 | 0.0031 | 0.7538 | +0.0211 |
| Factorized MMoE | 0.7211 | 0.0027 | 0.7529 | +0.0318 |

The single Seed AUROC averaged less than share headers and grandom forest. Ultimately, the Esmble competitiveness comes mainly from the five perceived gains of +0.0318, rather than from the consistent single model architecture advantage.

## 5. Final decision No. 9

♪ Claim ♪
|---|---|
Human start task studyability set up
Mouse start task searchability setup
Human master model better than tradition and simplicity
Human Auxiliary-subpervision between
Human multi-kernel encoder better than the most powerful auxiliary control
Human plain duan/ rank-fusion independent income  not established
Moe Accuracy is not established
Mouse Esmble beforefit
Unformed technical support

The test and statistical completion criteria for item 9 have been met. The final conclusions are:

> The system is built, but the overall strategic effectiveness is not established; the humans retain a clear characterization between, and the ultimate performance of the mouse benefits mostly from more than an esemble.

## 6. Available presentations of papers

### Results

> Under connected-component peptide-disjoint evaluation, all evaluated model families retained substantial ranking signal. In the human benchmark, TissuePMHC outperformed the shared-head, plain dual-branch, and traditional peptide baselines, but performed comparably to the auxiliary dual-branch model. The auxiliary objective, rather than the multi-kernel encoder or rank-fusion structure alone, accounted for the clearest retained architectural gain. In the mouse benchmark, the Factorized MMoE, shared-head model, and BLOSUM62 random forest achieved comparable five-seed ensemble performance.

### Discussion

> The standard-benchmark architectural advantage only partially persisted after peptide-identity separation. The strict results support task learnability and a reproducible benefit from auxiliary supervision in the human benchmark, but do not establish universal superiority of the full TissuePMHC or Factorized MMoE architectures. For the mouse model, most of the final performance gain was attributable to seed ensembling rather than consistently stronger single-model performance.

## 7. Output and traceability

Official output from Human:

```text
results/issue9_human_strict/
```

Mouse Official Output:

```text
results/issue9_mouse_strict/
```

Integrated analysis:

```text
results/issue9_analysis/
```

Core document:

- `strict_architecture_comparison.csv`;
- `paired_statistics.csv`;
- `per_task_differences.csv`;
- `seed_stability.csv`;
- `worst_group_metrics.csv`;
- `analysis_metadata.json`.

The papers have been updated on the basis of the actual results of the present report:

```text
paper.tex
```

## 8. Limitations on reservations

1. Only isolated activity paper identity, not isolated between parent protein, study, platform or similar motif;
2. All tasks are those that appear during training - MHC tasks;
3. If you want to be a part of the world, you can be a part of the world.
4. cross-modified tasks between task-bootstrap and Wilcoxon;
5. The results cannot be interpreted as protein-disjoint, external-cohort or clinical generativeization.

