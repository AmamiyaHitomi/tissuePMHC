# MousePMHC Phase 3 Experiment Report

## 1. Project objectives and final conclusions

The objective of Phase 3 is to establish reasonable minimum sample thresholds for the mouse tissue-H2-I pelican submission of preference tasks and to compare options such as single task learning, full sharing, multi-expert route, dynamic task weighting, task grouping and partial sharing.

The current conclusions are as follows:

1. The minimum task inclusion threshold is `min_pairs > 200`. It is not a cut-off point for the learning curve pre-registration stabilization rules, but a project selection that balances sample size, mission coverage and H2 diversity; the threshold continues to rise to 250, leaving only 20 tasks and 3 H2 and will be lost directly to H2-Kk.
2. Multitask sharing is important for low-resource mouse missions. E1 means more than the best traditional single task E0 means more than about `0.0543`.
3. E3b's task-balanced Factorized MMOE is a stable and strong baseline, and three sees mean task AUROC is ZXQ0QZ.
4. The apparent failure of the TAG-style hard grouping of E6 indicates that a complete cut-off of inter-group sharing would result in significant losses in the process of relocation, particularly to H2-Db.
5. E7 While retaining E3b global sharing, only H2-Kk increases the disability apperter to the current top threeseed OFOF mean task AUROC: `0.8180 ± 0.0013`. E7 increases ZXQ1QZ relative to E1 and `0.0031` relative to E3b.
6. E7 is a smaller increase than E3b, with an estimated 95% of the t-blocks at the three sighted margin of `[-0.0024, 0.0087]`, which cannot yet be claimed to be significantly better than E7. E7 should therefore be used as a simplified comparison and a fixed test set assessment should be conducted after the freeze.

## 2. Data sets and mission definitions

The data set is ZXQ0QZ, the task unit is `target_tissue × H2 restriction`. Each sample is a two-symmetrical category observation:

- Example: 9-mer peptide reported in target Tissue-H2;
- Negative example: peptide from the same UniProt and H2 reported in other organizations but not in the target organization;
- Data are limited to mice, mice host, MHC-I, H2 recovery, positive IEDB qualitative measurements, peptide with parent UniProt, unmodified standard amino acid and 9 lengths.

After using `min_pairs > 200` and keeping 100 test planes fixed for each task, the current benchmark contains:

Project  Value
|---|---:|
| tissue-H2 tasks | 24 |
| tissues | 13 |
| H2 restrictions | 4 |
| train pairs | 6,766 |
| train rows | 13,532 |
| fixed test pairs | 2,400 |
| fixed test rows | 4,800 |
Train ranges per task  124–470
Medium number per task  264.5

Data metadata are available at `data/mousePMHC/mousePMHC_metadata.json`.

## 3. Minimum threshold selection for pairs

A learning curve incorporates all eligible tasks under this threshold in each threshold point and samples the same number of pairs under each task; 3 repeats per point seed. The task covers the following:

Number of jobs per task
|---:|---:|---:|---:|
| 100 | 43 | 22 | 4 |
| 150 | 33 | 18 | 4 |
| 200 | 24 | 13 | 4 |
| 250 | 20 | 12 | 3 |
| 300 | 19 | 11 | 3 |
| 350 | 14 | 8 | 3 |
| 400 | 10 | 7 | 3 |

The curve cannot be interpreted as a pure sample volume effect on the same task. Preregistration stabilization rules do not return the automatic recommended values in 100-400. The main reason for the application of 200 is:

- 24 tasks and all 4 H2 remain;
- Increase from 200 to 250 to lose H2-Kk and cannot continue to study the trans-H2 negative migration;
- The sample of single-mission training is more adequate than 100 or 150 and the total cost of training remains acceptable.

200 is therefore a trade-off between coverage and single task data and should not be expressed as a performance platform point in the strict statistical sense.

## Uniform experimental agreement

- All current Phase 3 neural network comparisons use the same Train benchmark from `min_pairs > 200`.
- Use 3-old plain-grouped OOF, old standard saw for `20260711`, avoiding the same pair cross-training discount and validation discount.
- E3b, E6, E7 and supplement E1 use trainings seeds ZXQ0QZ, ZXQ1QZ, `20260706`.
- The main indicator is the macro-average of the tasks AUROC means task AUROC.
- It means task AUPRC, MCC, weakest task AUROC and worst-6 task AUROC.
- All model selection, gradient audit and task groups use only the Train/OOF data; as of this report, the cross-referenced test was not read.
- E2-E5 for single-seed filters is mainly used to phase out directions that are clearly inappropriate; E3b for filtering is confirmed in three.

## 5. E0 and E1 baseline

### 5.1 E0: Traditional task-by-task model

E0 compares five traditional candidates, each task-H2 independent training. The results are as follows:

♪ Mean task AUROC ♪ Mean task AUPRC ♪ Mean task MCC ♪
|---|---:|---:|---:|
| BLOSUM62 Random Forest | **0.7530** | **0.7292** | **0.3894** |
| BLOSUM62 Extra Trees | 0.7509 | 0.7259 | 0.3887 |
| BLOSUM62 HistGradientBoosting | 0.7412 | 0.7180 | 0.3678 |
| One-hot Logistic Regression | 0.7349 | 0.7094 | 0.3472 |
| BLOSUM62 Logistic Regression | 0.7250 | 0.6897 | 0.3446 |

BLOSUM62 Random Forest is the best traditional baseline for E0.

### 5.2 E1: share complete

E1 uses a global shared peptide encoder and 24 task-specific export heads. Three seeds mean task AUROC:

- `20260704`: 0.8057
- `20260705`: 0.8057
- `20260706`: 0.8107

III Seed average value ZXQ0QZ. E1 is significantly better than best E0 and proves that cross-task sharing works for current low-resource benchmark. But early H2 analysis shows uneven sharing of benefits: H2-Db benefits are most evident, while Kb/Kk has negative migration risks, so subsequent experiments focus on selective sharing.

## E2-E7 Methodology and results

The following table summarizes the current `min_pairs > 200` route. The single seed value is used for method screening; the third seed value is expressed in `mean ± sample SD`.

♪ The world's greatest ever ♪
|---|---|---:|---:|---:|---:|---:|---|
BLOSUM62 Random Forest 1  0.7530  0.7292  0.3894  Best traditional baseline
Sharad encoder + task heads 3 08073  0.0029  0.7929  0.0033  0.4930  0.0071  0.6617  0.0043  Shared baseline
E2  H2-grouped hard share 1  0.802  0.7887  0.4932    below E1, stop
E3  Factorized MMOE-lite  1  0.8132  0.8029  0.5176    potential but Kd line edge failed
E4  Factorized PLE-lite  0.7830  0.7670  0.4425
E5 control  Task-balanced MMOE  0.8164  0.8067  0.5244  0.6804   to win and upgrade to E3b
E5  FAMO + MMOE 10.79930.78850.4886 significantly lower control, stop
E3b Task-balanced MMOE communication308148  0.0033  0.0028  0.5176  00.070  0.677  0.0053  Stabilization baseline
E6  TAG-style grouped hard share 3  0.7403  0.0040  0.7172  0.0038  0.364  00.0076  0.614  00.0039
E7  E3b + H2-K residual apperter 3  0.8180  0.0013  **0.865  0.0025  0.5218  00.0058  ** 0.6816  00.0010  Current main candidate

### 6.1 E2: H2 group hard-shared

E2 creates a peptide encoder for each H2 with a shared task under the same H2. E2 means task AUROC for `0.8022`, lower than the Seed E1. H2-Db has slightly improved, but Kb, Kd, and Kk have all dropped, indicating that a simple H2 split would lose useful cross H2 being migrated.

### 6.2 E3/E3b: Factorized MMOE and task-balanced training

E3 uses 3 shared experts, Gate input Peptide, tessue embeding and H2 embedding to keep independent header for each task. The initial E3 list sees AUROC is `0.8132`.

E5 Equal rights task-balanced control reached `0.8164`, which is better than the original E3, and therefore named it E3b and supplemented it with three seeds. E3b III sees AUROC `0.8148 ± 0.0033`, which is a relative match to the E1 found, increases the average ZXQ2QZ; the three seeds are positive. Its World-6 AUROC also increases the average of about `0.0154` relative to E1. MMoEgate does not appear expert collapse, so E3b is a reliable share backbone.

E3b There are still residual H2 heterogeneity. The AUROC margin for E1 group III seed H2 is: Db ZXQ0QZ, Kb ZXQ1QZ, Kd ZXQ2QZ, Kk ZXQ3QZ. Kk slightly exceeded the predefined `-0.010` line, and became the main target for E6E7.

### 6.3 E4:Factorized PLE-lite

E4 Designs global, tissue and H2 categories of route, but AUROC is only `0.7830`. The average route weight is approximately tissue ZXQ1QZ, H2 ZXQ2QZ, global `0.0535`, with some tasks having the maximum route weight approaching 1, showing a clear path collapse. This direction stops.

### 6.4 E5: FAMO Dynamic Task Weight

E5 Compare control and FAMO on the same MMOE backbone and task-balanced batt. The same control AUROC is `0.8164`, FAMO is `0.7993`, with a margin of ZXQ2QZ; only 4 of the 24 missions have improved and 20 have decreased. The dynamic weights have not eased the current negative migration, but have undermined the right training, and therefore the FAMO has stopped.

### 6.5 E6: TAG-style effect hard group

E6 Undertake one-step task migration estimate on each seed-old little data, and form 4-6 task forces with each team independently trained and shared encoder.

E6 III Seed AUROC is `0.7403 ± 0.0040`, which is about ZXQ1QZ, World-6 AUROC, which is about `0.0626` lower than the average of E3b. H2 See, the AUROC difference relative to E3b is:

H2E6 − E3b AUROC  Explanation
|---|---:|---|
H2-Db - 0.105  Active migration from global shared severe loss
H2-Kb-0.0421
H2-Kd-0.0333
H2-Kk-0024

E6 shows that segregation sharing does protect the Kk mission, but hard cuts destroy the positive migration of Db/Kb/Kd. Autogrouping is also unstable between the seed-olds, so E6 is not suitable as a final model, but supports the E7 design "to keep sharing globally, adding only small private paths to the conflict group".

### 6.6 E7:Recon-style H2-Kk residual adapter

E7 Keeps the full E3b MMOE backbone and adds a 16 dimension to pre-declared H2-Kk aptter after sharing peptide encoder. Each seed-old still performs a train-only H2 gradient audit for the verification mechanism, but the formal structure is fixed at H2-Kk apperter, avoiding changing the candidate according to OOF fold dynamic.

The results of the three seeds are:

| Seed | E7 AUROC | E3b AUROC | E7 − E3b |
|---:|---:|---:|---:|
| 20260704 | 0.8179 | 0.8164 | +0.0016 |
| 20260705 | 0.8167 | 0.8110 | +0.0057 |
| 20260706 | 0.8193 | 0.8171 | +0.0021 |
| Mean | **0.8180** | 0.8148 | **+0.0031** |

E7 AUROC difference relative to E3b for group H2 is:

E7 − E3b AUROC
|---|---:|---:|
| H2-Db | +0.0021 | 20/36 |
| H2-Kb | -0.0020 | 5/12 |
| H2-Kd | +0.0086 | 10/15 |
| H2-Kk | +0.0050 | 6/9 |

E7 not only improved the target H2-Kk, but also Kd and Db; Kb 's average decline was only `0.0020`. The overall AUROC increase ZXQ1QZ compared to E1, E7 showed 49 improvements in 72 match-task-seed observations. The E7 difference against E1 was calculated by adding up to the E3b margin, with the E7 difference of about Db ZXQ2QZ, Kb `+0.0034`, Kd `-0.0002`, Kk `-0.0070`, all H2 meeting the protection requirements of "not less than E1 than 0.010".

E7 World-6 AUROC is `0.6816 ± 0.0010`, better than E3b, and about ZXQ2QZ of E1. This means that promotion is not at the expense of the weakest tasks.

In the gradient audit, the average cosine of H2-Kk and other H2s was `0.0035`, the lowest of the four H2, but only 4/9 seed-olds were negative; Kd also had similar fluctuations. Therefore, audit support for Kk is the weakest share group but does not constitute evidence of a strong and stable "continuing conflict of gradients". The main evidence for E7 should still be matching the perceived OOF performance, not overexplaining the gradient.

## 7. Final model selection and next steps

It is currently proposed to freeze the following options:

- Main model: E7, task-balanced Factorized MMOE + H2-Kk residual apter;
- Simplified comparison: E3b, task-balanced Factorized MMOE;
- Shared baseline: E1;
- Traditional baseline: E0 BLOSUM62 Random Forest.

E7 meets the overall upgrade, H2 protection and World-6 protection requirements and achieves the highest average OOF AUROC. But E7 has only 95% of the `0.0031`, III seed difference value, and should therefore be expressed as "the best candidate at present" instead of "substantially superior to E3b".

Follow-up recommendations:

1. No longer add E8 or adjust E7 deviation parameters based on OOF results, and avoid further model selection deviations.
2. E7 architecture, hyper-parameters, training seeds, training rounds and evaluation indicators.
3. Re-read E7, E3b, E1 and E0 on the complete train and make a final assessment of the never read cross test.
4. The final report also gives macrík aUROC/AUPRC, macro-average per H2, task-6, task-by-task indicators and seed fluctuations.
5. Fixed test results indicate that E7 is almost identical to E3b, preference should be given to E3b, which is simpler, and E7 should be chosen if Kk and World-6 improvements are restored.

## 8. Limitations on the interpretation of results

- The task of the pairs learning curves is not the same as the task is assembled at the points, so that changes in the curves cannot be attributed entirely to the sample number.
- E2-E5 Most of them are single-seed filters, suitable for phasing out obvious failures and not for precise sequencing.
- E6 is the TAG-style realized within the project to move the output group step by step, rather than to repeat the entire training process of the original paper.
- E7 is that the Recon-style gradient audit is appropriate for a partially shared project and is not a return to the original methodology.
- Phase 3 has been used on multiple rounds of OOF results for model selection, and the final generalization is based on frozen test.
- All conclusions are currently train-only OOF conclusions and cannot be described as independent tests of performance.

## 9. Location of main code and result

### Code

- Learning curves: ZXQ0QZ
- E0:`scripts/run_mousepmhc_phase3_e0_oof.py`
- E1:`scripts/run_mousepmhc_phase3_e1_oof.py`
- E1 Supplementary seeds: ZXQ0QZ
- E2:`scripts/run_mousepmhc_phase3_e2_h2_grouped_oof.py`
- E3:`scripts/run_mousepmhc_phase3_e3_factorized_mmoe_oof.py`
- E4:`scripts/run_mousepmhc_phase3_e4_factorized_ple_oof.py`
- E5:`scripts/run_mousepmhc_phase3_e5_famo_mmoe_oof.py`
- E3b:`scripts/run_mousepmhc_phase3_e3b_task_balanced_mmoe_oof.py`
- E6:`scripts/run_mousepmhc_phase3_e6_tag_grouped_oof.py`
- E7:`scripts/run_mousepmhc_phase3_e7_recon_h2_adapters_oof.py`

### Outcome

- Learning curves: ZXQ0QZ
- E0:`results/mousePMHC_phase3_e0_oof/`
- E1: ZXQ0QZ and ZXQ1QZ
- E2:`results/mousePMHC_phase3_e2_h2_grouped_oof/`
- E3:`results/mousePMHC_phase3_e3_factorized_mmoe_min200_oof/`
- E4:`results/mousePMHC_phase3_e4_factorized_ple_min200_oof/`
- E5:`results/mousePMHC_phase3_e5_famo_mmoe_min200_oof/`
- E3b:`results/mousePMHC_phase3_e3b_task_balanced_mmoe_min200_oof/`
- E6:`results/mousePMHC_phase3_e6_tag_grouped_min200_oof/`
- E7:`results/mousePMHC_phase3_e7_recon_h2_adapters_min200_oof/`
