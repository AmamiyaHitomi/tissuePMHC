# 2026.7.6 Research log

## 1. Targets for today

The objective today is to continue the next experiment presented in the log, focusing on completion and evaluation:

```text
E4: conditioned model with tissue embedding + HLA pseudo-sequence
E4b: conditioned model with tissue embedding + HLA ID embedding + HLA pseudo-sequence
E5: E2 shared peptide encoder + task-specific heads + FAMO
```

ZXQ0QZ is Human Leukocyte Antigen, human white cell antigen. Here, for example, `HLA-A*02:01`, which represents a specific HLA-like gene.

ZXQ0QZ refers to a small number of key amino acid sites extracted from the HLA protein sequence, usually selected for the remediate positions associated with the peptide binting. ZXQ1QZ refers to the amino acid number in the protein sequence.

ZXQ0QZ is Fast Adaptive Multitask Implementation, a self-adapted multitasking optimization method. It dynamically adjusts the loss weights of different task tasks with the goal of making training more balanced.

## 2. New and modified codes

Four additional scripts have been added today, either to the main document or to the results sheet.

### 2.1 HLA pseudo-security build script

Add script:

```text
scripts/build_hla_pseudo_sequences.py
```

Purpose:

```text
Take the HLA data items I pseudo-equality from the official IPD-IMG/HLA protein FASTA.
```

Enter FASTA:

```text
data/processed/A_prot.fasta
data/processed/B_prot.fasta
data/processed/C_prot.fasta
```

Output:

```text
data/processed/hla_pseudo_sequences.csv
```

Generated pseudo-equality overwrite all 12 HLA allele:

```text
HLA-A*02:01
HLA-A*24:02
HLA-B*07:02
HLA-B*15:01
HLA-B*15:02
HLA-B*27:05
HLA-B*40:01
HLA-B*40:02
HLA-B*51:01
HLA-C*03:04
HLA-C*05:01
HLA-C*12:02
```

The length of pseudo-equality per article is 34.

Build method:

```text
1. Reads the HLA-A, HLA-B, HLA-C protein sequence from IPD-IMGT/ HLA protein FASTA.
2. Removes the former 24 amino acids from N-terminal signature peptide.
3. The amino acid is extracted by using the NetMHCpan-style 34 MHC-I binding-site positions.
4. Write the phrase " hla, pseudo_segment, sourge " .
```

Here ZXQ0QZ refers to a short sequence of protein N end used to guide the positioning of the secretion or membrane. The sequence is contained in IPD-IMGT/HLA FASTA, and the common MHz-I pseudo-sequience position is usually numbered as mature proteins, so 24 amino acids need to be removed first.

### 2.2 E4 HLA pseudo-sequience script

Add script:

```text
scripts/run_tissuepmhc_hla_pseudoseq.py
```

Purpose:

```text
Compare E3 HLA ID embedding with E4 HLA pseudo-segregation encoder.
```

Output directory:

```text
results/tissuePMHC_hla_pseudoseq/
```

Main output:

```text
per_task_metrics.csv
summary_metrics.csv
stability_metrics.csv
comparison_metrics.csv
metadata.json
```

### 2.3 E4b hybrid script

Add script:

```text
scripts/run_tissuepmhc_hla_hybrid.py
```

Purpose:

```text
Compare E3, E4 and E4b.
```

Model definitions:

```text
E3:
peptide encoder + tissue embedding + HLA ID embedding

E4:
peptide encoder + tissue embedding + HLA pseudo-sequence encoder

E4b:
peptide encoder + tissue embedding + HLA ID embedding + HLA pseudo-sequence encoder
```

ZXQ0QZ means a training vector. HLA ID embedding is a vector that converts a category of ID from `HLA-A*02:01`. HLA pseudo-sequience encoder is a vector of amino acid from HLA.

Output directory:

```text
results/tissuePMHC_hla_hybrid/
```

### 2.4 E5 FAMO script

Add script:

```text
scripts/run_tissuepmhc_famo.py
```

Purpose:

```text
Test the FAMO-first effective task testing on E2 shared index + task-special headers.
```

For fair comparison, the script runs simultaneously:

```text
E2_task_balanced
E5_E2_FAMO
```

`task-balanced` means that each piece of Training Step draws the same number of samples. This way FAMO is constructed using the same catch pattern as a non-FAMO comparison.

Output directory:

```text
results/tissuePMHC_famo/
```

Additional output:

```text
task_weight_history.csv
```

This document records the FAMO weights of each epoch and task.

## 3. E4 Experimental results: HLA pseudo-security replacement HLA ID embedding

E4 The core issue is:

```text
If HLA replaces the normal HLA ID embedding with HLA pseudo-security encoded?
```

Experimental results:

| model | mean AUROC | mean AUPRC | mean accuracy | mean MCC | worst-10 mean AUROC |
|---|---:|---:|---:|---:|---:|
| E3 HLA embedding | 0.7833 | 0.7685 | 0.7152 | 0.4350 | 0.6992 |
| E4 HLA pseudo-sequence | 0.7728 | 0.7606 | 0.7042 | 0.4125 | 0.6705 |

`AUROC` is the Area Under the Refeiver Contracting Characteristic Curve, which measures the overall ability of the model to distinguish between positive and negative samples.

`AUPRC` is the Area Under the Precision-Recall Curve, more concerned with the ability to detect the samples.

ZXQ0QZ is Matthews Correlation Coefficency, a more robust composite indicator in the classification task II.

E4 compared to E3:

```text
I mean AUROC down by about 0.0105
I mean AUPPC drops about 0.0079
I mean MCC drops about 0.0225
World-10 means AUROC down by about 0.0287
```

Press task average:

```text
15 tasks improved
29 tasks degraded
```

E4 Upgraded to the greatest task:

| target_tissue | mhc_restriction | AUROC delta |
|---|---:|---:|
| breast | HLA-A*02:01 | +0.0243 |
| ovary | HLA-A*02:01 | +0.0224 |
| lymph node | HLA-A*02:01 | +0.0173 |
| thymus | HLA-A*24:02 | +0.0168 |
| blood | HLA-A*24:02 | +0.0150 |

E4 The greatest downfall:

| target_tissue | mhc_restriction | AUROC delta |
|---|---:|---:|
| uterine cervix | HLA-B*51:01 | -0.0851 |
| blood | HLA-B*15:01 | -0.0673 |
| lymphoid | HLA-C*12:02 | -0.0442 |
| blood | HLA-B*51:01 | -0.0414 |
| lymphoid | HLA-B*40:01 | -0.0368 |

The pattern of E4 by HLA grouping is more obvious:

```text
HLA-A*02:01 and HLA-A*24:02
Most HLA-B and HLA-C missions are down.
```

Conclusions

```text
HLA pseudo-equality cannot directly replace HLA ID embedding.
```

The reason is assumed:

```text
Current train/test is closed-set HLA setting.
```

ZXQ0QZ refers to the HLA allele in the test set that has been present in the training centre. In this case, HLA ID embedding can remember the mission characteristics of each allele directly, and pseido-segregation encoder needs to compress 12 HLAs into shared sequence encoder, instead creating information bottlenecks.

## 4. E4b Experimental results: HLA ID ebeding + HLA pseudo-sequence

The core issue of E4b is:

```text
Would you upgrade E3 if pseudo-equality were not allowed to replace HLA ID but to be added as additional biological information?
```

Experimental results:

| model | mean AUROC | mean AUPRC | mean accuracy | mean MCC | worst-10 mean AUROC |
|---|---:|---:|---:|---:|---:|
| E3 HLA embedding | 0.7833 | 0.7685 | 0.7152 | 0.4350 | 0.6992 |
| E4b hybrid | 0.7824 | 0.7704 | 0.7133 | 0.4303 | 0.6952 |
| E4 pseudo-sequence only | 0.7728 | 0.7606 | 0.7042 | 0.4125 | 0.6705 |

E4b compared to E3:

```text
I mean AUROC down by about 0.0009.
In the AUPPC upgrades
I mean MCC down by about 0.0047
World-10 means AUROC down by about 0.0039
```

E4b is purer than E4:

```text
The E4 performance loss was apparently restored.
```

Press task average:

```text
21 tasks improved
23 tasks degraded
```

E4b Upgraded to the greatest task:

| target_tissue | mhc_restriction | AUROC delta vs E3 |
|---|---:|---:|
| blood | HLA-A*24:02 | +0.0482 |
| breast | HLA-A*02:01 | +0.0246 |
| bone | HLA-A*02:01 | +0.0233 |
| lymphoid | HLA-B*51:01 | +0.0190 |
| uterine cervix | HLA-B*07:02 | +0.0164 |

E4b The greatest drop in tasks:

| target_tissue | mhc_restriction | AUROC delta vs E3 |
|---|---:|---:|
| uterine cervix | HLA-B*51:01 | -0.0481 |
| lymphoid | HLA-B*15:02 | -0.0345 |
| lung | HLA-B*15:01 | -0.0282 |
| lymphoid | HLA-A*24:02 | -0.0223 |
| lymphoid | HLA-B*40:02 | -0.0194 |

Grouping by HLA:

```text
HLA-B*07:02   +0.0071
HLA-A*24:02   +0.0062
HLA-A*02:01   +0.0040
HLA-B*15:02   -0.0345
HLA-B*40:02   -0.0126
HLA-B*15:01   -0.0115
```

Conclusions

```text
E4b states that pseudo-equality has a certain value as supporting information, especially for AUPRC.
But E4b didn't exceed E3 and did not exceed E2.
```

E4b could therefore be an exploratory outcome of the subsequent discussions, but not as the current master model.

## 5. E5 Experimental results: FAMO experimental task thinking

E5 The core issue is:

```text
E2 shared paper encoded + task-special headers, can you ease the speed transition?
```

ZXQ0QZ refers to the fact that some tasks in multitasking learning are disrupted by other tasks and performance is reduced.

For fair comparison, the E5 script used a task-balanced batt structure:

```text
Each piece of the train taking a sample of the same number.
```

Compare object:

```text
E2_task_balanced
E5_E2_FAMO
```

Experimental results:

| model | mean AUROC | mean AUPRC | mean accuracy | mean MCC | worst-10 mean AUROC |
|---|---:|---:|---:|---:|---:|
E2 shared headers 0.77777  0.7180 0.4404 0.71778
| E2 task-balanced | 0.7819 | 0.7639 | 0.7131 | 0.4290 | 0.7036 |
| E5 FAMO | 0.7758 | 0.7571 | 0.7077 | 0.4193 | 0.6979 |

E5 FAMO compared to E2 task-balanced:

```text
I mean AUROC down by about 0.0061
I mean AUPPC drops about 0.0009.
I mean MCC down by about 0.0097.
World-10 means AUROC down by about 0.0057
```

Press task average:

```text
13 tasks improved
31 tasks degraded
```

FAMO Upgraded the biggest task:

| target_tissue | mhc_restriction | AUROC delta vs E2_task_balanced |
|---|---:|---:|
| lymphoid | HLA-B*15:01 | +0.0288 |
| lymphoid | HLA-B*15:02 | +0.0248 |
| uterine cervix | HLA-B*07:02 | +0.0186 |
| brain | HLA-B*40:02 | +0.0150 |
| lung | HLA-B*15:01 | +0.0125 |

FAMO The greatest drop in tasks:

| target_tissue | mhc_restriction | AUROC delta vs E2_task_balanced |
|---|---:|---:|
| uterine cervix | HLA-B*51:01 | -0.0381 |
| thymus | HLA-A*02:01 | -0.0330 |
| blood | HLA-C*03:04 | -0.0324 |
| lymphoid | HLA-B*51:01 | -0.0227 |
| lymphoid | HLA-B*40:01 | -0.0220 |

FAMO has a smaller ultimate weight:

```text
approximately 0.0215 to 0.0271
uniform weight = 1 / 44 = 0.0227
```

This means that FAMO has not learned a strong task distinction. Some of the tasks with the highest weight include:

| task | final mean weight |
|---|---:|
| umbilical cord blood / HLA-A*02:01 | 0.0271 |
| thymus / HLA-A*24:02 | 0.0251 |
| lymphoid / HLA-A*24:02 | 0.0247 |
| lymphoid / HLA-B*51:01 | 0.0244 |
| lung / HLA-B*15:01 | 0.0241 |

Some of the tasks with the lowest priority include:

| task | final mean weight |
|---|---:|
| thymus / HLA-A*02:01 | 0.0187 |
| breast / HLA-A*02:01 | 0.0202 |
| uterine cervix / HLA-B*51:01 | 0.0208 |
| lymphoid / HLA-C*03:04 | 0.0212 |
| lung / HLA-A*02:01 | 0.0215 |

Conclusions

```text
The current E5 FAMO version is not successful.
```

More important observations are:

```text
E2_task_balanced itself is also significantly weaker than the original E2.
```

This means that forcing each step to sample all task equivalents changes the training distribution and may undermine the advantages of the original E2. The original E2 uses a mix of natural data distribution training that may be more appropriate for the current data configuration.

## 6. Current model sequencing

The current model is sorted as follows, in combination with the results of 2026.7.5 and 2026.7.6:

```text
E2 shared peptide encoder + task-specific heads
>
E3 conditioned tissue + HLA ID embedding
≈ E4b hybrid HLA ID + HLA pseudo-sequence
>
E2 task-balanced
>
E5 FAMO
>
E4 HLA pseudo-sequence only
```

Main performance comparisons:

| model | mean AUROC | mean AUPRC | mean MCC | worst-10 mean AUROC |
|---|---:|---:|---:|---:|
| E2 shared heads | 0.7927 | 0.7777 | 0.4404 | 0.7178 |
| E3 HLA embedding | 0.7833 | 0.7685 | 0.4350 | 0.6992 |
| E4b hybrid | 0.7824 | 0.7704 | 0.4303 | 0.6952 |
| E2 task-balanced | 0.7819 | 0.7639 | 0.4290 | 0.7036 |
| E5 FAMO | 0.7758 | 0.7571 | 0.4193 | 0.6979 |
| E4 pseudo-sequence only | 0.7728 | 0.7606 | 0.4125 | 0.6705 |

## 7. Main findings of the day

The most important conclusion today is that:

```text
E2 remains the strongest and most stable major.
```

More specifically:

1. HLA pseudo-equality cannot directly replace HLA ID embedding.
2. HLA ID embedding + HLA pseuddo-segregation 's hybrid model can repair the performance loss of pure pseudo-segrete, but it does not exceed E3.
3. The AUPRC of E4b is slightly higher than E3, indicating that pseudo-equality may have slightly increased information for the positive samples.
4. The current version of FAMO does not raise E2 but instead lowers the meaning of AUROC, mean AUPREC, MCC and World-10 AUROC.
5. The task-balanced mapling itself weakens E2, suggesting that the natural training distribution of current data may be more appropriate than the balance of mandatory tasks.
6. Negative transfer still exists, but may need to be solved by task grouping or serctive sharing rather than simply losing weighting.

## 8. Next steps: access to E6 task grouping

Based on today's results, it is not recommended to continue the adjustment of E4 or E5.

Recommended access:

```text
E6: HLA-based and tissue-based task grouping analysis
```

Specific experiments:

```text
1. Train shared peptide encoder + task-specific heads within each HLA group.
2. Train shared peptide encoder + task-specific heads within each tissue group.
3. Compare grouped training vs original all-task E2.
4. Analyze which tasks benefit from grouping and which tasks need global sharing.
```

Why E6 is more important:

```text
E2 has proven successful paper application.
But E2 also appeared 14 degraded tasks.
These degraded tasks may not be a loss question, but a question of task sharing structure.
```

ZXQ0QZ refers to which tasks should be shared with model parameters and which tasks should not be shared. The current task is naturally broken down to:

```text
task = tissue + HLA
```

The next step, therefore, is analysis:

```text
Should we share it?
Should we share it?
Which HLA or Tissue group are easily generated?
```

## 9. Key documents

HLA pseudo-security:

```text
scripts/build_hla_pseudo_sequences.py
data/processed/hla_pseudo_sequences.csv
data/processed/A_prot.fasta
data/processed/B_prot.fasta
data/processed/C_prot.fasta
```

E4 Code and results:

```text
scripts/run_tissuepmhc_hla_pseudoseq.py
results/tissuePMHC_hla_pseudoseq/per_task_metrics.csv
results/tissuePMHC_hla_pseudoseq/summary_metrics.csv
results/tissuePMHC_hla_pseudoseq/stability_metrics.csv
results/tissuePMHC_hla_pseudoseq/comparison_metrics.csv
results/tissuePMHC_hla_pseudoseq/metadata.json
```

E4b Code and results:

```text
scripts/run_tissuepmhc_hla_hybrid.py
results/tissuePMHC_hla_hybrid/per_task_metrics.csv
results/tissuePMHC_hla_hybrid/summary_metrics.csv
results/tissuePMHC_hla_hybrid/stability_metrics.csv
results/tissuePMHC_hla_hybrid/comparison_metrics.csv
results/tissuePMHC_hla_hybrid/metadata.json
```

E5 Code and results:

```text
scripts/run_tissuepmhc_famo.py
results/tissuePMHC_famo/per_task_metrics.csv
results/tissuePMHC_famo/summary_metrics.csv
results/tissuePMHC_famo/stability_metrics.csv
results/tissuePMHC_famo/comparison_metrics.csv
results/tissuePMHC_famo/task_weight_history.csv
results/tissuePMHC_famo/metadata.json
```

Current master baseline:

```text
scripts/run_tissuepmhc_neural_baselines_v2.py
results/tissuePMHC_neural_baselines_v2/stability_metrics.csv
```

## 10. E6 task grouping supplementation

When E6 is complete, compare grouping and original E2 all-task shared headers.

The overall results are as follows:

| model | mean AUROC | mean AUPRC | mean MCC | worst-10 mean AUROC |
|---|---:|---:|---:|---:|
| E2 all-task shared heads | 0.7927 | 0.7777 | 0.4404 | 0.7178 |
| E6 HLA-grouped | 0.7862 | 0.7725 | 0.4316 | 0.7037 |
| E6 tissue-grouped | 0.7387 | 0.7227 | 0.3548 | 0.6699 |

E6 HLA grouping

```text
I mean AUROC down by about 0.0005
I mean AUPPC drops about 0.0052.
I mean MCC down by about 0.0088
```

E6 compared to E2:

```text
I mean AUROC down by about 0.0541
I mean AUPRC drops by about 0.0550
I mean MCC drops about 0.0856.
```

So, E6. HLA grouping just approaches E2, and tissue grouping fails.

In task, HLA grouping in 44 tasks:

```text
18 tasks improved
26 tasks degraded
```

The biggest HLA-grouped task:

| target_tissue | mhc_restriction | AUROC delta vs E2 |
|---|---:|---:|
| lung | HLA-A*24:02 | +0.0399 |
| blood | HLA-B*51:01 | +0.0360 |
| lymphoid | HLA-B*51:01 | +0.0331 |
| small intestine | HLA-B*15:01 | +0.0293 |
| uterine cervix | HLA-B*07:02 | +0.0286 |

HLA-grouped task:

| target_tissue | mhc_restriction | AUROC delta vs E2 |
|---|---:|---:|
| uterine cervix | HLA-A*24:02 | -0.0564 |
| lymphoid | HLA-B*40:02 | -0.0540 |
| lymphoid | HLA-C*03:04 | -0.0540 |
| lymphoid | HLA-B*15:02 | -0.0490 |
| lymphoid | HLA-C*05:01 | -0.0419 |

On HLA group average, HLA grouping has performed relatively well group including:

```text
HLA-B*51:01   +0.0101 AUROC
HLA-C*12:02   +0.0099 AUROC
HLA-B*07:02   +0.0096 AUROC
HLA-B*27:05   +0.0048 AUROC
HLA-B*15:01   +0.0017 AUROC
```

The decrease is more pronounced:

```text
HLA-B*15:02   -0.0490 AUROC
HLA-B*40:02   -0.0360 AUROC
HLA-C*03:04   -0.0352 AUROC
HLA-C*05:01   -0.0243 AUROC
```

Conclusions

```text
HLA grouping has a local value but cannot be a global replacement for the main E2 model.
Tissue grouping does not recommend continuation.
```

The reason is assumed:

```text
The same HLA internal sharing can sometimes reduce the number of newer transfers across HLA.
But many task still need to go over the whole HLA, so all-task E2 is still stronger.
The system is a significant decline in performance because it's a big difference in sharing.
```

Next steps to:

```text
E7: validation-based selective HLA/global sharing
```

E7's core philosophy is:

```text
Do not choose all-task global sharing or HLA grouping.
The first step is to cut the value set inside the training set.
Compares global E2 branch to the value AUROC of HLA-grouped branch for each task.
If HLA-grouped branch is better on valueation, then task uses HLA branch;
If you do not, then you will continue to use the global E2 branch.
Retrain the global branch and HLA branch using the complete training set after each task has been identified.
Finally evaluate this task-level spatially model on test set.
```

This avoids the leakage of data from the test set selection model.
At the same time, it is more equitable to use the full training after re-training component in the final test assessment and the original E2 in the same amount of training data.

E6 Code and results:

```text
scripts/run_tissuepmhc_task_grouping.py
results/tissuePMHC_task_grouping/per_task_metrics.csv
results/tissuePMHC_task_grouping/summary_metrics.csv
results/tissuePMHC_task_grouping/stability_metrics.csv
results/tissuePMHC_task_grouping/group_summary_metrics.csv
results/tissuePMHC_task_grouping/comparison_metrics.csv
results/tissuePMHC_task_grouping/metadata.json
```

E7 code and planned output:

```text
scripts/run_tissuepmhc_selective_grouping.py
results/tissuePMHC_selective_grouping/per_task_metrics.csv
results/tissuePMHC_selective_grouping/summary_metrics.csv
results/tissuePMHC_selective_grouping/stability_metrics.csv
results/tissuePMHC_selective_grouping/candidate_metrics.csv
results/tissuePMHC_selective_grouping/selection_metrics.csv
results/tissuePMHC_selective_grouping/comparison_metrics.csv
results/tissuePMHC_selective_grouping/metadata.json
```

E7 Scripts take time to run a direct print training at the terminal:

```text
Training hours for every global branch
Training hours for each HLA branch
The total time of each seed
Total Run Time
```

## 11. E7 valuation-based results of private HLA/global share

E7 The core issue is:

```text
If not to force all task to use the same share scheme,
Select a global branch or HLA-grouped branch for each task on value set,
Can you get more than the original E2?
```

E7 Use two-stage process:

```text
1. Cuts value set out of the train.
2. Train with train-core value global brand and value HLA brand.
3. Select the branch for each task using value AUROC.
4. Retrain with complete global brand and final HLA brand.
5. Assesss the selected final branch on each test set.
```

This avoids the test limit and ensures that the final test evaluation is the same amount of training data as E2.

The overall results are as follows:

| model | mean AUROC | mean AUPRC | mean accuracy | mean MCC | worst-10 mean AUROC |
|---|---:|---:|---:|---:|---:|
| E2 all-task shared heads | 0.7927 | 0.7777 | 0.7180 | 0.4404 | 0.7178 |
| E6 HLA-grouped | 0.7862 | 0.7725 | 0.7148 | 0.4316 | 0.7037 |
| E7 selective HLA/global | 0.7904 | 0.7754 | 0.7184 | 0.4405 | 0.7143 |
| E6 tissue-grouped | 0.7387 | 0.7227 | 0.6765 | 0.3548 | 0.6699 |

E7 is a comparison with E2:

```text
I mean AUROC down by about 0.0023
I mean AUPPC drops about 0.00023
I mean an improvement 0.0005
means MCC upgrade 0.0001
World-10 means AUROC down by about 0.0035
```

Thus, E7 is clearly better than E6 HLA-grouped, but still does not exceed the original E2.

The Branch selection for E7:

| seed | global tasks | HLA-grouped tasks |
|---:|---:|---:|
| 20260704 | 25 | 19 |
| 20260705 | 26 | 18 |
| 20260706 | 32 | 12 |

HLAbranch was the most selected HLA group:

```text
HLA-A*02:01 13
HLA-A*24:02 7 times
HLA-B*15:01 7 times
HLA-B*07:02 4 times
HLA-C*03:04 4 times
HLA-B*51:01 4 times
```

E7 raise the biggest task:

| target_tissue | mhc_restriction | AUROC delta vs E2 |
|---|---:|---:|
| small intestine | HLA-B*15:01 | +0.0446 |
| lung | HLA-A*24:02 | +0.0444 |
| blood | HLA-A*24:02 | +0.0167 |
| blood | HLA-B*40:01 | +0.0141 |
| spleen | HLA-B*15:01 | +0.0133 |

E7 top drop task:

| target_tissue | mhc_restriction | AUROC delta vs E2 |
|---|---:|---:|
| blood | HLA-B*07:02 | -0.0511 |
| lymphoid | HLA-C*03:04 | -0.0332 |
| lymphoid | HLA-B*15:02 | -0.0329 |
| lymph node | HLA-C*03:04 | -0.0250 |
| blood | HLA-C*05:01 | -0.0217 |

Grouping by the last selected branch, the average delta is:

| selected branch | mean AUROC delta | mean AUPRC delta | mean MCC delta |
|---|---:|---:|---:|
| global | -0.0048 | -0.0068 | -0.0073 |
| HLA-grouped | +0.0019 | +0.0055 | +0.0127 |

This results state that:

```text
HLAbranch does have a partial benefit.
E7 value-based hard supply restores part of E6 loss.
But hard supply is still not stable enough for the whole body to exceed E2.
```

`hard selection` means that each task can only be selected by one option: either using global branch or HLA branch.
The problem with this is that the small fluctuations on the value set will directly change the ultimate branch selection.

The conclusions as of today are as follows:

```text
E2 remains the most powerful man in the world today.
E6 illustrates that HLA grouping has a local value and that it is clearly not appropriate.
E7 indicates that task-level selective share is valuable, but hard supply is not stable enough.
```

## 12. Next steps: E8 soft alternative

Next recommended entry:

```text
E8: validation-weighted soft ensemble of global branch and HLA branch
```

`soft ensemble` means that instead of choosing a branch for each task, you combine two branch prepositions score by weight.
Here `prediction score` is the positive probability of model output, indicating that peptide belongs to the possibility of this Tissue-HLA task.

E8 specific experimental design:

```text
1. Continues to use E7 phases of Train-core/ valuation/ full-training.
2. Assess global branch and HLA branch on value set.
3. For each task, the weight of integration is calculated on the basis of the value AUROC or AUPRC.
4. Train with complete global brand and complete HLA brand.
5. Output on test set:
   final_score = w * hla_score + (1 - w) * global_score
6. E8 soft Eemble and E2, E6, E7.
```

Three weight strategies can be tested as a matter of priority:

```text
E8a: fixed average
Global_score and hla_score each account for 0.5.

E8b: validation-delta clipped weight
Increase the weight of HLA if HLA valuation AUROC is higher than the global;
Otherwise, the weight of HLA is lowered, but no weight is given to 0 or 1.

E8c: validation-rank softmax weight
Generates global/HLA weights with value metric softmax.
Softmax is a function that converts multiple fractions to non-negative weights and totals 1.
```

It is expected that:

```text
E8 may be more stable than E7.
Because even if the validation misjudges, soft esmble will not completely abandon another branch.
```

E7 Code and results:

```text
scripts/run_tissuepmhc_selective_grouping.py
results/tissuePMHC_selective_grouping/per_task_metrics.csv
results/tissuePMHC_selective_grouping/summary_metrics.csv
results/tissuePMHC_selective_grouping/stability_metrics.csv
results/tissuePMHC_selective_grouping/candidate_metrics.csv
results/tissuePMHC_selective_grouping/selection_metrics.csv
results/tissuePMHC_selective_grouping/comparison_metrics.csv
results/tissuePMHC_selective_grouping/metadata.json
```

## 13. Description of follow-up number correction

The subsequent retray confirmed that the main lines of E5, E6 and E7 in this log are correct and that they should all be included in the E2 shared page encoded + task-special headers, rather than the E4 HLA pseudo-sequience line.

The current official number should be understood to read:

```text
E5: FAMO on E2
E6: HLA/tissue task grouping on E2
E7: selective HLA/global hard selection
E8: global/HLA soft ensemble
E9: E2 + CAGrad
E10: MMoE / PLE selective-sharing model on the E2/E8 line
E11: E2 + DB-MTL
```

Of these, ZXQ0QZ means Multi-gate Mixture-of-Experts, a multi-manophone expert hybrid model; ZXQ1QZ means Progress Layed Extraaction, a gradual layer extraction model. Both belong to the spatial-sharing model, which allows different tabs to automatically select models with different levels of sharing.

ZXQ0QZ is a dynamic balancing multi-task learning method that balances lossscale and gradient size of different task.

Therefore, E8 soft esmble in section 12 of this log should be followed by the official E9 CaGrad-on-E2, instead of the CAGrad line.
