# 2026.7.5 Research log

## 1. Targets for today

Today 's goal is to complete the first round of the Stage 1 in `NEW_RESEARCH_ROADMAP.md`.

The core issues are:

```text
Could the multitask neural network further enhance the prediction of the TissuePMHC after the traditional single task baseline?
```

The following models are being achieved and compared with this focus:

1. `neural_single_task`
2. `shared_peptide_encoder_task_heads`
3. `conditioned_tissue_hla`

Of these, `shared_peptide_encoder_task_heads` is determined to be the current E2 master baseline.

## 2. Data and task setting

The experiment follows the existing `tissuePMHC` standard train/test standard.

Data size:

```text
Number of tasks: 44
Training rows: 96,972
Test rows: 8,800
Test set per task: 100 positive + 100 negative
Peptide length: 9
```

The definition of the mandate remains:

```text
task = target_tissue + mhc_restriction
```

That is, each Tissue-HLA combination is a two-class task.

## 3. Code realization

Two PyTorch training scripts were added and sorted today.

First edition of script:

```text
scripts/run_tissuepmhc_neural_baselines.py
```

Purpose:

```text
One run each time, each operation is free, free, free, and free.
```

Script number two:

```text
scripts/run_tissuepmhc_neural_baselines_v2.py
```

Purpose:

```text
Keep E2 master baseline, repeat 3 seed and rotated to condition_issue_hla.
```

Release 1 remains as a repossessable experimental code, while Release 2 is exported to:

```text
results/tissuePMHC_neural_baselines_v2/
```

This would not cover the results of the first edition.

## Modelling

### 4.1 neural_single_task

`neural_single_task` means that each mission trains a small neural network separately.

Enter:

```text
peptide sequence
```

Output:

```text
binary label
```

This model is used to check whether simple neural networks can outpace traditional machine learning.

### 4.2 shared_peptide_encoder_task_heads

`shared_peptide_encoder_task_heads` is the most important E2 model of this time.

Structure:

```text
peptide sequence
-> shared peptide encoder
-> task-specific head
-> binary prediction
```

ZXQ0QZ means that all tasks are shared with the same thong feature extractor.

ZXQ0QZ means that each task has its own classification layer.

The meaning of this model is:

```text
Different tasks are shared by peptide, but the final classification boundaries are learned by each mission.
```

### 4.3 conditioned_tissue_hla

`conditioned_tissue_hla` is a condition model.

Structure:

```text
peptide encoder
+ tissue embedding
+ HLA embedding
-> shared classifier
-> binary prediction
```

`embedding` means a trainingable vector that is used to convert a class variable such as tissue or HLA into a numerical vector that can be used by a neural network.

The model aims at learning:

```text
Gives a peptide, tessue and HLA for predicting whether the peptide is biased towards the tessue-HLA condition.
```

## 5. Description of evaluation indicators

The core indicators of this record include:

```text
AUROC
AUPRC
accuracy
balanced accuracy
F1
MCC
worst-10-task mean AUROC
```

`AUROC` is the Area Under the Receiver Contracting Characteristic Curve, which indicates the ability of the model to distinguish between positive and negative samples as a whole.

`AUPRC` is the Area Under the Precision-Recall Curve, more concerned with the ability to detect the samples.

ZXQ0QZ is Matthews Correlation Coefficency, which combines the considerations of true realization, trueness, false position and false classivity, which are usually more stable than accuracy in category II tasks.

`worst-10-task mean AUROC` represents the average of the worst 10 AUROC missions, which is used to observe whether the model increases only average performance or also improves difficult tasks.

## 6. Results of the first version of the experiment

First version of the Experiment Output Directory:

```text
results/tissuePMHC_neural_baselines/
```

The overall results are as follows:

| model | mean AUROC | mean AUPRC | mean accuracy | mean MCC |
|---|---:|---:|---:|---:|
| `shared_peptide_encoder_task_heads` | 0.7944 | 0.7766 | 0.7225 | 0.4498 |
| `conditioned_tissue_hla` | 0.7756 | 0.7620 | 0.7053 | 0.4125 |
| `neural_single_task` | 0.7326 | 0.7223 | 0.6669 | 0.3356 |

The best average model of the original baseline is:

```text
onehot_logistic_regression
mean AUROC = 0.7558
mean AUPRC = 0.7384
mean accuracy = 0.6909
mean MCC = 0.3841
```

Therefore, the first version of the experiment states:

```text
The video is available on the Internet and is available on the Internet.
The viewed_issue_hla is also stronger than traditional baseline but weaker than traditional plande_peptide_encode_task_heads.
The traditional baseline is weak.
```

## 7. Experimental design for Release 2

The second version of the experiment will use E2 as the main baseline and repeat three grandom seen.

`random seed` is a random number of seeds that control random processes like model initialization, battling shuffle, etc. Repeats multiple seeds to determine whether the results are stable.

Use & & & seed:

```text
20260704
20260705
20260706
```

The second version of the experiment contains five sets of experiments:

```text
E2_shared_peptide_encoder_task_heads
E3_conditioned_default
E3_conditioned_wider_condition
E3_conditioned_wider_hidden
E3_conditioned_low_dropout
```

The reference settings for the conditioned model are:

| experiment | condition_dim | hidden_dim | dropout | learning_rate |
|---|---:|---:|---:|---:|
| `E3_conditioned_default` | 16 | 128 | 0.2 | 0.001 |
| `E3_conditioned_wider_condition` | 32 | 128 | 0.2 | 0.001 |
| `E3_conditioned_wider_hidden` | 32 | 256 | 0.2 | 0.0005 |
| `E3_conditioned_low_dropout` | 32 | 128 | 0.1 | 0.001 |

`dropout` is a regular method of training to randomly discard some neuron output to reduce the risk of collusiveness.

## 8. Second edition of the stabilization results

Version 2 Experiment Output Directory:

```text
results/tissuePMHC_neural_baselines_v2/
```

The stability of the three seeds has the following results:

| experiment | mean AUROC | AUROC std | mean AUPRC | mean accuracy | mean MCC | worst-10 mean AUROC |
|---|---:|---:|---:|---:|---:|---:|
| `E2_shared_peptide_encoder_task_heads` | 0.7927 | 0.0010 | 0.7777 | 0.7180 | 0.4404 | 0.7178 |
| `E3_conditioned_wider_condition` | 0.7833 | 0.0029 | 0.7685 | 0.7152 | 0.4350 | 0.6992 |
| `E3_conditioned_wider_hidden` | 0.7825 | 0.0025 | 0.7714 | 0.7102 | 0.4241 | 0.6882 |
| `E3_conditioned_default` | 0.7821 | 0.0020 | 0.7720 | 0.7112 | 0.4255 | 0.6912 |
| `E3_conditioned_low_dropout` | 0.7770 | 0.0039 | 0.7618 | 0.7088 | 0.4216 | 0.6860 |

`AUROC std` is a standard AUROC deviation of 3 seeds. The smaller the standard difference, the more stable the results are under different random seeds.

Key findings:

```text
The AUROC is the highest and AUROC std is only 0.010.
```

This suggests that E2 upgrades are not random luck but more stable model gains.

## 9. E2 Comparison with traditional baseline

Tradition's Top Average:

```text
onehot_logistic_regression mean AUROC = 0.7558
```

E2 Seed average:

```text
E2 mean AUROC = 0.7927
```

Upgrade:

```text
0.7927 - 0.7558 = 0.0369
```

This means sharing the peptide encoder is clearly better than the current traditional single task baseline.

E2 Best tradition for each mission

```text
30 tasks improved
14 tasks degraded
mean AUROC delta = +0.0201
mean AUPRC delta = +0.0199
```

## E2 Upgraded to maximum task

E2 Baseline, the highest priority for each task, is as follows:

| target_tissue | mhc_restriction | mean AUROC | baseline AUROC | AUROC delta |
|---|---:|---:|---:|---:|
| thymus | HLA-A*02:01 | 0.9017 | 0.7974 | +0.1043 |
| blood | HLA-C*05:01 | 0.8332 | 0.7440 | +0.0892 |
| lung | HLA-A*02:01 | 0.8272 | 0.7455 | +0.0817 |
| small intestine | HLA-B*15:01 | 0.8027 | 0.7218 | +0.0809 |
| umbilical cord blood | HLA-A*02:01 | 0.9511 | 0.8958 | +0.0553 |
| ovary | HLA-A*02:01 | 0.8440 | 0.7922 | +0.0518 |

These mission statements do learn from other tasks to migrate information.

## E2 Tasks with the greatest decline

E2 Compared to the best traditional baseline for each task, the most important reductions are as follows:

| target_tissue | mhc_restriction | mean AUROC | baseline AUROC | AUROC delta |
|---|---:|---:|---:|---:|
| lymphoid | HLA-C*12:02 | 0.7301 | 0.7798 | -0.0497 |
| brain | HLA-B*40:02 | 0.7035 | 0.7378 | -0.0343 |
| blood | HLA-B*27:05 | 0.7490 | 0.7818 | -0.0328 |
| uterine cervix | HLA-B*07:02 | 0.8003 | 0.8298 | -0.0295 |
| lymph node | HLA-A*02:01 | 0.8219 | 0.8491 | -0.0272 |
| lung | HLA-A*24:02 | 0.8297 | 0.8475 | -0.0178 |

These drop task alerts exist in `negative transfer`.

ZXQ0QZ refers to the fact that after learning multitasking, certain tasks are disrupted by other tasks and performance is reduced.

## 12. E2 Changes in HLA Grouping

E2 Average AUROC delta on different HLAs:

| HLA | n_tasks | mean AUROC delta |
|---|---:|---:|
| HLA-C*05:01 | 3 | +0.0477 |
| HLA-A*02:01 | 12 | +0.0411 |
| HLA-A*24:02 | 6 | +0.0298 |
| HLA-B*51:01 | 3 | +0.0280 |
| HLA-B*15:01 | 5 | +0.0251 |
| HLA-B*27:05 | 2 | -0.0192 |
| HLA-C*12:02 | 1 | -0.0497 |

Preliminary observations:

```text
The HLA-A*02:01 and HLA-C*05:01 tasks have benefited significantly from shared learning.
HLA-B*27:05 and HLA-C*12:02 may be more vulnerable to new transactions.
```

## 13. E2 Changes in organizational groupings

AUROC deta mean on different tissue:

| tissue | n_tasks | mean AUROC delta |
|---|---:|---:|
| small intestine | 1 | +0.0809 |
| thymus | 2 | +0.0763 |
| umbilical cord blood | 1 | +0.0553 |
| ovary | 1 | +0.0518 |
| breast | 1 | +0.0492 |
| lung | 3 | +0.0348 |
| blood | 9 | +0.0130 |
| lymphoid | 12 | +0.0083 |
| lymph node | 4 | -0.0022 |
| spleen | 1 | -0.0096 |

Preliminary observations:

```text
Some of the small organizational tasks benefit more clearly from shared learning.
The average increase is not the largest, although the number of tasks is high.
```

## 14. Convention_issue_hla cross-reference conclusions

The best configuration for this conditioned model is:

```text
E3_conditioned_wider_condition
condition_dim = 32
hidden_dim = 128
dropout = 0.2
learning_rate = 0.001
mean AUROC = 0.7833
```

Compared to defaulted model:

```text
0.7833 - 0.7821 = +0.0012
```

The lift is small.

Compare E2:

```text
0.7833 - 0.7927 = -0.0094
```

Therefore, this rotation factor does not allow the conditioned model to exceed E2.

Conclusions

```text
The normal question-embeding + HLA embedding conditioned model is not currently the strongest model.
```

However, the discussion continued to be valuable, as it was the infrastructure for the follow-up E4 HLA policy-making.

## 15. Main findings of the day

The most important conclusion today is that:

```text
E2 share_peptide_encoder_task_heads is the strongest and most stable early nervous network available.
```

More specifically:

1. `neural_single_task` is less than the traditional baseline, which indicates that the single task little neural network is unstable under the current data volume.
2. `conditioned_tissue_hla` exceeds the traditional baseline, but not as E2.
3. `shared_peptide_encoder_task_heads` is significantly more than traditional baseline and three seeds are very stable.
4. E2 has 30 tasks that exceed the best traditional baseline for each task, but 14 missions are down.
5. New Transfer has emerged and the follow-up needs to be task grouped and more detailed.
6. Normal HLA embedding conditioned model with limited benefits, and should be added to HLA pseuddo-sequience.

## 16. Current model sequencing

Based on today ' s results, the current model is sorted as:

```text
E2 shared peptide encoder + task-specific heads
>
E3 conditioned tissue-HLA model
>
traditional single-task baseline
>
E1 neural single-task baseline
```

Of which E2 is the current master baseline.

## 17. Next steps

Next move is recommended to enter E4:

```text
conditioned model with tissue embedding + HLA pseudo-sequence
```

Specific tasks:

1. Fetch or construct the HLA table.
2. Achieve HLA pseudo-sequience encoder.
3. Comparison:

```text
conditioned model + HLA embedding
vs
conditioned model + HLA pseudo-sequence
```

If HLA pseudo-sequience is upgraded, it is suggested that models do benefit from the more biological HLA expression.

After that, consider:

```text
E4 + FAMO
task grouping and negative transfer analysis
CAGrad
```

`FAMO` is Fast Adaptive Multitask Application, a loss weight that is used to customise different tasks.

ZXQ0QZ is the Confect-Averse Gradient Discent, which is used to mitigate conflicts between mission gradient directions.

## 18. Key documents

First-page code:

```text
scripts/run_tissuepmhc_neural_baselines.py
```

Second-round code:

```text
scripts/run_tissuepmhc_neural_baselines_v2.py
```

Results of Release 1:

```text
results/tissuePMHC_neural_baselines/per_task_metrics.csv
results/tissuePMHC_neural_baselines/summary_metrics.csv
results/tissuePMHC_neural_baselines/metadata.json
```

Results of Release 2:

```text
results/tissuePMHC_neural_baselines_v2/per_task_metrics.csv
results/tissuePMHC_neural_baselines_v2/summary_metrics.csv
results/tissuePMHC_neural_baselines_v2/stability_metrics.csv
results/tissuePMHC_neural_baselines_v2/metadata.json
```

## 19. Description of follow-up number correction

The subsequent retrieval confirmed that `E4 + FAMO / task grouping / CAGrad` in section 17 of this log was a provisional plan based on the information available at the time of the event and was no longer used as the current official number.

The reason for this is that the following experiments show that:

```text
The E4 HLA pseudo-security line does not exceed E2.
Therefore, the follow-up performance main line should continue along E2 shared pattern encoded + task-specific headers.
```

The current official number should be understood to read:

```text
E2: shared peptide encoder + task-specific heads
E3: conditioned tissue + HLA ID
E4: conditioned tissue + HLA pseudo-sequence
E4b: HLA ID + HLA pseudo-sequence hybrid
E5: FAMO on E2
E6: HLA/tissue task grouping on E2
E7: selective HLA/global hard selection
E8: global/HLA soft ensemble
E9: E2 + CAGrad
```

That is, the E4 line is now retained as HLA biological integration analysis, which is the HLA bioexpression analysis line; the E2/E8 line is the current performance main line.
