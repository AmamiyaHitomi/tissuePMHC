# TissuePMHC Phase 7 Experiment Report: Min_pairs=200 Sync, Model Recomposure and E29 Final Results

Update: 2026-07-17
Status: Phase 7 core experiment and E31 peptide-disjoint core field has been completed; E30 is not part of the core objective of this phase and has not been implemented.
Final model: E29 multi-kernel CNN, 3-seed mean.
Data definition: The original pair number for each Tissue-HLA task is strictly satisfied with `total_pairs > 200`.

## 1. Experimental objectives and final conclusions

The objective of Phase 7 is to synchronize ZXQ0QZ from the human project TissuePMHC with the mouse project of 200, to reconfigure the data and to re-establish key models E2, E14, E17, E26 and E29 without overlaying the original ZXQ1QZ code and results.

The final conclusions are as follows:

1. `min_pairs > 200` increases the number of benchmarks from 44 task to 157 task, trade in air from 48,486 to 73,798, and HLA allele from 12 to 35.
2. E14a 3-seed rank esmble from E17 to `+0.01069` as opposed to E2
3. The OOF greendy section of E26 ultimately only selects E14 3-seed mean, but only increases ZXQ0QZ AUROC as opposed to E17, and does not constitute a valid new model benefit.
4. E29's OOF CNN matches E14 OOF up ZXQ0QZ AUROC with all predefined filter conditions.
5. E29 Arrival on a fixed test set of means AUROC ZXQ0QZ, mean AUPRC ZXQ1QZ, mean MCC ZXQ2QZ, world-10 means AUROC `0.68339`, is the final master model of Phase 7.
6. E29 Increase ZXQ0QZ AUROC on the test set compared to E17, 126 of 157 tabs and 31 of the decreases; the expressions of gains observed in OOF are confirmed in the same direction during the testing phase.
7. Freezing E29 structures, hyperparameters and three seeds, E31 conned-componted peptide-disjoint OOF takes meaning task AUROC/ AUPREC ZXQ0QZ. The same complete train plain is still standard E29 OOF, AUROC/ AUPRCC is dropping ZXQ1 XZ, indicating that the overlap of entities is clearly elevated up the unseen-peptide estimate, but maintaining a sorting signal above the random level under strict division.

[Phase 7 Model Progress] (ZXQ0QZ)

## 2. Data build and benchmark extension

Phase 7 Use the same raw pair data, label definitions and filtering rules as the original human project to include only task into the threshold from `total_pairs > 500` to `total_pairs > 200`. 100 test pairs are still drawn for each task, with one positive and one pair of negative examples for each pair.

Enter a total of 125,649 pairs to read during the cleanup phase, delete 857 pairs without valid Tissue/ HLA task fields, and retain 124,792 pairs to be formally constructed. The final data are as follows:

min_pairs=500min_pairs=200 Change
|---|---:|---:|---:|
Tisue-HLA tasks 44 157  3.57 times
Tissues 1525 1.67 times
HLA Alles 12  35  2.92
Train pairs 48,486  73,798  1.52 times
Test days 4,400  15,700  3.57 times the amount of the
Train rows 96,972  147,596  1.52 times
Test rows 31,4003.57 times more than

Each task set maintains 100 positive-negative pairs of pair, so the task-macro indicator is not directly determined by the number of samples in large task. The 157 tabs in Phase 7 cover distribution is shown below; the sample is still highly concentrated in a few common Tissue-HLA combinations, and the low-resource task sets meet the threshold, but the training pairs are far less than the top bits.

[Phase 7 Data Overwrite] (../results/figures_phase7/04_dataset_coverage.png)

## 3. Code separation and operating chain

The entrance to the Phase 7 is located at `phase7/`, where the data is written to `data/tissuePMHC_phase7_min200/`, and the name is written to a separate directory containing `phase7_min200`. The original `min_pairs=500` data and results are not overwritten.

The core operating order is:

```powershell
& E:/ancd/envs/my_pytorch/python.exe d:/my_python/tissuePMHC/phase7/build_human_dataset_min200.py
& E:/ancd/envs/my_pytorch/python.exe d:/my_python/tissuePMHC/phase7/run_e2_shared_heads.py
& E:/ancd/envs/my_pytorch/python.exe d:/my_python/tissuePMHC/phase7/run_e14_auxiliary_soft_ensemble.py
& E:/ancd/envs/my_pytorch/python.exe d:/my_python/tissuePMHC/phase7/run_e17_seed_ensemble.py
& E:/ancd/envs/my_pytorch/python.exe d:/my_python/tissuePMHC/phase7/run_e26_all_in_one.py
& E:/ancd/envs/my_pytorch/python.exe d:/my_python/tissuePMHC/phase7/run_e29_multikernel_cnn_oof.py --device cuda
& E:/ancd/envs/my_pytorch/python.exe d:/my_python/tissuePMHC/phase7/run_e29_independent_test.py --device cuda
& E:/ancd/envs/my_pytorch/python.exe d:/my_python/tissuePMHC/phase7/run_e31_peptide_disjoint_oof.py --device cuda
```

E17 Instead of training the new network, the ZXQ0QZ, which relies on E14, completes the recommended match-matching. E26 retrains the match-matching candidate in 3-fold OOF, providing E29 with a clean-safe OOF filter. E29 completes the OOF filter first; after it is independently tested for re-use of frozen OOF decisions, only trains the full-train 3-seed model, and does not repeat OOF.

## 4. Phase 7 Core model results

Experimental Mean AUROC  Mean AUPRC  Accuracy  MCC  World-10 AUROC
|---|---|---:|---:|---:|---:|---:|
E2  Shared paper encoded + task heads, 3 seess mean 0.79723  0.78287 0.72771  0.45984  0.62994
E14a  Global Auxiliary + HLA plain, 3 Seeds mean 0.82386  0.81172  0.75516  0.51319  0.65006
| E17 | E14a 3-seed rank average | 0.83455 | 0.82334 | 0.76551 | 0.53131 | 0.66663 |
| E26 | OOF greedy task-rank selection | 0.83462 | 0.82350 | 0.76465 | 0.52933 | 0.66712 |
| E29 | Multi-kernel CNN 3-seed mean | **0.84478** | **0.83477** | **0.77723** | **0.55448** | **0.68339** |

### 4.1 E2: Shared encoder baseline

E2 Seed means AUROC is ZXQ0QZ, and Main AUPRC is `0.78287 ± 0.00062`. The result is that, after expanding to 157 tabs, sharing the peptide encoder can still stabilize training, but the weakest tabk has only limited performance, and the most-10 AUROC is only `0.62994`.

### 4.2 E14: Subsidiary supervision and dual branches

The meaning of the AUROC is ZXQ0QZ, which is an upgrade of `+0.02663` relative to E2. The meaning of the AUROC is `0.82017 ± 0.00140`; E14a is 394 won and 77 negative in 471 seed-task comparisons, and therefore continues to select global auxiliary + HLA plain.

### 4.3 E17: Multiple seed projections integration

E17 Averaged the global/HLA branches of three E14a seeds, and executed the task-wise rank Fusion. Ultimately, the AUROC is a ZXQ0QZ, with an increase in the average value of ZXQ1QZ relative to E14a III seed; 154 AUROCs were raised in 157 tabs, with only 3 declines. The standard difference in the stability table for an ensemble is only a fixed convergence point, without any uncertainty about repeated experiments.

### 4.4 E26:OOF greedy selection

E26 candidates include E14 final and E16 MC-dropout 's single seed and 3-sided mean. The XQ0QZ selection was eventually selected only for ZXQ; no second candidate reached the `0.0001` OOF AUROC minimum gain. The test means AUROC `0.83462` and E17 `0.83455`, which means that highly co-constructed candidates cannot generate new and valid information by reprocessing alone.

### 4.5 E29:multi-kernel CNN

E29 uses the position-positioning Conv1d peptide encoder of kennel size 2/3/5, and retains the two branches of `E14' of `global auxiliary 'and 'HLA plain '. 3-old, 3-seed OOF results:

OOF model  Mean AUROC Mean AUPPC  World-10 AUROC
|---|---:|---:|---:|
Match E14 baseline 081313 0.79800 0.64191
| E29 CNN | **0.82795** | **0.81436** | **0.65615** |
| E14 + E29 equal-rank fusion | 0.82508 | 0.81157 | **0.65680** |

E29 CNN increased ZXQ0QZ AUROC, out of 157 task units 144 increased and 13 decreased; with baseline, the task-rank corration is `0.93795`, below the predefined limit of 0.97. All four OF Gates passed. Since the standalone CNN's OFOFO Act AUROC is higher than the equitable-rankfosion, E29 standalone is eventually selected for entry testing under the freeze rule.

[E29 OOF and Test Confirm] (../results/figures_phase7/06_oof_test_confirmation.png)

## 5. E29 Independent test set validation

E29 full-training results are as follows:

Mean AUROC Mean AUPPC AccuracyMCCWork-10 AUROC
|---|---:|---:|---:|---:|---:|
| E17 3-seed | 0.83455 | 0.82334 | 0.76551 | 0.53131 | 0.66663 |
| E29 3-seed | **0.84478** | **0.83477** | **0.77723** | **0.55448** | **0.68339** |
| E29 − E17 | **+0.01024** | **+0.01142** | **+0.01172** | **+0.02317** | **+0.01677** |

On a task AUROC, E29 has 126 winners and 31 negatives; the median is raised to `+0.00980`. The maximum gain is in bLood-HLA-A*11:01 (ZXQ1QZ), lung-HLA-C*12:03 (ZXQ2QZ) and bLood-HLA-B*51:01 (ZXQ3QZ). The maximum is lymphoid-HLA-B*15:02 (`-0.02970`), lymph node-HLA-B*44:02 (ZXQ5QZ) and lung-HLA-B*27:05 (`-0.02175`).

[E29 and E17 task-by-task] (../results/figures_phase7/03_e29_vs_e17_task_gain.png)

Three E29s single-seed means AUROC is `0.83449`, ZXQ1QZ and `0.83299`; 3-seed means `0.84478`. This suggests that the error of initialization of independence is sufficiently complementary, and that the Seed Avelaging is an important part of E29 ' s final performance.

![E29 seed ensemble](../results/figures_phase7/05_e29_seed_ensemble.png)

## Min_pairs=500 extra comparison with min_pairs=200

To maintain consistency in model configuration, this section compares three E2, E14a, E17 and E29 that are the same seed (20260704/05/06). Results are divided into three categories: 44 task units in the original `min500`, 44 task units in the Phase 7 and all 157 task units in the original benchmark.

Model: 44 tasks    44 tasks      157 tasks  total changes in appearance
|---|---:|---:|---:|---:|
| E2 | 0.79273 | 0.79358 | 0.79723 | +0.00449 |
| E14a | 0.81158 | 0.81823 | 0.82386 | +0.01228 |
| E17 | 0.82428 | 0.83028 | 0.83455 | +0.01027 |
| E29 | 0.83415 | 0.83709 | 0.84478 | +0.01064 |

[min_pairs=500 and min_pairs=200] (../results/figures_phase7/02_minpairs_500_vs_200.png)

This set of differences cannot be interpreted as "lowening the mini-pairs leads to improved model performance", for two reasons:

1. ** Change in mission composition.** Add 113 task; its macro-average difficulty is not necessarily higher than the original 44 task. E2 common 44-task changes only `+0.00085`, while the composition effects of the new task contribute approximately ZXQ1QZ, indicating that most of the visual upgrades to E2 are from mission composition.
2. ** Test pair for common task also resampling.** Data builder continuously consumes the same random stream in the sequence of task; subsequent sample status changes for task after adding new tabs. The original 44 tabs have 100 test pairs each in both versions, but averages only 12.41, minimum 1 and maximum 23, and no test set is identical. Therefore, common 44-task is not a strict match on a fixed test sample.

By AUROC break-up, the total changes can be described as "common task changes + changes in the composition of new tasks":

Model Common 44-task Change New Task Composition Change Total Changes
|---|---:|---:|---:|
| E2 | +0.00085 | +0.00365 | +0.00449 |
| E14a | +0.00665 | +0.00563 | +0.01228 |
| E17 | +0.00600 | +0.00427 | +0.01027 |
| E29 | +0.00295 | +0.00769 | +0.01064 |

### 6.1 One of the main reasons is the macro-average mission composition effect

`min_pairs` is not a regular parameter within the model; it only determines which tissue-HLA task is included in the training and evaluation. Phase 7 adds 113 tabs, representing an even 157-task macro ZXQ1XZ. Macros give the same weight to each tab, so that new tabs, even with fewer trainings, decide together on the final score:

\[
M_{157}=\frac{44}{157}M_{\mathrm{common44}}+\frac{113}{157}M_{\mathrm{new113}}.
\]

Add 113 tabs easier on all four models than 44 tabs in Phase 7:

Phase 7 Common 44-task AUROC New 113-task AUROC New task - Common task
|---|---:|---:|---:|
| E2 | 0.79358 | 0.79865 | +0.00507 |
| E14a | 0.81823 | 0.82605 | +0.00783 |
| E17 | 0.83028 | 0.83621 | +0.00593 |
| E29 | 0.83709 | 0.84778 | +0.01068 |

Thus, the task added after the lower threshold does not bring down the macro evenly, but rather improves it. The `81.12%` for the E2 view of total gain, and `72.28%` for the E29, can be explained by the component effect of "Phase 7 All tasks are above the common 44-task average". Here the "interpretation" is arithmetic breakdown, not a causal judgement on the individual source of the difficulty of the task.

### 6.2 Lower number of pairs is not equal to more difficult classification boundaries

Of 157 task units in the Phase 7, the number of tradein pairs is weak and slightly negative in relation to the AUROC Spearman: E2 for ZXQ0QZ, E14a for ZXQ1QZ, E17 for ZXQ2QZ, E29 for `-0.109`. When looking at 113 tabs, the coefficient is still only `-0.088` to `-0.114`. This means that in the current benchmark, the "less sample" is not single-touched to "lower" in the "AUROC".

For example, the second four-point task (median approximately 208 days) on E29 means AUROC is `0.87375`, and is higher than the four-score ZXQ1QZ for the largest sample. The potential causes include: partially adding the positive peptide motif of the Tissue-HLA combination, stronger organizational signals or a pair of negative constructions forming a clearer sorting boundary on these tabs. The results are only proof that these new tabs are more easily distinguished under the current label and negative protocol, and that biological separation cannot be distinguished from data construction effects.

### 6.3 More sharing of training data may help E14/E17, but evidence is not strictly causal

From ZXQ0QZ to ZXQ1QZ, the number of trains increased from 96,972 to 147,596, the number of HLA alleles increased from 12 to 35, and the number of tessues increased from 15 to 25. The total amount of pair available for 44 joint taskers is essentially unchanged, but more task and more diverse training samples are shared, supported tissue/ HLA monitoring and cross-task parameters.

The common 44-task image AUROC changes only ZXQ0QZ on E2 and `+0.00665` and `+0.00600` on E14a/E17, respectively, and `+0.00295` on E29. This model is compatible with "subsidial supervision and multi-branch structures are more able to use extra trans-task information"; E17 reduces the training margin by seeing as avelaging. Since the common task test pair has been re-sampled, these values cannot alone demonstrate an increase in the sharing of trans-task data.

### 6.4 test redraw sufficient to create a clear common-task fluctuation

Common 44 task sets of old and new test averaged only ZXQ0QZ. The standard deviations in the ZXQ1QZ AUROC margin by task are approximately E2 `0.0351`, E14a ZXQ3QZ, E17 ZXQ4QZ, E29 `0.0326`, which is much larger than the average changes in the common tabk. The common task also does not have a single advantage: E2 won by X19, E14a by 24/20, E17 by 23/21 and E29 by 28/16. This indicates that the test redaw produced a magnitude of the task sample fluctuations that are not negligible.

Tests the pair overlap rate for Spearman, which is associated with AUROC changes, are close to zero (E2/E14a/E17 ZXQ0QZ, `0.036`, `0.011`, E29 `-0.104`). This does not mean that redraw is not affected, but that it is impossible to predict the direction of the change by relativity alone; the specific difficulty of being replaced is more important than the number of replacements.

### 6.5 Synthesis of judgements

The increase in the model score should be explained in the following order:

1. **The macro-average composition effect of adding task.** The addition of 113 task units, representing 71.97 per cent of the final weight, is on average easier on all models; this is the main component of the E2 and E29 appearance upgrades.
2. ** Extra task training information.** More samples of tisue, HLA and peptide may improve sharing indications that changes in the common task are compatible with the mechanism.
3. ** common task test redraw.** Only 12.41% top fair overlaps on average, which mixes common-task comparisons with clear sampling differences and cannot be directly attributed to the expansion of the training set.
4. ** The model structure is suitable for expanding the benchmark.** The supplementary monitoring of E14, the seed avelaging and the multiscale motif code of E17 are more able to use the additional data than E2; these are the reasons for the relative increase between models, not the evidence of the higher scores in `min_pairs` itself.

Therefore, the strict formulation should read:** On the reconfigured ZXQ0QZ benchmark, the task-macro scores for the four key models were higher than the original `min_pairs > 500` benchmark; the upgrades were made mainly of the difficulty of adding the task, more sharing of training data and more test redraw, and could not be interpreted as reducing the direct causal gains of `min_pairs` on the model performance**.

To strictly isolate the effects of `min_pairs`, the original 44 test plain IDs should be fixed and trained with `>500` and `>200` training tabpol, and then evaluated only on exactly the same 44-task test set. Future data construction can be changed to a stable hash seen for each task, avoiding adding a task to change other tabs.

## 7. E31: Humans peptide-disjoint

### 7.1 Freezing protocol and split effect

E31 does not read the Page 7 standard test, only `tissuePMHC_phase7_min200_train.csv.gz`. The model is fixed to E29 multi-kernel CNN's global-auxiliary/ HLA-plan double branch rank foundation, using seeds `20260704/05/06`, 3 golds, 25 epochs; structures, hyperparameters and integration rules are not adjusted to the strict indicator.

Splits the pair as atom: merge any shared pairs into agreed coponents and assign the full coponent certainty to three folds. The results are as follows:

- 147,596 rows, 73,798 pairs, 59,983 unique peptides;
- 17,036 components, of which 10,923 contain multiple pairs, the largest component is 755 pairs;
- All 157 tissue-HLA tasks;
- helps-out 24,598-24,600 pairs, getting 49, 198–49,200 pairs;
- Hold-out 33-2, 155 fairs, filling 67-4, 310 fairs;
- All folded global paper overlap and plain overlap are zero.

The agreement was seenn-task/unseen-peptide OOF; it was not certified by protein-disjoint, unseen-HLA, study-disjoint or the foreign forces.

### 7.2 Overall results versus standard OOF Gap

| Evaluation | Accuracy | Mean AUROC | Mean AUPRC | F1 | MCC | Worst-10 AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Standard pair-grouped E29 OOF | 0.76051 | 0.82795 | 0.81436 | 0.76319 | 0.52143 | 0.65616 |
| E31 peptide-disjoint E29 OOF | 0.69801 | 0.76520 | 0.74522 | 0.70242 | 0.39649 | 0.63817 |
| Strict − standard | -0.06250 | **-0.06275** | **-0.06914** | -0.06077 | -0.12494 | -0.01799 |

Two OOFs use the same full train plain number, same model, Seeds and old numbers, the difference being that E31 locks the full share of peptide into the same old. This is why the program AUROC keeps the standard OOF ZXQ0QZ, AUPRC keeps `91.51%`; this means that the model retains a signal for not seeing peptide, but cannot interpret the standard OOF 's 0.82795 as strictly unseen-peptide.

95% of the CI of nonparametric bootstrap, primary AUROC is ZXQ1QZ, AUPRC is ZXQ2QZ; 95% of the CI of the pair of strict-standard AUROC gap is ZXQ3QZ, AUPRC gap is `[-0.07989, -0.05880]`. The uncertainty of the typography between these compartments does not include the extra difference between retraining, sied or split selection.

The three singles that see mean task AUROC are ZXQ0QZ, ZXQ1QZ and ZXQ2QZ, III seed Esmble to `0.76520`, which increases the average value of ZXQ4QZ relative to the single, which indicates that there is a steady gain to seeaverising under the target level.

### 7.3 Task and HLA Heterotoxicity

The median number of the task-standard AUROC is ZXQ0QZ, the quadrant is ZXQ1QZ; the 12 of the 157 sets of tasks are up and down, and the relevant number of the Standard/Start task AUROC Pearson is `0.69325`. Summary by HLA locus:

| HLA locus | Tasks | Standard AUROC | Strict AUROC | ΔAUROC |
|---|---:|---:|---:|---:|
| HLA-A | 55 | 0.85405 | 0.78244 | -0.07161 |
| HLA-B | 79 | 0.82202 | 0.75070 | -0.07132 |
| HLA-C | 23 | 0.78589 | 0.77376 | -0.01212 |

The largest decline was concentrated in a number of high-value HLA-A*11:01 and HLA-B*15:01 tasks, such as small instine-HLA-A*11:01 (ZXQ0QZ), thyroid-HLA-A*11:01 (ZXQ1QZ) and ophagus-HLA-A*11:01 (ZXQ2QZ). These are descriptive heterogeneous results; the absence of a task-paired bootstrap, confidence zone or multiple tests does not explain the biological differences between locuss for cause and effect.

### 7.4 OPERATIONS AND PROPERTY

GPUs spend total time ZXQ0QZ (ZXQ1QZ). Three seeds spend time ZXQ2QZ, `29m 54s` and `30m 10s` respectively; the single old is ZXQ5QZ to ZXQ6QZ. Full epoch logs, old/seed/toral Timing, split manifest, member forecasts, Esmble projections and tabs are stored in `results/tissuePMHC_phase7_min200_e31_peptide_disjoint_oof/`.

## 8. Studying boundaries and limitations

1. The "independent test" of Phase 7 is relative to the OOF model selection of E29; it is still a fixed standard test within the project, not a foreign troop column.
2. Standard split does not force peptide-disjoint, protein-disjoint or unseen-HLA; the result cannot be interpreted directly as an extrapolation capacity for new peptide, new parent protein or new HLA.
3. Each task test contains only 100 positive and negative pairs, and the single task AUROC still has a sampling variance; preference should be given to macro averages, World-10 and a broad list of negative directions for the task.
4. Phase 7 has reported model results on the same standard test on several occasions. E29 has been upgraded by OOF date, reducing the risk of direct test penetration, but the entire research chain should still clearly distinguish between development findings and real external validation.
5. E30 's protein-disjoint OOF answered another general question, not the core experiment necessary to synchronize `min_pairs`, and therefore did not include the main result of Phase 7.
6. E31 The E29 freeze was answered under conditions seen-task/unseen-peptide and 95% CI was given task-bootstrap; but the same list of transactions/pHC/shared-encoder sensitives, PairAcc, core elimination and sensitivity analysis containing re-training differences cannot be claimed to have been preferred to all baselines under the Trict agreement.

## 9. Final conclusions and deliverables

Phase 7 Initial target: the Human Project uses the ZXQ0QZ threshold consistent with the Mouse project, and the code, data and results are separated from the original `min_pairs=500` project. E29 multi-kernel CNN 3-seed mean is the final master model, with a sixed-test means AUROC being `0.84478`, relative to E17 upgrade `+0.01024` and synchronized improvements by the weakest task force. The new E31 indicates that the same freezing model still reaches ZXQ4QZ AUROC under strict peptide-disjoint OF, but relative standard OOF drops ZXQ5XZ; the paper should therefore also retain the main findings of the Standard Benchmark and clarify the unseen-peptide border created by the overlap.

The report companion is visualized in `results/figures_phase7/`:

- ZXQ0QZ: Progress of the Phase 7 model;
- ZXQ0QZ: Min_pairs contrast to benchmark extension;
- `03_e29_vs_e17_task_gain.png`: profit distribution by task;
- ZXQ0QZ: 157 listage headmap;
- `05_e29_seed_ensemble.png`:E29 seed averaging;
- ZXQ0QZ: OOF filter and test confirmation;
- `tissuepmhc_phase7_figures.pdf`: a merger of all six maps PDF;
- CSV: Recoverable source data for each graph.

E31 The accessories are located in `results/tissuePMHC_phase7_min200_e31_peptide_disjoint_oof/`: ZXQ1QZ, `split_audit.json`, `pair_fold_assignments.csv`, `member_oof_predictions.csv`, `ensemble_oof_predictions.csv`, ZXQ6QZ, `summary_metrics.csv`, `timing.csv` and `run.log`.

All maps are regenerated by `scripts/build_tissuepmhc_phase7_figures.py` from the freeze, without training models or modifying the experimental output.
