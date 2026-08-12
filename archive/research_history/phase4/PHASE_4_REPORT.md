# Phase 4 Study: Current progress

Update: 2026-07-13
Status: E8, E15 adopted; E9, E11, E12 stopped; E10, E13, E14 not implemented; crossed test not read
Freezing data: ZXQ0QZ, 24 items-H2 tasks, 13 items, 4 H2 responses

## 1. Study of boundaries and agreements

Phase 4 only uses `data/mousePMHC/mousePMHC_train.csv.gz` for model development. All reported results are 3-old plain-grouped, leading-only OOF; fixed test files are not read by E8, E9, E11, E12 or E15.

- Training data: 13,532 rows, 6,766 fairs.
- Data per task training: 124–470 days, median 264.5.
- OOF folds:3,split seed `20260711`.
- Independent training: ZXQ0QZ, ZXQ1QZ, `20260706`.
- Main indicator: Mean task AUROC; also reporting AUPRC, World-6 AUROC, H2 and task results.

The current Phase 3 single model anchor is E3b task-balanced Factorized MMOE:3-set means task AUROC is 0.8148.

## 2. E8: E3b independent of the

### Methodology

E8 No re-training model, strictly reuses three independent OOF projections for E3b. Each sample is predicted by three E3b models that have not seen the help-out pair, and is integrated with the equivalent rights.

Two candidates are pre-established:

1. (a) Probability mean: three seeds of equal probabilities, as the main candidate;
2. tab-rank means: to change the fractions of the seed to the average of the rights after the percentile rank in each tissue-H2 tab, as a fixed rodile dissipation.

E8 code reads only frozen E3b OOF files, authenticates that each seed has a single projection for each `sample_id`; does not read cross test, or chooses to seed or adjust weights.

### Outcome

Mean task AUROC  Mean task AUPRC  World-6 AUROC
|---|---:|---:|---:|
E3b single-seed indicator mean 08148 0.80430.6771
| E8 3-seed probability mean | 0.8350 | 0.8246 | 0.7050 |
| E8 3-seed task-rank mean | 0.8353 | 0.8245 | 0.7068 |

The AUROC gain of the E3b single document is ZXQ0QZ, the task-rank mean gain of ZXQ1QZ relative to the average value of E3b single document. Both candidates are higher than the average AUROC of the three seed single model of this tabk on 24/24 task.

By task, use the average AUROC variance across the Seed to botstrap:

\[
\Delta \mathrm{AUROC}_{\text{probability}} \in [0.0156, 0.0250],
\]

\[
\Delta \mathrm{AUROC}_{\text{rank}} \in [0.0165, 0.0244].
\]

Both are completely larger than zero. E8 explicitly uses the low-risk IPA4 threshold to become the anchor for the next global branch.

By H2, the AUROC gain of probability means relative to the average value of the single seed is: H2-Db ZXQ0QZ, H2-Kb ZXQ1QZ, H2-Kd ZXQ2QZ, H2-Kk `+0.0350`. This shows that the Esmble not only improves the overall average, but also significantly improves the non-Db H2 groups, which have fewer samples and higher single model margins.

The sample-by-species score correlation of three E3b seeds is 0.764–0.776 and not entirely redundant; it predicts a standard deviation average of 0.1065 persample. This explains why independence sees a large and wide-ranging gain on average.

## 3. E9:Multi-kernel CNN shared task heads

### Methodology

E9. Remobilize multi-kernel CNN peptide encoder. It retains E1 shared tabs, sample-level BCE, AdamW, epochs, folds and seeds, replacing only `Embedding -> Flatten/MLP` encoder with the position reserved Conv1d encoder: kernel sizes `2/3/5`, each kernel 32 rolls re-trew 9-mer and then fold and enter MLP.

E9 Using three saved OOF seeds that are fully aligned to E1. The E9 parameter is 119,368; the E1 counterpart is approximately 21,992. Therefore, its failure cannot be attributed to the fact that the CNN parameter is too small.

### Outcome

E1 3  E9 CNN 3 E9 − E1
|---|---:|---:|---:|
| Mean task AUROC | 0.8073 | 0.7888 | **−0.0186** |
| Mean task AUPRC | 0.7929 | 0.7740 | **−0.0190** |
| Worst-6 AUROC | 0.6617 | 0.6447 | **−0.0170** |
| Worst-task AUROC | 0.6155 | 0.5951 | −0.0204 |

For the average AUROC pair difference for each task overseed, the rootstrap 95% CI is:

\[
\Delta \mathrm{AUROC}_{E9-E1} \in [-0.0240, -0.0130].
\]

Only 2/24 task increased after crossing the average seed, while the remaining 22/24 decreased; three seeds combined nine task-level upgrades and 63 declines.

H2  task E9 - E1 means AUROC
|---|---:|---:|
| H2-Db | 12 | **−0.0280** |
| H2-Kb | 4 | −0.0121 |
| H2-Kd | 5 | −0.0121 |
| H2-Kk | 3 | −0.0001 |

E9 is only almost flat for H2-Kk as a whole, where skin-H2-Kk average gains are about `+0.0108`; however, only H2-Kk only has three sets of data, which cannot be used as a global encoder. The systematic decline on H9 indicates that currently, under the low resource setting of 24-task, E1 Flatten-MLP has been able to effectively use fixed 9-mer position patterns, while the local motif of CNN is not able to stabilize the concentration bias into gains.

### Decision-making

E9 does not meet any key promotion conditions for Stase 4: AUROC and AUPRC have dropped significantly, World-6 has declined, H2-Db/Kb/Kd has exceeded the permitted decline in the group and is running short of the list. Therefore, the CNN main line ** does not execute E10 CNN-E3b**. No more channels, changing kenel or repeating to chase the same OF results.

## 4. E11: Global/H2 Fixed soft integration

E11 Freezing global branch with three-seed plain H2-grouped branch using E8 3-seed probability mean. Within each tissue-H2 task, two branches are converted to percentile rank and fixed to `0.5/0.5` average; probability average is fixed and reduced. No per-task weight search is performed.

Method
|---|---:|---:|---:|---:|
| E8 global | 0.8350 | — | 0.8246 | 0.7050 |
| H2 branch 3-seed | 0.8174 | −0.0176 | 0.8084 | 0.6687 |
| probability fusion | 0.8344 | −0.0006 | 0.8260 | 0.6975 |
| rank fusion | 0.8351 | +0.0001 | 0.8262 | 0.6986 |

The AUROC increment for rank funsion is only ZXQ0QZ, task bootstrap 95% CI is `[-0.0048, +0.0047]`, while the work-6 lowers `−0.0065`. It does not meet the low risk integration threshold and therefore does not remain the final structure.

H2 branch score collation with global branch is 0.8736: there is a small amount of complementary information in the branch, but the whole branch is clearly weak and the fixed 0.5 mix is not able to offset the loss. The only positive signal is from H2-Kd:rank fact average ZXQ0QZ AUROC; however, the remaining H2 group does not have a consistent benefit and cannot be used as a basis for adding H2/task special weight search to the same OOF.

** Decision-making:** E11 Stop, do not enter E14.

## 5. E12:H2/tissue auxiliary supervision

E12 Add the tissue and H2 headers to E1 shared encoder + task headers with fixed losses as follows:

\[
L=L_{\mathrm{main}}+0.02L_{\mathrm{tissue}}+0.10L_{\mathrm{H2}}.
\]

E1 3 Seed Average  E12 3 E12 − E1
|---|---:|---:|---:|
| Mean task AUROC | 0.8073 | 0.7755 | **−0.0318** |
| Mean task AUPRC | 0.7929 | 0.7568 | **−0.0362** |
| Worst-6 AUROC | 0.6617 | 0.6390 | **−0.0227** |

The task-level bootstrap 95% CI is:

\[
\Delta \mathrm{AUROC}_{E12-E1} \in [-0.0425, -0.0214].
\]

Only 4/24 task increased after crossing the average seed; 12 task posts in H2-Db fell all down under three seeds, with the average change in H2-Db being ZXQ0QZ AUROC.

The auxiliary diagnosis supports the explanation of "invalidity/conflict signal": the mean value of the tissue auxiliary accuracy is 0.2143, below the ratio of 0.205 per cent for the maximum tissue class; H2 auxiliary accillary at 0.6926, higher than the ratio of most H2-Db classes 0.5531, but its shared encoder gradient and the average cosine of the main mission gradient is `−0.0116`. The average of the weighted aid gradient is also negative (`−0.0130`). The secondary target does not enhance the primary task ranking but instead introduces a stable negative migration.

** Decision-making:** E12 Stop; not implement E13. No H2 group shows a stable, clear, positive pattern to the contrary, and no H2-only auxiliary elimination is implemented.

## 6. Basis for discontinuation of E13/E14

- E13 Auxiliary signature, which relies on E12, has passed; E12 has failed significantly and therefore does not execute E13 Auxiliary MMoE.
- E14 Dependencies on effective E11 H2 fact or E13 auxiliary global branch; neither premise is valid and therefore E14 is not implemented.

This is not an omission of experiments, but a predefined reliance and cessation rule: to avoid continuing to stack on the same OOF benchmark mechanisms that have proved ineffective.

## 7. E15: Pre-registration 5-seed confirmation

E15 Freezing the only winning structure is E3b task-balanced Factorized MMOE, with a fixed entitlement to all members, etc., probability mean. The original member is seeds ZXQ0QZ, with the new member being pre-registered `20260707/08`; no member has been removed, a 4-seed subset has been selected or the weight has been modified.

Mean task AUROC  Mean task AUPRC  World-6 AUROC
|---|---:|---:|---:|
3-seed E8 freeze means 0.8350  0.8246  0.7050
E15 Pre-registration 5-seed probability means ** 0.8392** ** 0.8316** ** 0.7101** **
| 5-seed − 3-seed | **+0.0042** | **+0.0069** | **+0.0051** |

The pre-registered OOF date three have passed: AUROC gains exceed `+0.0010`, AUPRC and World-6 have not decreased. On a tab, AUROC is 19 won and 5 negative; task bootstrap 95% CI is:

\[
\Delta \mathrm{AUROC}_{5-3} \in [0.0016, 0.0066].
\]

In four H2 groups, the average gain of H2-Db, H2-Kb, H2-Kd was ZXQ0QZ, `+0.0024`, `+0.0109`, H2-Kk were average `−0.0038`, but only three tasks were included, and the increase was still higher. The new single model of the Seed meant 0.8102 and 0.8127 were all in the original E3b single model distribution; the 5-seed gain was from a lower margin of power, rather than from an unusually strong new member.

The official OOF date of E15 has been adopted, but as of this report has been updated, the Metadata record ZXQ0QZ. The final conclusions currently confirmed are therefore strictly limited to the train-only OOF. The code has been secured by the gaté-test runner; the final confirmation is a follow-up stand-alone decision.

## 8. Phase 4 Summary and freezing status

Current evidence is sorted as:

\[
\text{E15 5-seed E3b ensemble} > \text{E8 3-seed E3b ensemble} > \text{E3b single model} > \text{E1 shared heads} > \text{E9/E11/E12 negative candidates}.
\]

The core conclusion of Phase 4 is not that more complex encoder, hard H2 branches or auxiliary labels improve mouse tasks; none exceeds simple global E3b. The real repossessable performance sources are:

1. task-balanced Factorized MMOE reserves a global sharing across the issue/H2 and absorbs some heterogeneity with light gate;
2. 5 independent trainings seed equivalents prepare intervention targeting significantly lower margins;
3. Additional structural constraints are more likely to generate negative migration under the current set of fixed 9-mer, small samples, and more than a task.

The current Phase 4 OOF main result is:

```text
E15 preregistered 5-seed task-balanced Factorized MMoE probability ensemble
mean task AUROC = 0.8392
mean task AUPRC = 0.8316
worst-6 mean AUROC = 0.7101
```

Conclusive boundary: These values are for current 24-task, ZXQ0QZ, Pair-grouped OOF, balance balance benchmark. They are not yet confirmed results of a combination-test, strict peptide-disjoint, protein-disjoint, unseen H2 or foreign forces.

## 9. Key documents

- E8 runner:`scripts/run_mousepmhc_phase4_e8_e3b_seed_ensemble_oof.py`
- E8 Result: `results/mousePMHC_phase4_e8_e3b_seed_ensemble_oof/`
- E9 runner:`scripts/run_mousepmhc_phase4_e9_multikernel_cnn_shared_oof.py`
- E9 Result: `results/mousePMHC_phase4_e9_multikernel_cnn_shared_oof/`
- E11 runner:`scripts/run_mousepmhc_phase4_e11_global_h2_soft_fusion_oof.py`
- E11 Result: `results/mousePMHC_phase4_e11_global_h2_soft_fusion_oof/`
- E12 runner:`scripts/run_mousepmhc_phase4_e12_auxiliary_shared_oof.py`
- E12 Result: `results/mousePMHC_phase4_e12_auxiliary_shared_oof/`
- E15 runner:`scripts/run_mousepmhc_phase4_e15_five_seed_confirmation.py`
- E15 Result: `results/mousePMHC_phase4_e15_five_seed_confirmation/`
- Phase 4 Route: ZXQ0QZ
