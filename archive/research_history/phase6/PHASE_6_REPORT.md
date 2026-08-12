# MousePMHC Phase 6 Experiment Report: Difficulties H2 Targeted Uplifting and Stopping Conclusion

Status: E25-E28 completed; E29-E31 not implemented; E32 Freezing match-test completed; E33 peptide-disjoint split audio completed and model training not currently implemented.
Development anchor: Phase 4 E15,5-seed task-balanced Factorized MMOE probability ensemble.
Development data boundary: The TRUNK feature of the official mouse UniProt reference protein group is used only to predict `data/mousePMHC/mousePMHC_train.csv.gz`, its train-only OOF. The model and the five seed etc. are frozen before E32 can read `data/mousePMHC/mousePMHC_test.csv.gz` once it is frozen.

## 1. Objectives and conclusions

The objective of Phase 6 is to improve the relatively weak H2-Kb and H2-Kd in E15 without compromising H2-Db/H2-Kk. The results are as follows:

1. E25 Excludes pair construction errors, peptide tag conflicts in task as the main explanation for Kb/Kd low scores; the hard sample also shows higher perceived differences.
2. The E26 unified plank/position proposal failed clearly: Macro AUROC is `-0.01110` for E3b, all four H2 are down, stop.
3. Kd rank-8 apperter for E27 has a weak Kd local positive signal (`+0.00498`), but the overall AUROC only has `+0.00079` and does not pass.
4. E28 Independent Kb/Kd rank-8 adapters is the best new candidate for this phase, Macro AUROC ZXQ0QZ, AUPRC `+0.00288`, but is still below the pre-registered ZXQ2QZ AUROC threshold, the Bootstrap CI cross-zero and the Kb/Kd group is well below the targeted threshold. Therefore, do not enter the five seeds to confirm.

Thus, the only winner of the freeze development phase at the end of the Phase 6 is still E15:

\[
\mathrm{mean\ task\ AUROC}=0.8392,\quad
\mathrm{mean\ task\ AUPRC}=0.8316,\quad
\mathrm{worst6\ AUROC}=0.7101.
\]

The above values are the development phase of the development process -- total OFF. The E32 fixed test obtained a one-time after the freeze means task AUROC ZXQ0QZ, mean task AUPRC ZXQ1QZ, World-6 AUROC ZXQ2QZ, confirming that E15 is not a pan-disjoint; it is not still a peptide-disjoint, protein-disjoint, unseen-H2 or an extra-force feature.

## Uniform agreements

- 24 tissue x H2 task, ZXQ0QZ; 6,766 train ranges, 13,532 rows.
- 3-fold pair-grouped OOF,split seed `20260711`.
- E26-E28 uses seeds ZXQ0QZ and compares it with E3b of the seed and task.
- The main indicators are the mean task AUROC; the protection indicators are the mean task AUPRC, the World-6 AUROC, the H2 macro average and the task AUROC.
- E25–E28 The sixed test was not read during the development and decision-stop; E32 is read once after all the rights of E15 structure, hyperparameter, five seeds and etc.

The pre-registration promotion threshold for candidates is: macro AUROC at least ZXQ0QZ, macro AUPRC at least ZXQ1QZ, World-6 at least ZXQ2QZ; Db/Kk at least no decrease above ZXQ3QZ; Kb or Kd at least one increase at ZXQ4QZ; Kb/Kd at least 6/9 task improvements; task bottom of Bootstrap AUROC CI at least `-0.0010`.

## 3. E25: Kb/Kd Data and Disability Audit

E25 No training model, read only the Train and frozen E15 OOF members forecast. All 6,766 pairs are satisfied: two lines, one plus one, different from the Tissue-H2 task, and more or less from the parent UniProt and more than the peptide. No invalid pair, and no Peptide in the task has the same positive or negative label.

E15 means AUROC, train days/task means five-said score std
|---|---:|---:|---:|---:|
| Db | 12 | 0.8946 | 311.8 | 0.0924 |
| Kb | 4 | 0.7335 | 317.0 | 0.1771 |
| Kd | 5 | 0.7970 | 272.4 | 0.1718 |
| Kk | 3 | 0.8292 | 131.3 | 0.1647 |

In Kb/Kd, the members of the E15 misallocated pair are more divided than correct: Kb for ZXQ0QZ vs ZXQ1XZ, Kd for ZXQ2QZ vs ZXQ3XZ. The three most difficult tasks are pancreas-Kd (AUROC ZXQ4QZ, 140 train pairs), colon-Kb (ZXQ5QZ, 208 pairs) and liver-Kb (ZXQ6QZ, 367 pairs). This supports the diagnosis of "stability in difficult samples" but does not support the misinterpretation of data.

## 4. E26:flank/position processing branch

E26 Using the official UniProt reference protein group to send 10 aa flank, relative position and N/C end distance each. 13,430/13,532 rows (`99.2462%`) can be uniquely mapped; the rest of the rows are retained for missing token and missing indicator. The model is added to the organization conditions at the E3b main export office, and initializes to zero entry to the backbone scale.

Indicator E26  met E3b
|---|---:|---:|---:|
| Mean task AUROC | 0.80374 | 0.81484 | **-0.01110** |
| Mean task AUPRC | — | — | **-0.01252** |
| Task wins/losses | 3 / 21 | — | — |

AUROC task team 95% CI is `[-0.01653, -0.00611]`, which is completely below zero. H2 AUROC changes to Db ZXQ1QZ, Kb ZXQ2QZ, Kd ZXQ3QZ, Kk `-0.01504`. Moreover, the processing scale is negative and negative in different folds and does not learn to stabilize the direction.

** Decision-making: ** Stop harmonizing flank/position brach. No search of flank length, branch width, scale initialization or training wheels on the same OOF. E26 failed, so the combination with aadapter E29 is not performed.

## 5. E27: Zero initialisation for H2-Kd only

E27 After E3b peptide encoder only increases the ark-8 projection apter for H2-Kd. The initial model is identical to E3b as a result of the zero initialization of the project on adapter; training continues with the E3b 25 epoch task-balanced protocol.

Indicator E27 − met E3b
|---|---:|
| Mean task AUROC | +0.00079 |
| Mean task AUPRC | +0.00030 |
| AUROC bootstrap 95% CI | `[-0.00125, +0.00280]` |
| Task wins/losses | 14 / 10 |

H2 AUROC changes to Db ZXQ0QZ, Kb ZXQ1QZ, Kd ZXQ2QZ, Kk ZXQ3QZ. The pancreas, liver, thymus improved but the spleen-Kd dropped. E27 does not meet the overall ZXQ4QZ, Kd `+0.0120` or CI threshold.

** Decision-making:** diagnostic retention as "Kd private capacity may have a local value", but not promotion.

## 6. E28: Independent Kb/Kd Zero Initializations

E28 Insert separate Kb and Kd rank-8 adapter with the same agreement; private parameters are not shared and Db/Kk does not use adapter.

Indicator E28  meted E3b
|---|---:|---:|---:|
| Mean task AUROC | 0.81705 | 0.81484 | **+0.00221** |
| Mean task AUPRC | — | — | **+0.00288** |
| Task wins/losses | 14 / 10 | — | — |
| AUROC bootstrap 95% CI | — | — | `[-0.00026, +0.00489]` |

H2 AUROC changes to Db ZXQ0QZ, Kb ZXQ1QZ, Kd ZXQ2QZ, Kk `+0.00248`. Local improvements in Kb/Kd are concentrated on colon-Kb (ZXQ4QZ), pancreas-Kd (ZXQ5QZ), liver-Kd (`+0.00786`) and skin-Kd (`+0.00763`); but the skin-Kb (ZXQ8QZ) and spleen-Kd (`-0.00907`) are down.

E28 Meets the directional requirements for AUPPC, World-6 and Db/Kk protection, but fails in three key areas: Macro AUROC is smaller than `+0.0030`, Bootstrap CI is lower than the stability requirements of `-0.0010`, and Kb/Kd has not met `+0.0120`. It cannot therefore be described as an confirmed structure beyond E3b.

** Decision-making:** E28 stopped at three seed OOF; E30 conditionality integration not implemented or E31 V seed confirmed.

## E32: Freezing E15 lump-test communication

E32 Verify the OOF date of the Phase 4 E15, model structure, five pre-declarations seed (ZXQ0QZ) and probability mean. Each seed will be trained only once on the complete train and read once in a single reading of the sixed test containing 24 task units, each task 100 days. The Metadata records ZXQ1QZ, `model_selection_on_test=false`, data and freeze date save SHA-256; the total training and reasoning time is `271.36 s`, with a parameter ZXQ4QZ.

Trans-only OOF  Frozen fix descriptive deviation
|---|---:|---:|---:|
| Mean task AUROC | 0.8392 | **0.8562** | +0.0170 |
| Mean task AUPRC | 0.8316 | **0.8506** | +0.0190 |
| Worst-6 AUROC | 0.7101 | **0.7245** | +0.0144 |
| Worst-task AUROC | 0.6666 | **0.6840** | +0.0174 |

Of the 24 task task that was fixed, 18 AUROC were higher than their OOF, and 6 were lower. The descriptive score H2 was:

H2 Task OOF AUROC Fixed-test AUROCBog
|---|---:|---:|---:|---:|---:|
| H2-Db | 12 | 0.8946 | **0.9152** | +0.0206 | 0.9154 |
| H2-Kb | 4 | 0.7335 | **0.7404** | +0.0069 | 0.7404 |
| H2-Kd | 5 | 0.7970 | **0.8397** | +0.0426 | 0.8283 |
| H2-Kk | 3 | 0.8292 | **0.8024** | -0.0268 | 0.7757 |

The upgrades were mainly from H2-Kd and H2-Db, H2-Kb, with only minor improvements, and H2-Kk, with an overall decline. The hardest task was the colon-Kb (AUROC ZXQ0QZ), pancreas-Kd (ZXQ1QZ), Skin-Kb (ZXQ2QZ), colon-Db (`0.7277`), Skin-Kk (`0.7405`) and liver-Kb (`0.7639`). Thus, E32 supported "the learningability of mice on benchmark is recognized as being frozen", but not "all H2 are equally stable".

The full `pair_id` cluster 500 diagnostics were given to mean task AUROC 95% interval ZXQ1QZ, mean task AUPRC ZXQ2QZ, World-6 AUROC ZXQ3QZ. Macro PairAcc is ZXQ4QZ, World-6 PairAcc is `0.7117`. If the official paper takes CI as its main assumption, the number of bootstrap should be increased and the complete analytical product preserved.

### 7.1 Broadened Borders of Fix-test

E32 Guarantee that the train/test ZXQ0QZ does not overlap, but not strictly unseen-peptide or unseen-protein test:

Audit item Results
|---|---:|
| Train/test pair_id overlap | 0 |
| Test unique peptides also seen in train | 2,453 / 3,011(81.47%) |
| Test rows whose peptide is seen in train | 84.31% |
| Test unique parent UniProt also seen in train | 967 / 1,088(88.88%) |
| Test rows whose parent UniProt is seen in train | 93.04% |

E32 should therefore be called **frozen international pair-disjoinfixed-test verification**, not peptide-disjoint, protein-disjoint or external independent troupe validation.

## 8. E33: peptide-disjoint split audio and current stop-decision

The current case of whited benchmark relies on inter-organizational relationships "with H2, with parent UniProt, but reported by other organizations." The introduction of a pixel formula "Deep deletes the relevant pair directly after discovering the cross-split piptide" does cut off a large number of cross-organizational pairs, could cause a sharp decline in the size of the benchmark and a partial low-resource Tissue-H2task sample. Therefore, eliminating overlapping entities to force the creation of a peptide-disjoint subset.

However, E33's read-only alternative tested: to merge all pairs shared peptide connections into connection fractions and then allocate the full fractions to fold. The audit found that the programme could ** at the current train to keep pair separate from peptide without deleting data:

Audit item Results
|---|---:|
Totals / Keeps 6,766 / 6,766
| Unique peptides | 5,041 |
| Pair–peptide connected components | 1,844 |
| Largest component | 50 pairs |
1 2 255–2 256  per head-out old
held-out old
Xiaobing:
♪ Take a look at the airs 82–314 ♪
| Peptide overlap / pair overlap | 0 / 0 |

Therefore, the phrase "benchmark inevitably descends or task disappears" cannot be written as the fact that the complete version of the program is only described as a simple deletion risk. The current decision is:** to retain the completed split Feiability audit, but not to perform the 5-seed x 3-fold model training, without producing the peptide-disjoint performance conclusion.** If implemented in the future, it should be used as a predefined robess protocol, and also to report the difference between the completed sample size, the task fold and the sandard OF. This stop decision does not eliminate the peptide limit of the standard match test.

## Summary and follow-up

Phase 6 shows that in the current low resource 24-task benchmark, Kb/Kd cannot be improved by expanding the same MMOE, joining the unified source-protein flank message, or adding a small H2 private apter. The weak positive signal of E28 suggests that there is no private expression space at all, and the effect is not sufficient to offset the optimism gap created by the repeated OOF selection.

The most conservative conclusion is that E15 benefits are mainly derived from stable global sharing and independent seed average probabilities; no Kb/Kd specialization structure has yet demonstrated the value of a wide range of stable additions to the current data. E32 further confirms that E15 freezes maintain good performance on internal pair-disjoined test, but H2-Kk declines and significant amounts of Train/test paper, protein overlap are limited to extrapolation.

Next steps should not continue to be made on the same OOF apperter rank, leaving rate, epoch, H2 target pool or integrated weights. If the project continues, priority should shift to:

1. (b) Use E32 as the only frozen fix-test primary result for mice, without modifying the model based on test;
2. (c) Priority is given to restoring the study/PMID profile or to establishing external validation from a truly independent source;
3. If a rigorous physicalization is undertaken, priority is given to the auditable possibility of adopting a negotiated-componid-disjoint OOF; a large number of cross-organizational relationships cannot be deleted directly from the current benchmark. Independent peptide-ZXQ0QZ fixed benchmark still needs to be designed separately;
4. Redefinition of pre-training and aadaptation studies after more high-quality Kb/Kd immunopeptidomics data are collected or integrated.

## 10. Recoverable documents

- E25:`scripts/run_mousepmhc_phase6_e25_kb_kd_audit.py`,`results/mousePMHC_phase6_e25_kb_kd_audit/`
- E26 features:`scripts/prepare_mousepmhc_phase6_e26_flanks.py`,`data/mousePMHC/mousePMHC_train_flank_features.csv.gz`
- E26:`scripts/run_mousepmhc_phase6_e26_flank_mmoe_oof.py`,`results/mousePMHC_phase6_e26_flank_mmoe_oof/`
- E27:`scripts/run_mousepmhc_phase6_e27_kd_adapter_oof.py`,`results/mousePMHC_phase6_e27_kd_adapter_oof/`
- E28:`scripts/run_mousepmhc_phase6_e28_kb_kd_adapters_oof.py`,`results/mousePMHC_phase6_e28_kb_kd_adapters_oof/`
- E32:`scripts/run_mousepmhc_phase6_e32_e15_fixed_test.py`,`results/mousePMHC_phase6_e32_e15_fixed_test/`
- E33 split audio (completed, not implemented model training): ZXQ0QZ, ZXQ1QZ
- Chance 6 figures: ZXQ0QZ, ZXQ1QZ (5 PNG, source CSV, merge PDF with metadata; including milestone figure in AUROC descending order, E3b dashing and colouring of evidence type)
