# HumanPMHC premium: full model results and cause analysis

## 1. Overview of results

All of the completed single-seed tests were used for this comparison; the internal model was trained using premium train and evaluated in premium test, MHCflurry and NetMHCpan were not reading premium train external models.

Models Average task AUROC  Average task AUPRC  PairAcc  worst 10 task average AUROC
|---|---:|---:|---:|---:|
| E29 multi-kernel CNN | **0.7040** | **0.6929** | **0.7040** | 0.5421 |
| E14a auxiliary dual branch | 0.6984 | 0.6865 | 0.6971 | **0.5557** |
| E2 shared heads | 0.6827 | 0.6720 | 0.6896 | 0.5436 |
| E0 random forest | 0.6779 | 0.6629 | 0.6835 | 0.5446 |
| MHCflurry presentation | 0.6677 | 0.6568 | 0.6781 | 0.5175 |
| NetMHCpan EL rank | 0.6545 | 0.6406 | 0.6659 | 0.4971 |
| NetMHCpan BA rank | 0.6332 | 0.6225 | 0.6496 | 0.4552 |

The internal model still maintains an average ranking of E0  E2  E14a  E29, but the phase gains are significantly compressed:

premium AUROC gain 75 tasks win/ level/ negative  task blockstrap 95%
|---|---:|---:|---:|
| E2 − E0 RF | +0.0048 | 40 / 0 / 35 | [-0.0066, 0.0161] |
| E14a − E2 | **+0.0157** | **49 / 1 / 25** | **[0.0071, 0.0243]** |
| E29 − E14a | +0.0056 | 42 / 0 / 33 | [-0.0043, 0.0152] |

The area here is only a fixed list of seed results that re-sampling 75 tasks to describe the stability between tasks cannot substitute for more perceived training uncertainties. Strictly speaking, E14a is the only clear and stable gain in the three upgrades; the average gain for E2 and E29 is less than 0.006 and the direction between task is inconsistent.

The original 44-task and ad-seed AUROC process was 0.7558  0.7918  0.8102  0.8212, corresponding to gains +0.0360, +0.0185, +0.0110. Premium was 0.6779  0.68227  0.6984 § 0.7040, so it was not "upgrading sequences that failed", but the available signals were compressed at all stages, and E2 compression was most visible.

## 2. Is it a question of model structure or data processing?

The conclusion was that:** no significant data processing bug was found; the main reason was a more stringent and thinner data division, coupled with the difficulty of the premium data itself and its proximity to organizational specificities; and the model structure was clearly not matched to this new difficulty.**

### 2.1 No apparent processing error found

- The 7,500 lines, 3,750 pairs and 75 tasks of the premium test are fully covered.
- Each pair has a positive or negative balance, with no trade/test leak; the sequence is 9-mer.
- Both MHz and NetMHCpan over 32 HLA, 7,204 to heavy peptide-HLA queries are 100% overlayed.
- External models have declined at the same time as internal models, so the result cannot be interpreted by an internal training script with a single error.
- The UniProt of source molecule is missing about 23.3 per cent, but the current E0/E2/E14a/E29 does not use the field, so it is not a direct cause of the decline in the score in the current cycle.

### 2.2 Data disaggregation is one of the primary reasons, but not equal to "Errors"

The median training pair for each task is only 137, the original 44-task is 644; the lower quadrilogy of Premium is 76 and the original data is 562. Premium each task is only 50 pairs, the original 100 pairs, so the task indicator itself is more volatile.

More importantly, the test relative to the train has significantly decreased the overlap between entities:

Dataset  peptide Overlap  HLA–peptide Overlap  Protein Overlap
|---|---:|---:|---:|
| premium | **43.1%** | **28.2%** | 85.2% |
44-task 77.3%  71.3%  95.8%
| Phase 7 | 78.6% | 68.3% | 96.0% |

This brings premium closer to extrapolating the unseen peptide/HLA-peptide combination, while the original project is more often inserted near the covered entity. E29's PairAcc rises from 0.6786, "neither of the peptide" to 0.7633, "both of the two"; E29's advantage over E14a is also mainly on the seen peptide. This suggests that the additional capacity of CNN is better at reusing the motif that has emerged in the training than at resolving the real unseen extrapolation.

The four-digit mission training volume, when it was small to large, was E29 AUROC 0.6744, 0.6850, 0.7126, 0.7443; the training volume was 0.320 (p = 0.005) related to the Spearman of E29 task AUROC, and therefore there was direct evidence of the limitations of data fragmentation on the deep model.

### 2.3 Data are also more difficult per se, but the evidence is now more like "weak labels and task definition difficulties", not dirty data

The positive example of the premium is the one reported in the target organization, and the negative one reported in the other organization but not in the target organization.

When the number of submissions in the peptide in other organizations is increased from 1 to 2 and 3+:

- E29 PairAcc:0.7197 → 0.6462 → 0.6484;
- MHCflurry presentation:0.6973 → 0.6220 → 0.5824;
- NetMHCpan EL rank:0.6922 → 0.5698 → 0.5714.

The more common it is across tissues, the more difficult it is to separate between positive and negative. This may be both more real biological ambiguity and include weak label noises caused by "undetected by the target organization". It explains why all models, including freezing external models, are falling on premium.

## 3. Why are the stages of the gains small?

### E2

E2 shared peptide encoder does move across task, but premium spreads training data to 75 tissue-HLA tasks, each smaller and unseen more. Sharing represents benefits that are offset by extrapolation by a small and stronger entity with a small sample of the top of the tabk, so that there is only E0 RF + 0.0048.

### E14a

E14a is not entirely "small up": +0.0157 is the only steady phase gain. The HLA/tissue branch provides a more structured general bias than a pure task head, which reduces data scarcity in small task. It still relies mainly on peptide and discrete task information, without direct protein expression, tissue expression, antigen processing or experimental sources, and thus the ceiling is still limited by mission signals.

### E29

E29 multi-kernel CNN increases local motif modelling capabilities for fixed 9-mer, but E14a has been able to learn a lot about location-specific patterns; adding new CNN capacity is difficult to estimate with a very small amount of HLA/task data. E29 relative to E14a, while average +0.0056, only 42/75 task has been raised, with the lowest 10-task mean still falling from 0.555 to 0.5421, indicating that it increases averages while increasing tail instability.

So the structural problem is not that CNN is wrong, but that ** there is a mismatch between the existing input features and the organizational speciality objectives of premium**. The marginal benefits of continuing to pile the peptide-only capacity are likely to be limited.

## 4. Why are external models low?

Most notably, the forecast objectives are not consistent:

- NetMHCpan BA Main Learning peptide-HLA combination;
- NetMHCpan EL and MHz reservation added a signal closer to efface/presentation;
- Premium compares organizational preferences within the same tissue-HLA task and is controlled by HLA/protein background.

The external model does not see the target target, source protein, protein enzyme processing, samples and experimental context, and cannot answer: "Why is this submitting peptide more likely to be in tissue A than in tissue B?" Pair structures also proactively control a large number of general pMHC signals, so they can only use residual general combination/transmission tendencies.

The results are consistent with this interpretation: MHCflurry preparation AUROC 0.667, higher than its level of coverage 0.635; NetMHCpan EL 0.6545, higher than BA 0.6332. The scores include better targets for delivery training, but still lack the conditions for the presentation.

The low score is not due to HLA's unsupported, because all 32 HLAs and all test lines have been successfully predicted. The AUROC on the same freezing model on the original Phase 7 is 0.6851, NetMHCpan EL 0.683, BA 0.6577, all down to 0.667, 0.6545, 0.633 on premium. This further proves that the premium itself has removed more generic PHC shortcuts.

## 5. Suggested next steps

1. Retain the current division as a strict gender equality benchmark, with additional reporting of the tiered indicators seen/unseen peptide and seen/unseen HLA-peptide, rather than just reporting the overall average.
2. If the objective is to determine structural ceilings, priority is given to adding real tissue/protein conditionalities rather than continuing to expand peptide-only CNN: for example, tissue expression, protein abundance, antigen processing and experimental sources.
3. Independent review or low-weight sensitivity analysis of `other_tissue_presentation_count >= 2` samples to judge the impact of weak labels on results.
4. 3–5 Seed validation of E2 and E29 using fixed divisions; the current single document is sufficient to illustrate average trends, but not sufficient to assert that the small gains of the two can be duplicated.

## Chart File

- `01_performance_overview`: Full model, stage process, phase gain stability, tail staleness.
- `02_task_auroc_heatmap`: 75 task x 7 main model AUROC thermal map.
- `03_data_split_diagnostics`: Training volume, cross-organizational ambiguity, overlap of the Train/test entity and E14aE29 transverse.
- ZXQ0QZ: external model objectives mismatch, inter-organizational degradation and comparison of original data.
- `premium_results_all_figures.pdf`: Combined Pages of the four above.
