# TissuePMHC Phase 2 Report

Update: 2026-07-12
Status: E15-E27 and E29 completed; E29 5-seed is the final master result of standard split; pre-registration confirmation extension closed, E28 not implemented

## Summary

The objective of Phase 2 is to verify whether a more robust fractional integration and low-cost integration approach can yield additional benefits in the two-branch structure of E14a. E14a consists of a global auxiliary branch with a Tissue/ HLA-assisted supervision and a regular branch by HLA grouping (HLA plain branch); the original integration is a fixed 0.5 probability average.

The phase has now completed the integration of fixed integration rules (E15), MC Dropout (E16), multi-random seed predictions (E17), the global integration weight of valueration selection (E18), the training period checkpoint/snapshot integration (E19), SWA (E20), five auxiliary learning/structual exploration (E21-E25), the strict OOF ' s gready selection and stacked generalization (E26-E27), and Multi-kernel CNN peptide encoder (E29). The strongest frozen standard-split results are pre-registered E295-seed meanings: mean AUROC 0.8373, mean AURRC 0.8259, world-10 weekly AUROC 0.770.

The main conclusion is that the margin between the independent random seed and the expression peptide encoder can be superimposed. task-wise rank fasion remains a solid branch integration rule; MC Dropout, Checkpoint/snapshot ensemble, SWA, dynamic auxiliary weights and complex two-tier integration do not yield the same level of benefits. E26/E27 shows that the same candidate cannot be able to create new information by reprocessing; E29 shows that the introduction of multiscale local motif codes while preserving the strong E14a two-branch structures can achieve stable, aggregate gains on the current closed-set standard standard split. The early MLP ' s Flatten code itself has been retained, so the added value of E29 should be attributed to local volume, multiscale field and parameter sharing bias rather than "first-time reserve".

After completing the full-link audit, the report strictly limits the conclusions to the current benchmark: E29's preservation projections and aggregate indicators can be independently calculated and the E17 upgrades are consistent at both mission and random seed levels; but 77.69 per cent of the test lines in the field of standard split have been peptide, 97.36 per cent of the test lines parent UniProt has appeared in other tasks of the training set and the negative example is "recorded in other organizations and unrecorded in the target organization." This result cannot therefore be directly extrapolated as proof of "non-submission" of unseen peptide, unseen protein, unseen HLA or real biology.

## 1. Experimental scope and co-setting

The base model for Phase 2 is E14a. Each task is defined by tisue and HLA allele; Standard split contains 44 tasks. With the exception of E17 's 5-seed extension, E15, E16, E18, E19, E20 uses three random seeds for training: 20260704, 206075, 20260706. The training configuration maintains 25 epochs, AdamW, leaving ratting 0.001, batt size 512.

The main indicators are the EuroCs and AUPRCs, which are ambitiously balanced; and the report is the first-10 means AUROC, which avoids the average indicator masking the degradation of difficult tasks.

All integration experiments are forecasted in ZXQ0QZ, ZXQ1QZ, `mhc_restriction` alignment branch; the same labels alone are not sufficient to ensure sample-by-sampling. The procedures involve random reasoning or the uniform control of the experiment of Python, NumPy, CPU Torch and CUDA RNG status, avoiding extra reasoning or training paths to change the random sequence of subsequent models. E18 uses valueation in train, E26, E27 and E29 to complete local selection using pair-grouped OOF; these rules are experimental designs and are reserved for the audit of section 8 for long-term reuse of Standard test, pre-registration credibility and cross-dataset extrapolation restrictions.

E14a The results as a starting point for this phase are:

Mean AUROC Mean AUPPC Mean AccuracyMean MCC World-10 Mean AUROC
|---|---:|---:|---:|---:|---:|
E14a: auximical global + plain HLA, average fixed probability 0.8116  0.79778  0.73772 0.4769  0.7349

## 2. Phase 2 completes the experiment and results

Experiment,  Core Comparison  Best configuration  Mean AUURPC  World-10 Mean AUROC
|---|---|---|---:|---:|---:|---|
E15  Fixed integration rules 0.8130  0.0012  0.79990  0.00008  0.73881  0.0023  Retention
E16 Single model MC Dropout  20 predictions + rank Fusion 0.811 0.0015  0.791  0.0019  0.73776  0.0044  not better than E15
E17  multi-seed forecast average 5-seed + rank forecast0.8263 0.8139  0.75773
E18  valuation selected weights of global weights
E19  Training period integration 0.8122  0.0017  0.798  00.0036  0.7385  0.0038  checkpoint/ snapshot ensemble
E20  SWA blending  final checkpoint  0.8150  0.0004  0.805  0.0011  0.74220  0.0021  SWA not retaining
E21  Gradient Similarity Support Door Control (1-seed screen)  Global retrain + Fixed HLA plain  0.8069  0.79332  0.7328  Assisted date near zero, stop
E22 Periodic Nash-MTL (1-seed screen)
ForkMerge (1-seed screen)  pair-grouped value fork/merge  0.8031  0.7895  0.7270  valueation filter is valid, external generalization is ungained
E24  Auto-Lambda (1-seed screen)  valueation prism meta-loss  0.8056  0.7931  0.7289  weight almost unchanged, stop
E25 HLA-Structured PLE(3-seed)  2 global + 12 HLA-private experts  0.7938  0.00030  0.7769  0.0033  0.7212  Gate 0.0034  health but not better than E10/E14a
E26  3-old plain-grouped OOF greeny section  E14 final 3-seed meant 0.8246  0.8116  0.7535  slightly higher than E17 3-seed but not E17 5-seed
E27  OOF task-rank Logist stacking  L2, C= 0.1  0.8243  0.8109  0.7535   not better than E26/E17; candidate colinearity is clear
E29 Multi-kernel CNN E14a(3-seed) Kernel 2/3/5, position maintained  0.8341 0.8228 0.7634 intermediate confirmation of result
E29  Multi-kernel CNN E14a (pre-registration 5-seed)  new fixed seeds 20260707/08; no change of weight/member  0.837 **  0.8259 **0.7670  standard split final best

Note: E17 's 5-seed refers to the average of five independent training model forecasts in each branch before making a task-rank Fusion. It is a fixed 5-seed integration result, so that there is only one polymer in the stability table; its "0-seed standard deviation" cannot be misinterpreted as having proved zero-variencies in duplicate experiments.

## 3. E15: De-consolidation of the rules of fixed integration

E15 Compares three rules on a sample-by-sample projection of a fully aligned E14a branch: average probability, logit and tab-wise percentile-rank average. The ranking is as follows:

♪ Mean AUROC ♪ Mean AUPPC ♪ Mean MCC ♪ World-10 Mean AUROC ♪
|---|---:|---:|---:|---:|
| task-rank average | 0.8130 | 0.7990 | 0.4800 | 0.7381 |
| logit average | 0.8118 | 0.7981 | 0.4769 | 0.7337 |
| probability average | 0.8116 | 0.7978 | 0.4769 | 0.7349 |

The base probability of the task increases on average to about 0.0014 AUROC relative to E14a. It does not require a complete alignment of probabilities calibration scales for the two branches, but only information on the ranking of the branches in task, and is therefore more suitable for the default integration rules as a follow-up integration.

** Decision-making:** Follow-up E16–E20 Default use task-rank Fusion.

## E16-E20: Uncertainty and Training Track Integration

### 4.1 E16: MC Dropout Incorporation of the Modition Period

E16 The same training model is activated during the reasoning period with an average of 5, 10 and 20 random predictions, followed by E15's task-rank Fusion.

MCC Number of Mean AUROC  Mean AUPPC  World-10 Mean AUROC
|---:|---:|---:|---:|
| 5 | 0.8107 | 0.7971 | 0.7360 |
| 10 | 0.8114 | 0.7973 | 0.7361 |
| 20 | 0.8121 | 0.7981 | 0.7376 |

More MC Draws have slightly improved, but 20 predictions are still below the normal task-rank Fusion of E15. The uncertainty of the current model is almost non-existent to provide sufficient independent integrated members.

Random number audits: MC reasoning consumes random numbers of dropout mask. Codes have been saved and restored to Python, NumPy, CPU Torch and all CUDA RNG status to ensure that MC reasoning does not change the subsequent initialization of HLA branches or DataLoader shuffle.

** Decision-making:** Not as an independent master model; may be used as a low priority complementary candidate in the follow-up pool of genuine OOFs.

### 4.2 E17: Multi-random seed predictions

E17 Provides for the independent E14a model preparation exercise targeting: the probability of having an average of the global and HLA branches, respectively, and then task-rank foundation. The results are as follows:

Seed scale of the integration
|---:|---|---:|---:|---:|---:|---:|
| 3-seed | 20260704–20260706 | 0.8243 | 0.8109 | 0.7467 | 0.4936 | 0.7530 |
| 5-seed | 20260704–20260708 | 0.8263 | 0.8139 | 0.7492 | 0.4986 | 0.7573 |

This is the most visible and stable source of performance for Phase 2: 0.8116 from E14a to 0.8263 from 5-seed. 5-sided further increases to 0.0020 AUROC than 3-bed and the weakest 10 missions from 0.7530 to 0.7573.

** Decision-making:** E17 5-seed was the most powerful anchor for Phase 2; E29 was the best compared to the previous one. The results should still be compared to 0.8263.

### 4.3 E18: Value Selection Select global integration weights

E18 Cuts value by grouping in train by pair_id; selects only one global/ HLA global rank weight with only value selection, and eventually re-reads it with complete Train and test evaluation. The design avoids the test weighting.

Mean AUROC Mean AUPPC Mean MCC World-10 Mean AUROC
|---|---:|---:|---:|---:|
| fixed 0.50 rank average | 0.8130 ± 0.0012 | 0.7990 ± 0.0008 | 0.4800 ± 0.0015 | 0.7381 ± 0.0023 |
| validation-selected global weight | 0.8137 ± 0.0016 | 0.7990 ± 0.0020 | 0.4787 ± 0.0039 | 0.7411 ± 0.0035 |

The selection weight of the value brings a very small average AUROC gain (approximately 0.0007), and improves the lower limit of the difficult task, but does not approach more perceived gains of E17. The use of single global weights for 44 tabs also limits their ability to express.

** Decision-making:** as a reference-safe weighting reservation; not replacing the E175-seed baseline.

### 4.4 E19: Integrating checkpoint with snapshot during training

E19 Normal checkpoint, checkpoint Espoons, and cosine-restart snapshot essemble.

Main AUROC  Mean AUPPC  Mean MCC  World-10 Mean AUROC
|---|---:|---:|---:|---:|
| final checkpoint | 0.8122 ± 0.0017 | 0.7980 ± 0.0036 | 0.4720 ± 0.0011 | 0.7385 ± 0.0038 |
| checkpoint ensemble | 0.8096 ± 0.0018 | 0.7951 ± 0.0039 | 0.4670 ± 0.0032 | 0.7387 ± 0.0021 |
| snapshot ensemble | 0.7962 ± 0.0023 | 0.7805 ± 0.0031 | 0.4507 ± 0.0056 | 0.7310 ± 0.0017 |

The results show that the error of the adjacent training checkpoint is not independent enough; the average checkpoint dilutes the effective ranking of the final checkpoint. The snapshot esempmble is clearly the weakest, suggesting that the end-of-cycle model under the current cosine-restart configuration does not constitute a complementary member.

** Decision-making:** E19 only reserved financial final checkpoint; checkpoint/snapshot esmble not placed in the preferred candidate pool.

### 4.5 E20: SWA pairing

E20 From the same training prefix, strictly compares the normal AdamW final checkpoint with the SWA conversion. Normal and SWA paths save and restore Python, NumPy, CPU Torch, CUDA RNG status at the forkpoint; each late epoch checks RNG alignment. The observed differences are therefore attributable to the SWA path, not to the initialization or batting order.

Global branch  Main AUROC branch  Mean AUPPC  Mean MCC  World-10 Mean AUROC
|---|---|---:|---:|---:|---:|
| final | final | 0.8150 ± 0.0004 | 0.8005 ± 0.0011 | 0.4786 ± 0.0052 | 0.7420 ± 0.0021 |
| SWA | final | 0.8141 ± 0.0003 | 0.7992 ± 0.0018 | 0.4756 ± 0.0016 | 0.7415 ± 0.0029 |
| final | SWA | 0.8121 ± 0.0000 | 0.7972 ± 0.0014 | 0.4765 ± 0.0059 | 0.7398 ± 0.0020 |
| SWA | SWA | 0.8107 ± 0.0007 | 0.7956 ± 0.0019 | 0.4717 ± 0.0020 | 0.7393 ± 0.0029 |

The global-only SWA AUROC drops on average by 0.010; HLA-only SWA by 0.0030; and the two branches decline simultaneously by SWA by 0.0044. The degradation of the HLA branch is in the same direction in three seeds.

E20 final/ final results are higher than E19 final checkpoint, but the difference cannot be attributed to SWA. E20 The valid conclusion for SWA can only be derived from the same round final/SWA blending.

** Decision-making:** Not using SWA; following up with normal checkpoint. SWA does not enter E21 candidate pool.

## E21-E25: Support weights and structural exploration

E21–E24 only retrains the global auxiliary branch of E14a and retsk-rank forecast after the same ed & retrend HLA plain projection. E23/E24 uses the train pair-grouped valueration, so its model is only updated with 80% of the little data; their relative complete test gap in E14a cannot be attributed to the algorithm itself, but does not meet the screen condition for entering 3-seed. E25 is an independent single model structure experiment that is directly compared with E10 MMoE and E14a global branch.

♪ Experiment, tact, tact, tact, t-- ♪ Results, and decision-making ♪
|---|---|---|
E21 Gradient-Similarity Gating Average Tissue/HLA Gate
E22 Periodic Nash-MTL 475 updates were successful; the effective gradient for later macro targets contributed approximately 35% ZXQ0QZ 1-seed integration AUROC 0.8051 compared to E14-0065; balance optimized at the expense of the task generalization and stopped
E23 ForkMrage  4 cycles after 5 cycles do not translate 0.00012/00082 BCE into est gains, stop
E24 Auto-Lambda  400 times meta update; tissue weight of about 0.100 ~ 0.1036, HLA about 0.100 ~ 0.107 -seed Integration AUROC 0.8056 compared to E14 − 0.0060; gradient signal is weak, stop
E25 HLA-Structured PLE2 global experts + 12 HLA-private experts; gate per task; 3-side no gate > 0.9 3-seed AUROC 0.7938, below E10 0.7948 and E14a global 0.8023; does not extend to 5 seeds or enter integration

E25 The final date mean entropy is 0.858–0.931 (theoretically caps log(3)1.099) and the maximum gain weight is 0.802–0.838, which indicates that failure is not an expert collapse. Its HLA-private weight is 0.501, 0.508, 0.566: The model does use the HLA-private route, but the additional specialization does not stabilize better than cross-HLA sharing.

E21–E25 The combined evidence shows that the auxiliary tasks are effective as a fixed light positive in the current peptide-only expression and standard split; that further dynamic magnification, compression or structural decomposition of the aid signal is not better than E14a. E26/E27 then provides rigorous OOF evidence that these directions and the co-constituent E14/MC candidates do not generate additional benefits.

## E26-E27: Strict OOF integration

E26 Use 3-old plain-grouped OOF. Each held-out sample is predicted by a model that has not been seen; the test forecast, which is generated by the complete train-retraining model, is read only after all the candidates have been selected. The candidate pool contains three separate documents, seed and two types of 3-seed mean.

OOF Mean AUROC  OOF World-10 AUROC  Test Mean AUROC  Test World-10 AUROC
|---|---:|---:|---:|---:|
| E14 final 3-seed mean | 0.8042 | 0.7257 | 0.8246 | 0.7535 |
| E16 MC-20 3-seed mean | 0.8038 | 0.7254 | 0.8239 | 0.7531 |

The original selection of E14 final 3-seed means; continuing to add no candidate to the preset minimum gain of 0.0001 means AUROC, so eventually there is only one member for Eemble. E26 is raised to only 0.0003 against E17 3-seed 's test of the means AUROC, which out of 44 missions is 21 won, 22 negative, 1 flat, and cannot be considered a new advantage of stability.

E27 Characterizes the eight candidates ' tabs as the percentile rank, and combines the total OOF labels with the fixed ZXQ0QZ L2 Logist Regulation, then evaluates the test. E27 test means AUROC for 0.8243, mean AUPRC for 0.8109, and world-10 for AUROC for 0.7535. The coefficient is clearly offset by the fact that E14 final 3-sed meant with MC-203-seed mean OOF task-rrelation for 0.9987, which indicates that the second-tier model is facing a duplicate information of a high-combined line.

** Decision-making at the time:** E26/E27 retained as negative result of the leakage-safe, E17 5-seed temporarily maintained the main result; no review of observed test was carried out by modifying the griddy threshold, L2C or the candidate cluster. This conclusion directly contributed to the subsequent use of OOF filters only for E29.

## 7. E29 Outcome, 5-seed pre-registration extension and E28 disposal

E29 Based on E26/E27 exposed homogenous problem design. Former E14a peptide encoder is `Embedding → Flatten → MLP`; E29 reserves global auxiliary, HLA plain and task-rank finance, replacing encoder with a light volume of a single volume of kernel size 2, 3, 5 and retaining post-volume location information. This design is more consistent with the evidence than the current 12-allele closed-set priority HLA pseudo-sequence cross-attention: E4/E4b has shown that pseudo-sequence CONDINATION does not exceed the powerful shared model, while 9-mer local motif still has untested encumberors.

E29 First round saw 202,60704 3-old plain-grouped OFF completed and unreaded test. The CNN single model OOF means AUROC was 0.807, higher than the matching E14 Seed 0.7915; task-rank corration was 0.8727. After the integration of rights like E14 3-seed meant rank, the main AUROC was 0.8097, up from 0.8042 to 0.0056; mean AUPRC was raised from 0.7855 to 0.7918; World-10 means AUROC was increased from 0.725 to 0.7314. The integration was 39 successes, 5 negatives, and all four pre-registration conditions were adopted in 44 missions.

E29 then entered the 3-seed phase and adopted E14 final 3-seed means making baseline and fasion baseline. 3-seed CNN OF mean AUROC for 0.8138, up from 0.8042 for E14 to 0.0096; relevance to 0.942. The E14/E29 rights, etc., are integrated into 0.8132, up from 0.090 for E14. The scripts are read after the four conditions have been adopted again and implemented full-train 3-seed training.

The results of the official test were as follows:

Mean AUROC Mean AUPPC Mean AccuracyMean MCC World-10 Mean AUROC
|---|---:|---:|---:|---:|---:|
E17 5-seed (previous best) 0.8263  0.8139  0.74992  0.4986  0.7573
E29 CNN single seen average performance 0.8212 ± 0.0008  0.0009  0.7423  0.00037  0.4846  00.007  0.7447  00.008
| E29 CNN 3-seed mean | 0.8341 | 0.8228 | 0.7542 | 0.5084 | 0.7634 |
E14-3 + E29-3 power RK integration 0.8340  0.8229

E29 The AUROC is 34 won, 10 negative, compared to E17 5-seed; the average AUROC is 0.0079, and the task bootstrap 95% is `[0.0041, 0.0116]`. The E14/E29 integration slightly improved AUPREC and World-10, but AUROC is 0.002 lower than E29 alone, and therefore does not replace the E29 Master Outcome and is no longer based on test-based integration weighting.

Before adding the ED, E29 5-second was pre-registered as the last stopdsplit confirmation extension. Added to reads is 20660707/20708, which is used only without re-rehearsing three seeds; member weights are fixed as equal rights, prohibiting the deletion of the ED, re-weighting or selection of a subset of 4-bed. The three OOF conditions for entering test are: mean AUROC gain at least 0.010, world-10 AUROC gain at least − 0.010, and mean AUPRC gain at no less than −0.0005; any condition is not stopped at OOF.

The pre-registered OOF threshold was fully adopted: 5-seed means AUROC at 0.8157 vs 0.8138 (+0.00019), mean AUPPC at 0.79995 vs 0.7971 (+0.00024), and the working-10 means AUROC at 0.7379 vs 0.7349 (+0.0030). The script therefore performed only one fixed 5-seed test assessment, with the following official results.

Mean AUROC Mean AUPPC Mean AccuracyMean MCC World-10 Mean AUROC
|---|---:|---:|---:|---:|---:|
| E29 3-seed | 0.8341 | 0.8228 | 0.7542 | 0.5084 | 0.7634 |
E29 Pre-registration 5-seed  0.837**  0.8259** **0.7588** ** 0.5175** **0.76770** **
| 5-seed − 3-seed | +0.0032 | +0.0031 | +0.0045 | +0.0091 | +0.0036 |

E29 5-seed Job-by-task AUROC 37 won, 7 negative, with an average increase of 0.0110, mission botstrap 95% ZXQ0QZ; AUPRC averaged an increase of 0.0121 (35 won, 9 negative). This means that the gain is not only from adding two random seeds: CNN indicates that the improvement still has a steady presence after 5-seed recognition of sexual expansion.

** Final decision:** E29 pre-registration 5-seed means that the freezing result is a sandard split. As a pre-registration commitment, the addition, modification or continuation of the split is no longer required. E28 National record Learning is not implemented and is not included in the present stage of the conclusion.

## 8. Audit and conclusions boundary for full-link reliability

### 8.1 Data definition, splitting and leakage boundaries

The data concentration pair contains one positive and one false example: the two have the same HLA and parent UniProt; the positive is reported in target target-HLA and the negative is reported in other organizations but not in target target-HLA. The design controls some of the protein/HLA mix and is suitable for studying relative organizational preferences, but label 0 is not an "no-submission" as confirmed by the experiment. Therefore, model output should be interpreted as `tissue-HLA-specific presentation preference` or evidence priority rather than absolute establishment.

Standard split randomly extracts from each task the complete pair, trade/test `pair_id` without overlap, and no repeats in the same task. Independent audit gets:

Check item  Results
|---|---:|
| Train rows / test rows | 96,972 / 8,800 |
| Tasks | 44 |
| Train/test `pair_id` overlap | 0 |
| Same-task peptide overlap | 0 |
| Test rows whose peptide appears in training in another task | 6,837 / 8,800(77.69%) |
| Test rows whose parent UniProt appears in training | 8,568 / 8,800(97.36%) |
| Test composition per task | 100 positive + 100 pseudo-negative |

The task is therefore a strong closed-set benchmark. Overlap across task entities is not a direct label leak, but it can result in significantly higher outcomes than the true peptide-disjoint or protein-disjoint generalization. Nor should the AUPREC, accuracy, F1 and MCC in the Balance Test Set be interpreted directly as a precision or clinical accuracy rate under the reality prevalence.

### 8.2 Review of indicators and outcome documents

The audit covered 244 results, and no empty, empty or unread documents were found. The official E295-seed candidate included 8,800 single projection keys, no duplication, no missing, and all 44 tasks. The saved sample-by-sample forecast was independently calculated:

```text
macro AUROC = 0.83730568
macro AUPRC = 0.82593372
```

The E17 5-seed and E29 5-seed are 37 win, 7 negative, with an average AUROC gain of 0.01102; the mission bootststrap 95% is `[0.00717, 0.01495]`. The same five Seed single models are all positive. The gains are 00959, 00690, 0.0815, 0.0726 and 0.0195, respectively. This supports "stable gains in the current benchmark, rather than individual solicited by chance".

E17 The order of integration across the seed/ cross-branch E29s is slightly different. Replace E14 with the same E29 "branch-rank foundation first, seed average" after AUROC is 0.82589, while formal E17 is 0.82629, the difference is only 0.00039, which does not explain the main gain of E29.

### 8.3 Key-phased upgrade evidence of matching

Compared with AUROC, which is seen with the visue-HLA task, the main upgrades are as follows:

Compare Mean AUROC gain  Mission victory/ negative  Job pair test p value
|---|---:|---:|---:|
| E8a − E2 | +0.01223 | 33 / 11 | 4.18×10⁻⁵ |
| E13 − matched E2 | +0.00779 | 35 / 9 | 2.63×10⁻⁶ |
| E14a − E8a | +0.00662 | 34 / 10 | 2.44×10⁻⁴ |
| E14a − E13 | +0.00928 | 34 / 10 | 4.41×10⁻⁴ |
| E15 rank − probability fusion | +0.00142 | 31 / 13 | 0.00126 |

E8, E13 is more consistent with the direction of the phased upgrading of E14. E15 benefits are rediscoverable but have a small impact, which should be expressed as "small integration improvements" rather than major model breakthroughs. Most of the results of E16, E18–E27 are more suitable as negative results or boundary experiments of methodological value.

### 8.4 Seen/unseen diagnostic

Test if peptide has a layering of other tasks in the training set:

Subset E29 AUROC E17 AUROCE29 − E17
|---|---:|---:|---:|
| Seen peptide | 0.85328 | 0.83966 | +0.01362 |
| Unseen peptide | 0.74199 | 0.73761 | +0.00438 |

E29 A certain sorting capacity is retained on the unseen-peptide subset, but absolute AUROC is significantly reduced and relative gains are reduced. Thus, "multiscale CNN may enhance local motif generalization" is a supported working assumption, but it is not a fact that has been externally confirmed by peptide-disjoint.

### 8.5 Pre-registration, test reuse and repossibility boundaries

E18, E26, E27 and E29 scripts choose to use valueation in the train or pair-grouped OOF, respectively, and E29 read the corresponding test only after the OOF condition was passed; these local processes are reasonable. But, in the overall view of the project, E0-E29 has long been observing the same standard test, so the test cannot be considered a complete independent collection of external confirmations that have never participated in the research decision-making. E29 5-sed is the freezing end of the split, and all observed test results cannot be used for secondary cross-referencing, adding members or selecting new structures.

The time order of the local files for E295-seed supports "pre-registration before OOF, then test": `preregistration.json` pre-creates OOF and test output before the new Seed. The code also specifies that the test stage is entered after the OOF date passes. However, the project directory lacks a verifiable version of history or external time stamp, so that it can only be called local pre-registration evidence and cannot be considered an irrevocable third-party pre-registration.

The code sets Python, Numpy, CPU Torch and CUDA seed with sufficient sample-by-sampling predictions, but does not uniformly enable the deterministic algorits/cudNdeterministic configuration, which is not guaranteed to be consistent in place in different CUDA/cudNN environments. The standard difference in the Esmble Stability Table is 0, which means only that it is a convergence point and does not mean that statistical uncertainty is zero.

## Model positioning and final conclusions

The main findings of the current recommended reports are:

```text
E29 5-seed Multi-kernel CNN E14a extenuating pre-registration confirmation
mean AUROC = 0.8373
mean AUPRC = 0.8259
worst-10 mean AUROC = 0.7670
```

The core finding of Phase 2 is not a more complex OStimizer or training track weight average, but rather: task-rank Fusion can reduce branch scale differences; independently trained seed projections averagely reduce the variance; model members ' independence is more important than the same checkpoint, snapshot or SWA on the same training track; changes in the E14a bi-branch structure can increase individual model strength and diversity of membership. In contrast, neither the dynamic support task weighting, HLA-private structure, nor the OOF remote action/steaking on highly homogeneous candidates yield the same level of benefits.

Based on the current evidence, the recommended tier is:

```text
E29 5-seed Multi-kernel CNN task-rank ensemble
> E29 3-seed Multi-kernel CNN task-rank ensemble
> E17 5-seed task-rank ensemble
> E26 OOF greedy 3-seed mean ≈ E17 3-seed task-rank ensemble ≈ E27 OOF stacking
> E20 final/final and E18 value-soleted rank user
> E15 task-rank fusion
> E16 MC Dropout-20
> E19 final checkpoint
> Single model exploration for checkpoint/snapshot ensemble, SWA, E21-E25
```

Note: The above tiers are first sorted according to the current standard standard formula, which means AUROC; E29 3-seed (0.8341) is higher than E17 5-seed (0.8263) and therefore the order is consistent with the values.

Summary: A multitask II classification framework is currently being constructed for the sharing of the Tissue with HLA dimensions and the current standard division of 44 Tissues-HLAtask has resulted in a broad and stable mission-by-task performance enhancement of the relatively strong baseline.

### Current evidence supports and is not supported by conclusions

** Current evidence supports:**

> At the current 44-task, closed-set, balanced tissepHC standard split, E29 Multi-kernel CNN E14a 5-seed task-ensk esmble is the best model in the evaluated model. Its upscaling relative to E17 is in the same direction as most tasks and all five of the same seeds.

** Current evidence not supported:**

- (a) The same performance for unseen HLA, strict peptide-disjoint, protein-disjoint or foreign ranks;
- Label 0 for biologically confirmed "no submission"
- AUPRC/precision for balancing benchmark is equivalent to a precision for reality;
- E29 The proceeds demonstrate a specific biological mechanism or a causal organization regulatory mechanism.

## Key codes and outcome documents

Contents
|---|---|
Phase 2 Road Map ZXQ0QZ
| E15 | `scripts/run_tissuepmhc_e15_fusion_ablation.py`;`results/tissuePMHC_e15_fusion_ablation/` |
| E16 | `scripts/run_tissuepmhc_e16_mc_dropout_ensemble.py`;`results/tissuePMHC_e16_mc_dropout_ensemble/` |
| E17 | `scripts/run_tissuepmhc_e17_seed_ensemble.py`;`results/tissuePMHC_e17_seed_ensemble/` |
| E18 | `scripts/run_tissuepmhc_e18_global_weight_selection.py`;`results/tissuePMHC_e18_global_weight_selection/` |
| E19 | `scripts/run_tissuepmhc_e19_training_ensemble.py`;`results/tissuePMHC_e19_training_ensemble/` |
| E20 | `scripts/run_tissuepmhc_e20_swa.py`;`results/tissuePMHC_e20_swa/` |
| E21 Gradient-Similarity Gating | `scripts/run_tissuepmhc_e21_gradient_similarity_auxiliary.py`;`results/tissuePMHC_e21_gradient_similarity_auxiliary/` |
| E22 Periodic Nash-MTL | `scripts/run_tissuepmhc_e22_periodic_nash_mtl.py`;`results/tissuePMHC_e22_periodic_nash_mtl/` |
| E23 ForkMerge | `scripts/run_tissuepmhc_e23_forkmerge.py`;`results/tissuePMHC_e23_forkmerge/` |
| E24 Auto-Lambda | `scripts/run_tissuepmhc_e24_auto_lambda.py`;`results/tissuePMHC_e24_auto_lambda/` |
| E25 HLA-Structured PLE | `scripts/run_tissuepmhc_e25_hla_structured_ple.py`;`results/tissuePMHC_e25_hla_structured_ple/` |
| E26 OOF Greedy Selection | `scripts/run_tissuepmhc_e26_all_in_one.py`, `scripts/run_tissuepmhc_e26_greedy_ensemble_selection.py`;`results/tissuePMHC_e26_greedy_ensemble_selection/` |
| E27 Stacked Generalization | `scripts/run_tissuepmhc_e27_stacked_generalization.py`;`results/tissuePMHC_e27_stacked_generalization/` |
E29 Multi-kernel CNN OOF Screen ZXQ0QZ; 1-side results ZXQ1QZ; 3-side default output `results/tissuePMHC_e29_multikernel_cnn_3seed/`
E29 5-seed pre-registered incremental extension  ZXQ0QZ; ZXQ1QZ; default output `results/tissuePMHC_e29_multikernel_cnn_5seed/`

## 11. Reference methodology

1. Dietterich, T. G. *Ensemble Methods in Machine Learning* (2000).
2. Gal, Y. & Ghahramani, Z. *Dropout as a Bayesian Approximation* (2016).
3. Lakshminarayanan, B. et al. *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles* (2017).
4. Chen, H. et al. *Training Neural Networks with Multi-branch Architectures* / checkpoint ensemble literature (2017).
5. Huang, G. et al. *Snapshot Ensembles* (2017).
6. Izmailov, P. et al. *Averaging Weights Leads to Wider Optima and Better Generalization* (2018).
7. Caruana, R. et al. *Ensemble Selection from Libraries of Models* (2004).
8. Wolpert, D. H. *Stacked Generalization* (1992).
9. Du, Y. et al. *Adapting Auxiliary Losses Using Gradient Similarity* (2018).
10. Navon, A. et al. *Multi-Task Learning as a Bargaining Game* (2022).
11. Jiang, J. et al. *ForkMerge: Mitigating Negative Transfer in Auxiliary-Task Learning* (2023).
12. Liu, S. et al. *Auto-Lambda: Disentangling Dynamic Task Relationships* (2022).
13. Tang, H. et al. *Progressive Layered Extraction* (2020).
14. Liu, Y. & Yao, X. *Ensemble Learning via Negative Correlation* (1999).
15. Kim, Y. *Conventional Neural Networks for Unity Transportation* (2014); E29 extracts ideas using local sequence mode of multi-kernel 1D CNN and indicates a fixed 9-mer reservation position.
