# Issue 5 Experiment Report: Generic pMHC comparison

Update: 2026-07-25
Status: MHCflurry 2.2.1, NetMHCpan 4.1b, coverage audit, task-to-task statistics and incremental stacking analysis completed.

## 1. Experimental objectives and conclusions

Issue 5 Tests whether the ability of TissePMHC to predict is derived mainly from the general peptide-MHC
The external control of both frozen controls is not read.
The tissue, nor does it use the tabs of this post to fine-tune:

1. (a) MHzFlurry 2.2.1 predominantly `presentation_score`;
2. NetMHCpan 4.1b, predominantly `EL_Rank`, ZXQ1QZ as a binding-only supplemental result.

The final conclusions are as follows:

1. Both external models have medium predictive capability, task-macro AUROC about 0.65 - 0.69, which describes general pMHC
   The combination or presentation tendency does explain some of the benchmark signals.
2. Human on MHCflury reservation is roughly the same as NetMHCpan EL; the former AUROC is slightly higher and the latter is the same.
   PairAcc is slightly higher, with very little difference.
3. Mouse on NetMHCpan EL is significantly better than MHCflury preparation and is a more suitable mouse common pMHC
   Contrast.
4. NetMHCpan EL is generally better than BA, especially human, so the text should read EL as the main result of NetMHCpan, and the text should read as follows:
   BA Incendiary or sensitivity analysis.
5. Both generic models are significantly lower than the complete TissePMHC; after adding the generic score to TissePMHC, the U.S.A.
   AUROC increases up to about 0.0001 and indicates that the complete model has absorbed the vast majority of the general pMHC signals available.

Therefore, Issue 5 supports the following restrictive conclusion:

> General peptide–MHC binding/presentation propensity explains part of the benchmark signal but does not fully account for the performance of the tissue-conditioned model.

## 2. Data, models and evaluation protocols

The only number of peptide-MHC queries is as follows:

species
|---|---:|---:|
| Human | 79,759 | 35 |
| Mouse | 6,663 | 4 |

All fractions are converted to the direction "The higher the point, the stronger " .
NetMHCpan runs EL and BA at the same time. All only queries are rated successfully:

Mouse Grow coverage
|---|---:|---:|
| MHCflurry presentation | 100% | 100% |
| NetMHCpan EL rank | 100% | 100% |
| NetMHCpan BA rank | 100% | 100% |

The evaluation covered the following agreements:

- standard fixed test;
- matched standard OOF;
- connected-component peptide-disjoint OOF.

Calculate AUROC, AUPRC and PairAcc for each Tissue-MHC task, and then take a task-macro aggregation. The external model is in
Matched standard OOF used the same complete train row point on peptide-disjoint OOF, so it freezes points
The sandalone indicator is the same; the difference between the two agreements is reflected in a matching comparison with the respective TissePHC OOF projections.

## 3. Key results of the two common models

### 3.1 Standard fixed test

Mean AUROC  Mean AUPRC  Mean PairAcc
|---|---|---:|---:|---:|
| Human | MHCflurry presentation | **0.68511** | **0.67830** | 0.68860 |
| Human | NetMHCpan EL | 0.68332 | 0.66863 | **0.68968** |
| Mouse | MHCflurry presentation | 0.65643 | 0.64357 | 0.67333 |
| Mouse | NetMHCpan EL | **0.68934** | **0.69263** | **0.69167** |

The two are very close. The AMOC of the MSCFlurry reservation is 0.00179 higher than NetMHCpan EL,
However, the PairAcc of NetMHCpan EL is high 0.00108, and this difference in magnitude should not be interpreted as a stable model advantage.

The advantages of NetMHCpan EL on Mouse are clearer: relative to MHz reservation, AUROC, AUPREC and
PairAcc increases 0.03291, 0.04906 and 0.01833 respectively. Therefore, the text should be used as a priority.
NetMHCpan EL.

### 3.2 Train-pool OOF

Mean AUROC  Mean AUPRC  Mean PairAcc
|---|---|---:|---:|---:|
| Human | MHCflurry presentation | **0.68709** | **0.67511** | **0.68975** |
| Human | NetMHCpan EL | 0.68228 | 0.66324 | 0.68751 |
| Mouse | MHCflurry presentation | 0.64735 | 0.62625 | 0.65649 |
| Mouse | NetMHCpan EL | **0.68019** | **0.67845** | **0.68877** |

OOF results are in line with the order of the match test: humans two models are close and MHz is slightly higher, and the use is
NetMHCpan EL is clearly stronger.

## NetMHCpan EL and BA

species and protocols
|---|---:|---:|---:|
| Human fixed test | 0.65773 | 0.68332 | +0.02559 |
| Human train-pool OOF | 0.65585 | 0.68228 | +0.02643 |
| Mouse fixed test | 0.68474 | 0.68934 | +0.00460 |
| Mouse train-pool OOF | 0.67780 | 0.68019 | +0.00239 |

El is no less than BA in either species, with a human advantage of about 0.026 AUROC. This is closer to the mission.
The settings are consistent, both for the establishment/ranking and not for the purely binting effect. NetMHCpan EL should be the main external baseline, and the main external baseline should be the main external baseline.
BA is used to indicate the performance threshold for the binting-only signal that can be explained.

## 5. Comparison with complete TissePMMHC

The following comparison is made using exactly the same rating.
And change the master model to evaluate the sample.

species protocol  most common model  common model AUROC  TissePHC AUROC master model increment
|---|---|---|---:|---:|---:|
| Human | Fixed test | MHCflurry presentation | 0.68511 | 0.84478 | +0.15967 |
| Human | Standard OOF | MHCflurry presentation | 0.68709 | 0.82795 | +0.14086 |
| Human | Peptide-disjoint OOF | MHCflurry presentation | 0.68709 | 0.76520 | +0.07811 |
| Mouse | Fixed test | NetMHCpan EL | 0.68934 | 0.85622 | +0.16688 |
| Mouse | Standard OOF | NetMHCpan EL | 0.68019 | 0.83922 | +0.15903 |
| Mouse | Peptide-disjoint OOF | NetMHCpan EL | 0.68019 | 0.75293 | +0.07274 |

In the combination test and standard OOF, the more robust models are about 0.14 - 0.17 AUROC.
The gap in the peptide-disjoint OOF is reduced to about 0.07 - 0.08, but the macro average of the complete model is still higher. This means that the entity is strict.
Separation weakens the advantage of the Tissue-conventioned model, but does not transform it into a simple general pMHC predictor.

The main model advantage of the Human protocol and the mouse match/standard is passed in task-to-task statistics.
BH-FDR.Mouse peptide-disjoint OOF needs to be interpreted with caution:

- MHCflurry preparation, main model means AUROC 0.10558, Wilcoxon
  `q = 0.000646`;
- NetMHCpan EL, main model means AUROC 0.07274, task bootstrap 95% interval
  ZXQ0QZ, but Wilcoxon ZXQ1QZ, of 24 tabs, 11 won, 13 negative.

So it cannot be claimed that the complete model is consistently superior to NetMHCpan EL on most of the base character. A more accurate explanation is:
The macro average difference is positive, but the task is more heterogeneous and the rank-based pair is not well supported.

## 6. Increments to control generic scores

Cross-mixing logistic staff uses "outer" and "outer + TissuePMHC scores" respectively.
The main AUROC is as follows:

Extranal only external + TissuePMTHC
|---|---|---|---:|---:|
| Human | Fixed test | MHCflurry presentation | 0.68511 | 0.84526 |
| Human | Fixed test | NetMHCpan EL | 0.68332 | 0.84469 |
| Human | Peptide-disjoint OOF | MHCflurry presentation | 0.68673 | 0.76929 |
| Human | Peptide-disjoint OOF | NetMHCpan EL | 0.67404 | 0.76514 |
| Mouse | Fixed test | MHCflurry presentation | 0.65643 | 0.85648 |
| Mouse | Fixed test | NetMHCpan EL | 0.68934 | 0.85639 |
| Mouse | Peptide-disjoint OOF | MHCflurry presentation | 0.64643 | 0.75642 |
| Mouse | Peptide-disjoint OOF | NetMHCpan EL | 0.64195 | 0.75339 |

The largest AUROC increment after adding the outer part is found in humans compared to the TissePMHC results alone
The remaining major increment is the MHCflurry stack: ZXQ0QZ, or +000409.
Even smaller, some combinations are almost unchanged. This indicates that external predicor and complete models are not completely redundant, but their independent contribution is limited;
The slight changes in the stack cannot be interpreted as new model improvements.

## 7. Presentation of the paper.

The text of the recommendation also reports:

- MHCflurry 2.2.1 presentation score;
- NetMHCpan 4.1b EL rank;
- NetMHCpan BA rank supplementing-only results;
- 100% row/pair/task coverage;
- The task-by-task matching of TissePMHC;
- Cross-cutting to be added after the number of controls.

It is recommended that the text be avoided as "the generic predicor cannot predict the mission." The actual results show that they have a stable medium performance.

> Both frozen general pMHC predictors achieved moderate discrimination, indicating that generic binding or presentation propensity contributes to the benchmark. However, neither approached the tissue-conditioned model, and adding the external scores to TissuePMHC produced only marginal further gains.

## 8. Limitations

1. External presence predicor may have used IEDB or related Immunopeptidomics training data.
   Peptide-disjoint split is created only in relation to in-house training sets and does not guarantee that no pre-training data are available against external training
   overlap.
2. MHCflurry and NetMHCpan should be described as frozen general-signral controls, not completely leak-free.
   Fair model competition.
3. Task-bootstrap belongs to nominal task-level inference.
   The parent protein and peptide component cannot be considered separate foreign ranks.
4. The differences between the two external models in Human are minimal; unless the predefined direct matching tests between the models are further implemented, they should not be claimed
   MHCflurry is significantly superior to NetMHCpan.

## Outcome document

- External model summary: `results/issue5_general_pmhc/external_evaluation/summary_metrics.csv`
- Audit of coverage: `results/issue5_general_pmhc/external_evaluation/coverage_audit.csv`
- Mission-by-mission result: `results/issue5_general_pmhc/external_evaluation/per_task_metrics.csv`
- Main model pairing statistics: `results/issue5_general_pmhc/external_evaluation/paired_statistics.csv`
- incremental stacking: `results/issue5_general_pmhc/stack_increment/summary_metrics.csv`
- Stacking pair statistics: `results/issue5_general_pmhc/stack_increment/paired_statistics.csv`
- MHCflurry score cache:`results/issue5_general_pmhc/score_cache/{human,mouse}_mhcflurry.csv.gz`
- NetMHCpan score cache:`results/issue5_general_pmhc/score_cache/{human,mouse}_netmhcpan.csv.gz`
- Version with allele snapshot: ZXQ0XZ

