# TissuePMHC Phase 1 Report

## Summary

This report summarizes new studies completed by the ZXQ0QZ project after the traditional baseline experiment E0. The data set was constructed by HLA-I 9-mer peptide log with 44 taxonomic tasks. Each task is defined by a target organization and a HLA limit equivalent gene. The standard train/test split contains 96,972 training samples and 8,800 test samples, with 100 positive samples and 100 false load samples fixed for each task.

The "new study" in this report refers to E1-E14. The research route has been gradually expanded from a single and multitask task of the neural network to HLA bionomization, task weighting, task grouping, selective sharing, soft ensemble, multitasking optimization, pair ranking, tissue/HLA auxiliary preparation, and a combination of auxiary subvision and soft ensemble. The most powerful results are currently E14a auxiliary soft ensemble, i.e., a global sha branch with tissue/ HLA auxiliary subvision, twirious mod, mcC0747, ww10-0.049.

The overall conclusion is that under the current closed-set tessuquePMHC standard model, the best performer is not a single global structure, a single HLA grouping, or a more complex generic multitask optimiser, but auxiary-enhanced global sharing with HLA-special sharing soft essemble. The conclusion is that the exploratory model within benchmark is comparable and priced at different prices than the recognition of a completely new peptide, protein, HLA or external data generalization capability.

## 1. Background and definition of the mandate

The `tissuePMHC` project studies the organizational speciality preference of HLA-I peptide. The model mission is to assign a peptide, a tessue and a HLA allele, predicting whether the peptide is more likely to be presented under the Tissue-HLA conditions.

Each task is defined as:

$$
\text{task} = (\text{target tissue}, \text{HLA restriction})
$$

Each task is a two-class issue:

$$
y \in \{0,1\}
$$

ZXQ0QZ indicates that there are positive samples reported under target target text-HLA conditions; \(y=0\ indicates that there are identical HLA and parent UniProt, but not under target target item-HLA conditions. Label 0 is not an experimentally confirmed biology "non-presentation".

The current standard data is divided as follows:

Project  Value
|---|---:|
Number of missions
Training sample  96,972
♪ Test sample number ♪ 8,800 ♪
Each mission test sample  100 positive + 100 false
peptide length 9
HLA Allele count 12

It should be noted that the current standard split is a closed-set setup: testing centralized Tissue, HLA Allele and Tissue-HLA Task have all appeared in the training centre; 77.69% of test lines peptide and 97.36% of test lines parent UniProt have also appeared in other tasks of the training set. The results thus show mainly that the model has the same sorting and information-sharing capability within the benchmark, and that it does not directly prove that it is equally valid for all new peptide, new source protein, new HLA or external data.

## Reference baseline: E0

E0 is a traditional single task baseline. Each tessue-HLA task trains a traditional machine learning model, with input features including one-hot peptide encoding or BLOSUM62 peptide encoding, with models including bloglistic restatement, rranom forest, extratrees, etc.

The best model of average performance in E0 is one-hot blog response:

Mean AUROC  Mean AUPPC  Mean Accuracy  Mean MCC
|---|---:|---:|---:|---:|
| E0 one-hot logistic regression | 0.7558 | 0.7384 | 0.6909 | 0.3841 |

E1-E14, which follows this report, is a new research element.

## 3. E1-E3: Neuronet baseline and shared peptide

The new question for the first stage is whether the neuronet model and multitask sharing signage can go beyond the traditional E0 baseline.

Three directions were tested at this stage:

♪ Experiment, experiment, model thinking, ♪
|---|---|
E1  Each tissue-HLA task train the little neural network separately
E2  All task shares peptide encoder, each task has its own head platinum
E3 eeptide encoder plus tissue ID embedding and HLA ID embedding

E2 is the most important model at this stage.

$$
\hat{y}_{t}=h_t(f_\theta(p))
$$

ZXQ0QZ is a peptide, ZXQ1QZ) is a peptide encoder, ZXQ2QZ) is a task task for \(t\ task task task task-special disclosure head.

The stabilization results of the three seeds are as follows:

Mean AUROC Mean AUPPC Mean AccuracyMean MCC World-10 Mean AUROC
|---|---:|---:|---:|---:|---:|
| E2 shared peptide encoder + task heads | 0.7927 | 0.7777 | 0.7180 | 0.4404 | 0.7178 |
| E3 best conditioned tissue/HLA ID model | 0.7833 | 0.7685 | 0.7152 | 0.4350 | 0.6992 |

E2 means AUROC to be upgraded to:

$$
0.7927 - 0.7558 = 0.0369
$$

This means that the share of the paper is valid. E3 is also more than the traditional baseline, but not as much as E2, which explains that in the current task, the task-specific head is more effective than simply using the textue/HLA as a condition input.

## 4. E4-E4b: HLA bionometric analysis

E4 and E4b check if HLA pseudo-equality can improve models. The project builds 34-redue HLA

The experiment is set as follows:

♪ Experiment, experiment, model thinking, ♪
|---|---|
| E4 | tissue embedding + HLA pseudo-sequence encoder |
| E4b | tissue embedding + HLA ID embedding + HLA pseudo-sequence encoder |

The results are as follows:

Mean AUROC Mean AUPPC Mean AccuracyMean MCC World-10 Mean AUROC
|---|---:|---:|---:|---:|---:|
| E3 HLA ID embedding | 0.7833 | 0.7685 | 0.7152 | 0.4350 | 0.6992 |
| E4 HLA pseudo-sequence only | 0.7728 | 0.7606 | 0.7042 | 0.4125 | 0.6705 |
| E4b HLA ID + pseudo-sequence hybrid | 0.7824 | 0.7704 | 0.7133 | 0.4303 | 0.6952 |

E4, which is no more than E3, suggests that HLA pseudo-security cannot directly replace HLA ID embedding. The main reason is that the current list-set HLA setting, which tested the concentration of HLA allele, has already appeared in the training pool, so that HLA ID embedding can learn about each allele, and pseudo-sequire encoder needs 12 alleles to be compressed into shared sequence encoders, which may instead create information bottlenecks.

E4b is significantly more resilient than E4 and AUPRC is slightly higher than E3. This means that pseudo-equience has a certain value as an auxiliary biological information, but it cannot be the current main energy line.

## 5. E5-E7: Task weight, task grouping and hard section

E2 is strong, but there is a newcomtive transfer: some missions are falling after multiple tasks are shared. So the core of the next phase of the study is how parameters should be shared between missions.

This phase includes:

♪ Experiment, experiment, model thinking, ♪
|---|---|
E5  Add FAMO-style Agressive task Weighting 2
E6  Training by HLA or Tissue
E7  for each tab hard-select global branch or HLA branch

The main results are as follows:

Mean AUROC Mean AUPPC Mean AccuracyMean MCC World-10 Mean AUROC
|---|---:|---:|---:|---:|---:|
| E2 shared heads | 0.7927 | 0.7777 | 0.7180 | 0.4404 | 0.7178 |
| E5 FAMO | 0.7758 | 0.7571 | 0.7077 | 0.4193 | 0.6979 |
| E6 HLA-grouped | 0.7862 | 0.7725 | 0.7148 | 0.4316 | 0.7037 |
| E6 tissue-grouped | 0.7387 | 0.7227 | 0.6765 | 0.3548 | 0.6699 |
| E7 selective HLA/global | 0.7904 | 0.7754 | 0.7184 | 0.4405 | 0.7143 |

E5. The final task of the FAMO is near even distribution and the task-balanced training itself weakens the original E2 training distribution.

E6 shows that HLA grouping is of local value but not a substitute for global sharing. The loss of the tissue grouping suggests that HLA binting motif is forced to combine tasks that are very different.

E7 Using value-based hard supply restores a portion of E6 HLA grouping, but it still does not exceed E2. The reason is that hard supply retains only one branch per task, and if the choice of value is loud, it completely throws away another brach useful information.

## 6. E8:Global/HLA Soft Ensemble

E8. Change the hard section of E7 to softise. It retains E7 's two-stage process: cutting out the value from the train to decide on the integration strategy, retraining the final global brand and final HLA brach, test only for final evaluation.

The integration formula is:

$$
s_{\text{final}}=w_{\text{HLA}}s_{\text{HLA}}+(1-w_{\text{HLA}})s_{\text{global}}
$$

Of which ZXQ0QZ is HLAbnch output fraction, ZXQ1QZ is global output fraction.

E8 tested three strategies:

Policy  Definition
|---|---|
| E8a fixed average | \(w_{\text{HLA}}=0.5\) |
E8b valuation-delta clipped ZXQ0QZ, 0.15, 0.85) where ZXQ1QZ, val
E8c valuation softmax  do softmax generation weights for value AUROC

The results are as follows:

Mean AUROC Mean AUPPC Mean AccuracyMean MCC World-10 Mean AUROC
|---|---:|---:|---:|---:|---:|
| E2 shared heads | 0.7927 | 0.7777 | 0.7180 | 0.4404 | 0.7178 |
| E7 selective HLA/global | 0.7904 | 0.7754 | 0.7184 | 0.4405 | 0.7143 |
| E8a fixed average | 0.8050 | 0.7925 | 0.7313 | 0.4657 | 0.7295 |
| E8b validation-delta clipped | 0.8046 | 0.7916 | 0.7314 | 0.4660 | 0.7304 |
| E8c validation softmax | 0.8020 | 0.7890 | 0.7274 | 0.4583 | 0.7279 |

E8a is upgraded as follows from E2:

Indicators
|---|---:|
| Mean AUROC | +0.0122 |
| Mean AUPRC | +0.0149 |
| Mean accuracy | +0.0133 |
| Mean MCC | +0.0253 |

E8a upgrades 91 AUROCs out of 132 task-seed rows and works 10 means AUROC from 0.7178 to 0.7295. This means that E8 benefits are not higher averages for a few tasks, but improved below the overall and difficult tasks.

The most important observation is that the simplest fixed average E8a is the best. This suggests that there is complementary information between the global branch and HLA branch, and that there is still noise.

## 7. E9-E14: E8 expansive experiments

After confirming E8 as a strong structure, the project continued to test several more complex multitasking learning methods and monitoring signals, and eventually merged E13 's auxiliary subvision with E8 's soft E14.

♪ Experiment, experiment, model thinking, ♪
|---|---|
| E9 | E2 + CAGrad gradient conflict handling |
| E10 | MMoE selective-sharing model |
E10b Adjust number, width and date regularation of MMOE
| E11 | DB-MTL dynamic task loss balancing |
| E12 | paired ranking loss |
E13 + task/ HLA auxiliary preparation
| E14 | auxiliary global branch + HLA-specific branch soft ensemble |

The overall results are as follows:

Mean AUROC Mean AUPPC Mean AccuracyMean MCC World-10 Mean AUROC
|---|---:|---:|---:|---:|---:|
| E14a global auxiliary + HLA plain soft ensemble | 0.8116 | 0.7978 | 0.7372 | 0.4769 | 0.7349 |
| E14b global auxiliary + HLA auxiliary soft ensemble | 0.8093 | 0.7955 | 0.7348 | 0.4735 | 0.7372 |
| E8a fixed soft ensemble | 0.8050 | 0.7925 | 0.7313 | 0.4657 | 0.7295 |
| E13 auxiliary tissue/HLA | 0.8023 | 0.7856 | 0.7292 | 0.4640 | 0.7306 |
| E10 MMoE | 0.7948 | 0.7804 | 0.7210 | 0.4474 | 0.7195 |
| E2 sample BCE | 0.7945 | 0.7793 | 0.7197 | 0.4441 | 0.7211 |
| E10b 4 experts, width 256 | 0.7928 | 0.7788 | 0.7170 | 0.4393 | 0.7104 |
| E10b 6 experts, width 128 | 0.7913 | 0.7768 | 0.7178 | 0.4423 | 0.7122 |
| E11 DB-MTL | 0.7817 | 0.7643 | 0.7144 | 0.4313 | 0.6956 |
| E9 CAGrad | 0.7810 | 0.7649 | 0.7106 | 0.4239 | 0.6979 |
| E12 pair ranking | 0.7807 | 0.7618 | 0.7102 | 0.4227 | 0.7007 |

E9, E11 and E12 are interpretive results. They suggest that the main bottlenecks are unlikely to be solved only by the ranking of gradient conflict management, dynamic loss bailing or pair.

E10 MMOE is slightly stronger than the sample-level E2 baseline, which suggests that automatic automatic share has some effect. It is still below the simpler E8 soft Eemble and the larger expert or wider expert of E10b does not further narrow the gap.

E13 is the most valuable single model experiment since E8. The APURC upgrade + 0.0078, mean AUPRC upgrade + 0.0063, mean awareness upgrade + 0.0095, mean MCC upgrade + 0.0200, world-10 means AUROC upgrade + 0.0005. The Auxiliary Job Diagnostic shows that the benefits of the E13 are mainly derived from HLA-aware recovery.

E14 further demonstrates that the benefits of E8 and E13 can be added up. E14a uses auxiliary global brandy to achieve a combination-overage soft economy, and 0.7116 compared to E8a, E14a average AUROC uplifted +0.02116 using the average AUROC upgraded from 132 seed-task points, with 85 ups and 47 downs; E13, E14a average AUROC ups and 00928, 97 ups and 35 downs from 132 points. E14b also increases the AUROC up to 0.000lical subvision plus 0.00662, with HLA-specific branch, and a hen AUROC more than 0.8093, below E14a, but higher than the AUROC 10, which explains that it is more efficient than the average of AUROC.

## 8. Algorithm selection versus method positioning

The algorithm is not just for the newest methods, but for the task structure of the algorithm to match `tissuePMHC`. Specific criteria include: the algorithm must be suitable for 44 tissue-HLA II tasks; it can use shared information between tasks; it cannot be forced to mix all tasks into one task; it is better to ease the nigative transition; it can be compared fairly with the current baseline; and the difficulty of being able to move forward as a scientific research project.

From this standard, early candidate directions include: share paper application encoded + task-special headers, agreed model, HLA pseuddo-sequire conversion, FAMO, CAGrad, task grouping, MMoE/PLE, directing lost, auxiliary tasks and DB-MTL. The results of E1-E14 show that these directions are not of the same value.

The methodology is as follows:

The current judgment suggests the use of the proposed purpose
|---|---|---|
Auxiliary global branch + HLA-special brand soft Esmble  as the current main line model
♪ The ♪ ♪ The ♪ ♪ The ♪ ♪ The ♪ ♪ The ♪ ♪ The ♪ ♪ The ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the the the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪
♪ The ♪ ♪ The ♪ ♪ The ♪ ♪ The ♪ ♪ The ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the ♪ the
HLA grouping tile with local value as source of HLA-specialbranch
auxiliary tissue/ HLA preposition  add value to the enhanced use of the shared re-representation
conditioned test/ HLA ID model  reasonable but not superior  as a control experiment
HLA pseudo-security conditioning  with biointerpretation value but without sub-optimal  for unseen-HLA generalization or interpretation analysis
FAMO, DB-MTL  Unenhanced Main Line Performance as a newative result
CAGrad  does not enhance main line performance  as a major condition orientation Negative result
MMoE/ PLE  Autoselective shareing is valuable but less stable than the visible structure  as an option, not as the current main line
paired ranking loss  matches the data construction logic but not up on the indicator  as a newative result reserve
Protein laguage model embeddings  with late potential and high cost of engineering  in the phase of generalization capability upgrading

The algorithm choice of this report therefore concludes that the main line that is most worthy of retention under the current sandard split is not a more complex generic multitask optimizer, but a soft ensable for `auxiliary-enhanced global sharing + HLA-specific sharing`. The failure results of FAMO, DB-MTL, CAGrad and paired shifting loss are also valuable because they suggest that the current bottlenecks are more like task-sharing architecture design than simply loss weighting or gradient control.

## 9. Overall ranking and interpretation

The current phase is sorted as:

```text
E14a auxiliary global + HLA plain soft ensemble
>
E14b auxiliary global + auxiliary HLA soft ensemble
>
E8a fixed soft ensemble
≈ E8b validation-delta clipped ensemble
>
E13 auxiliary tissue/HLA prediction
>
E8c validation softmax ensemble
>
E10 MMoE
≈ E2 sample BCE
>
E10b tuned MMoE
>
E11 DB-MTL
≈ E9 CAGrad
≈ E12 pair ranking
>
E4/E5/E6 and other earlier branches
```

The core findings can be summarized as follows:

```text
The blogger writes about the current situation of the tissue-set tissuePHC standard split, which is a very important tool for the development of the new society.
The strongest structures are the auxiliary-enhanced global supply and HLA-special supply software.
```

From a model perspective, the global branch learns more than the universal peptide paper; HLA branch learns from the same HLA allele HLA-special peptide motif. E14a further allows the global branch learns stronger structural expressions through the tissue/ HLA auxiliary submission, while HLA Branch maintains plain standard headers. Soft ensemble retains both types of information, thus superior to a single global share, hard supply and unintegrated auxiliary model.

## 10. Reliability of audit and evidence of milestones

The calculation of the indicators and the key stages of the results are reviewed independently.

Compare Mean AUROC gain  Mission victory/ negative  Job pair test p value
|---|---:|---:|---:|
| E8a − E2 | +0.01223 | 33 / 11 | 4.18×10⁻⁵ |
| E13 − matched E2 | +0.00779 | 35 / 9 | 2.63×10⁻⁶ |
| E14a − E8a | +0.00662 | 34 / 10 | 2.44×10⁻⁴ |
| E14a − E13 | +0.00928 | 34 / 10 | 4.41×10⁻⁴ |

These results show that E8, E13 and E14 upgrades are not driven by individual seeds or a few tasks alone, and that the upward orientation within benchmark is consistent. These p values measure only the difference in task pairs on fixed test, and do not eliminate the deviation in project-level selection: the direction of E1-E14 research and model sequencing have repeatedly observed the same Standard test. The E14a values are therefore real and recouptable, but the test is no longer a completely untouched external confirmation set.

E8 valuation policy, complete train retrain, and final test evaluation process is within a single experiment; this should not be interpreted as expanding to mean that the entire E1-E14 research program does not use test feedback. All rankings for Phase 1 should be considered as explarian benchmark evidence, confirming that frozen outer split or stand-alone data must be used.

## Limitations

The results need to be understood within the benchmark boundary:

1. The current split is closed-set for the tissue, HLA and task.
2. 77.69% of the peptide in the test line was present in other tabs in the training set.
3. 97.36% of the test lines were used to show up in training centres.
4. E14 The current proven advantage under sstandsplit has not been proven to be the best in terms of peptide-disjoint, protein-disjoint, unseen-HLA or external data.
5. Label 0 is a false negative based on a "not reported" construction of the target organization, not an experimental confirmation of non-presentation; the model is closer to predicting organizational preferences or evidence priorities.
6. Each test tasker maintains a 1:1 balance, so AUPRC, accuracy, F1 and MCC cannot be interpreted directly as actual deployment performance under prevalence.
7. Standard test was repeated in E1-E14 studies to compare results and there was a deviation in the selection of project-level models.

These limitations do not negate the present conclusions, but limit their scope of application.

## 12. Follow-up recommendations

The benefits of continuing the complex model of the non-differentiated pile are low under the current standard split. The next step should be to conduct a reliability and border-wide analysis around E14/E8:

1. Freezes the current standard test, and no longer selects models, integration weights or research directions.
2. Creates a new outer contact set that you never view and builds a peptide-disjoint, parent-protein-disjoint, unseen-HLA and study/assay-disjoint plain.
3. Do nigative control, like HLAbrabranch score after disruption.
4. Assess the complete model selection process using the developed cross-value system, not just the individual final model.
5. Report uncertainty using the task/tassue/protein/pair level
6. The label is explicitly called pseudo-negative and is subject to a sensitivity-analysis.
7. Analyses the relevance of the global score to HLA score and centrally saves sample-by-sample predictions, environment and data hash.

## Conclusions

The new E0 study has significantly increased the current prediction indicators on the TissuePHC standard split. E2 supports multi-task shared supplemental information on the plaza task assessed; E6 and E7 explain that HLA-specific share has a local value but hard stability; E8 supports global Branch and HLA Branch with complementary information on the entourage of the entourage. E9-E13 indicates that simply changing the optimist or lOSS bailing is less effective than the auxiliary structure of the graphic design; E13 supports the tissue/HLA auxiliary assessment of the most valuable model of learning. E14a will add the complementary software version of E8 to the kernel ed edible edible version of the enlist1 1

Thus, at the present 44-task, closed-set, balanced sample, stand-up standard model for Phase 1 is E14a auxiliary global + HLA plain fixed-average soft enemy. This conclusion should not be extrapolated to an identified advantage on the unseen peptide, unseen protein, unseen HLA or foreign forces.

## 14. References

1. Caruana, R. Multitask Learning  
   https://link.springer.com/article/10.1023/A:1007379606734

2. Ruder, S. An Overview of Multi-Task Learning in Deep Neural Networks  
   https://arxiv.org/abs/1706.05098

3. NetMHCpan: pan-specific MHC class I binding prediction and HLA pseudo-sequence representation  
   https://doi.org/10.1371/journal.pone.0000796

4. NetMHCpan-4.0: improved peptide-MHC class I interaction prediction  
   https://doi.org/10.1093/nar/gkx276

5. FAMO: Fast Adaptive Multitask Optimization  
   https://arxiv.org/abs/2306.03792

6. DB-MTL: Dual-Balancing for Multi-Task Learning  
   https://arxiv.org/abs/2308.12029

7. CAGrad: Conflict-Averse Gradient Descent for Multi-task Learning  
   https://arxiv.org/abs/2110.14048

8. MMoE: Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts  
   https://dl.acm.org/doi/10.1145/3219819.3220007

9. PLE: Progressive Layered Extraction for Multi-Task Learning  
   https://dl.acm.org/doi/10.1145/3383313.3412236

10. RankNet: Learning to Rank using Gradient Descent  
    https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/

11. Dietterich, T. G. Ensemble Methods in Machine Learning  
    https://link.springer.com/chapter/10.1007/3-540-45014-9_1

12. BLOSUM62: Amino acid substitution matrices from protein blocks  
    https://doi.org/10.1073/pnas.89.22.10915

13. ESM: Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences  
    https://doi.org/10.1073/pnas.2016239118

14. ProtTrans: Towards Cracking the Language of Life's Code Through Self-Supervised Deep Learning and High Performance Computing  
    https://doi.org/10.1109/TPAMI.2021.3095381

15. Continued domain-specific pre-training of protein language models for pMHC-I binding prediction  
    https://arxiv.org/abs/2507.13077

16. Gradient-Based Multi-Objective Deep Learning: Algorithms, Theories, Applications, and Beyond  
    https://arxiv.org/abs/2501.10945
