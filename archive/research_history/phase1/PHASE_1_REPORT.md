# tissuePMHC Phase 1 Report

## Abstract

This report summarizes the new research conducted after the initial traditional baseline experiment (E0) for the `tissuePMHC` project. The dataset contains 44 tissue-HLA binary classification tasks built from human HLA-I 9-mer peptide ligand records. Each task is defined by a pair of target tissue and HLA restriction. The standard train/test split contains 96,972 training rows and 8,800 test rows, with each task having 100 positive and 100 pseudo-negative test samples.

The new experiments start from neural single-task and multi-task baselines (E1-E3), then explore biological HLA representation (E4/E4b), task weighting (E5), task grouping and selective sharing (E6-E8), additional multi-task extensions (E9-E13), and finally the combination of auxiliary supervision with soft ensembling (E14). The strongest result is E14a, a fixed-average soft ensemble of an auxiliary global branch and a plain HLA-specific branch. It achieves mean AUROC 0.8116, mean AUPRC 0.7978, mean accuracy 0.7372, mean MCC 0.4769, and worst-10-task mean AUROC 0.7349.

Overall, the main conclusion is that, under the current closed-set tissuePMHC standard split, the strongest evaluated structure is a soft ensemble that combines auxiliary-enhanced global sharing and HLA-specific sharing. This is an exploratory within-benchmark conclusion, not confirmation of generalization to unseen peptides, proteins, HLA alleles, or external cohorts.

## 1. Background and Task Definition

The `tissuePMHC` project studies tissue-specific presentation preferences of human HLA-I peptides. The prediction target is whether a peptide is likely to be presented in a given tissue under a given HLA allele.

Each task is defined as:

$$
\text{task} = (\text{target tissue}, \text{HLA restriction})
$$

For each task, the model predicts a binary label:

$$
y \in \{0, 1\}
$$

where \(y=1\) denotes a peptide reported in the target tissue-HLA context, and \(y=0\) denotes a paired pseudo-negative with the same HLA and parent UniProt that was reported in another tissue but not reported in the target context. Label 0 is not experimentally confirmed biological non-presentation.

The current dataset uses a standard split:

| Item | Value |
|---|---:|
| Number of tasks | 44 |
| Training rows | 96,972 |
| Test rows | 8,800 |
| Test samples per task | 100 positive + 100 pseudo-negative |
| Peptide length | 9 |
| HLA alleles | 12 |

The split is closed-set: test tissues, HLA alleles, and tissue-HLA tasks all appear in training. In addition, 77.69% of test rows contain a peptide observed in another training task and 97.36% contain a parent UniProt observed in training. The current results therefore measure within-benchmark ranking and information sharing rather than confirmed generalization to unseen peptides, proteins, HLA alleles, or external cohorts.

## 2. Reference Baseline: E0

E0 refers to the traditional single-task baseline experiments. These models were trained independently for each task using one-hot or BLOSUM62 peptide features and classical classifiers.

The strongest E0 average baseline was one-hot logistic regression:

| Model | Mean AUROC | Mean AUPRC | Mean Accuracy | Mean MCC |
|---|---:|---:|---:|---:|
| E0 one-hot logistic regression | 0.7558 | 0.7384 | 0.6909 | 0.3841 |

All following experiments, E1-E14, are treated as new research in this report.

## 3. E1-E3: Neural Baselines and Shared Peptide Representation

The first new research question was whether neural models and multi-task representation learning can improve over the traditional E0 baseline.

The tested models were:

| Experiment | Model idea |
|---|---|
| E1 | Independent neural model per tissue-HLA task |
| E2 | Shared peptide encoder with task-specific heads |
| E3 | Peptide encoder conditioned on tissue ID and HLA ID embeddings |

The key model, E2, uses:

$$
\hat{y}_{t} = h_t(f_\theta(p))
$$

where \(p\) is the peptide, \(f_\theta\) is the shared peptide encoder, and \(h_t\) is the task-specific classification head for task \(t\).

The stable 3-seed results were:

| Model | Mean AUROC | Mean AUPRC | Mean Accuracy | Mean MCC | Worst-10 Mean AUROC |
|---|---:|---:|---:|---:|---:|
| E2 shared peptide encoder + task heads | 0.7927 | 0.7777 | 0.7180 | 0.4404 | 0.7178 |
| E3 best conditioned tissue/HLA ID model | 0.7833 | 0.7685 | 0.7152 | 0.4350 | 0.6992 |

E2 improved mean AUROC by 0.0369 over the strongest E0 baseline. This established shared peptide representation learning as the main neural baseline. E3 showed that explicit tissue/HLA ID conditioning was useful, but less effective than task-specific heads on top of a shared peptide encoder.

## 4. E4-E4b: HLA Biological Representation

E4 and E4b tested whether HLA pseudo-sequences could improve the conditioned model. A 34-residue HLA class I pseudo-sequence was built for all 12 HLA alleles in the dataset.

The models were:

| Experiment | Model idea |
|---|---|
| E4 | Tissue embedding + HLA pseudo-sequence encoder |
| E4b | Tissue embedding + HLA ID embedding + HLA pseudo-sequence encoder |

Results:

| Model | Mean AUROC | Mean AUPRC | Mean Accuracy | Mean MCC | Worst-10 Mean AUROC |
|---|---:|---:|---:|---:|---:|
| E3 HLA ID embedding | 0.7833 | 0.7685 | 0.7152 | 0.4350 | 0.6992 |
| E4 HLA pseudo-sequence only | 0.7728 | 0.7606 | 0.7042 | 0.4125 | 0.6705 |
| E4b HLA ID + pseudo-sequence hybrid | 0.7824 | 0.7704 | 0.7133 | 0.4303 | 0.6952 |

The pseudo-sequence-only model did not outperform HLA ID embedding. The likely reason is the closed-set HLA setting: every HLA allele in test also appears in train, so an HLA ID embedding can directly learn allele-specific behavior. The hybrid E4b recovered most of the E4 loss and slightly improved AUPRC over E3, suggesting that pseudo-sequences carry some biological signal, but not enough to define the main performance path.

## 5. E5-E7: Task Weighting, Grouping, and Hard Selection

After E2 showed strong multi-task performance but also some negative transfer, the next experiments asked whether task weighting or task sharing structure could reduce this problem.

E5 tested FAMO-style adaptive task weighting on top of E2. E6 tested task grouping by HLA or tissue. E7 tested validation-based hard selection between the global branch and the HLA-grouped branch.

| Model | Mean AUROC | Mean AUPRC | Mean Accuracy | Mean MCC | Worst-10 Mean AUROC |
|---|---:|---:|---:|---:|---:|
| E2 shared heads | 0.7927 | 0.7777 | 0.7180 | 0.4404 | 0.7178 |
| E5 FAMO | 0.7758 | 0.7571 | 0.7077 | 0.4193 | 0.6979 |
| E6 HLA-grouped | 0.7862 | 0.7725 | 0.7148 | 0.4316 | 0.7037 |
| E6 tissue-grouped | 0.7387 | 0.7227 | 0.6765 | 0.3548 | 0.6699 |
| E7 selective HLA/global | 0.7904 | 0.7754 | 0.7184 | 0.4405 | 0.7143 |

E5 did not improve E2. Its task weights stayed close to uniform, and the task-balanced sampling used for FAMO weakened the original E2 training distribution.

E6 showed that HLA grouping has local value, but does not replace global sharing. Tissue grouping was clearly unsuitable, likely because it forces together tasks with different HLA binding motifs.

E7 recovered part of the HLA-grouped loss by choosing between global and HLA branches for each task, but hard selection remained unstable. A task could only keep one branch, so validation noise could cause the model to discard useful complementary information.

## 6. E8: Global/HLA Soft Ensemble

E8 replaced hard selection with soft score fusion. It uses the same leakage-safe two-stage design as E7: validation is used to define the ensemble strategy, and final test evaluation uses branches retrained on the full training set.

The ensemble score is:

$$
s_{\text{final}} = w_{\text{HLA}} s_{\text{HLA}} + (1-w_{\text{HLA}})s_{\text{global}}
$$

Three weighting strategies were tested:

| Strategy | Definition |
|---|---|
| E8a fixed average | \(w_{\text{HLA}} = 0.5\) |
| E8b validation-delta clipped | \(w_{\text{HLA}}=\operatorname{clip}(0.5+5\Delta,0.15,0.85)\), where \(\Delta=\text{AUROC}_{\text{HLA,val}}-\text{AUROC}_{\text{global,val}}\) |
| E8c validation softmax | Softmax over validation AUROC values |

Results:

| Model | Mean AUROC | Mean AUPRC | Mean Accuracy | Mean MCC | Worst-10 Mean AUROC |
|---|---:|---:|---:|---:|---:|
| E2 shared heads | 0.7927 | 0.7777 | 0.7180 | 0.4404 | 0.7178 |
| E7 selective HLA/global | 0.7904 | 0.7754 | 0.7184 | 0.4405 | 0.7143 |
| E8a fixed average | 0.8050 | 0.7925 | 0.7313 | 0.4657 | 0.7295 |
| E8b validation-delta clipped | 0.8046 | 0.7916 | 0.7314 | 0.4660 | 0.7304 |
| E8c validation softmax | 0.8020 | 0.7890 | 0.7274 | 0.4583 | 0.7279 |

E8a improved over E2 by:

| Metric | Improvement |
|---|---:|
| Mean AUROC | +0.0122 |
| Mean AUPRC | +0.0149 |
| Mean accuracy | +0.0133 |
| Mean MCC | +0.0253 |

E8a improved AUROC in 91 of 132 task-seed rows. It also increased worst-10 mean AUROC from 0.7178 to 0.7295, showing that the gain was not limited to a few easy tasks.

The most important observation is that the simplest fixed-average ensemble performed best. This suggests that global and HLA-specific branches contain complementary signal, while validation-based per-task weighting remains noisy.

## 7. E9-E14: Completed Extensions After E8

After E8 became a strong model, additional extensions tested whether more complex multi-task optimization or representation supervision could further improve the result. E14 then combined the useful E13 auxiliary signal with the E8 global/HLA soft-ensemble structure.

| Experiment | Model idea |
|---|---|
| E9 | E2 with CAGrad gradient conflict handling |
| E10 | MMoE selective-sharing model |
| E10b | Tuned MMoE variants |
| E11 | DB-MTL dynamic task loss balancing |
| E12 | Paired ranking loss |
| E13 | Main task plus tissue/HLA auxiliary prediction |
| E14 | Auxiliary global branch plus HLA-specific branch soft ensemble |

Main results:

| Model | Mean AUROC | Mean AUPRC | Mean Accuracy | Mean MCC | Worst-10 Mean AUROC |
|---|---:|---:|---:|---:|---:|
| E14a auxiliary global + plain HLA soft ensemble | 0.8116 | 0.7978 | 0.7372 | 0.4769 | 0.7349 |
| E14b auxiliary global + auxiliary HLA soft ensemble | 0.8093 | 0.7955 | 0.7348 | 0.4735 | 0.7372 |
| E8a fixed soft ensemble | 0.8050 | 0.7925 | 0.7313 | 0.4657 | 0.7295 |
| E13 auxiliary tissue/HLA | 0.8023 | 0.7856 | 0.7292 | 0.4640 | 0.7306 |
| E10 MMoE | 0.7948 | 0.7804 | 0.7210 | 0.4474 | 0.7195 |
| E2 sample BCE | 0.7945 | 0.7793 | 0.7197 | 0.4441 | 0.7211 |
| E10b 4 experts, width 256 | 0.7928 | 0.7788 | 0.7170 | 0.4393 | 0.7104 |
| E10b 6 experts, width 128 | 0.7913 | 0.7768 | 0.7178 | 0.4423 | 0.7122 |
| E11 DB-MTL | 0.7817 | 0.7643 | 0.7144 | 0.4313 | 0.6956 |
| E9 CAGrad | 0.7810 | 0.7649 | 0.7106 | 0.4239 | 0.6979 |
| E12 pair ranking | 0.7807 | 0.7618 | 0.7102 | 0.4227 | 0.7007 |

E9, E11, and E12 are useful negative results. They indicate that the main bottleneck is unlikely to be solved by only changing gradient conflict handling, dynamic loss balancing, or pairwise ranking supervision.

E10 MMoE slightly improved over the sample-level E2 baseline, showing that automatic selective sharing has some value. However, it remained below the simpler E8 ensemble, and larger tuned MMoE variants did not improve it.

E13 was the strongest later single-model extension. Compared with E2 sample BCE, it improved mean AUROC by 0.0078, mean AUPRC by 0.0063, mean accuracy by 0.0095, mean MCC by 0.0200, and worst-10 mean AUROC by 0.0095. Its auxiliary diagnostics showed tissue auxiliary accuracy around 0.30 and HLA auxiliary accuracy around 0.77, suggesting that its gain mainly comes from HLA-aware representation learning.

E14 shows that the E8 and E13 gains can be combined. E14a uses an auxiliary global branch and a plain HLA-specific branch with fixed-average score fusion. It improves mean AUROC by 0.00662 over E8a and by 0.00928 over E13. Across 132 seed-task rows, E14a improves over E8a in 85 rows and over E13 in 97 rows. E14b also improves over E8a on mean AUROC, but it is slightly below E14a; its main advantage is a higher worst-10 mean AUROC. This suggests that auxiliary supervision is most useful for the global shared representation, while the HLA-specific branch benefits from remaining plain.

## 8. Algorithm Selection Rationale and Method Positioning

The algorithm selection in this project is not driven only by novelty. The priority is whether a method matches the structure of `tissuePMHC`. A suitable method should handle 44 tissue-HLA binary tasks, use shared information across tasks, avoid collapsing all tasks into one undifferentiated task, reduce negative transfer, allow fair comparison with current baselines, and remain implementable for an undergraduate research project.

From this perspective, the early candidate directions included shared peptide encoders with task-specific heads, conditioned models, HLA pseudo-sequence conditioning, FAMO, CAGrad, task grouping, MMoE / PLE, paired ranking loss, auxiliary tasks, and DB-MTL. The E1-E14 results show that these directions have very different value.

Current method positioning:

| Direction | Current judgment | Suggested role |
|---|---|---|
| Auxiliary global branch + HLA-specific branch soft ensemble | Most effective | Main model |
| Global shared branch + HLA-specific branch soft ensemble | Very effective | Core anchor baseline |
| Shared peptide encoder + task-specific heads | Stable and useful | Core baseline |
| HLA grouping | Locally useful | Source of the HLA-specific branch in the ensemble |
| Auxiliary tissue/HLA prediction | Useful complement | Representation enhancement |
| Conditioned tissue/HLA ID model | Reasonable but not dominant | Control experiment |
| HLA pseudo-sequence conditioning | Biologically interpretable, but not dominant in closed-set split | Unseen-HLA generalization or interpretation |
| FAMO and DB-MTL | Did not improve the main line | Negative results for dynamic task weighting |
| CAGrad | Did not improve the main line | Negative result for gradient-conflict handling |
| MMoE / PLE | Automatic selective sharing is useful, but less stable than explicit structure | Backup direction, not current main line |
| Paired ranking loss | Matches data construction logic, but did not improve metrics | Negative result |
| Protein language model embeddings | Potential for later work, higher engineering cost | Generalization upgrade stage |

Therefore, the algorithm-selection conclusion is that the current standard split is best served by `auxiliary-enhanced global sharing + HLA-specific sharing` with soft ensembling, rather than a more complex generic multi-task optimizer. The negative results from FAMO, DB-MTL, CAGrad, and paired ranking are still useful because they indicate that the bottleneck is more about sharing-structure design than simple loss weighting or gradient-conflict handling.

## 9. Overall Ranking and Interpretation

The final stage-wise ranking is:

```text
E14a auxiliary global + plain HLA soft ensemble
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
earlier branches such as E4, E5, and E6
```

The central finding is:

```text
Under the current closed-set tissuePMHC standard split, the best-performing structure is a soft ensemble of auxiliary-enhanced global sharing and HLA-specific sharing.
```

This can be interpreted biologically and computationally. The global branch learns peptide patterns shared across tissues and HLA alleles, and E14 strengthens this branch with auxiliary tissue/HLA supervision. The HLA branch learns allele-specific peptide motif preferences. A soft ensemble preserves both signals instead of forcing a single shared structure.

## 10. Reliability Audit and Evidence for Stage-wise Gains

The saved metrics and key stage comparisons were independently checked. Paired AUROC comparisons using the same seed and tissue-HLA task produced:

| Comparison | Mean AUROC gain | Task wins/losses | Paired-task p value |
|---|---:|---:|---:|
| E8a − E2 | +0.01223 | 33/11 | 4.18×10⁻⁵ |
| E13 − matched E2 | +0.00779 | 35/9 | 2.63×10⁻⁶ |
| E14a − E8a | +0.00662 | 34/10 | 2.44×10⁻⁴ |
| E14a − E13 | +0.00928 | 34/10 | 4.41×10⁻⁴ |

The improvements in E8, E13, and E14 are therefore not driven by a single seed or a small number of tasks. However, these p values quantify paired differences on a fixed test set; they do not eliminate project-level selection bias. The same standard test was repeatedly observed while E1–E14 research directions and model rankings were developed.

E8 uses a sensible leakage-aware procedure within the individual experiment: validation selects the ensemble strategy, full training data are used for retraining, and test data are used for final evaluation. This local control must not be generalized into a claim that the complete E1–E14 research program was independent of test feedback. Phase 1 rankings should be treated as exploratory benchmark evidence and confirmed on a frozen outer split or external data.

## 11. Limitations

The current results should be interpreted within the limits of the benchmark:

1. The split is closed-set for tissues, HLA alleles, and tasks.
2. 77.69% of test rows contain a peptide observed in another training task.
3. 97.36% of test rows contain a parent UniProt observed in training.
4. E14 currently proves superiority under the standard split, not yet under peptide-disjoint, protein-disjoint, unseen-HLA, or external validation settings.
5. Label 0 is a pseudo-negative based on missing target-tissue evidence, not confirmed non-presentation.
6. Every test task is artificially balanced 1:1, so AUPRC, accuracy, F1, and MCC are not deployment-prevalence metrics.
7. The standard test was repeatedly used for comparisons during E1–E14, creating project-level model-selection bias.

These limitations do not invalidate the current conclusion, but they define its scope.

## 12. Recommended Next Steps

The standard-split model-stacking phase can be considered temporarily complete. The next research should focus on reliability and generalization:

1. Freeze the current standard test and stop using it for model or research-direction selection.
2. Create a never-inspected outer confirmation set and peptide-, parent-protein-, unseen-HLA-, and study/assay-disjoint splits.
3. Run negative controls by shuffling HLA branch scores before ensembling.
4. Use nested cross-validation for the complete model-selection pipeline.
5. Report hierarchical cluster-bootstrap intervals over tasks, tissues, proteins, and pairs.
6. Explicitly call label 0 a pseudo-negative and perform positive-unlabeled sensitivity analyses.
7. Analyze global/HLA score correlation and save sample-level predictions, environment information, and data hashes.

## 13. Conclusion

The research after E0 substantially improved predictive metrics on the current tissuePMHC standard split. E2 supports multi-task shared peptide representation over the evaluated traditional single-task baselines. E6 and E7 indicate that HLA-specific sharing has local value but hard selection is unstable. E8 supports complementary global and HLA branch information on this benchmark. Later extensions indicate that generic optimization changes are less effective than explicit sharing-structure design, while E13 supports auxiliary tissue/HLA supervision as useful representation regularization. E14a combines these ideas and is the strongest Phase 1 model evaluated.

Therefore, on the current 44-task, closed-set, balanced standard split, the best Phase 1 model is the E14a auxiliary global plus plain HLA fixed-average soft ensemble. This conclusion must not be extrapolated into confirmed superiority for unseen peptides, proteins, HLA alleles, or external cohorts.

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
