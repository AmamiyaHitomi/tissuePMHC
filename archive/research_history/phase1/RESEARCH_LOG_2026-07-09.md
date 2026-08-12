# 2026.7.9 Research log

## 1. Targets for today

Today's goal is to complete the planned E9-E13 mainline after 7.8 and to determine whether the current `standard split` still needs to be stacked up on complex models.

ZXQ0QZ refers to the current fixed train/ test division. The teste here has been present in the train, HLA and task, so it mainly assesses the internalization of tasks under the closed-set scenario.

The experiments that have been added or completed today are:

```text
E9:  E2 + CAGrad
E10: MMoE selective-sharing
E10b: tuned MMoE configs
E11: E2 + DB-MTL
E12: paired ranking loss
E13: auxiliary tissue/HLA prediction
```

## Key terms

ZXQ0QZ is Confect-Averse Gradient Discent, which is understood in Chinese as "avoiding a decline in the gradient of conflict." It attempts to alleviate the problem of conflicting gradient directions in different tasks.

ZXQ0QZ is Multi-gate Mixture-of-Experts, which is understood in Chinese as " Multi-TextMix Models." It allows different taskers to select multiple versions with different Gates, and thus learn from the single share.

`DB-MTL` is a dynamic balancing multi-task learning method with the goal of dynamic balance of different loss or gradient scales for the task.

ZXQ0QZ means using positive or negative sample pairing to make positive or negative sample scores higher than negative sample fractions in the same pair.

`auxiliary task` refers to the supporting tasks. The main task in E13 remains the positive and negative classification of each Tissue-HLA task, with additional projections for Tissue label and HLA label, hoping to share encoder to learn the task structure.

ZXQ0QZ is the area below the ROC curve, and the measurement model distinguishes the overall sorting capacity of the positive and negative samples. ZXQ1QZ is the area below the Accuracy-Recall Rate curve, with greater attention to the quality of the positive sample retrieval. ZXQ2QZ is the Matthews coefficient, an indicator for the combined consideration of the four categories of classification results.

## 3. Add a new directory of codes and results today

The experiment, the experiment, the script, the results, the results, the purpose, the purpose.
|---|---|---|---|
E9  ZXQ0QZ  `results/tissuePMHC_cagrad/`  Check for gradient conflict management improvements E2
E10  ZXQ0XZ  `results/tissuePMHC_mmoe/`  Check for automatic proximity to E8
E10bZXQ0XZZXQ1QZ Resize number, width and date regularization
E11 ZXQ0XZ ZXQ1QZ
ZXQ0QZ`results/tissuePMHC_pair_ranking/` Checks for validity of the sorting in the pair
ZXQ0QZ ZXQ1QZ Check whether improvement is indicated by the / HLA improvement

All official comparisons are available using three seed:

```text
20260704
20260705
20260706
```

## Overall results

The most important summary today is as follows:

I mean AUROC means AUPRC means transparency means MCC means worst-10 means AUROC
|---|---:|---:|---:|---:|---:|---|
E8a fixed soft E.0.8050  0.7925  0.7313  0.4657  0.7295  Current most powerful primary model
E13 auxiliary tissue/ HLA 0.8023  0.7856  0.7292  0.4640  0.7306  is significantly better than E2/E11/E12 and slightly weaker than E8
E10 MMOE  0.7948  0.7804  0.7210  0.4474  0.7195  slightly more than E2/E9/E11/E12 but not close to E8
| E2 sample BCE | 0.7945 | 0.7793 | 0.7197 | 0.4441 | 0.7211 | sample-level E2 baseline |
E10b 4 experts, width 256 0.79228  0.7788  0.7170  0.4393  0.7104  tuned MMOE not exceeding E10
E10b 6 experts, width 128 0.7913  0.7768 0.4423  0.7122 tuned MMOE not exceeding E10
E11 DB-MTL  0.7817  0.7644  0.4313  0.6956  Unimproved E2
E9 CAGrad  0.7810  0.7649  0.7106  0.4239  0.6979  not improved E2
E12 fair running 0.7807 0.7618 0.7102 0.4227 0.707

## 5. E9 CAGrad Conclusions

The core question for E9 is whether CAGrad can improve the performance of the main line if there is a gradient conflict between task in E2.

The results were:

```text
E9 mean AUROC = 0.7810
E9 mean AUPRC = 0.7649
E9 worst-10 mean AUROC = 0.6979
```

It is significantly lower than E8 and lower than E2 sample BCE. Therefore, over the current task, there is no benefit from visible gradient conflict resolution.

This may mean that the performance bottlenecks are not simple task-gradient condition, but rather how the task-specific rule is expressed. The success of E8 supports this: the global branch and HLA-special mix are more effective than direct optimizer changes.

## 6. E10 and E10b MMOE Conclusions

E10 The objective is to allow models to learn automatically instead of designing global branch and HLA brach manually.

E10 Results:

```text
mean AUROC = 0.7948
mean AUPRC = 0.7804
mean accuracy = 0.7210
mean MCC = 0.4474
```

E10 is stronger than E9, E11, E12 and slightly stronger than E2 sample BCE, but is still significantly weaker than E8a.

E10b tested for more complex MMOE configurations:

```text
e10b_4experts_256 mean AUROC = 0.7928
e10b_6experts_128 mean AUROC = 0.7913
```

The increase in the number or width of the extert does not narrow the gap with E8 and indicates that MMOE 's automatic gate learning is less stable than E8.

## 7. E11 DB-MTL Conclusions

E11 Check if the dynamic task loss can improve E2.

The results were:

```text
E11 mean AUROC = 0.7817
E11 mean AUPRC = 0.7643
E11 worst-10 mean AUROC = 0.6956
```

DB-MTL does not exceed E2 and does not exceed E8. The results show that the main performance gap in the tissuePHC standlist is not solved by a mere dynamic balance.

## 8. E12 fair running conclusions

E12 Adds a paired ranking loss, hoping to use relative order information from the positive and negative sample.

The results were:

```text
E12 mean AUROC = 0.7807
E12 mean AUPRC = 0.7618
E12 worst-10 mean AUROC = 0.7007
```

E12 is weaker than E2 sample BCE. It is explained that the current sorting oversight in the pair_id does not enhance the main task, possibly because:

```text
1. BCE classification signals are strong enough to interfere with probabilities calibration.
2. pair is not necessarily a fully applicable learning boundary for positive and negative samples.
3. Ripping loss weights or pair sampling may not be appropriate.
```

E12 is therefore more appropriate to write analysis as a national result rather than as a primary model.

## 9. E13 auxiliary tissue/HLA preposition conclusion

E13 is today's most valuable new experiment.

E13 Average increase over E2 sample BCE:

E2 sample BCE  E13 auxiliary  E13 - E2
|---|---:|---:|---:|
| mean AUROC | 0.7945 | 0.8023 | +0.0078 |
| mean AUPRC | 0.7793 | 0.7856 | +0.0063 |
| mean accuracy | 0.7197 | 0.7292 | +0.0095 |
| mean MCC | 0.4441 | 0.4640 | +0.0200 |
| worst-10 mean AUROC | 0.7211 | 0.7306 | +0.0095 |

E13's promotion is stable, with 3 seeds meaning AUROC standard deviations of only 0.0018.

By task-seed rows, E13 relative to E2 AUROC:

```text
86 / 132 improved
46 / 132 decreased
```

E13 Relative to other models:

```text
Relative to E10 MMoE: AUROC average + 0.0075
Compare E12 fair ranking: AUROC average +0.0216
AUROC average -0.0027
```

The final training diagnosis of the auxiliary mission itself is also clear:

```text
tissue auxiliary accuracy ≈ 0.30
HLA auxiliary accuracy ≈ 0.77
```

This suggests that E13 has learned more about HLA structures, and the tissue structure is also weak. So E13 benefits are more like a global/HLA architecture that replaces E8.

## 10. Current sequencing

After completing E9-E13, the current Standard standard standard string order can be:

```text
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

And to be more concise:

```text
E8 is the current best performance master model.
E13 is the most valuable new feature.
E9/E11/E12 is of interpretative value.
E10 indicates that the automatic automatic share has some effect, but it is not as good as the manual global/HLA soft Esmble.
```

## 11. Core findings of today

7.8 The post-planned roadmap is almost finished. The most important conclusion is:

```text
Under the tissuePMHC standard split, the strongest structure remains E8a fixed overage soft Esmble.
```

E13 gives a valuable addition:

```text
The visual feature/ HLA auxiliary submission improves the quality of the video,
But it is not yet possible to replace E8 with a global branch + HLA-special brand soft E.
```

The main line can therefore be phased in. It is not recommended that the complex multitask approach continue without any distinction. The next logical step is to conduct a reliability and a broader border analysis around E8:

```text
1. E8 sample_id alignment and nigative control.
2. peptide-disjoint / protein-disjoint split.
3. More E8a/E8b stability validation for seeds.
4. Global score correlation analysis with HLA score.
5. PLE is considered again if necessary, but with a lower priority than E8 priority.
```

## 12. Recommendations on the methodology of the report

The official report could be written as a result of a "planneed effect after E8":

```text
After identifying E8 as the strongest soft-ensemble model, we evaluated several planned extensions, including gradient conflict handling, dynamic task balancing, automatic expert-based selective sharing, paired ranking supervision, and auxiliary tissue/HLA prediction.
```

The Chinese narrative can read:

```text
After confirming that E8 soft Esmble is the best structure at the moment, we have tested CAGRAD, MMOE, DB-MTL, paid rans and Tissue/HLA auxiliary preposition.
Of these methods, only E13 auxiliary preparation has brought about a steady additional increase, but it is still slightly weaker than E8.
This suggests that the manual introduction of global/HLA competitive structure is still the most effective inductive bias under the current standard split.
```

`inductive bias` can be understood as a pre-inclusion preference or hypothesis in the model structure. Here E8 inventive bias is: peptide indicates that both the global share and the HLA-specific share are needed.
