# 2026.7.8 Research log

## 1. Targets for today

Today ' s objective is to complete and analyse:

```text
E8: validation-weighted soft ensemble of global branch and HLA branch
```

E8 Continues the two-stage design of E7 but no longer has design.

`hard selection` means that each task can only be selected by one option: either using global branch or HLA branch.
The results of E7 indicate that this alternative mechanism is not stable enough.

E8 to soft answer:

```text
final_score = w_hla * hla_score + (1 - w_hla) * global_score
```

ZXQ0QZ refers to the integration of the projected scores of multiple models by weight, rather than one of the models.
Here `prediction score` is the positive probability of model output.

## 2. E8 Code

Add script:

```text
scripts/run_tissuepmhc_soft_ensemble.py
```

Output directory:

```text
results/tissuePMHC_soft_ensemble/
```

Main output file:

```text
per_task_metrics.csv
summary_metrics.csv
stability_metrics.csv
candidate_metrics.csv
weight_metrics.csv
comparison_metrics.csv
metadata.json
```

E8 Use 3 seed:

```text
20260704
20260705
20260706
```

Training parameters:

```text
device: cuda
epochs: 25
batch_size: 512
learning_rate: 0.001
weight_decay: 0.0001
embedding_dim: 16
hidden_dim: 128
dropout: 0.2
validation_fraction: 0.2
selection_metric: AUROC
```

## 3. E8 Experimental design

E8 Use two-stage process:

```text
1. Cuts the Train-core and value from the Train.
2. Train with train-core valuation global branch.
3. Train each HLA group in the train-core.
4. Get every global_validation_metric and hla_validation_metric on the value.
5. Retrain with complete train.
6. Retrain every HLA group in the complete train.
7. Get global_score and hla_score on test.
8. The weights decided by the validation merge.
```

The purpose of this is to:

```text
The value is used only to determine the weight of integration.
The test is only used for the final evaluation.
Final branch uses complete trainee training and E2 uses the same amount of training data.
```

## 4. E8 Three strategies

E8 tests three integration strategies:

```text
E8a: e8a_fixed_average
E8b: e8b_validation_delta_clipped
E8c: e8c_validation_softmax
```

### 4.1 E8a fixed average

Fixed average:

```text
hla_weight = 0.5
global_weight = 0.5
```

That is:

```text
final_score = 0.5 * hla_score + 0.5 * global_score
```

### 4.2 E8b validation-delta clipped

Adjusting the HLA weight to value AUROC margin:

```text
delta = hla_validation_auroc - global_validation_auroc
hla_weight = clip(0.5 + 5.0 * delta, 0.15, 0.85)
```

`clip` means that values are limited to the specified range.
Here HLA weights are not below 0.15 or 0.85.

### 4.3 E8c validation softmax

Generates weights with value AUROC's softmax.

`softmax` is a function that converts multiple fractions to non-negative weights and totals 1.
The softmax experiment is:

```text
0.02
```

`temperature` controls the sharpness of the softmax.
The smaller the weight, the easier it is to approach 0 or 1.

## 5. E8 Main findings

The overall results are as follows:

| model | mean AUROC | mean AUPRC | mean accuracy | mean MCC | worst-10 mean AUROC |
|---|---:|---:|---:|---:|---:|
| E2 shared heads | 0.7927 | 0.7777 | 0.7180 | 0.4404 | 0.7178 |
| E7 selective HLA/global | 0.7904 | 0.7754 | 0.7184 | 0.4405 | 0.7143 |
| E8a fixed average | 0.8050 | 0.7925 | 0.7313 | 0.4657 | 0.7295 |
| E8b validation-delta clipped | 0.8046 | 0.7916 | 0.7314 | 0.4660 | 0.7304 |
| E8c validation softmax | 0.8020 | 0.7890 | 0.7274 | 0.4583 | 0.7279 |

E8 is clearly above E2 and E7.

E8a compared to E2:

```text
mean AUROC  +0.0122
mean AUPRC  +0.0149
mean accuracy +0.0133
mean MCC    +0.0253
```

E8b is compared to E2:

```text
mean AUROC  +0.0119
mean AUPRC  +0.0140
mean accuracy +0.0134
mean MCC    +0.0256
```

E8c compared to E2:

```text
mean AUROC  +0.0092
mean AUPRC  +0.0114
mean accuracy +0.0094
mean MCC    +0.0179
```

The number of AUROC upgrades by task-seed rows is:

```text
E8a fixed average: 91 / 132 improved
E8b delta clipped: 88 / 132 improved
E8c softmax:       84 / 132 improved
```

`task-seed row` means a task under an evaluation record in a seed.
44 tasks multiplied by 3 seeds, 132 total.

## E8 weight analysis

The distribution of HLA weights for three strategies:

| strategy | mean HLA weight | std | min | max |
|---|---:|---:|---:|---:|
| E8a fixed average | 0.5000 | 0.0000 | 0.5000 | 0.5000 |
| E8b delta clipped | 0.4576 | 0.1461 | 0.1500 | 0.8500 |
| E8c softmax | 0.4228 | 0.2567 | 0.0073 | 0.9868 |

E8c weight is too extreme, close to E7 hard section.
This may explain why E8c is weaker than E8a and E8b.

Most important observation:

```text
The simplest E8a fixter performance.
```

This suggests that there is still noise in the estimation of the weight of the task-level.
The fixed average does not over-construct value, so it's more stable.

## 7. E8 task-level results

E8a Uplifts the biggest task:

| target_tissue | mhc_restriction | AUROC delta vs E2 |
|---|---:|---:|
| small intestine | HLA-B*15:01 | +0.0541 |
| lung | HLA-A*24:02 | +0.0460 |
| blood | HLA-B*51:01 | +0.0410 |
| lymphoid | HLA-B*51:01 | +0.0369 |
| lymphoid | HLA-B*27:05 | +0.0342 |

E8a: The biggest drop:

| target_tissue | mhc_restriction | AUROC delta vs E2 |
|---|---:|---:|
| blood | HLA-B*07:02 | -0.0182 |
| uterine cervix | HLA-A*24:02 | -0.0170 |
| lymphoid | HLA-B*15:02 | -0.0147 |
| blood | HLA-C*05:01 | -0.0144 |
| lymphoid | HLA-C*05:01 | -0.0122 |

E8a is not just raising a few tasks.
It is raised on most task-seed rows and works-10 means AUROC from 0.7178 of E2 to 0.7295.

This suggests that E8 has improved overall performance and has improved the bottom of the difficult task.

## 8. Why E8 is higher than E2

E2 is:

```text
shared peptide encoder + task-specific heads
```

E2 has the advantage of global shareing.
All 44 tissue-HLA task share a peptide encoder, thus learning generic peptide paper from across HLA, and across the tessué.

But E2 limits are:

```text
All task shares the same peptide re-entry.
```

This may make encoder more global averages and less sensitive to HLA-special peptide methodology.

E8 added HLA brach:

```text
within-HLA shared peptide encoder + task-specific heads
```

HLAbranch can learn more easily about some of the alpha alleles in the medium.

`motif` refers to a recurring, functional pattern in the sequence.
In HLA-I9-mer peptide, amino acids are often critical to HLA binting at certain locations.
These key positions are also often referred to as `anchor residues`.

E8 Success statement:

```text
Complementarity information exists for global branch and HLA brach.
```

Global branch learns universal delivery signals across HLA.
HLAbranch learns HLA-special finding preference.
Soft Esmble merges the two, so it exceeds E2.

## 9. Why E8 is higher than E7

E7 with hard section:

```text
Only global or HLA brach is selected for each task.
```

If the selection on validation set is noisey, E7 will completely lose another branch message.

E8 with soft Esmble:

```text
final_score = w_hla * hla_score + (1 - w_hla) * global_score
```

Even if some branch is weak on value, it can still contribute some information.

So E8 is much more stable than E7.

From the results:

```text
E7 mean AUROC = 0.7904
E8a mean AUROC = 0.8050
```

The problem is not that HLA branch is worthless, but that E7 uses HLA branch in a way too hard.

## E8 Logical credibility audit

Today E8 is being analysed for logical credibility.

Conclusions

```text
The logic of E8 is credible.
E8 The results under the current TissuePMHC standandard standard are probably true.
```

### 10.1 Not evident

E8 weights are from value, not test.

The code process is:

```text
train → train-core + validation
validationbranch only trains on train-core
validation metric only calculates on validation
Full train retrain
Test only for final evaluation
```

So E8 doesn't directly use test set weight.

In particular, E8a:

```text
hla_weight = 0.5
```

E8a does not use value metric weighting at all.
It's still the strongest model.
This further supports the E8 upgrade from the true complementarities of the global/HLA score, rather than the incidental benefits of the validation.

### 10.2 Branch coverage complete

Results of the inspection:

```text
Seed:
validation global: 44 tasks
validation HLA:    44 tasks
test global:       44 tasks
test HLA:          44 tasks
```

Eventually, every Seed, every E8 Strategy has 44 tabs.

### 10.3 score alignment is largely credible

E8 Integration requirements:

```text
The global_score[i] and hla_score[i] correspond to the same test screen.
```

Checked in code:

```text
global_test["y_true"] == hla_test["y_true"]
```

Meanwhile, the global branch and HLA brach sets of test subsets are from the same test_df and the filtering operation retains the original line order.
So the current score alignment is basically credible.

However, for the sake of paper-level rigour, it is suggested that the following be added with a stronger assertion:

```text
The order of sample_id must also be fully consistent.
```

### 10.4 E8 didn't end up like a bug.

If E8 upgrades are from bugs, the common phenomenon is that:

```text
Only one saw is unusually high;
Only average AUROC is high, but not worth-10;
(a) Complex weight strategies with low fixed averages;
The result was concentrated by a few tasks pulling up.
```

The result is not that:

```text
3 seeds are stable and high;
The first-10 AUROC is also upgraded;
Simplest fixover strongest;
91/132 task-seed rows up.
```

The E8 result is therefore more in line with the real ensemble gain than with the obvious bug.

## 11. E8 Ending boundary

E8 is credible on the current benchmark, but still needs to be noted:

```text
The current TissuePMHC split is closed-set.
```

ZXQ0QZ means that the Tissue, HLA and task in the test have all appeared in the train.

Previous inspection found:

```text
About 77.3% of the test peptide was found in other task in the train.
About 95.8% of the test source protein appears in the train.
```

Thus, E8 is currently proving:

```text
The blogger writes about the current situation of the tissue-set tissuePHC standard split, which is a very important tool for the development of the new society.
The global structure + HLA-special structure is the most powerful.
```

But it has not yet proved:

```text
The same is true for completely new peptide, new source protein, new HLA or external data.
```

This is not a logical error for E8 but a generalized boundary for the current benchmark.

## 12. Current model sequencing

As of E8, the current model is sorted as:

```text
E8a fixed soft ensemble
≈ E8b validation-delta clipped ensemble
>
E8c validation softmax ensemble
>
E2 shared peptide encoder + task-specific heads
>
E7 selective HLA/global
>
E3 conditioned tissue + HLA ID
≈ E4b hybrid HLA ID + HLA pseudo-sequence
>
E6 HLA grouped
>
E5 FAMO
>
E6 tissue grouped
>
E4 HLA pseudo-sequence only
```

Current suggested master model:

```text
E8a fixed average soft ensemble
```

Reason:

```text
(a) The simplest structure;
means AUROC and means AUPRC highest;
(b) Upgrading stability;
Do not rely on value metric weighting, so less than the valueation.
```

E8b can be a sound alternative:

```text
MCC and World-10 AUROC are slightly higher than E8a.
```

## 13. Next steps

It is not recommended that the complex model continue to be built immediately.
The credibility of E8 should be confirmed and the border generalized analysis undertaken.

Suggesting next steps:

```text
Provisional recommendation: E8 valuation and stand measures
(Return to formal E9 after the subsequent retreading; for official E9 mainline see section 15.
```

These include:

```text
1. Add sample_id alignment assertion.
   Confirms that the global_score and hla_score blends correspond to exactly the same test screen.

2. Do the nigative control.
   Randomly disrupt HLAbranch score before opening.
   If performance falls back near or lower than E2, then E8 upgrades come from real complementary information.

3. Do peptide-disjoint or protein-disjoint split.
   Test whether E8 still exceeds E2 under a more stringent generalization set.

4. Increase the number of seeds.
   Expand E8a/E8b from 3 seeds to 5 or 10 seeds.

5. Analyses the relevance of global_score to hla_score.
   If the two are relevant but not identical, indicate that the gains are reasonably generated by the Esemble.
```

`negative control` means the intentional destruction of a key structure in the model to check for performance decline.
If performance does not decline after destruction, this may indicate that the original upgrade may not be credible.
If the performance is significantly reduced after destruction, this suggests that the original structure does work.

## 14. Conclusion today

The most important conclusion today is that:

```text
E8 soft Emble is clearly above E2, becoming the strongest model at present.
```

More specifically:

```text
E2 proves multi-task validated peptide encoded.
E6 proves that HLA-special share has a partial value but that its use alone would have lost global information.
E7 proves hard supply is unstable.
E8 proves that global supply and HLA-special supply can be complemented by soft additional.
```

The current tissuePMMHC core findings are therefore:

```text
The best structure is not a single global structure, nor is it a single HLA grouping,
It's a global branch + HLA-special branch soft ensemble.
```

The result is now credible under the current standard split.
However, when writing an official report or paper, it must be clearly stated that:

```text
The current conclusions apply mainly to closed-set issuesPHC benchmark.
A more rigorous new peptide/ new protein/ external data aggregation still needs to be validated.
```

## 15. Description of follow-up main line correction

The follow-up retune confirmed that the main line of the item, from E6 to E8, should be clearly classified as E2 instead of E4.

E2 Thread:

```text
shared peptide encoder + task-specific heads
```

This line is concerned with:

```text
How to share the picture between tasks;
Which is more appropriate for a global share, HLA-special share, slective share, soft esmble.
```

The spectrometry of E6, E7 and E8 should therefore be understood as:

```text
E2 shared heads
→ E6 HLA/tissue task grouping
→ E7 global branch vs HLA branch hard selection
→ E8 global branch + HLA branch soft ensemble
```

E4 Thread:

```text
peptide encoder + tissue embedding + HLA pseudo-sequence encoder
```

This line is concerned with:

```text
HLA pseudo-equality is a value expressed by this creature;
Which would be more appropriate for HLA ID, HLA pseudo-equality, hybrid HLA representation?
```

E4 is not the main follow-up performance line for the time being, as E4 is not above E2 on standard split.
E4 should be retained as biological representational branch, i.e., biological expression analytical line.

The base of the follow-up optimization method in the original roadmap should therefore be corrected to E2.
E7 hard addition and E8 solid update,
The more backward method numbering in the original roadmap needs to be delayed:

```text
E6: HLA/tissue task grouping on E2
E7: validation-based selective HLA/global sharing
E8: validation-weighted soft ensemble of global branch and HLA branch
E9: E2 + CAGrad
E10: MMoE / PLE selective-sharing model on the E2/E8 shared-head line
E11: E2 + DB-MTL
```

ZXQ0QZ means Confect-Averse Gradient Descent, used to mitigate gradient conflicts between different tabs.
`DB-MTL` is a dynamic multitasking learning method that balances lossscape and gradient magnitude for different task.
ZXQ0QZ and ZXQ1QZ are both part of the secret-sharing mode that allows different tabs to automatically choose different levels of sharing.

The order of follow-up studies should therefore read:

```text
Continue with the original roadmap content of E9 CAGrad, E10 MMOE/ PLE, E11 DB-MTL, etc. along the E2/E8 performance main line;
The new roadmap extension E8 reliability, stress tess, disjoint split.
```
