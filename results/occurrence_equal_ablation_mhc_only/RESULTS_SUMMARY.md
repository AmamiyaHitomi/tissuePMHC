# Experiment 3/4: Component Decomposition versus MHC-only

# Experimental Settings

- Data: Fixed division of occurrence-equival Human/Mouse train/test.
- Random seeds: 20260704, 2026005, 202600606.
- Human: 20 epochs; Mouse: 25 epochs; all species are locked in hyperparameters.
- Full model: multiscale CNN, tessue/MHC subsidiary supervision, global branch and MHC branch, rank foundation in task.
- `no_mhc_branch ' and `no_rank_fusion ' are derived directly from the same complete set of models, avoiding any random deviations resulting from training avoidance.
- `no_auxiliary ' and `no_multikernel ' re-training under the same data, secret and training budgets.
- `mhc_only_nn ' only receives the peptide sequence and selects MHC-special head, not Tissue or Tissue --MHC task identity.
- Do not perform the required comparison of the strict generalization or strict protocol structure of new data.

# Three seed summary

| Species | Model | Mean task AUROC | Mean task AUPRC |
|---|---|---:|---:|
| Human | full rank fusion | 0.8136 ± 0.0021 | 0.8126 ± 0.0021 |
| Human | no MHC branch | 0.7923 ± 0.0017 | 0.7913 ± 0.0014 |
| Human | no rank fusion | 0.8115 ± 0.0027 | 0.8117 ± 0.0022 |
| Human | no auxiliary | 0.8169 ± 0.0023 | 0.8156 ± 0.0025 |
| Human | no multikernel | 0.7999 ± 0.0037 | 0.7989 ± 0.0036 |
| Human | MHC-only CNN | 0.2925 ± 0.0062 | 0.3932 ± 0.0040 |
| Mouse | full rank fusion | 0.8194 ± 0.0157 | 0.8279 ± 0.0121 |
| Mouse | no MHC branch | 0.8133 ± 0.0142 | 0.8218 ± 0.0074 |
| Mouse | no rank fusion | 0.8163 ± 0.0160 | 0.8254 ± 0.0124 |
| Mouse | no auxiliary | 0.8098 ± 0.0147 | 0.8233 ± 0.0141 |
| Mouse | no multikernel | 0.7949 ± 0.0178 | 0.8043 ± 0.0159 |
| Mouse | MHC-only CNN | 0.2168 ± 0.0052 | 0.3686 ± 0.0064 |

# Relatively complete model task-by-task

| Species | Candidate | Mean delta | Bootstrap 95% CI | W/T/L | Wilcoxon p |
|---|---|---:|---:|---:|---:|
| Human | no MHC branch | -0.02134 | [-0.02647, -0.01643] | 11/0/66 | 1.11e-10 |
| Human | no rank fusion | -0.00212 | [-0.00361, -0.00066] | 29/0/48 | 0.00557 |
| Human | no auxiliary | +0.00326 | [-0.00165, +0.00798] | 45/0/32 | 0.179 |
| Human | no multikernel | -0.01369 | [-0.01895, -0.00840] | 19/0/58 | 7.09e-06 |
| Human | MHC-only CNN | -0.52109 | [-0.54887, -0.49270] | 0/0/77 | 2.46e-14 |
| Mouse | no MHC branch | -0.00613 | [-0.01442, +0.00171] | 4/0/7 | 0.278 |
| Mouse | no rank fusion | -0.00316 | [-0.00672, +0.00062] | 3/0/8 | 0.102 |
| Mouse | no auxiliary | -0.00962 | [-0.01550, -0.00347] | 2/0/9 | 0.0244 |
| Mouse | no multikernel | -0.02453 | [-0.03329, -0.01484] | 1/0/10 | 0.00293 |
| Mouse | MHC-only CNN | -0.60259 | [-0.71828, -0.46165] | 0/0/11 | 0.000977 |

# Conclusion

1. Multiscale sequence codes contribute steadily to both species; the removal of Human/Mouse AUROC has decreased by 0.0137/0.0245, and the task-by-task testing is significant.
2. The contribution of the MHC-special branch of Human is clear; Mouse is in the same direction but has not been visible under 11 missions.
The Rank Fusion has brought small but significant AUROC improvements to Human; the same direction is not significant in Mouse.
4. There are species differences in subsidiary supervision: AUROC has declined significantly after Mouse removed the auxiliary loss; the removal in Human has risen slightly, but confidence is 0 across the zone and the contribution of the auxiliary supervision in Human cannot be claimed to be positive.
5. MHC-only can be trained to show that the current task is not explained separately by the peptide-MHC information by contrast to the complete model that is behind all Human and all Mouse tasks.

The AUROC of MHz-only is significantly less than 0.5 and should not be read as "models inverted" in the ordinary sense. In the occurence-equivalement construction, the same peptide - MHC can match conflict labels in different tissues; models that do not contain tissues can only give these lines the same scores, thus systematically violating the tissue-conditioned sorting objectives. The audit covered the length of 685 cross-tissue queries per Human, Mouse 104 per ed, `1e-6 ' tolerances are 0 non-transformation irregularities; the actual maximum scores ranged from 5.96e-8 and 1.19e-7, respectively.

# Recoverable and produced

- Predicted completeness: Human 138,600 rows, Mouse 19,800 rows; none missing scores.
- Task-by-task indicators: Human 1,386 lines, House 198 lines; coverage of 6 models, 3 seeds and all tasks.
- Human complete model saw 202,60704 AUROC 0.811247, AUPRC 0.810163.
- Mouse complete model III seesd exact recovery of lock averages AUROC 0.8199442424242, AUPRC 0.82740265.
- Formal correct training takes time: Human 2454.96 s; revised seed Mouse 196/95 s; epoch, ed and master run records are available at `timing_results.csv'.
