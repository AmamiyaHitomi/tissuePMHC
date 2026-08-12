# RESEARCH_LOG_2026-07-10

## 1. E14: auxiliary soft ensemble

E14 full run today. E14 is designed to check if E13's tissue/ HLA auxiliary supervision can be added to E8.

Results catalogue:

```text
results/tissuePMHC_auxiliary_soft_ensemble/
```

Experimental settings:

Project Settings
|---|---|
| seeds | 20260704, 20260705, 20260706 |
| tasks | 44 |
| ensemble formula | `0.5 * global_score + 0.5 * hla_score` |
| E14a | global auxiliary branch + HLA plain branch |
| E14b | global auxiliary branch + HLA auxiliary branch |

## Core results

Mean AUROC Mean AUPPC Mean AccuracyMean MCC World-10 Mean AUROC
|---|---:|---:|---:|---:|---:|
| E14a global auxiliary + HLA plain soft ensemble | 0.8116 | 0.7978 | 0.7372 | 0.4769 | 0.7349 |
| E14b global auxiliary + HLA auxiliary soft ensemble | 0.8093 | 0.7955 | 0.7348 | 0.4735 | 0.7372 |
| E8a fixed soft ensemble | 0.8050 | 0.7925 | 0.7313 | 0.4657 | 0.7295 |
| E13 auxiliary tissue/HLA | 0.8023 | 0.7856 | 0.7292 | 0.4640 | 0.7306 |

E14a is the new strongest outcome of the current era.

## 3. Comparison with existing models

E14a vs E8a:

```text
mean delta AUROC = +0.00662
wins/losses over 132 seed-task rows = 85 / 47
median delta AUROC = +0.00645
```

E14a vs E13:

```text
mean delta AUROC = +0.00928
wins/losses over 132 seed-task rows = 97 / 35
median delta AUROC = +0.00765
```

E14b vs E8a:

```text
mean delta AUROC = +0.00433
wins/losses over 132 seed-task rows = 84 / 48
median delta AUROC = +0.00520
```

## Interpretation

E14 Indicates that the gains of E8 and E13 can be superimposed. The best combination is not both of the branch plus auxiliary, but:

```text
global branch: auxiliary tissue/HLA supervision
HLA branch: plain supervised heads
fusion: fixed average soft ensemble
```

This suggests that the auxiliary subvisition is better suited to enhance global shared remission. HLA-special Branch is already grouped by HLA, adding an additional marginal benefit to HLA auxiliary subvision, and may even introduce too much structural constraints.

## 5. Changes in mandate levels

E14a is the most ambitious task compared to E8a:

| Task | Mean AUROC delta |
|---|---:|
| blood + HLA-B*51:01 | +0.0294 |
| kidney + HLA-A*24:02 | +0.0265 |
| uterine cervix + HLA-A*24:02 | +0.0228 |
| blood + HLA-C*03:04 | +0.0213 |
| blood + HLA-B*27:05 | +0.0206 |

E14a, the biggest drop in tasks compared to E8a:

| Task | Mean AUROC delta |
|---|---:|
| lymphoid + HLA-B*40:02 | -0.0205 |
| brain + HLA-B*40:02 | -0.0118 |
| thymus + HLA-A*24:02 | -0.0112 |
| lung + HLA-A*02:01 | -0.0105 |
| uterine cervix + HLA-B*51:01 | -0.0104 |

## 6. Current ranking

```text
E14a auxiliary global + HLA plain soft ensemble
>
E14b auxiliary global + auxiliary HLA soft ensemble
>
E8a fixed global/HLA soft ensemble
≈ E8b validation-clipped ensemble
>
E13 auxiliary tissue/HLA prediction
>
E8c validation softmax ensemble
>
E10 MMoE
≈ E2 sample BCE
>
E11 / E9 / E12 et al.
```

## 7. Current conclusions

Under the TissuePMHC standard split, the best structure at present is:

```text
auxiliary-enhanced global sharing + HLA-specific sharing + fixed soft ensemble
```

E14a should be included in the official report as a new main result. E8a is reduced from the "most powerful" to "most powerful before" E13 from "supplementary experiments of no more than E8" to "key component of E14".
