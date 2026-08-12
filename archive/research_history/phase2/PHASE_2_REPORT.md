# tissuePMHC Phase 2 Report

Updated: 2026-07-12  
Status: E15–E27 and E29 completed; the preregistered E29 5-seed ensemble is the frozen primary result on the standard split; E28 was not executed.

## Abstract

Phase 2 investigated whether score fusion, low-cost ensembling, auxiliary-task optimization, and a stronger peptide encoder could improve the E14a dual-branch model. E14a combines a global branch trained with tissue/HLA auxiliary supervision and an HLA-grouped plain branch. The original model fused the two branch probabilities with equal weights.

Experiments covered fixed fusion rules (E15), MC Dropout (E16), multi-seed prediction averaging (E17), validation-selected fusion weights (E18), checkpoint and snapshot ensembles (E19), stochastic weight averaging (E20), auxiliary-weighting and structured-sharing methods (E21–E25), OOF greedy selection and stacking (E26–E27), and a position-sensitive multi-kernel CNN peptide encoder (E29).

The frozen standard-split result is the preregistered E29 5-seed ensemble:

```text
mean AUROC            = 0.8373
mean AUPRC            = 0.8259
worst-10 mean AUROC   = 0.7670
```

Relative to the E17 5-seed ensemble, the gains are 0.0110 AUROC, 0.0121 AUPRC, and 0.0098 worst-10 AUROC. The E29-versus-E17 AUROC gain is positive on 37 of 44 tasks, with a task-bootstrap 95% interval of `[0.00717, 0.01495]`. The five matched single-seed comparisons are also all positive.

These findings support a stable improvement within the current closed-set benchmark. They do not establish equivalent generalization to unseen peptides, proteins, HLA alleles, or external cohorts. In the standard split, 77.69% of test rows contain a peptide observed in another training task, and 97.36% contain a parent UniProt observed in training. In addition, negative labels are pseudo-negatives: peptides reported in another tissue but not reported in the target tissue-HLA context.

## 1. Experimental scope and common setup

Each task is defined by a target tissue and a four-digit HLA allele. The standard split contains 44 binary tasks. Every test task contains 100 positives and 100 pseudo-negatives. Unless otherwise stated, E15–E20 use seeds 20260704, 20260705, and 20260706; the E17 and E29 final ensembles use two additional seeds, 20260707 and 20260708.

The common training configuration is 25 epochs, AdamW, learning rate 0.001, weight decay 0.0001, dropout 0.2, and batch size 512. The principal metrics are task-macro AUROC and AUPRC, supplemented by accuracy, MCC, and worst-10 mean AUROC.

All fusion experiments align branch predictions by `sample_id`, `target_tissue`, and `mhc_restriction`; matching labels alone is not sufficient. Experiments involving stochastic inference or paired training control Python, NumPy, CPU Torch, and CUDA RNG states. E18 uses inner validation, while E26, E27, and E29 use pair-grouped OOF predictions for local selection. The broader limitations from repeated standard-test reuse, preregistration provenance, and cross-dataset extrapolation are evaluated in Section 8 rather than interrupting the experiment sequence.

E14a is the Phase 2 starting point:

| Model | Mean AUROC | Mean AUPRC | Mean accuracy | Mean MCC | Worst-10 AUROC |
|---|---:|---:|---:|---:|---:|
| E14a: auxiliary global + plain HLA, probability average | 0.8116 | 0.7978 | 0.7372 | 0.4769 | 0.7349 |

## 2. Completed experiments

| Experiment | Main comparison | Selected configuration | Mean AUROC | Mean AUPRC | Worst-10 AUROC | Decision |
|---|---|---|---:|---:|---:|---|
| E15 | Fixed fusion rules | Task-wise rank average | 0.8130 ± 0.0012 | 0.7990 ± 0.0008 | 0.7381 ± 0.0023 | Retain |
| E16 | MC Dropout | 20 draws + rank fusion | 0.8121 ± 0.0015 | 0.7981 ± 0.0019 | 0.7376 ± 0.0044 | No advantage over E15 |
| E17 | Seed prediction ensemble | 5-seed + rank fusion | 0.8263 | 0.8139 | 0.7573 | Strong reference |
| E18 | Validation-selected global weight | Rank-weight selection | 0.8137 ± 0.0016 | 0.7990 ± 0.0020 | 0.7411 ± 0.0035 | Small gain |
| E19 | Checkpoint/snapshot ensemble | Final checkpoint | 0.8122 ± 0.0017 | 0.7980 ± 0.0036 | 0.7385 ± 0.0038 | Do not retain snapshots |
| E20 | SWA paired ablation | Final/final | 0.8150 ± 0.0004 | 0.8005 ± 0.0011 | 0.7420 ± 0.0021 | Do not use SWA |
| E21 | Gradient-similarity auxiliary gating | One-seed screen | 0.8069 | 0.7932 | 0.7328 | Stop |
| E22 | Periodic Nash-MTL | One-seed screen | 0.8051 | 0.7939 | 0.7304 | Stop |
| E23 | ForkMerge | One-seed screen | 0.8031 | 0.7895 | 0.7270 | Stop |
| E24 | Auto-Lambda | One-seed screen | 0.8056 | 0.7931 | 0.7289 | Stop |
| E25 | HLA-structured PLE | Three seeds | 0.7938 ± 0.0030 | 0.7769 ± 0.0033 | 0.7212 ± 0.0034 | Stop |
| E26 | Pair-grouped OOF greedy selection | E14 final 3-seed mean only | 0.8246 | 0.8116 | 0.7535 | No new information |
| E27 | OOF rank logistic stacking | Fixed C=0.1 | 0.8243 | 0.8109 | 0.7535 | No advantage |
| E29 | Multi-kernel CNN E14a | Three seeds | 0.8341 | 0.8228 | 0.7634 | Intermediate confirmation |
| E29 | Preregistered extension | Five fixed seeds | **0.8373** | **0.8259** | **0.7670** | Frozen primary result |

## 3. E15: fixed fusion-rule ablation

E15 compared equal probability averaging, equal logit averaging, and task-wise percentile-rank averaging on exactly aligned E14a branch predictions.

| Fusion rule | Mean AUROC | Mean AUPRC | Mean MCC | Worst-10 AUROC |
|---|---:|---:|---:|---:|
| Task-wise rank average | 0.8130 | 0.7990 | 0.4800 | 0.7381 |
| Logit average | 0.8118 | 0.7981 | 0.4769 | 0.7337 |
| Probability average | 0.8116 | 0.7978 | 0.4769 | 0.7349 |

Rank fusion reduces sensitivity to branch calibration differences. Its gain is consistent but small: +0.00142 AUROC versus probability averaging. It should be described as a modest fusion improvement rather than a major modeling breakthrough.

## 4. E16–E20: uncertainty and training-trajectory ensembles

MC Dropout improved slightly as the number of draws increased, but 20 draws remained below ordinary E15 rank fusion. Checkpoint and snapshot averaging diluted useful final-checkpoint rankings, and the configured snapshot ensemble performed substantially worse. Paired SWA comparisons also favored the ordinary final checkpoints, particularly for the HLA branch.

The successful approach in this group was E17: averaging predictions from independently trained seeds. The 3-seed and 5-seed results were:

| Members | Mean AUROC | Mean AUPRC | Mean accuracy | Mean MCC | Worst-10 AUROC |
|---:|---:|---:|---:|---:|---:|
| 3 seeds | 0.8243 | 0.8109 | 0.7467 | 0.4936 | 0.7530 |
| 5 seeds | 0.8263 | 0.8139 | 0.7492 | 0.4986 | 0.7573 |

This supports the interpretation that independently trained members provide more useful diversity than nearby checkpoints or stochastic passes through one trained model.

E18 selected one global/HLA rank weight using a pair-grouped validation split inside training data. The mean AUROC gain was only about 0.0007. It remains a leakage-aware weighting baseline but does not replace E17.

## 5. E21–E25: auxiliary weighting and structured sharing

E21–E24 retrained the global auxiliary branch and reused a fixed HLA plain branch. Gradient gating nearly disabled the auxiliary objectives; Nash-MTL balanced gradient contributions but reduced primary-task generalization; ForkMerge selected several validation branches without translating the small validation loss improvements to test gains; Auto-Lambda changed the auxiliary weights only minimally.

E25 used two global experts and twelve HLA-private experts. Its gates remained active and did not collapse, but the additional specialization did not outperform simpler shared representations. These are informative negative results: on the current peptide-only closed-set benchmark, lightweight fixed auxiliary supervision is more reliable than dynamically amplifying or structurally decomposing the auxiliary signals.

E23 and E24 fit on only 80% of the training data because of their internal validation design. Their gap to full-training E14a therefore cannot be attributed entirely to the algorithms themselves.

## 6. E26–E27: OOF ensemble selection

E26 used three-fold pair-grouped out-of-fold predictions. Greedy selection retained only the E14 final 3-seed mean because adding any other E14/MC candidate failed to exceed the prespecified 0.0001 AUROC threshold. E27 fitted an L2 logistic stacker to within-task candidate ranks using OOF labels only. It also failed to improve on E17.

The OOF correlation between the E14 final and MC-20 3-seed means was 0.9987. This explains why post-processing highly homogeneous candidates did not create meaningful new information.

## 7. E29: multi-kernel CNN encoder

E29 retained the E14a global/HLA dual-branch structure but replaced the flattened MLP peptide encoder with embedding, learnable positional offsets, Conv1d kernels of sizes 2, 3, and 5, and a flattened MLP projection. The principal new inductive bias is local convolution with shared multi-scale motif detectors. The earlier flattened MLP was already position-sensitive, so E29 should not be described as the first model to preserve peptide position.

The three-seed E29 OOF mean AUROC was 0.8138, compared with 0.8042 for the matched E14 OOF baseline. After the prespecified OOF criteria passed, the full-training test result was evaluated. A later preregistered extension added exactly seeds 20260707 and 20260708, retained equal member weights, and prohibited seed removal or subset selection.

The five-seed OOF gate passed:

| Metric | E29 3-seed OOF | E29 5-seed OOF | Delta |
|---|---:|---:|---:|
| Mean AUROC | 0.8138 | 0.8157 | +0.0019 |
| Mean AUPRC | 0.7971 | 0.7995 | +0.0024 |
| Worst-10 AUROC | 0.7349 | 0.7379 | +0.0030 |

The resulting frozen test comparison is:

| Model | Mean AUROC | Mean AUPRC | Accuracy | MCC | Worst-10 AUROC |
|---|---:|---:|---:|---:|---:|
| E17 5-seed | 0.8263 | 0.8139 | 0.7492 | 0.4986 | 0.7573 |
| E29 3-seed | 0.8341 | 0.8228 | 0.7542 | 0.5084 | 0.7634 |
| E29 5-seed | **0.8373** | **0.8259** | **0.7588** | **0.5175** | **0.7670** |

## 8. Full-chain reliability audit and conclusion boundaries

### 8.1 Data definition and split audit

Every pair contains one positive and one pseudo-negative with the same HLA allele and parent UniProt. A positive peptide was reported in the target tissue-HLA context. A pseudo-negative peptide was reported for the same HLA/protein in another tissue but was not reported in the target context.

This design partially controls protein/HLA confounding and is appropriate for relative tissue preference. It does not establish biological non-presentation. Missing evidence can reflect incomplete assay coverage, detection limits, laboratory protocols, or reporting bias.

Independent split checks produced:

| Check | Result |
|---|---:|
| Training/test rows | 96,972 / 8,800 |
| Tasks | 44 |
| Train/test pair-ID overlap | 0 |
| Same-task peptide overlap | 0 |
| Test rows with a peptide observed in another training task | 6,837 / 8,800 (77.69%) |
| Test rows with a parent UniProt observed in training | 8,568 / 8,800 (97.36%) |
| Per-task test composition | 100 positive + 100 pseudo-negative |

These properties make the standard split a strong closed-set benchmark rather than a strict entity-disjoint generalization benchmark. The balanced test prevalence also means AUPRC, accuracy, F1, and MCC should not be interpreted as deployment-prevalence metrics.

### 8.2 Result integrity and statistical audit

The audit covered 244 result CSV files and found no empty files, all-null rows, or unreadable tables. The formal E29 5-seed candidate has 8,800 unique sample/task keys, no duplicate or missing scores, and all 44 tasks. Independent recomputation from saved predictions produced AUROC 0.83730568 and AUPRC 0.82593372, exactly matching the formal summary.

For E29 versus E17:

- Task wins/losses: 37/7.
- Mean paired AUROC gain: 0.01102.
- Task-bootstrap 95% interval: `[0.00717, 0.01495]`.
- Paired-task t-test: p = 1.93×10⁻⁶.
- Wilcoxon signed-rank test: p = 1.82×10⁻⁶.

Matched single-seed AUROC gains for seeds 20260704–20260708 were +0.00959, +0.00690, +0.00815, +0.00726, and +0.01595. Thus the E29 improvement is not explained by selecting one favorable seed.

Key earlier paired comparisons were also directionally consistent:

| Comparison | Mean AUROC gain | Task wins/losses | Paired-task p value |
|---|---:|---:|---:|
| E8a − E2 | +0.01223 | 33/11 | 4.18×10⁻⁵ |
| E13 − matched E2 | +0.00779 | 35/9 | 2.63×10⁻⁶ |
| E14a − E8a | +0.00662 | 34/10 | 2.44×10⁻⁴ |
| E14a − E13 | +0.00928 | 34/10 | 4.41×10⁻⁴ |
| E15 rank − probability fusion | +0.00142 | 31/13 | 0.00126 |

### 8.3 Seen- and unseen-peptide analysis

| Test subset | E29 AUROC | E17 AUROC | E29 − E17 |
|---|---:|---:|---:|
| Peptide observed in training | 0.85328 | 0.83966 | +0.01362 |
| Peptide not observed in training | 0.74199 | 0.73761 | +0.00438 |

E29 retains some ranking ability on unseen peptides, but absolute performance drops substantially and the relative gain narrows. Multi-scale convolution improving local-motif generalization is therefore a supported working hypothesis, not yet a confirmed peptide-disjoint result.

### 8.4 Test reuse and preregistration boundary

E18, E26, E27, and E29 implement sensible local selection controls using inner validation or pair-grouped OOF predictions. E29 reads the test data only after its OOF gate passes. The local file timestamps for the five-seed extension are also ordered consistently with preregistration, then new-seed OOF generation, then test prediction.

However, the same standard test was observed repeatedly throughout the broader E0–E29 research program. The E29 architecture was proposed after substantial feedback from earlier test-side experiments. Consequently, the final standard test value is numerically valid but should not be described as a fully untouched external confirmation. The local preregistration files also lack an independently verifiable repository history or third-party timestamp.

The standard split is now frozen. It must not be used to select additional members, architectures, fusion weights, or follow-up methods.

### 8.5 Reproducibility

The code sets Python, NumPy, CPU Torch, and CUDA random seeds and saves sample-level predictions. All 43 Python scripts passed syntax-tree parsing during the audit. Prediction alignment and OOF pair-integrity checks are implemented in the later experiments.

Remaining limitations are:

- Deterministic Torch/cuDNN algorithms are not enabled globally, so identical seeds are not guaranteed to be bitwise reproducible across CUDA environments.
- The project does not yet provide a complete locked environment, data hashes, prediction hashes, and command manifest.
- A reported ensemble standard deviation of zero means that only one aggregated ensemble point exists; it does not imply zero statistical uncertainty.
- Tasks share tissues, HLA alleles, peptides, and proteins, so simple task-level tests do not capture every dependence structure.

## 9. Model positioning and final conclusions

The recommended primary result is the preregistered E29 5-seed multi-kernel CNN E14a ensemble: mean AUROC 0.8373, mean AUPRC 0.8259, and worst-10 mean AUROC 0.7670.

The central Phase 2 finding is not that a more complex optimizer or training-trajectory average is intrinsically better. Task-rank fusion reduces branch-scale mismatch; prediction averaging across independently trained seeds reduces variance; independent members are more useful than nearby checkpoints, snapshots, or SWA states; and changing the peptide encoder while retaining the E14a dual-branch structure can improve both member strength and diversity. Dynamic auxiliary weighting, HLA-private structure, and OOF post-processing of highly homogeneous candidates did not produce gains of the same magnitude.

The current standard-split ranking is:

```text
E29 5-seed multi-kernel CNN task-rank ensemble
> E29 3-seed multi-kernel CNN task-rank ensemble
> E17 5-seed task-rank ensemble
> E26 OOF greedy 3-seed mean ≈ E17 3-seed task-rank ensemble ≈ E27 OOF stacking
> E20 final/final and E18 validation-selected rank fusion
> E15 task-rank fusion
> E16 MC Dropout-20
> E19 final checkpoint
> checkpoint/snapshot ensembles, SWA, and E21–E25 single-model explorations
```

This order is based primarily on mean AUROC on the current standard split; in particular, E29 3-seed (0.8341) ranks above E17 5-seed (0.8263).

### Supported and unsupported conclusions

The evidence supports the following statement:

> On the current 44-task, closed-set, balanced tissuePMHC standard split, the E29 multi-kernel CNN E14a 5-seed task-rank ensemble is the best evaluated model. Its improvement over E17 is directionally consistent across most tasks and all five matched seeds.

The current evidence does not support claims that:

- the same performance holds for unseen HLA alleles, strict peptide-disjoint or protein-disjoint splits, or external cohorts;
- label 0 represents experimentally confirmed biological non-presentation;
- balanced-benchmark precision equals real-prevalence precision;
- E29 proves a specific biological or causal tissue-regulation mechanism.

## 10. Key code and result files

| Component | Location |
|---|---|
| Dataset construction | `scripts/build_tissue_specificity_pairs.py`; `scripts/build_tissuepmhc_dataset.py` |
| E14 dual-branch model | `scripts/run_tissuepmhc_auxiliary_soft_ensemble.py` |
| E15 rank fusion | `scripts/run_tissuepmhc_e15_fusion_ablation.py` |
| E17 seed ensemble | `scripts/run_tissuepmhc_e17_seed_ensemble.py` |
| E18 validation weighting | `scripts/run_tissuepmhc_e18_global_weight_selection.py` |
| E26 OOF selection | `scripts/run_tissuepmhc_e26_all_in_one.py`; `scripts/run_tissuepmhc_e26_greedy_ensemble_selection.py` |
| E27 stacking | `scripts/run_tissuepmhc_e27_stacked_generalization.py` |
| E29 CNN and OOF screen | `scripts/run_tissuepmhc_e29_multikernel_cnn_oof.py` |
| E29 five-seed extension | `E29_5SEED_PREREGISTRATION.md`; `scripts/run_tissuepmhc_e29_incremental_5seed.py` |
| Frozen E29 result | `results/tissuePMHC_e29_multikernel_cnn_5seed/` |

## 11. Method references

1. Caruana R. Multitask Learning.
2. Ruder S. An Overview of Multi-Task Learning in Deep Neural Networks.
3. Nielsen et al. NetMHCpan and HLA pseudo-sequence modeling.
4. Sener and Koltun. Multi-Task Learning as Multi-Objective Optimization.
5. Liu et al. Conflict-Averse Gradient Descent for Multi-task Learning.
6. Liu et al. Multi-gate Mixture-of-Experts.
7. Tang et al. Progressive Layered Extraction.
8. Caruana et al. Ensemble Selection from Libraries of Models.
9. Izmailov et al. Averaging Weights Leads to Wider Optima and Better Generalization.
10. Dietterich TG. Ensemble Methods in Machine Learning.
