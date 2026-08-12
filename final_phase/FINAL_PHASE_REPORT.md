# TissePMHC Final Phase No training analysis report

Update: 2026-07-17
Status: 01-09 Default analysis completed; peptide-component bluespoltstrap** not yet produced**.
Data boundaries: This report compares the same train fair pool between standard plain-grouped OOF and agreed-component peptide-disjoint OOF, and does not subtract internal cross test from the target OFF.

## 1. Scope and products of implementation

Nine of the `final_phase/` entrances are actually operational, with the default analysis taking approximately 93 seconds, and do not include any model training:

1. PairAcc;
2. standard/strict task-fold size matching audit;
3. parent-UniProt overlap audit;
4. PMID/study/assay/date provenance feasibility audit;
5. task-paired bootstrap, Wilcoxon, Hodges–Lehmann, win/tie/loss and BH-FDR;
6. The main table is with the General Security CSV.
7. (a) Standard/standard visualization;
8. Environment, parameter volume, time and documentation hash;
9. Drafts fordaset card, reproducability state and ethics/intended-use.

All machine generation results are located at `results/final_phase/`. The uniform code entry is:

```powershell
& E:/ancd/envs/my_pytorch/python.exe final_phase/run_all.py
```

Note: `run_all.py` uses `--component-bootstrap 0` and does not run copontstrap.

## 2. Standard and peptide-disjoint main result

| Species | Protocol | Tasks | Accuracy | Mean AUROC | Mean AUPRC | F1 | MCC | Worst group AUROC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Human | Standard OOF | 157 | 0.76051 | 0.82795 | 0.81436 | 0.76319 | 0.52143 | 0.65616 |
| Human | Peptide-disjoint OOF | 157 | 0.69801 | 0.76520 | 0.74521 | 0.70242 | 0.39649 | 0.63817 |
| Mouse | Standard OOF | 24 | 0.77953 | 0.83922 | 0.83158 | 0.77806 | 0.55987 | 0.71012 |
| Mouse | Peptide-disjoint OOF | 24 | 0.68771 | 0.75293 | 0.73222 | 0.68680 | 0.37608 | 0.64808 |

The species of Human AUROC/AURPC is ZXQ0QZ for the start-up - standard gap; the mouse is `-0.08629/-0.09936`. The two species are still significantly above random sorting under unseen-peptide conditions, but the entity of the standard split will clearly be used to increase the performance estimates.

Source table: `results/final_phase/06_tables/table_4_standard_strict_summary.csv`.

## 3. PairAcc

PairAcc compares positive and positive scores for each `pair_id`; positive scores are higher by 1, equal points to 0.5.

| Species | Standard PairAcc | Strict PairAcc | Strict − standard | Strict worst task | Strict median task |
|---|---:|---:|---:|---:|---:|
| Human | 0.82323 | 0.77026 | -0.05297 | 0.61059 | 0.76471 |
| Mouse | 0.82112 | 0.75962 | -0.06150 | 0.58590 | 0.78645 |

PairAcc is in the same direction as AUROC: strict segregation reduces the sorting performance, but two species still retain a ratio of about 0.76 - 0.77.

Source table: `results/final_phase/01_pairacc/pairacc_summary.csv`.

## 4. Matched-standard gold audit

| Species | Task-folds | Exact held-size matches | Exact match rate | Mean absolute pair difference | Maximum difference | Mean relative difference |
|---|---:|---:|---:|---:|---:|---:|
| Human | 471 | 333 | 70.70% | 0.293 | 1 pair | 0.377% |
| Mouse | 72 | 48 | 66.67% | 0.333 | 1 pair | 0.432% |

Standard matches the maximum size of one pair for each task, old-held-out. Therefore, there is a reasonable size-matched descriptive control for the standandard OOF; there is still a need to recognize that the gap is a different sample composition and cannot be interpreted as a causal effect of pure, no fold-compostion effects.

In the head-out unit peptides of Standard OOF, humans appear simultaneously in the little; three-fold pair overlap with the peptide overlap at every discount ZXQ0QZ, mouse about `71.43%–72.12%`.

Source table:

- `results/final_phase/02_matched_fold_audit/matching_summary.csv`;
- `results/final_phase/02_matched_fold_audit/protocol_overlap_audit.csv`;
- `results/final_phase/02_matched_fold_audit/standard_vs_strict_task_fold_comparison.csv`.

## 5. Parent-protein overlap

Peptide-disjoint does not amount to protein-disjoint.

| Species | Protocol | Held-out unique parent proteins seen in fitting | Held-out rows with seen parent protein |
|---|---|---:|---:|
| Human | Standard | 90.64% | 96.92% |
| Human | Peptide-disjoint | 65.26% | 75.42% |
| Mouse | Standard | 80.69% | 87.36% |
| Mouse | Peptide-disjoint | 14.37% | 13.18% |

Human project still has a significant profile, so 0.76520 can only be interpreted as seeing-task/unseen-peptide generalization. Mouse project is low, but still not strict protein-disjoint.

Source table: `results/final_phase/03_parent_protein_overlap/protein_overlap_summary.csv`.

## 6. Task-level statistical analysis

Default statistics use fixed seed ZXQ0QZ, 10,000 times task-paired bootstrap. BH-FDR has harmonized correction on 8 of the four indicators for two species.

### 6.1 Human

| Metric | Standard | Strict | Mean delta | Median delta | Hodges–Lehmann delta | Win/tie/loss | Strict 95% CI | Delta 95% CI | FDR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AUROC | 0.82795 | 0.76520 | -0.06275 | -0.04836 | -0.05523 | 12/0/145 | [0.75511, 0.77519] | [-0.07283, -0.05336] | 1.29e-22 |
| AUPRC | 0.81436 | 0.74521 | -0.06914 | -0.05091 | -0.06213 | 10/0/147 | [0.73517, 0.75530] | [-0.07990, -0.05881] | 1.29e-22 |
| Accuracy | 0.76051 | 0.69801 | -0.06250 | -0.04545 | -0.05631 | 17/2/138 | [0.68926, 0.70693] | [-0.07265, -0.05286] | 2.12e-22 |
| MCC | 0.52143 | 0.39649 | -0.12494 | -0.09072 | -0.11247 | 18/0/139 | [0.37892, 0.41440] | [-0.14527, -0.10567] | 2.12e-22 |

### 6.2 Mouse

| Metric | Standard | Strict | Mean delta | Median delta | Hodges–Lehmann delta | Win/tie/loss | Strict 95% CI | Delta 95% CI | FDR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AUROC | 0.83922 | 0.75293 | -0.08629 | -0.08322 | -0.08592 | 2/0/22 | [0.72300, 0.77991] | [-0.11062, -0.06322] | 6.81e-07 |
| AUPRC | 0.83158 | 0.73222 | -0.09936 | -0.10399 | -0.10019 | 2/0/22 | [0.70401, 0.75834] | [-0.12928, -0.07058] | 1.67e-06 |
| Accuracy | 0.77953 | 0.68771 | -0.09182 | -0.08040 | -0.09021 | 1/0/23 | [0.66247, 0.71092] | [-0.12069, -0.06523] | 4.77e-07 |
| MCC | 0.55987 | 0.37608 | -0.18379 | -0.16322 | -0.18019 | 1/0/23 | [0.32572, 0.42242] | [-0.24145, -0.13056] | 4.77e-07 |

The result supports the "wide-ranging and consistent decline" of the Standard & Stric, rather than being driven by a small number of anomalies. Statistical visibility does not equal clinical or external validity; task Bootstrap does not include differences in re-training, different split seed or data source changes.

Source table: `results/final_phase/05_statistics/paired_statistical_tests.csv`.

## 7. Peptide-component bluespool status

There is no current **pptide-component standard bootstrap **:

```json
{
  "human": {"status": "not_run"},
  "mouse": {"status": "not_run"},
  "component_bootstrap_repeats": 0
}
```

As of 2026-07-17 19:33:55, ZXQ0QZ is still `not_run`, and there is no second outcome document in the warehouse. The task-bootstrap CI cannot be misspelled as the component-cluster CI.

Correct to-run command:

```powershell
& E:/ancd/envs/my_pytorch/python.exe final_phase/05_statistical_analysis.py --component-bootstrap 1000
```

Do not execute the default `run_all.py` after the run-off, otherwise the current implementation will cover the JSON with `--component-bootstrap 0`. If the official draft uses the component-cluster CI, this section should be replaced with a reference to the repeats and seed in the report.

## 8. Provenance feasibility

Checked the human/mouse trade data and two profiled data. All documents do not have candidate fields such as PMID, study ID, sassay ID, publicisation/submission date, so it is not possible to recreate the stuff based on the current product.

The paper should read: The current processing product is not kept enough study-level proposal to build a reliable study-disjoint plain; this is a clear data limitation rather than replacing the study-disjoint with a peptide-disjoint.

Source: `results/final_phase/04_provenance_feasibility/provenance_feasibility.json`.

## 9. Recoverability and computational information

- Python:3.13.11;
- PyTorch:2.10.0+cu126;
- CUDA runtime:12.6;
- GPU:NVIDIA GeForce RTX 4060 Laptop GPU,8,188 MiB;
- NumPy/Pandas/scikit-learn/SciPy:2.4.1/3.0.0/1.8.0/1.17.0;
- (a) Human E29: global average 160,921 tables; 35 HLA plain models total 4,672,733; and all separate training models total 4,833,654;
- Mouse E33:68,475 parameters;
- Human E31 trade total time: 5,425,718 seconds (1h 30m 26s);
- pak GPU memory: original operation not recorded and cannot be accurately restored ex post facto;
- git committee: the current environment is not solved, the name of which is ZXQ0QZ in the manifest;
- Data, predictions and split files SHA256 are written in `results/final_phase/08_reproducibility/file_hashes.csv`.

Source: `results/final_phase/08_reproducibility/reproducibility_manifest.json`.

## 10. Draft figures, tables and documents

### 10.1 Figure

`results/final_phase/07_figures/` Generated:

- `01_standard_vs_strict_scatter.png`;
- `02_task_delta_distribution.png`;
- `03_mhc_group_gap.png`;
- `04_parent_protein_overlap.png`;
- `final_phase_figures.pdf`;
- `figure_source_task_comparison.csv`.

### 10.2 Table

ZXQ0QZ has generated nine CSVs for benchmark statistics, standard/standard training, PairAcc, old matching, overlap aid, protein overlap, statistical tests, task metrics and training time.

### 10.3 Draft Documents

`results/final_phase/09_dataset_cards/` Generated:

- `DATASET_CARD.md`;
- `REPRODUCIBILITY_STATEMENT.md`;
- `ETHICS_AND_INTENDED_USE.md`.

## 11. Consolidated conclusions

1. Human and Mouse retain AUROC ZXQ0QZ and PairAcc `0.77026/0.75962` under conditions of peptide-disjoint, indicating that the signal seen-task/unseen-peptide is still available.
2. The AUROC, AUPRC, PairAcc and MCC of Standard & Stric have all declined significantly, and the need for a task-level win/loss is widely consistent, indicating that the peptide of the standard spliter will systematically overestimate the generalization of the entity.
3. The task-old help-out size is the maximum difference of 1 pair, which supports the descriptive control that already exists as a high match.
4. Human project still has 65.26% unique parent-protein overlap, which cannot be extrapolated to unseen-protein capacity; use overlap is lower but not zero.
5. The current data cannot restore the study/PMID profile, and therefore cannot claim that the study-disjoint or independent source is authenticated.
6. task-bootstrap and Wilcoxon/FDR have been completed; paptide-compont standard bootstrap is not yet completed and cannot be quoted in the official draft.
7. Until the strict baselines, external pMHC predictors and cores are completed, it cannot be claimed that the current master model is still superior to all alternatives under the unseen-peptide agreement.

## 12. Still to be completed

- peptide-component cluster bootstrap;
- Human/mouse start baselines and external pMHC predicors;
- (a) the recovery of the strigger array;
- shuffled tissue/branch negative controls;
- (b) The presentation, layout and body integration of papers that can be continued without training;
- If you want a peak GPU memory, you can only run a representative operation with a clear mark as profiling and not disguise it as a original experimental record.

