# Premium experiment A0-A5

The status, design and implementation sequence of the follow-up experiment is recorded in the following table:
[`EXPERIMENT_CHECKLIST_A_TO_E.md`](EXPERIMENT_CHECKLIST_A_TO_E.md).

All these entrances keep the original E14a structure: a global branch, a plain trained with HLA
The experiment changes only the tissue of the global branch.
auxiliary objective.

♪ The world's branches are under supervision ♪
|---|---|---|
ZXQ0QZ HLA+ All row forecasts query ZXQ1QZ
A1ZXQ0XZ HLA only, not calculating the Tissue loss
A2 ZXQ0QZ  tissue loss only supervise `label=1`
A3 ZXQ0QZ rain-only observed-tassue multi-label BCE, unobserved organization considered 0
A4 ZXQ0QZ Supervise only observed active and current query tissue, remaining organizations mask
ZXQ0QZ rain-only Other Organizations Category III: 0/1/2+

## Three, seed.

All entrances run the same three by default:

```text
20260704 20260705 20260706
```

Every operation will reset the full random state before training the global branch and train HLA
Reset before branch. This ensures that the new count number number number numberfiler will not change the HLA
Initialization of branches, data shuffle or other random state.

Could temporarily overwrite default values with ZXQ0QZ, for example, seeing smoke test:

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\experiments\run_a5_other_tissue_count_auxiliary.py --device cuda --epochs 1 --seeds 20260704
```

## All-in-one

The following command runs all 18 training sessions in order of A0 to A5:

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\experiments\run_all_a_experiments.py --device cuda
```

Or even adjust the training cycle or catch size:

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\experiments\run_all_a_experiments.py --device cuda --epochs 25 --batch-size 512
```

## And it's isolated.

The result is not overwritten. The new result is written:

```text
extra_premium/results/experiments/<experiment_id>/seed_<seed>/
```

Each directory of the seed contains:

- `test_predictions.csv`
- `per_task_metrics.csv`
- `summary_metrics.csv`
- `run_settings.json`
- `training_diagnostics.csv`

Additional generation per experimental directory:

- `seed_summary.csv`
- `seed_aggregate.csv`

All-in-one will be generated at ZXQ0QZ after completion:

- `A_experiments_seed_summary.csv`
- `A_experiments_seed_aggregate.csv`

A3/A4/A5 Prepolymers without direct use of full data
`reported_tissues_same_hla_uniprot` or
`other_tissue_presentation_count`, the auxiliary tag is only reconstructed from the normal premium train.

## B1-B3 diagnosis-only OOF

Group B does not read premium fix test, default runs ZXQ0QZ
Three seeds, using fixed 3-old plain-grouped OOF:

♪ The experiment, the portal, the output focus, the ♪
|---|---|---|
B1 ZXQ0QZStep, tissue and other-account stratification Accuracy/loss
B2ZXQZ shared paptide encoder main–tissue/HLA gradient cosine
B3ZXQ0QZ The correct tissue tag matches the OOF tags in the fold

The use of shared portals is recommended, B1/B2 repeats the correct label model for B3 and does not repeat training:

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\experiments\run_all_b_experiments.py --device cuda
```

Default result written:

```text
extra_premium/results/experiments/B_all_auxiliary_diagnostics/
```

The main documents include:

- `b1_auxiliary_predictions.csv`
- `b1_auxiliary_metrics.csv`
- `b2_gradient_cosines.csv`
- `b2_gradient_summary.csv`
- `oof_predictions.csv`
- `per_task_metrics.csv`
- `summary_metrics.csv`
- `b3_matched_task_comparison.csv`
- Tier indicators, old indicators, training diagnostics and running settings

The correct label for B3 is used in the same initialization and batting order as the global model for the disorganized tag and shared
The same HLA plain branch. Disruption occurs only within every little old and does not modify the main task
tab. B2 closes the dropout on the help-out cold and only audits shared paptide embedding and
Encoder parameters.

## Co-C4 condition/HLA Conditional Structure

Group C continues to use A0 query-target-tassue and HLA auxiliary losses, changing only the main branch of the global:

♪ The world's gonna be so big ♪
|---|---|---|
ZXQ0QZ Current peptide encoder + task-specific head
C1ZXQZpeptide and query tissue/HLA embedding late
C2ZXQZtissue/ HLA FiLM modem + shared glassifier
| C3 | `run_c3_conditional_task_residual.py` | FiLM + shared/HLA/tissue/task residual |
C4 ZXQ0QZ C3 Remove task residual

Recommended running shared access; training of HLA plain branch only once per Seed/fold and for C2-C4
Common use:

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\experiments\run_all_c_experiments.py --device cuda
```

Default `20260704/20260705/20260706` 3-bed, 3-fold
The results are written in:

```text
extra_premium/results/experiments/C_all_conditioning_models/
```

Main output:

- `oof_predictions.csv`
- `per_task_metrics.csv`
- `summary_metrics.csv`
- `matched_c0_comparison.csv`
- `other_count_metrics.csv`
- `seen_unseen_metrics.csv`
- `per_hla_metrics.csv`
- `per_tissue_metrics.csv`
- `training_diagnostics.csv`
- `parameter_counts.csv`
- `fold_assignments.csv`
- `run_settings.json`

All global models reset random status before initialization and prior training, so that batt order and dropout
Random stream matches. The C3/C4 residual header is initialized from zero and is punished with a fixed L2. Group C does not read
premium fixed test.

## D1-D3 Organizational conditions dependent on diagnosis

All-in-one entrances in Group D only use premium Train and match C0/C2/C3
3-seed OOF:

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\experiments\run_all_d_experiments.py --device cuda
```

The portal also generates the tisue-swap fraction matrix, value tisue shufble, tissue/HLA/
The retrenchment of the reasoning of the task rediscovery and the retraining auxiary-off.
`extra_premium/results/experiments/D_all_condition_diagnostics/`.

## Pre-E biological data preparation

Pre-E is not a model experiment, and does not use an F number. It only uses premium Train, which downloads or reads.
HPA presents data, audits UniProtEnsembl maps and generates a manual tissue map template:

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\experiments\run_all_pre_e_preparation.py
```

ZXQ0QZ generated manually reviewed and passed
`--approved-tissue-mapping-csv` runs once to generate a preprocessor that can be delivered directly to E.
ZXQ0QZ, ZXQ1QZ, `tissue_mapping.csv` and
`expression_metadata.json`. Pre-E is a definitive data processing without Seed; follow-E default
Single-seed ZXQ0QZ. Time-consuming printing to terminal and not to be included in the outcome document.

## E0-E4 processing-first mechanism experiments

The new version E group is C4 as a skeleton and uses the Pair-aware BCE + ranking loss:

Number  Contents
|---|---|
E0  Group E baseline matching C4 structure
E1 real N/C plans + MHzFlurry working code
E2  parent-protein query-tassuque expression-only negative
E3  processor expressing a FiLM modem for flank processing
E4  E3 plus full model of expression

You need to prepare external features before you start running. The first step is to download from the parent UniProt in the premium train
Protein sequence and Ensembl map:

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\experiments\download_e_uniprot_inputs.py
```

Then download and freeze from Human Protein Atlas official
ZXQ0QZ (`https://www.proteinatlas.org/download/tsv/rna_tissue_consensus.tsv.zip`), copying and manual review
ZXQ0QZ. Prepares real flanks, expressions, process machine expressions, and belts.
The MSCflurry project core:

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\experiments\prepare_e_mechanism_features.py `
  --protein-fasta extra_premium\external\mechanism\premium_train_parent_uniprot.fasta `
  --uniprot-gene-map data\expression\hpa_v25_1\e_ready\protein_mapping.csv `
  --expression-table data\expression\hpa_v25_1\e_ready\expression.csv.gz `
  --tissue-mapping data\expression\hpa_v25_1\e_ready\tissue_mapping.csv `
  --expression-metadata-json data\expression\hpa_v25_1\e_ready\expression_metadata.json
```

First perform read-only audits:

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\experiments\run_all_e_experiments.py `
  --validate-inputs-only
```

After the audit, the whole group is run. By default, only single-seed ZXQ0QZ:

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\experiments\run_all_e_experiments.py --device cuda
```

Each E1-E4 model produces an extra negative comparison of reasoning with itself: processing press task
Disruption, disruption to complete pair, or the replacement of the tissue machinery with another organization.
Continuous external features only use fitting fold to align standardized parameters. Group E does not read premium test,
Default result is written in `extra_premium/results/experiments/E_all_processing_mechanisms/`.

Organisational mapping uncertainty using conservative protocol: `low_proxy` (bLood, Bone and uniical cod
The query/relative expression and the missing value of the main training, which is the most important part of the training, is the "freedom of the people" of the country.
But keep the objects not dependent on the target organization.
ZXQ0QZ is not involved in the main training.
ZXQ0QZ, reporting exact/synonym, aggregate proxy,
Low proxy and non-low-proxy subsets. ZXQ0QZ will be visible in the model,
The use to distinguish between true low expression and tissue mapping is not available.

B-E is printed only at the terminal and does not write `run_settings.json` or CSV results.
