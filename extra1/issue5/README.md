# Issue 5: Generic pMHC comparison with MHC-only

This directory achieves item 5 of ZXQ0QZ. The code will not fine-tune the labels of this document
NetMHCpan or MHz and unify all external parts to "higher and stronger".

## Experimental Composition

External freeze baseline:

1. NetMHCpan-4.1 BA: main score ZXQ0QZ;
2. NetMHCpan-4.1 EL: main score ZXQ0QZ;
3. (a) MHCflurry preparation: main score `presentation_score`, which officially disables the flank;
4. MHCflurry effect percentile can be an additional binding sensitive result.

Internal comparison:

- human:E29-compatible position-preserving multi-kernel CNN encoder + HLA heads;
- mouse:E15-compatible peptide MLP/expert structure + H2-only gate/heads;
- Neither type of textue, use of tissue embedding, tessue auxiliary loss, tessue-MHC
  I'm not sure you're gonna be able to get a job.

## Environment

Project Python:

```powershell
E:\ancd\envs\my_pytorch\python.exe
```

NetMHCpan-4.1 requires users to get standalone UNIX packages with DTU permission. Suggested in Linux/ WSL
running. MHCflurry should fix the official stabilization version and model bindle; do not automatically upgrade during the same formal experiment
Software or model file.

## 1. Verify and create a unique query

```powershell
E:\ancd\envs\my_pytorch\python.exe extra\issue5\run_tests.py
E:\ancd\envs\my_pytorch\python.exe extra\issue5\build_queries.py
```

Expected generation:

```text
results/issue5_general_pmhc/queries/
├── human_unique_peptide_mhc.csv.gz
├── mouse_unique_peptide_mhc.csv.gz
├── human_mhcflurry_input.csv
├── mouse_mhcflurry_input.csv
├── human_netmhcpan_manifest.csv
├── mouse_netmhcpan_manifest.csv
└── netmhcpan/{human,mouse}/*.pep
```

The official data are expected to be the only peptide-MHC query for human ZXQ0QZ, mouse ZXQ1QZ.

## 2. Run MHzflurry

Installation and fixed version, downloading of the position bindle in a separate environment, for example:

```powershell
python -m pip install mhcflurry==2.2.1
mhcflurry-downloads fetch models_class1_presentation
mhcflurry-downloads path models_class1_presentation
```

If Python 3.13 and ZXQ0QZ is used
ZXQ0QZ, using project-in-compatible starters:

```powershell
$env:PYTHONUTF8 = "1"
E:\ancd\envs\my_pytorch\python.exe extra\issue5\mhcflurry_downloads_py313.py fetch models_class1_presentation
E:\ancd\envs\my_pytorch\python.exe extra\issue5\mhcflurry_downloads_py313.py path models_class1_presentation
```

The starter only maps ZXQ0XZ from Python 3.13 to the equivalent price
`shlex.quote` does not modify the MHzFlurry model or predictive code.

Send the ZXQ0QZ directory display to Runner for the last command. Examples of stable-text compatibility command:

```powershell
python extra\issue5\run_predictors.py mhcflurry --legacy-cli `
  --executable mhcflurry-predict `
  --input results\issue5_general_pmhc\queries\human_mhcflurry_input.csv `
  --output results\issue5_general_pmhc\raw_outputs\human_mhcflurry.csv `
  --metadata results\issue5_general_pmhc\raw_outputs\human_mhcflurry.metadata.json `
  --model-dir <models_class1_presentation model directory>ZXQ0QZ
```

Mouse uses the corresponding input/output repeat. If the frozen version explicitly provides a new version of the CLI, it can be omitted
`--legacy-cli` and replace execable with `mhcflurry`. Runner fixed:

```text
--no-throw --no-flanking
```

The policy predict CLI of MHz 2.2.1 does not accept `--num-jobs`, so runner defaults not to pass it
argument. Only visible settings are available when the version is confirmed.

Before being formally operational, it shall also be preserved:

```powershell
mhcflurry-predict --version
mhcflurry-predict --list-supported-alleles
mhcflurry-downloads info
```

## 3. Run NetMHCpan-4.1

Run in Linux/WSL environment that can execute standalone NetMHCpan:

```bash
python extra/issue5/run_predictors.py netmhcpan \
  --executable /absolute/path/to/netMHCpan \
  --manifest results/issue5_general_pmhc/queries/human_netmhcpan_manifest.csv \
  --metadata results/issue5_general_pmhc/raw_outputs/human_netmhcpan.metadata.json
```

Mouse manyefest repeats. Each allele runs independently, the complete output already exists is defaulted; only clear
Only fax `--force` when it needs to be overwritten. AlleListing is required to check all before it is fully run.
35 HLA and 4 H2 maps, in particular:

```text
H2-Db -> H-2-Db
H2-Kb -> H-2-Kb
H2-Kd -> H-2-Kd
H2-Kk -> H-2-Kk
```

If WSL cannot read Windows paths in manifest, rerun within WSL
ZXQ0QZ, lets the manifest save the absolute path visible for WSL.

## 4. Importing uniform score cache

MHCflurry:

```powershell
E:\ancd\envs\my_pytorch\python.exe extra\issue5\import_scores.py mhcflurry `
  --queries results\issue5_general_pmhc\queries\human_unique_peptide_mhc.csv.gz `
  --predictions results\issue5_general_pmhc\raw_outputs\human_mhcflurry.csv `
  --version 2.2.1 `
  --output results\issue5_general_pmhc\score_cache\human_mhcflurry.csv.gz
```

NetMHCpan:

```powershell
E:\ancd\envs\my_pytorch\python.exe extra\issue5\import_scores.py netmhcpan `
  --queries results\issue5_general_pmhc\queries\human_unique_peptide_mhc.csv.gz `
  --manifest results\issue5_general_pmhc\queries\human_netmhcpan_manifest.csv `
  --version 4.1 `
  --output results\issue5_general_pmhc\score_cache\human_netmhcpan.csv.gz
```

Mouse repeats the same operation. Importers do not fill missing predictions with extremes; unsupported/missing combinations are kept
It's NaN and has access to coverage audits.

## 5. External predictor evaluation

```powershell
E:\ancd\envs\my_pytorch\python.exe extra\issue5\evaluate_external.py `
  --score-cache results\issue5_general_pmhc\score_cache\human_netmhcpan.csv.gz `
  --score-cache results\issue5_general_pmhc\score_cache\human_mhcflurry.csv.gz `
  --score-cache results\issue5_general_pmhc\score_cache\mouse_netmhcpan.csv.gz `
  --score-cache results\issue5_general_pmhc\score_cache\mouse_mhcflurry.csv.gz
```

Output:

```text
results/issue5_general_pmhc/external_evaluation/
├── row_predictions.csv.gz
├── per_task_metrics.csv
├── summary_metrics.csv
├── coverage_audit.csv
├── paired_statistics.csv
├── per_task_differences.csv
└── metadata.json
```

AUROC, AUPRC and PairAcc only use the complete pair with both positive and negative lines. The main model indicator will be the same.
Full pair pool recalculates. Coverage reports both row, complete-pair, task and full-task.

Matched-standard OOF uses the same train-row pool as peptide-disjoint OOF, so freezes the outside
The stand-alone indicator for the predictor should be the same; the difference between the two columns is that the complete model predictions used for the matching comparison differ.

## 6. Increments to control generic scores

```powershell
E:\ancd\envs\my_pytorch\python.exe extra\issue5\stack_increment.py `
  --row-predictions results\issue5_general_pmhc\external_evaluation\row_predictions.csv.gz
```

The analysis is combined with:

```text
external score
external score + TissuePMHC score
```

Standard/standard OOF always uses two other outer foundations to align logist staff, and predicts help hold.
The match test staff only fits the match-standard OOF line.
or task intercept. This is a secondary increment analysis after controlling the generic score and does not replace the baseline for the original outer fraction.

## 7. MHC-only internal comparisons

Formally functioning:

```powershell
E:\ancd\envs\my_pytorch\python.exe extra\issue5\run_mhc_only.py --species human --device cuda
E:\ancd\envs\my_pytorch\python.exe extra\issue5\run_mhc_only.py --species mouse --device cuda
```

Runner overwrite:

- deterministic standard pair-grouped OOF,split seed `20260711`;
- Freezing-component peptide-disjoint folds;
- full-train → fixed test;
- Three pre-declarations;
- Five pre-declarations;
- Each protocol x seed x old checkpoint, can be safely continued.

Analysis:

```powershell
E:\ancd\envs\my_pytorch\python.exe extra\issue5\analyze_mhc_only.py `
  --predictions results\issue5_general_pmhc\human_mhc_only\ensemble_predictions.csv.gz `
  --predictions results\issue5_general_pmhc\mouse_mhc_only\ensemble_predictions.csv.gz
```

## 8. Statistical calibre

Each species x protocol x metric report:

- task-macro mean/median;
- I'm not sure what you mean by the fact that you're a man of your mind.
- Hodges–Lehmann difference;
- win/tie/loss;
- 10,000 times task trotstrap means-distant interval;
- Wilcoxon signed-rank;
- BH-FDR.

These tests are nominal task-level inference. Task sharing tisue, MHC, peptide and parent
Protein cannot be interpreted as evidence of an independent foreign force.

## 9. Pre-formal completion inspections

- (a) A record tool with a precise version, running date, complete command;
- Recording the MHCflurry model bindle path and file SHA256;
- Record the NetMHCpan data version, but do not redistribute in violation of permissions;
- Saves the original output of the supported-allele;
- Check all missing allele in `coverage_audit.csv`;
- Selects from the results of the match test or the poollist OOF;
- The external model is called frozen general-signal controls in the paper;
- Clearing out external models may overlap with pre-training data on IEDB/PIG training.
