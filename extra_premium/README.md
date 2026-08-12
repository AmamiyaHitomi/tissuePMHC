# simple test for humanpHC premium single seed

This directory only tests four stage entry points for the original human project:

1. `E0`: Five traditional baselines based on the independence of the tissue-HLA;
2. ZXQ0QZ: Sharing of peptide MLP encoder and tissue-HLA taskhead;
3. ZXQ0QZ: integration of the two branches of the global auxiliary MLP with HLA plain MLP;
4. `E29`: Final multi-kernel CNN dual-branch model.

All models are fixed by one only:

```text
20260704
```

Do not run more Seed integration, OOF, screen, & interpolating, & ;bootstrap; or complex test comparisons. Only for each script
`humanPMHC_train.csv.gz` training once and then evaluation on `humanPMHC_test.csv.gz`.

## Data is isolated from output

The input data is fixed as:

```text
data/humanPMHC_premium/humanPMHC_train.csv.gz
data/humanPMHC_premium/humanPMHC_test.csv.gz
```

All new output fixed to write:

```text
extra_premium/results/<model_name>/
```

The code will not be written in the original ZXQ0QZ, ZXQ1QZ or `results/tissuePMHC*`.

Only four basic documents are generated at each entrance:

- ZXQ0QZ: forecast scores per row;
- ZXQ0QZ: Basic indicators for each Tissue-HLA task;
- `summary_metrics.csv`: Average summary of single-line macros;
- ZXQ0QZ: Seed, data path and actual training parameters.

The underlying indicators include AUROC, AUPRC, accuracy, MCC and PairAcc.

## Run by

Run separately in the project root directory without all-in-one entry:

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\run_e0_traditional.py
```

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\run_e2_shared_heads.py --device cuda
```

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\run_e14a_auxiliary_dual_branch.py --device cuda
```

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\run_e29_multikernel_cnn.py --device cuda
```

Use 25 epochs by default. If you want to confirm that the code can run fast, you can significantly reduce the epoch, for example:

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\run_e2_shared_heads.py --device cuda --epochs 3
```

The result of reducing epoch can only be as smoke test and cannot be compared directly to the results of the original item 25-epoch.

## Annotations

- E0 Retain all five components of the original project: BLOSUM62 blog response, one-hot blog
  Repression, BLOSUM62 grandom forest, BLOSUM62 Extra Trees and BLOSUM62 Histogram
  The five models share an E0 output directory, but are distinguished in CSV by the `model` column.
- E17 and E26 did not join: E17 cores are multiple seeds and E26 cores are OOF candidates, none of which are compatible with this list.
  Seed, short time, simple fix-test requirements.
- E29 Retain original model structure and task-rank two-branch integration, but not run the original OOF screen, or do three seeds
  ensemble.
- The test data are used only for final projections and indicator calculations and are not involved in hyperparametric selection or model screening.

## NetMHCpan and MHzflurry

These models are generic pMHC forecasters that are frozen, do not read premium Train, or use this data label training.
Only the forecast for the peptide-HLA combination after the re-entry.

All queries, original output and score cache write:

```text
extra_premium/external/
```

The final underlying indicators are set out in:

```text
extra_premium/results/external_predictors/
```

Runs the following entry points separately, without all-in-one:

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\build_external_queries.py
```

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\run_mhcflurry.py
```

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\run_netmhcpan.py
```

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\evaluate_external_predictors.py
```

The installation used when the default application has completed the original project Issue 5:

- MHCflurry 2.2.1 legacy CLI;
- ZXQ0QZ bindle, disableing the planking;
- `/home/hitomi/netMHCpan-4.1/netMHCpan` in Ubuntu WSL;
- NetMHCpan-4.1b, with EL and BA scores to be exported simultaneously.

If the installation is changed, both Runners provide the corresponding command line parameters. NetMHCpan already exists by allele
Output default reuse; `run_netmhcpan.py` is given to `--force` only if it is clearly required to cover.

No uniform probabilities threshold for the outer portion, so only AUROC, AUPRC and PairAcc on the complete pair are not reported
The main results are MHCflury preparation code, NetMHCpan EL rank and BA.
rank; the rest is sensitive/EL score as an additional result.
