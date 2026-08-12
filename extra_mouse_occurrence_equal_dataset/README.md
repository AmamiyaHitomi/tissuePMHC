# Mouse occurence-equal experiment

This directory uses the user-provided occurence-equival trade/test files to keep the same-label, or intellectual cross-valued or peptide-disjoint splits. All training is done using seeds ZXQ0QZ, ZXQ1QZ, `20260706`.

## Basic experiments

ZXQ0QZ runs E0, E2, E14a and E29. External forecasters rate only on fixed test sets, including MHzFlurry 2.2.1 and NetMHCpan 4.1b. Results are available at ZXQ1XZ, ZXQ2XZ, `results/e14a_auxiliary_dual_branch`, `results/e29_multikernel_cnn` and `results/external_predictors`.

## V7 Missing Experiment Relay

Mouse entrance:

```powershell
E:\ancd\envs\my_pytorch\python.exe extra_mouse_occurence_equal_dataset\run_v7_remaining_experiments.py --device cuda
```

The entrance is fixed:

- `data/mousePMHC_occurence_equal_dataset/mousePMHC_train.csv.gz`
- `data/mousePMHC_occurence_equal_dataset/mousePMHC_test.csv.gz`
- seeds `20260704 20260705 20260706`
- New Output Directory `extra_mouse_occurence_equal_dataset/results/v7_full_rerun`

It does not read or overwrite the results of the old data set. E0, E2, E14a, E29 and external forecasters already completed in this occurence-equivalencies directory are used. H2 pseido-equality uses four 34-bit sequences from NetHCpan 4.1b version resources; derivative tables, source files Hashi and aliases are recorded in `data/processed/h2_pseudo_sequences.csv` and `h2_pseudo_sequence_provenance.json`, respectively.

Each training log is kept in `results/v7_full_rerun/run_logs`, containing each epoch, Seed and running time. Full running takes time to aggregate at `results/v7_full_rerun/timing_results_complete.csv`.

## Summary of the results of the paper

```powershell
E:\ancd\envs\my_pytorch\python.exe extra_mouse_occurence_equal_dataset\aggregate_v7_paper_results.py
```

The only source of the paper is `results/v7_full_rerun/paper_results`. Data Hashi, Seed, each source and the refill contract are recorded in `paper_results_provenance.json`.
