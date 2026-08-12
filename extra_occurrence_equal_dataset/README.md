# Human experiment-equival

This directory is only used for the new human data set.
ZXQ0QZ, `results` under the project root directory, is old data results,
This set of runners will actively refuse to read it.

## Fixed agreement

- Training session: `data/humanPMHC_occurence_equal_dataset/humanPMHC_train.csv.gz`
- Test set: `data/humanPMHC_occurence_equal_dataset/humanPMHC_test.csv.gz`
- seeds:`20260704 20260705 20260706`
- Total task: 77
- HLA pseudo-sequence:`data/processed/hla_pseudo_sequences_occurrence_equal.csv`
- New remaining experimental output: `results/v7_full_rerun`

New data completed and directly reused are E0, E2, E14a, E29 and external precursors;
No rerun or copying of results from old data sets. E14b and E17 require E14 branch forecast, which was previously E14a
The running only keeps integration scores, so ZXQ0QZ recalculates these shared branches, but in the paper
E14a Lines still use new data results that have been completed.

## Command

Run all missing entries:

```powershell
E:\ancd\envs\my_pytorch\python.exe extra_occurrence_equal_dataset\run_v7_remaining_experiments.py --device cuda
```

The same command is used for breakpoints; experiments with complete contract and submary skip automatically. View the plan and then
_Other Organiser

```powershell
E:\ancd\envs\my_pytorch\python.exe extra_occurrence_equal_dataset\run_v7_remaining_experiments.py --dry-run --device cuda
```

Run one method alone:

```powershell
E:\ancd\envs\my_pytorch\python.exe extra_occurrence_equal_dataset\run_human_experiment.py hla_pseudoseq --device cuda
```

Source of the synthesis of papers after all has been completed:

```powershell
E:\ancd\envs\my_pytorch\python.exe extra_occurrence_equal_dataset\aggregate_v7_paper_results.py
```

Prints seed and & epoch information per source trainer; the total package is written separately
ZXQ0QZ, log saved in `run_logs`.
