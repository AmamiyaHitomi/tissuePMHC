# Mouse occurrence-equal tissuePMHC intertwining

> ** Version notation:** Results produced in this directory are referred to as " Results " .
> **tissuePMHC (ref. following; mouse training set CV selection)**, with English label
> `TissuePMHC (tuned; mouse-training-CV-selected)`. No contact with original tisuePMHC
> The default configuration result is commingled to the same version.

This directory uses only ZXQ0QZ and does not overlay the results of the experiments that have been performed.

`run_tuning.py` executes the following processes:

1. Test sampling strategies and model capacity by triming up the task and pair in the training set;
2. Search for assistive losses, integration weights, learning rates and regularization around optimal configuration in phase I;
3. Sorting the validation results in all training sets and locking a configuration;
4. Only seeds 20260704, 2026005, 2026006 for locking configurations are used for full training re-training;
5. Report independent averages and line-by-line projections on the fixed test set and compare them with the results of DB-MTL already available.

Formally functioning:

```powershell
E:\ancd\envs\my_pytorch\python.exe extra_mouse_occurence_equal_dataset\adjusting\run_tuning.py --device cuda
```

Smoke test:

```powershell
E:\ancd\envs\my_pytorch\python.exe extra_mouse_occurence_equal_dataset\adjusting\run_tuning.py --device cuda --smoke
```

Every epoch, old, Seed and total run time is printed to the terminal and written to the result directory
`timing_results.csv`; complete progress is also written in `run.log`.
