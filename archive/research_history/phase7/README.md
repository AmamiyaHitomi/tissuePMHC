# Phase 7: Humans tissuePMMHC (min_pairs=200)

This directory is the recovery experimental entry for existing human projects. It does not modify the original code in `scripts/` or reuse the original project ZXQ1QZ or `results/tissuePMHC_*` output.

The data integration rule is fixed to ZXQ0QZ (i.e., command line parameter ZXQ1QZ). The new benchmark data set is marked to ZXQ2QZ, with all results written to `results/tissuePMHC_phase7_min200_*`.

## Execute Order

```powershell
python phase7/build_human_dataset_min200.py
python phase7/run_e2_shared_heads.py
python phase7/run_e14_auxiliary_soft_ensemble.py
python phase7/run_e17_seed_ensemble.py
python phase7/run_e26_all_in_one.py
python phase7/run_e29_multikernel_cnn_oof.py
python phase7/run_e29_independent_test.py
python phase7/run_e31_peptide_disjoint_oof.py --device cuda
python phase7/build_protein_disjoint_split.py
python phase7/run_e30_interaction_oof.py --encoder e29_cnn --skip-screen
python phase7/run_e30_interaction_oof.py --encoder e30_interaction `
  --baseline-oof-predictions results/tissuePMHC_phase7_min200_e30_interaction_oof/oof_predictions.csv `
  --baseline-candidate e29_cnn_seed_20260704
```

E2, E14, E17 is a complete Min200 standard split link: E14 reading E2 indicators, E17 reading E14 branch forecast. E8/E13 comparison files for E14 are optional and are not generated with an empty matching line, without affecting training or E17.

E26 will retrain its OOF E14/E16 candidate on the same Min200 data and generate the OOF baseline required for E29; it is not the E14/E17 file that reverts to the previous standard standard split. E29 will only complete OOF preen by default; after screen, if a Phase 7 test forecast is generated, it can be extra-transmitted to `--run-test`. No entry to the OOF result of the original project (min500). E30 uses an independent protein-disjob development template to run its ZQ1QXZ matching baseline and then `e30_interaction`.

E31 is the frozen E29 train-only frozen-componted PPPD-disjoint OF fundamentalness experiment, which does not read standard test. It produces in real time every branch/ group epoch, every fold, every seed and full time, and saves the terminal records to `results/tissuePMHC_phase7_min200_e31_peptide_disjoint_oof/run.log`;old/seed/total aggregate to `timing.csv`. The official results are completed: mean task AUROC/ AUPREC for ZXQ2QZ,worth-10 AUROC for ZXQ3QZ, three old peptide/pair overlap for 0, with total time ZXQ4QZ.
