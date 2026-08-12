# Final phase: No training analysis code

Harmonized analysis report: [FINAL_PHASE_REPORT_LEGACY.md] (FINAL_PHASE_REPORT_LEGACY.md).

This directory contains 9 separate read-only analytical portals. They read the existing human/mouse standard and peptide-disjoint OOF predictions by default, without training models; output is written in `resources/final_page/<number_name>/ `.

It is recommended that the existing environment be used:

```powershell
& E:/ancd/envs/my_pytorch/python.exe final_phase/01_compute_pairacc.py
& E:/ancd/envs/my_pytorch/python.exe final_phase/02_audit_matched_standard_folds.py
& E:/ancd/envs/my_pytorch/python.exe final_phase/03_audit_parent_protein_overlap.py
& E:/ancd/envs/my_pytorch/python.exe final_phase/04_audit_provenance_feasibility.py
& E:/ancd/envs/my_pytorch/python.exe final_phase/05_statistical_analysis.py
& E:/ancd/envs/my_pytorch/python.exe final_phase/06_build_supplementary_tables.py
& E:/ancd/envs/my_pytorch/python.exe final_phase/07_build_visualizations.py
& E:/ancd/envs/my_pytorch/python.exe final_phase/08_collect_reproducibility.py
& E:/ancd/envs/my_pytorch/python.exe final_phase/09_build_dataset_cards.py
```

It can also be run:

```powershell
& E:/ancd/envs/my_pytorch/python.exe final_phase/run_all.py
```

The numbers correspond to: PairAcc, Matched-old audio, parent-protein overlap, provenance finance, statistical tests, supplementary tables, visualization, repossibility materials, text of dataset/reproduciability/ethics.

`05_statistical_analysis.py` will execute 10,000 times task-paired bootstrap by default. If you also have to run slower peptide-component bluestrap, you can use it separately:

```powershell
& E:/ancd/envs/my_pytorch/python.exe final_phase/05_statistical_analysis.py --component-bootstrap 1000
```
