# Final phase：无训练分析代码

统一分析报告：[FINAL_PHASE_REPORT_zh.md](FINAL_PHASE_REPORT_zh.md)。

本目录包含 9 个彼此独立的只读分析入口。它们默认读取现有 human/mouse standard 与 peptide-disjoint OOF 预测，不训练模型；输出统一写入 `results/final_phase/<编号_名称>/`。

建议使用现有环境：

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

也可运行：

```powershell
& E:/ancd/envs/my_pytorch/python.exe final_phase/run_all.py
```

编号对应：PairAcc、matched-fold audit、parent-protein overlap、provenance feasibility、统计检验、补充表、可视化、可复现性材料、dataset/reproducibility/ethics 文本。

`05_statistical_analysis.py` 默认执行 10,000 次 task-paired bootstrap。若还要运行较慢的 peptide-component cluster bootstrap，可单独使用：

```powershell
& E:/ancd/envs/my_pytorch/python.exe final_phase/05_statistical_analysis.py --component-bootstrap 1000
```
