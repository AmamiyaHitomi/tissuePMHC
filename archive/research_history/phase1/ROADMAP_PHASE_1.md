# TissuePMHC Research Road Map

The present document is the main road map for the current project, highlighting:

```text
1. Which experimental line does each code file belong to?
2. Each experiment is improved on what model.
3. Each results catalogue corresponds to what is experimental.
4. How to use the current follow-up experimental number.
```

## Current mainline judgement

The current item has two lines:

```text
E2/E8 Performance Main Line
E4 Biological expression line
```

The main performance lines are:

```text
E2 shared peptide encoder + task-specific heads
→ E6 task grouping
→ E7 selective HLA/global sharing
→ E8 soft ensemble
→ E9 E2 + CAGrad
→ E10/E11/E12/E13 planned extensions
```

The biological expression line is:

```text
E3 tissue + HLA ID embedding
→ E4 tissue + HLA pseudo-sequence
→ E4b HLA ID + HLA pseudo-sequence hybrid
```

The strongest models are:

```text
E8a fixed average soft ensemble
```

It is an E2/E8 performance main line and is not an E4 bioexpression line. The E9-E13 is complete and the current conclusion remains unchanged: E8a is still the best performance master model under Standard split.

## 2. Data construction codes

Code file  Output/Use  Experimental meaning
|---|---|---|
`scripts/extract_iedb_human_mhci_ligands.py`  extracts data sources from IEDB MHC ligand original table for all subsequent experiments
ZXQ0QZ  Statistical cover-HLA-UniProt Check for sufficiency of task and protein cover
ZXQ0QZ  Builds the Tissue Speciality sample pair  Generates the second classification task sample
ZXQ0QZ  Checked peptide Length distribution  Confirmed 9-mer data is well constructed
ZXQ0QZ  Generates a sandard split  for all models
ZXQ0QZ  Generate HLA pseudo-equality table E4/E4b bio-expression line input

## Model codes and numbers

Number  Code file  Main line  Model base  Improvement of content/experimental meaning
|---|---|---|---|---|
ZXQ0QZ  Traditional baseline
ZXQ0QZ Neurocyte Baseline  Single Task Small Neuronet  Easy mode check for validity
E2 ZXQZ
E3  ZXQ0QZ  Conditional Line  peptide + Tissue ID + HLA ID  Check whether the visible condition tissue/ HLA is valid
E4  ZXQ0QZ  Biological expression line peptide + Tissue ID + HLA pseudo-sequire  Check if HLA sequence information is helpful
E4bZXQXZ Biological expression line  HLA ID + HLA pseudo-segment  Check if ID and biological sequences indicate complementarity
E5 ZXQ0XZE2 performance mainline E2 task-balanced shared headers
E6  ZXQ0XZ  E2 main energy line  E2 shared headers  to share encoded  by HLA or Tissue
E7  ZXQ0XZ  E2 global branch + E6 HLA branch  value-based hard section
E8  ZXQ0 XZ  E2/E8 Performance Main Line + E6 HLA Branch  soft integration global/HLA score
E9  ZXQ0XZ  E2 Performance Main Line  E2 shared headers  Join CAGrad Grading Conflicting
ZXQ0QZ  E2/E8 performance mains peptide experts + task sets + task headers  MMOE auto-learning task secret share
E10bZXQ0XZE2/E8 Performance Main Line  tuned MMOE configs Adjusted number of experts, extert width and date entry regularization
E11 ZXQ0XZ  E2 performance mains  E2 shared headers  DB-MTL
E12  ZXQ0XZ  E2/E8 Performance Main Line  E2 shared headers + pair_id  Added range loss
E13  ZXQ0QZ  E2/E8 Performance Main Line  E2 shared headers + auxiliary headers  Add to the Tissue/ HLA Support Prediction Task to help share encoder learning task structure

The old `scripts/run_tissuepmhc_neural_baselines.py` is the first version of E1/E2/E3 runner, which is retained for recurrence; the official priority is `run_tissuepmhc_neural_baselines_v2.py`.

## 4. Catalogue of results and meaning

Results catalogue  corresponding code  corresponding experiment  current conclusion
|---|---|---|---|
ZXQ0QZ ZXQ1QZ E0  Traditional base means AUROC approximately 0.75558
ZXQ0QZ  ZXQ1QZ  E1/E2/E3  First version results, only 1 seed
ZXQ0QZ  ZXQ1QZ  E1/E2/E3E2 means AUROC about 0.7927, the most powerful early master baseline
ZXQ0QZ  ZXQ1QZ  E4  E4 means AUROC approximately 0.7728, not exceeding E2
ZXQ0QZ  ZXQ1QZ  E4b  E4b means AUROC about 0.7824, close to E3 but weak than E2
ZXQ0QZ  ZXQ1QZ  E5  FAMO-on-E2 means AUROC approximately 0.7758, not exceeding E2
ZXQ0QZ  ZXQ1QZ  HLA grouping with local value and evident weakness of
ZXQ0QZ ZXQ1QZ E7  hard supply is not stable enough, mean AUROC is about 0.7904
ZXQ0QZ ZXQ1QZ  E8  E8a means AUROC about 0.8050, currently the strongest
ZXQ0QZ  ZXQ1QZ  E9  E2 + CAGrad not exceeding E8 but can be analysed as a gradient conflict
ZXQ0QZ  ZXQ1QZ  E10  MMoE is stronger than E9/E7/E5, but still weaker than E8
ZXQ0QZ  ZXQ1QZ  E10b  tuned MMOE not exceeding original E10 and not close to E8
ZXQ0QZ  ZXQ1QZ  E11 DB-MTL means AUROC about 0.7817, no improvement E2
ZXQ0QZ ZXQ1QZ E12  paired running loss means AUROC approximately 0.7807, weaker than E2
ZXQ0QZ  ZXQ1QZ  E13 means AUROC approximately 0.8023, significantly better than E2/E11/E12 but slightly weaker than E8

## 5. Annotations to the outcome document

File name, meaning, meaning, meaning,
|---|---|
`per_task_metrics.csv`  Detailed indicators for each seed, model, task
ZXQ0QZ  by average indicator after the concentration of the seed and model
ZXQ0QZ  More Seed Stability Statistics
`metadata.json`  Experimental parameters, input output paths, task making
ZXQ0QZ  candidate model per-task variance relative to baseline
`external_comparison_metrics.csv`  Comparison of external experiments
ZXQ0QZ E6 for each HLA/tissue group
ZXQ0QZ E7 tab for each task choose global or HLAbranch records
ZXQ0QZ  Global brand and HLA branch in E7/E8
ZXQ0QZ HLA/global ensemble weight in E8
`task_weight_history.csv` E5 FAMO dynamic weight for each task
ZXQ0QZ E9 CaGrad
ZXQ0QZ  E9 CAGrad Medium Grad Diagnosis Indicator for Conflict
ZXQ0QZ E10 MMoE for each task to add each
ZXQ0QZE11 DB-MTL dynamic weights for each task
ZXQ0QZ loss and dynamic weight diagnostic indicators in E11 DB-MTL
Training diagnostics for BCE, ranking loss and pair accuracy in ZXQ0QZ  E12
ZXQ0QZ E13 main task loss, tisse/ HLA auxiliary loss and auxiliary accuracy

## 6. Current model sequencing

As of E13, the current sorting is:

```text
E8a fixed soft ensemble
≈ E8b validation-delta clipped ensemble
>
E13 auxiliary tissue/HLA prediction
>
E8c validation softmax ensemble
>
E10 MMoE
≈ E2 sample BCE
>
E10b tuned MMoE
>
E11 DB-MTL
≈ E9 CAGrad
≈ E12 pair ranking
>
E4/E5/E6 and other earlier branches
```

Therefore:

```text
The current standard-split performance main line can be phased in.
E8a fixed above service version continues as the current best performance master model.
E13 as the most valuable mention of analysis.
E9/E11/E12 as interpretative results.
E4 lines are maintained as HLA bioexpression lines and are not the current performance mains.
```

## 7. Follow-up experimental number

Completed or already available code:

```text
E0: traditional single-task baseline
E1: neural single-task baseline
E2: shared peptide encoder + task-specific heads
E3: conditioned tissue + HLA ID
E4: conditioned tissue + HLA pseudo-sequence
E4b: HLA ID + HLA pseudo-sequence hybrid
E5: FAMO on E2
E6: HLA/tissue task grouping on E2
E7: selective HLA/global hard selection
E8: global/HLA soft ensemble
E9: E2 + CAGrad
E10: MMoE selective-sharing model on the E2/E8 line
E10b: tuned MMoE configs
E11: E2 + DB-MTL
E12: E2/E8-line paired ranking loss
E13: auxiliary tissue/HLA prediction
```

Current status:

```text
E9-E13 Completed.
E13 is significantly better than E2/E11/E12, but is still slightly weaker than E8.
So the current standard-split main line is ready to be phased in, with priority for subsequent E8 compliance analysis instead of continuing the non-differentiated stacking model.
```

## 8. Next steps

The next step, as it is most natural, is no longer to continue running the new model, but to conduct a reliability and generalized border analysis around E8:

```text
1. E8 sample_id alignment assertion.
2. E8 new control, for example, re-insemble after HLAbrabrant score.
3. peptide-disjoint / protein-disjoint split.
4. E8a/E8b extends to more seeds.
5. Global_score relevance analysis for hla_score.
```

`negative control` refers to the intentional destruction of key structures in the model and the checking of performance decline. If the E8 performance has dropped significantly after the disruption of HLAbranch, this suggests that the original E8 upgrades are indeed derived from complementary information from the global/HLAbranch.

`peptide-disjoint split` means peptide in test does not appear in train to test the generalization of the new peptide.
`protein-disjoint split` means source protein in test is not present in train and is used to test the generalization of new protein sources.

E13 The conclusions are clear:

```text
E13 over sample BCE baseline.
E13 exceeds E11 and E12.
E13 is slightly stronger than E10.
E13 is slightly weaker than E8.
About 0.30 for the reason that the HLA auxiliary accuracy is about 0.77.
```

Therefore:

```text
E8 Continue to serve as the current best performance master model.
E13 as a subsidiary re-education analysis.
PLE may be retained as an alternative, but with a lower priority than E8 priority.
```

## 9. Report narrative recommendations

The official report could be structured as follows:

```text
1. Dataset Build
2. Traditional single task
3. E1/E2/E3 Neuropathy
4. E4/E4b HLA Biological expression line
5. E5 FAMO-on-E2
6. E6 task grouping
7. E7 hard selection
8. E8 soft ensemble
9. E9 CAGrad-on-E2
10. E10 MMoE selective-sharing
11. E10b tuned MMoE
12. E11 DB-MTL
13. E12 paired ranking loss
14. E13 auxiliary tissue/HLA prediction
```

Core storyline:

```text
E2 proves that the power base is strong.
E4 proves that HLA pseudo-equality is not currently above HLA ID/E2, but has biological value.
E6 illustrates that HLA-special share has a local value.
E7 indicates that hard supply is not stable enough.
E8 illustrates that global share and HLA-special share can be the strongest models available by complementarities with softness.
E9 Check with CAGrad whether the open gradient conflict processing can further improve the E2/E8 line.
E10 Check with MMOE whether the model automatically learns.
E10b Checks whether more experts, wider experts and gate entropy recovery can narrow the gap between E10 and E8.
E11 Check with DB-MTL whether the dynamic task loss can improve the neetive transfer in the E2 mainline.
E12 Check with the paired ranking loss whether the relative order of the positive and negative samples in the pair_id can improve the E2/E8 mainline.
E13 Check with the tissue/HLA auxiliary tasks whether the oversight of the visible task structure can be improved.
```
