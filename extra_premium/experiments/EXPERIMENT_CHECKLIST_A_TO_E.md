# HumanPMTHC Premium Experiment A-E List

Final update: 2026-07-31

## 1. Scope of the study

This list addresses only two issues, not extending to general references, external model integration or unrelated structural searches:

1. The current tissue auxiliary forecast for E14/E29 actually predicts query `target_tissue`, but no negative cases are presented in this organization and the supporting labels do not correspond to strict biological semantics.
2. The current model uses mainly peptide and task ID; the solution/ HLA selects only the last part of the text but also the tissue conditionality, let alone the tissue specific biology mechanism, which the tissue ID claims to have learned.

## 2. Description of the status

- ZXQ0QZ: Code, operation and results check completed.
- ZXQ0QZ: The code has been completed and passed via smoke test, but is not yet fully operational.
- `[-]`: Keeps a cross-reference or diagnosis, but no longer stands as a candidate for promotion on the basis of the results obtained.
- `[ ]`: Not achieved or running.
- `Standing ' : Lack of necessary external data or mapping.

## 3. Uniform experimental agreements

### Completed Group A

- Data: `data/humanPMHC_premium/humanPMHC_train.csv.gz` and frozen top.
- seeds:`20260704`, `20260705`, `20260706`.
- epochs:25.
- batch size:512.
- Global/HLA integration: fixed ZXQ0QZ.
- Random segregation: Complete randomity is replaced before training for the global branch and the HLA branch, respectively.
- observed-tissue tag: only recreates from the current premium train, without reading the full data prepolymization group.

### Follow-up Group B-E

Premium test has been observed several times and B-E cannot continue to use it to select structures or hyper-parameters.

- Development data: use premium Train only.
- Main development division: fixed 3-old fair-grouped OOF.
- Strict inspection: supplemented by peptide-concepted-component-disjoint OOF.
- The B-E code entry is running ZXQ0QZ three-bed by default.
- If you need a fast smoke test, you can overwrite it on a temporary basis via `--seeds 20260704`; the official result must be three-bed.
- All observed-tassue labels must be re-constructed in each old biting part.
- The HLA branch, training parameters, epoch and integration modalities in the same comparison must be consistent.

### Harmonization of indicators

- mean task AUROC.
- mean task AUPRC.
- mean task PairAcc.
- mean task MCC.
- worst-10 mean AUROC.
- 75 tasks won/pixed/no.
- PairAcc, `other_tissue_count=1/2/3+` stratification.
- Seen-peptide, unseen-peptide.
- Average of macros per HLA and per tessue.
- Task pairs of Bootstrap blocks.

---

## 4. Experiment A: semantic decomposition of tissue auxiliary

Purpose: To judge whether the negative case of misdirection of the supporting supervision of the present model is the main performance bottleneck.

State  Experiment  Core changes 3-seed AUROC 3-seed AUPPC PairAcc Conclusion
|---|---|---|---|---:|---:|---:|---|
ZXQ0QZ  A0  Current auxiliary baseline  All sample forecast query `target_tissue`, with HLA auxiliary  0.69820  ** 0.68596  ** 0.70293  Current default baseline
ZXQ0QZ  A1  HLA-only  Delete tessue auxiliary  0.66995  0.68399  0.70151  Delete tissue supervision without raising it and MCC significantly drops
ZXQ0QZ  A2
ZXQ0QZ  A3  observed-tassue multi-label rain-only multi-label BCE, unobserved tissues treating 0 0 0.69725 0.68389 0.70258 single-seseed advantage not recreated in 3-side
ZXQ0QZ  A4  observed-tassue masked  only monitors the active organizations and query tissue, while the rest mask  0.69721  0.68517  0.70267  is more educative but not migrated to the main task
ZXQ0QZ  A5  Number of other organizations  Forecasting the number of cross-organizations in the number of barrels in the range of the organization  ZXQ1QZ  0.69558  0.68337  0.70178  Total worst, number signal close to most of the baseline categories

### Alpha team has completed inspection.

- [x] 18 run files complete.
- [x] Each time it contains 7,500 test forecasts and 75 tasks.
- [x] All projections are limited and located in `[0,1]`.
- [x] Under three Seeds, the HLA branch of A0-A5 has a consistent track of training.
- [x] A5 random state mix has been repaired.
- [x] Complete the stratification of numbers by type, by task, by Bootstrap and by organization.

### Final conclusion of group A

1. A0 is the average best for 3-bed on AUROC, AUPRC, PairAcc and MCC.
2. A2 and A0 were very close, indicating that the deletion of negative cases of tissue loss did not yield a clear benefit.
3. A3/A4 amended the semantic language of some of the supporting labels without breaking the input and structural limitations of the main model.
4. AUROC in A5 drops to about ZXQ0QZ relative to A0 and can stop.
5. "Little case tissue aid label error" is not the main current performance bottleneck; the subsequent focus is shifted to group C's conditionality structure.

Relevant results:

- `extra_premium/results/experiments/A_experiments_seed_summary.csv`
- `extra_premium/results/experiments/A_experiments_seed_aggregate.csv`

---

## Experiment B: Direct measurement of auxiliary mission conflicts

Purpose: Not only do we compare the final scores, but we will directly judge whether the negative example of tissue auxiliary is in conflict with the main task.

The test must be exported.
|---|---|---|---|---|
ZXQ0QZ  B1  Auxiliary diagnostics separated by positive/negative cases
ZXQ0QZ  B2  Main task conflicts with the ticue auxiliary gradient  positive/negative cosine; conflict ratio; other-count stratification  Negative cosine is lower or more negative to directly support semantic conflict
ZXQ0QZ  B3  tissue tags negative control  The correct tags and the OOF differences in folds close to the indication that the tissue auxiliary is primarily used as a normal regular typographical typographical typographical typographical typographical typographical typographical typographical typographical typographical typographical typographical typographical typographical tactiles

### Group B implementation requirements

- B1/B2 uses A0 matching structures.
- B2 only uses a gradient of shared peptide encoder parameters, and cannot mix independent head parameters into cosine.
- Calculates the main BCE, tissue auxiliary and HLA auxiliary gradients, respectively.
- Disruption of labels must be confined to the fitting fold and cannot be exchanged across the fold.
- Even if B2 finds a conflict, the results of Group A's "deletion of the oversight did not improve the final performance" cannot be overridden; the two answer separately the questions of mechanism and effect.

---

## Experiment C: query tissue/HLA entry into the main model

Purpose: To involve the tissue/HLA in the calculation at the formation stage, rather than just the final choice of task head.

Group A did not find an ancillary formula better than A0, so Group C defaulted on the A0 auxiliary configuration; and A2 was used as a more semantic sensitivity comparison if necessary.

State  Number  Structure  Role
|---|---|---|---|
ZXQ0QZ  C0  ZXQ1QZ  OOF structure baseline
ZXQ0QZ  C1  peptide embedding with tissue/HLA embedding simple
`[c]`  C2  tissue/HLA FiLM or gating modeping peptide means  Validity of the information entry feature extraction of the condition information
ZXQ0QZ C3 Shared  share/shared/HLA/tissue/task priority  master candidate; balance sharing and task localization
ZXQ0QZ  C4  C3 removes the task residual  to determine whether the independent tarsk composition is still necessary

### Group C fixed comparison

```text
C0 → C1 → C2 → C3
                 C4 melts.
```

### Team C must control.

- Co-C4 uses exactly the same OOFs and Seeds.
- The parameters must be reported; if necessary, a comparison between parameter-matched is added.
- The tissue/HLA embedding cannot carry the held-out statistics directly.
- C3 task reversion needs to be regularized to avoid re-degradation into 75 unshared task head.
- The same peptide conditions under different query tissues indicate that they must be different in practice and confirmed through unit testing.

### C-group promotion conditions

- AUROC and PairAcc are superior to C0 at the same time.
- AUPRC cannot decline significantly.
- Unseen-peptide and small samples task must not be systematically degraded.
- At least the same direction is found on most Seed and most task.
- Group D ' s counter-fact diagnosis can only be interpreted as using the tissue condition.

---

## Experiment D: Verify whether the model actually uses the organizational conditions

Purpose: To distinguish between "models enter tessue ID" and "models do rely on the tessue condition for predictions".

Only for C0, C2 and C3; if C2/C3 does not exceed C0, the extension of Group D is stopped.

The  state, , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
|---|---|---|---|---|
ZXQ0QZ  D1  Tissue-swap Counterfactual Test  Fixed peptide/HLA, toggle query tissue array  Report whether the organization is changing direction consistent with observed tissues
ZXQ0QZ  D2
ZXQ0QZ  D3  Close model fraction  Turn off indicator changes after Tissue, HLA, task residual, auxiliary  Positioning final revenue source

### Group D attention.

- The C0 switch to task head also changes scores, so that the "change of scores" is not in itself evidence of a mechanism.
- D1 must be interpreted in conjunction with the real observed-issue orientation, D2 shuffle and D3 integration.
- Only valid input at the reasoning stage is used for the time of thesis-swap.
- Group D can only prove that the organizational conditions are dependent and that biological mechanisms cannot be alone.

---

## 8. Experiment E: features of the organizational mechanism of the processing-first

Note: The acquisition and audit of HPA, UniProt and artificial tissue mapping were for `Pre-E` data preparation, not
Independent F experiments; model experiments are still starting with E0. Pre-E is a definitive treatment, not set.

Purpose: Three-seed top C4 bone, first added to the peptide that can be distinguished from two peptide in proteins
flank/processing information, which then tests the relationship between the parent interpretation and its interaction with the tissue processing environment.
All candidates are to use the same pair-aware BCE + rang loss, HLA/tissue reversion and not to use
The list of names is used for the list of items.

State  No.  Experiment  Enter or change  Main problem
|---|---|---|---|---|
ZXQ0QZ/ Data Block  E0  matchd C4 base  C4 structure + Group E unified Pairwise protocol  Establishment of a strictly matching baseline
ZXQ0QZ/ Data Block  E1flank processing N/Cflanks, Location, MHCFlurry processing scorre peptide Process Information Can improve PairAcc directly
ZXQ0QZ/ Data Block  E2  parent expression observation comment comment query, cross-organizational mean, relative expression and missing mask  Validation of the same protein under
ZXQ0QZ/ Data block  E3  process-processing interposition  PSMB/TAP/ERAP expression flank
ZXQ0QZ/ Data Block  E4  full metanismE3 + authentic expression  only effective after interacting with the process

E1-E4 automatically produces negative matching: disassembly by task, disruption by complete pair,
and tissue-machinery swap. Negative contrasts change only the help-out input, do not retrain, do not change the label.

### Group E data audit

- [ ] Determine the expression data version, download date and licence.
- [ ] Establish manual audit mapping of the premium tissue to the expression database.
- [ ] Express expression with a level of gene, transcript or protein.
- [ ] Audit of the mapping coverage and multi-linkage of UniProt to Gene/Protein.
- [ ] Download and freeze the parent protein FASTA, audit canonical/second access.
- [ ] Report peptide's locational coverage of the exact and unique in the parent levelence.
- [ ] Retention of the missing/mbiguity sign is not possible, isoform is ambiguous and repeated.
- [ ] Recalculate the MHCflurry process using the real N/C flanks.
- [ ] Saves the missing ratio and always inputs missing mask.
- [x] ZXQ0QZ tissue expression and processing machines are used as missing values in the main training; original proxy only
  A comparison of reasoning sensitivity.
- [x] Keep exact/synonym, aggregate layer and low layer and output on a sample-by-sampling basis
  `mapping_quality_metrics.csv`.
- [x] `machinery_missing_fraction` visible entry model.
- [ ] The expression of the reclassification uses only the information permitted by training.
- [ ] UniProt ID is not directly a rememable category feature.
- [ ] Leaks audits of study, samples and organizational sources.
- [x] All positive or negative pair's target tisue, HLA and parent UniProt are verified to be the same.
- [x] The code enforces the checking of the parent-expression in the pair.

### Group E conclusion request

The following conditions are also met to support the "process/expression mechanism providing effective information":

1. E1/E3/E4 exceeds the match E0 in OOF; E2 is not considered a failure.
2. The resulting disruptions have disappeared or significantly diminished.
3. Unseen-peptide and peptide-disjoint OOF still have benefits.
4. The proceeds are not concentrated in just one organization or a small amount of large task.
5. The missing value subsets and the complete map subsets are in the same direction.

---

## 9. Recommended order of implementation

### Current status

- [x] A0-A5 completed and reached 3-seed conclusions.
- [x] B1-B3III was formally run.
- [x] C0-C4III-Eed officially run complete.
- [x] D1-D333-bed officially run complete.
- [c] Processing-first E0-E4 code completed and complete with process synthesis of smoke test;
  The synthesis results were deleted and the real FASTA/expression data and artificial tissue mapping still blocked the formal operation.

### Next Order

1. B1: Supplement positive and negative cases of supplementary diagnosis.
2. B2: Direct measurement of gradient conflicts on shared encoder.
3. B3: Complete the sussue-label shuffle comparison.
4. C0: Establish a strictly matching baseline for the train-only OOF structure.
5. C1: Simple matching required.
6. C2: Conditional.
7. C3/C4: Conditions shared, disabilities and their elimination.
8. Only C2-D3 is executed when C2 or C3 exceeds C0.
9. Complete audit of UniProt sequence, peptide positioning, HPA expression and manual tissue mapping.
10. E0E1E2E3E4; only candidates pass negative contrasts before 3 seeds.

## 10. Files to be saved for each experiment

- `oof_predictions.csv`
- `per_task_metrics.csv`
- `summary_metrics.csv`
- `other_count_metrics.csv`
- `seen_unseen_metrics.csv`
- `per_hla_metrics.csv`
- `per_tissue_metrics.csv`
- `training_diagnostics.csv`
- `run_settings.json`
- fold assignments
- Data file shash
- Code version or script shash

## 11. Current overall judgement

Group A has indicated that the separate correction of the tissue auxiliary label is not sufficient to improve the prediction effect.

1. (b) Group B is used to answer whether there is a measurable gradient conflict for the auxiliary task;
2. (b) Group C for the resolution of problems expressed by the absence of access to peptide;
3. (b) Use group D to confirm whether the model is truly dependent on organizational conditions;
4. Introduce a team E to the flank processing, tissue processing machines and a parent interpretation, distinguishing statistics from statistics
   Conditionalization and biological mechanisms.
