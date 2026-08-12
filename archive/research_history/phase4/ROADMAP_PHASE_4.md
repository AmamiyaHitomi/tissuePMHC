# Phase 4 Roadmap: Mouse model migration, indicating enhancement and freezing confirmation

Update: 2026-07-13
Status: Planning completed, not implemented
Number range: E7 for the continuation of the Phase 3 starting with **E8**

## 1. Phase 4 Objective

Phase 4 no longer adds complex multitask structures without distinction, but moves the most well-documented and compatible mouse data features of the human tissuePMHC project to the mouse project on the current ZXQ0QZ benchmark.

Four questions were answered at this stage:

1. E3b independently sees projected a further reduction in the difference between low-resource task?
2. Can the human E29 multi-kernel CNN peptide encoder improve mouse 9-mer motif?
3. Global sharing and H2-specific indicate that it is possible to complement each other by means of soft foundation, not by having a win grouping?
4. H2/ tissue auxiliary representation can enhance mouse sharing expressions and add them to the soft user, CNN and seened additional?

The objective of Phase 4 is not to retain all candidates, but to screen a frozen structure through a rigorous train-only OOF, which is finally pre-registered only once with a 5-seed confirmation.

## Freezing of benchmark and existing evidence

- Data: `data/mousePMHC/mousePMHC_train.csv.gz`.
- Threshold: `min_pairs > 200` when building data; 100 spaces test for each task.
- Current size: 24 tissue-H2 taskes, 13 tissues, 4 H2 reductions.
- Scale of training: 6,766 fairs, 13,532 rows; median 264.5 per task 124 - 470 train days.
- Fixed test cannot be read during the Phase 4 model development and OOF filter.
- Fixed OOF: 3-old plain-grouped, standard seen ZXQ0QZ.
- Initial training: ZXQ0QZ, ZXQ1QZ, `20260706`.

Current anchor:

OOF means task AUROC State
|---|---|---:|---|
BLOSUM62 Random Forest  0.7530  Traditional single-mission baseline
E1  share all of the peptide encoder + task headers  0.8057  for humans
E3b  task-balanced Factorized MMOE 08148(3-seed mean) Phase 4 single model main anchor

E3b sees three references to task AUROC for 0.8164, 0.8110, 0.8171, Seed SD for 0.0033, indicating that independent seed ensemble has a clear range compression space.

Phase 3 has proven that H2 hard grouping, PLE, FAMO and TAG hard grouping do not exceed E3b. Therefore, Phase 4 does not repeat these directions again, nor does it prioritize the relocation of CAGRAD, Nash-MTL, Auto-Lambda, SWA, snapshot Emble or complex stacking.

## 3. Uniform assessment agreements

### 3.1 OOF and data access

- All models are selected using only the Train-only OOF.
- Each help-out pair can only be predicted by a model that has not been seen.
- Output must cover all training lines in each seed and `(seed, sample_id)` should not be repeated.
- Test runner separates from OOF runner; code cannot read the match test until the candidate has passed the final pre-registration threshold.
- All integrations must be based on a sample-by-sampling OOF projection based on ZXQ0QZ, ZXQ1QZ, `mhc_restriction`.

### 3.2 Key indicators

- Main indicator: Mean task AUROC.
- Key sub-indicators: mean task AUPRC, work-6 means AUROC.
- Diagnostic indicators: per H2 means AUROC, per task AUROC, seed SD, parameter and training time.
- Each official comparison reports a task-by-tek pairing and a task Bootstrap 95% CI.

### 3.3 Universal threshold for promotion

The structural candidates must meet both their pre-declared needs:

1. means task AUROC upgrade at least `+0.0030`;
2. means task AUPRC does not drop more than ZXQ0QZ;
3. The first 6 means AUROC does not drop over ZXQ0QZ;
4. Any H2 group means AUROC does not drop more than `0.0100`;
5. At least 14/24 tasks' AUROC is in the right direction.

For purely integrated or increased seed low-risk candidates, use a smaller but prefixed threshold: mean task AUROC at least `+0.0010` and mean AUPRC and World-6 do not decline above `0.0010`.

The threshold is used only to determine whether to move to the next stage and not to observe the change after the match test.

## E8–E15 Official experimental sequence

Name  Human project source  Core comparison  Role
|---|---|---|---|---|
E8  E3b 3-seed protection E17  E3b single seed mean vs III seed average per sample Phase 4
E9  Multi-kernel CNN shared headers E29 encoded CNN-E1 vs matched E1  sequestered authentication CNN encoded
E10  Multi-kernel CNN Factorized MMOE  E29 encoder + mouse E3b CNN-E3b vs matched E3b  indicates enhanced lead candidate
E11 Global/H2 soft sharing E8 global E3b + H2 brand vs global E3b
E12  H2/tissue auxiliary examined mode  E13  auxiliary-E1 vs matched E1  Isolated auxiliary supervision
E13  Auxiliary Factorized MMOE  E13 + Mouse E3b  Auxiliary-E3b vs Matched E3b  Auxiliary Supervisory Candidate
Auxiliary global + plain H2 soft E14a
E15  Freezing the winning structure 5-said E17/E295-seed5-seed vs and structure 3-seedPhase 4 Final confirmation

### E8:E3b 3-seed prediction ensemble

Purpose: To build the lowest cost and strongest Phase 4 anchor point using the OOF forecast of three Seeds using the E3b first.

Achieved:

- Average rights for sample-by-sample probabilities, etc. for three seeds; no re-training, no-selection, no weighting.
- Seed mean, third document statement mean, and list-wise pre-declarations meant.
- Defaults the main candidate is probability means; rank means only as a fixed digestion, and is not allowed to use test selection.

Decision-making: If the average of 3-seed indicators reaches the low risk promotion threshold relative to the three single ones, E8 replaces the average of E3b single sees as the subsequent integration with the final E15 Eemble baseline.

### E9:Multi-kernel CNN shared heads

Purpose: Change the peptide encoder only to separate the human E29 main indicator of gain to be moved to mice.

Structure:

- Keep the task-specific liner headers for E1 and the training protocol.
- Replace `Embedding -> Flatten -> MLP` with position-presenting multi-kernel Conv1d.
- The first version is fixed at the Kernel sizes ZXQ0QZ, 32 chennes per kernel, dropout 0.2.
- Volume output saves position information before entering a small MLP without doing the global max popping.

Matched baseline: A double-run E1 using the same folds,seeds,epochs, batts sampling and comparable parameters.

Decision-making: E9 only implements E10 through the generic structural threshold. If E9 fails, stop CNN mainline, not chasing OOF by expanding the chennels, adding the kenel or repeatedly interrogating the OOF.

### E10:Multi-kernel CNN Factorized MMoE

Purpose: Verify whether CNN motif indicates that it is superimposed with E3b soft optional sharing.

Structure:

- Keeps E3b's task-balanced batches, 3 experts, tessue/H2 conditioned date and task headers.
- Replace shared paper encoder only with the frozen CNN configuration of E9.
- Not to adjust the number of times, or to adjust the number of times, or to add value or loss weighting.

Matched baseline: same olds/seeds under E3b.

Decision-making: If E10 passes the generic structural threshold, enter the final pool; otherwise the E9 encoder conclusions are retained, but the CNN-E3b combination is not retained.

### E11:Global/H2 soft rank fusion

Purpose: To test global sharing and H2-specific indicating complementarity and avoid loss of information on the Phase 3 hard grouping.

Phase I does not train new structures, priority is given to the re-establishment of a strictly aligned OOF:

- Global branch: E8, or a stronger E10 3-seed forecast adopted at the time.
- H2 Branch: Phase 3 E2 H2-grouped encoder; if its OOF seed is insufficient, only the same seeds are filled, without changing the structure.
- Converts each tissue-H2 task to percentile rank, and fixes the ZXQ0QZ average.
- Probability means as a way to accommodate; no combination test is used to select integration rules.

If the integration of E2 fails to be repeated, a pre-declared light H2 default version can be executed; it cannot be extended to large per-task applications.

Decision: The low risk integration threshold is maintained. If the cross-fix 0.5 rank foundation fails, stop the branch and do not search for per-task weights.

### E12:H2/tissue auxiliary shared model

Purpose: To isolate the auxiliary supervision itself, rather than to change the auxiliary, MMoE and the two branches simultaneously at the beginning.

Structure:

- The base seat is made E1 shared encoded + task heads.
- Add H2 auxiliary head and tissue auxiliary head.
- Initial fixed ZXQ0QZ, ZXQ1QZ reflecting the a priori strength of H2 motif over the tissue.
- The main task is still the original sample-level BCE, not the FAMO or other dynamic weights.

H2/tissue auxiliary accuracy, ancillary gradient and a main task gradient cosine, and H2 sets of gains must be reported.

Decision-making: E12 Implementation of E13 is done only through the generic structural threshold. If the overall upgrade is insufficient but there is a stable opposite direction in H2-Db/Kb/Kd/Kk, only one `H2-only auxiliary` pre-declaration is allowed to be implemented.

### E13:Auxiliary Factorized MMoE

Purpose: Add the confirmed auxiliary signal of E12 to the current strongest single model E3b.

Structure:

- The base is E3b or E10 using the frozen CNN-E3b encoder when winning.
- Only the Auxiliary configuration that E12 won.
- MMOE, Gate, Sampling and training super-parameters remain unchanged.

Matched baseline: same encoder, folds and seeds, but auxiliary loss weight 0.

Decision-making: access to the E14/E15 pool through the generic structural threshold; otherwise, auximary will only remain the institutional outcome of E12.

### E14:Auxiliary global + plain H2 soft ensemble

Purpose: To migrate the core idea of E14a for humans, while re-engineering for mice H2 heterotoxicity, rather than directly copying old 5-task E14a.

Branch:

- Global branch: E13; if E13 is not passed, use the most powerful freeze in E10 or E8 at the time.
- H2 Branch: plain H2-grouped fanch, not add auxiliary loss.
- Integration: task-wise percentile-rank fix fixed by weight ZXQ0QZ.

E14 Allows only the combination of the OOF freezing branches that have previously been completed separately, and cannot re-search encoder, auxiliary weights or per-taskfosion weight at E14.

Decision-making: If E14 exceeds the optimal single branch and reaches the low risk integration threshold, it becomes the preferred structure for E15; otherwise E15 uses the best single model structure.

### E15: Frozen winning structure 5-seeted Esemble

Purpose: Finally confirm the extension of the Phase 4 instead of continuing the model search.

Pre-registration requirements:

- Freeze structures, hyper-parameters, integration rules and membership weights before training is done on new seeds.
- Former III seeds fixed to ZXQ0QZ; new seeds fixed to `20260707/08`.
- All members are not allowed to delete the seed, select a 4-bed subset or change weights according to test.
- New train-only OOFs are completed; access to the final full-train/fixed-test runner is only allowed after the pre-registered OOF threshold is adopted.

5-seed relative to structure 3-seed OOF threshold:

1. means task AUROC upgrade at least `+0.0010`;
2. means task AUPRC does not drop more than ZXQ0QZ;
3. The world-6 means AUROC doesn't drop over ZXQ0QZ.

Any failure stops at OOF and freezes the 3-bed structure as a result of the Phase 4 freeze.

## 5. Implementing the rules on reliance and cessation

```text
E8:E3b 3-seed ensemble anchor
 ♪ E9: CNN on E1 - ♪ E10: CNN on E3b ♪
 ├─ E11:global + H2 soft fusion
 E13:Auxily on E3b/CNN-E3b

E8/E10/E11/E13 Freezing the strongest single branch and effective complementary branch
                         │
                         └─> E14:auxiliary global + plain H2 fusion

Optimal freezing structure (E8/E10/E11/E13/E14)
                         └─> E15:5-seed preregistered confirmation
```

Stop rule:

- E9 Failure does not execute E10.
- E12 and its only H2-only failed to implement E13.
- E11 The fixed integration failed to search for per-task weights.
- E14 No more than the best single branch is not the base of E15.
- E15 Freeze Phase 4 after completion, without continuing to add seeds, integration members or structures to the same benchmark.

## 6. Interpretation of boundaries with changes in the number of tasks

The formal selection for Phase 4 is for 24-task, ZXQ0QZ benchmark. Model priority is not automatically extrapolated to other thresholds:

task range  Expected strategy
|---:|---|
33–43  Sharing of benefits may be greater, but low resources and imbalance are more severe; priority E1/E3b class light sharing with task-balanced sampling
24  Current Times 4 official range; suitable for E3b, CNN, soft Fusion, auxiliary and seen Esemble
14–20  Reduction of extert/ encoder width; H2 branch effective sample starts to run short
7-10  Co-collage tree models with small shared networks; no MMOE, PLE or complex dual branch
5 Shares task too few; old 5-task results are recorded in history only and not used for screening 24-task models

If the sensitivity of the number of task is to be studied, an independent learning curve/external validation phase should be established after the Phase 4 freeze; the current 24-task model should not be selected backwards using results from different thresholds.

## 7. Planned products and naming

Suggested scripts:

- `scripts/run_mousepmhc_phase4_e8_e3b_seed_ensemble_oof.py`
- `scripts/run_mousepmhc_phase4_e9_multikernel_cnn_shared_oof.py`
- `scripts/run_mousepmhc_phase4_e10_multikernel_cnn_mmoe_oof.py`
- `scripts/run_mousepmhc_phase4_e11_global_h2_soft_fusion_oof.py`
- `scripts/run_mousepmhc_phase4_e12_auxiliary_shared_oof.py`
- `scripts/run_mousepmhc_phase4_e13_auxiliary_mmoe_oof.py`
- `scripts/run_mousepmhc_phase4_e14_auxiliary_h2_ensemble_oof.py`
- `scripts/run_mousepmhc_phase4_e15_five_seed_confirmation.py`

The recommended results catalogue is used uniformly:

```text
Reults/mousePMHC_phase4_esnées short name
```

At least one experiment is saved:

- `*_oof_predictions.csv`
- `*_oof_per_task_metrics.csv`
- `*_oof_summary_metrics.csv`
- `*_oof_stability_metrics.csv`
- `*_oof_metadata.json`
- corresponding date, auxiliary,fosion or seletion diagnostics

Metadata must record: data path, task number, pair number, old seed, training seeds, hyperparametered base, test read, promotion threshold and actual judgement.

## 8. Phase 4 Expected final reporting sequence

The final report is not described as "as large as possible by experimental number", but organized as follows:

1. E8: Whether integration in independent seed reduces the variance;
2. E9/E10: Whether CNN offer independent motifs for gain;
3. E11: Whether the global and H2 branch are complementary;
4. E12/E13: Additional monitoring to enhance the sharing of expressions;
5. E14: The ability of the mechanism to be cumulative without any control has been identified;
6. E15: The 5-seed confirmation and final test results of the frozen structure;
7. The definitive conclusion is only applicable to the current 24-task closed-set, balance benchmark.

The ideal endpoint for Phase 4 is not the most complex model, but an interpretable model that exceeds the E3b/E8 anchor point on OOF means AUROC, the lower boundary of the difficult task, H2 group stability and the perceived stability, and is confirmed by a fixed 5-side agreement.
