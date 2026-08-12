# MousePMHC Phase 5 Experiment Report: Arithmetical Gain Diagnostics and Cessation Conclusions

Update: 2026-07-13
Status: E16, E17, E19 Completed; E18, E20, E21, E22, E23 not implemented; E24 No new candidates to confirm
Data boundary: ZXQ0QZ only; crossed test not read

## 1. Objectives and conclusions

Phase 5 Tests three pre-registered master assumptions: BCE mismatched the sorting target with task-macro AUROC, local stabilization gradient conflicts, and tissue x H2 level head partially pool values. All experiments maintain 24 tasks, 6,766 trade terms, 3-old plain-grouped OFF (split seed ZXQ0QZ) and E3b 's encoder/experts/gate training budget.

The conclusion was that none of the three major assumptions had been made for new candidates for promotion at the current benchmark.

1. E16 No stable conflict layer was identified and therefore no implementation was carried out for the residual apperter, PCGrad, CAGrad or RotoGrad.
2. E17 match-pair and all-pair ranking are not screened through a single seed; neither do you replace the two seeds nor search for ZXQ0QZ, ZXQ1QZ or other ranking variants.
3. E19 's hierarchical tissue-H2 head is significantly worse than matched E3b under a single, and small sample-H2-Kk is dominated by local interest.
4. E23 Preconditions (E18-H2 or E19 only seed) are not valid; E24 no new Phase 5 candidate can be confirmed.

Thus, the winner of the development phase at the end of Phase 5 is still Phase 4 E15:5-set task-balanced Factorized MMoE development outcome. This conclusion is still limited to the principal-only OOF and cannot replace a one-time fixed test.

## Uniform agreements and determinations

- Data: 13,532 traffic rows, 6,766 spaces, 24 items-H2 taskes, 13 items, 4 H2 recoverys.
- OOF: 3olds, pair-grouped; same `pair_id` does not cross
- Develop seeds: ZXQ0QZ; E17/E19 run `20260704` first, and only meet the non-poor conditions of the list is completed with two more seeds.
- Main indicators: mean task AUROC; protection indicators: mean task AUPRC, World-6 AUROC, macro average per H2, MCC and Brier score.
- All E16/E17/E19 metadata recorded `test_data_read=false`.

The formal threshold for the third seed candidate is: AUROC is at least higher than the match-E3b ZXQ0QZ, AUPRC is no lower than ZXQ1QZ, World-6 is no less than `-0.0030`, H2 is no less than `-0.0100`, at least 14/24 task AUROC is improved and the lower boundary of task-paired bootstrap AUROC is greater than `-0.0010`.

## 3. E16: Audit of the hierarchical main task gradient

E16 Audit of the 24 main tasks BCE gradient for E3b. Each seed-old repeats each task 4 fixed batting gradients on epoch ZXQ0XZ; first crosses the average bandees gradient and then calculates the peptide embedding, share encoder, three experts and gate tab/ H2/tissue cosine matrices. Dropout closes at the time of the audit and excludes taskheads and any auxiliary loss.

E16 OOF projections are identical to the freeze E3b, III seed mean task AUROC for `0.8148 ± 0.0033`, indicating that the diagnosis did not change the training model.

The X-rays, the X-rays, the X-rays, the X-rays, the X-rays, the X-rays, the X-rays, the X-rays, the X-rays, the X-rays, the X-rays, the X-rays, the X-rays, the X-rays, the X-rays, the X-rays, the X-rays, the X-rays, the X-rays, the X-rays, the X-rays, the X-rays, the X-rays, the X-rays, the X-rays, the X-rays, the X-rays, the X-rays, the X-rays, the X-rays, the X-rays, the X-rays, the X-rays, the X-rays, the X-rays, the X-rays, the X-The, the X-rays, the X-rays, the X-ray, the
|---|---:|---:|---:|---|
peptide embedding No 0/30/3 Unstable
shared encoder0/3 instability
ext 0 No 0/3 Unstable
ext 1 No 1/311111
ext2no 0/30/3ununstable
gate No 0/30/3 Unstable

The negative cosine strength of Gate is often high, but the average correlation value of the adjacent conflict matrix at the task stage is only about `0.103`, the most about `0.397`, and below `0.5`. Therefore, the change in its task-pair identity with the seed/old/epoch cannot be interpreted as an operational stable structural conflict. While there is a marked gradient imbalance in the middle and later stages, E22 needs to be accompanied by a steady directional conflict and a positive signal of E20/E21 first, conditions are not established.

** Adjudication:** Skip E18, E20, E21, E22; H2 adapter cannot be assigned manually on the basis of individual OOF task.

## 4. E17:task-wise ranking loss

E17 Use the same pair-awardee task-balanced batties as match-based BCE-only E3b. Check block contains 8 complete pair (16 rows):

- E17a: Sort of positive or negative logit within each original `pair_id`;
- E17b: All positive and negative logit in the same task block in two order;
- Fixed ZXQ0QZ, ZXQ1QZ.

The results are shown below. The AUROC of E17b is higher than the ZXQ0QZ, which is less than the pre-registered `0.0010` flat threshold, and therefore selects a faster E17a to be screened according to the rules.

Xiaojiang (seed 20260704)
|---|---:|---:|---:|---:|---:|
| matched E3b | 0.817311 | 0.806068 | 0.523612 | 0.192269 | 0.681798 |
| E17a matched-pair | 0.815397 | 0.802412 | 0.520579 | 0.192304 | 0.678108 |
| E17b all-pair | 0.816224 | 0.805488 | 0.523690 | 0.192435 | 0.678227 |

E17a AUROC ZXQ0QZ, AUPRC ZXQ1QZ, World-6 `-0.003690`, Task AUROC 9 won, 15 negative, AUROC bootstrap 95% `[-0.004345, 0.000466]`, which violates the single eded line of AUROC, AUPRC and World-6.

Even if E17b is examined without regard to the draw rule, its AUROC is ZXQ0QZ, World-6 `-0.003571`, and is equally not successful. Both ranking loss lifts H2-Kk, but offsets by other H2 drops, and cannot search H2-specific ranking or hyperparameters.

The training sequencing loss has fallen normally, so failure is not an optimistic collapse: E17a has dropped from about ZXQ0QZ to about ZXQ1QZ, and E17b has dropped from about `0.683` to about `0.046`. The conclusion is that the current surgate ranking optimization in the pair-batch has not been converted into a benefit of the help-out task-macro ranking, but rather has slightly damaged the AUPRC, calibration and difficult tasks.

** Adjudication:** Stop E17; Not Run the remaining two seed; Not Enter E24.

## 5. E19:hierarchical tissue-H2 head

E19 Retain E3b backbone, replace only 24 independent headers with:

\[
z_{t,h}(x)=z_0(x)+z_t^{\mathrm{tissue}}(x)+z_h^{\mathrm{H2}}(x)+z_{t,h}^{\mathrm{int}}(x).
\]

The primary effects of the tissue and H2 are zero-sum binding by centralizing the parameter group mean; the rank-4 interposition is generated by the dual centralization of the tissue/H2 bilinear factor. All non-global logit fractions are zero at initialization, and the output projection zero-initialization is interactivity L2 coefficient ZXQ0QZ is higher than the main effect `1e-4`. Candidates are 68,413 parameters, slightly less than the 68,475 parameters of E3b.

Xiaojiang (seed 20260704)
|---|---:|---:|---:|---:|---:|
| matched E3b | 0.816357 | 0.806668 | 0.524407 | 0.191574 | 0.680400 |
| E19 hierarchical head | 0.807132 | 0.796679 | 0.508397 | 0.198385 | 0.678657 |
| E19 − E3b | **-0.009225** | **-0.009989** | **-0.016010** | **+0.006811** | -0.001743 |

The AUROC bootstrap of E19 95% CI is ZXQ0QZ, AUPRC CI is `[-0.019827, -0.000574]`, both are completely below zero; 8 out of 24 tasks are won and 16 are negative. H2 AUROC is changed to Db ZXQ2QZ, Kb ZXQ3QZ, Kd ZXQ4QZ, Kk `-0.03461`, which clearly violates the formal line of protection.

In the polymerization diagnosis, the ratio of interaction/globallogit RMS to about ZXQ0QZ is not dominant overall. However, the tabk-by-temper diagnosis shows small samples of liver-H2-Kk (per fold 82-783 sitting pairs) interaction/global is the largest number of parameters compared to three folds, respectively, of approximately ZXQ1QZ, whose parameters of interaaction are also the largest. The task's AUROC ZXQ2QZ, AUPRC `-0.02930`, Brieer `+0.01918`. Therefore, although the standard threshold value of the polymer script is marked as mechanism pass, the "interaaction explosion or small sample tabk monopolistic parameter" should be interpreted as a partial failure of E19.

** Adjudication:** Stop E19; do not run the remaining two seeds; E23 precondition is not valid.

## 6. Phase 5 Summary and follow-up status

♪ Experiments, results, decisions, decisions ♪
|---|---|---|
E16 gradient audio 0/6 Stable conflict layer
E17 running single-seed harm AUROC, AUPRC and World-6 stop
E19 hierarchy single-seed AUROC/AUPRC CI is totally negative; Kk/Kd is significantly down
E23 hyper-head  E18-H2/E19 with no single perceived non-bad signal  not executed
E245-sead communication

The negative evidence given by Phase 5 is also valuable: on the current fixed 9-mer, 24-task, low resource, pair-grouped OOF benchmark, the profit of task-balanced Factorized MMoE is derived from the average probability of retaining sufficient global sharing and subsequent independence seen; the optimisation target is directly changed to catch running, surgery based on an unstable gradient, or the forced separation of the tesk head is not yielding generalized benefits.

The current freeze development phase is selected as:

```text
E15: 5-seed task-balanced Factorized MMoE probability ensemble
mean task AUROC = 0.8392
mean task AUPRC = 0.8316
worst-6 AUROC = 0.7101
```

Any cross-checked test assessment can only confirm this unique frozen model and its predefined E15 comparison once and can no longer select structure, Seed or integration weights with test.

## Outcome document

- E16:`results/mousePMHC_phase5_e16_gradient_audit/`
- E17:`results/mousePMHC_phase5_e17_taskwise_ranking_mmoe_oof/`
- E19:`results/mousePMHC_phase5_e19_hierarchical_heads_oof/`
- E16 runner:`scripts/run_mousepmhc_phase5_e16_gradient_audit.py`
- E17 runner:`scripts/run_mousepmhc_phase5_e17_taskwise_ranking_mmoe_oof.py`
- E19 runner:`scripts/run_mousepmhc_phase5_e19_hierarchical_heads_oof.py`
