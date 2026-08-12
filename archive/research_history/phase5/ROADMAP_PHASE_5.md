# Phase 5 Roadmap: Arithmetic gain study of multitask models in mice

Update: 2026-07-13
Number range: Continue Phase 4, starting with **E16**
Current anchor: E15,5-set task-balanced Factorized MMOE probability ensemble
Current OOF: mean task AUROC ZXQ0QZ, mean task AUPRC ZXQ1QZ, first-6 AUROC ZXQ2QZ
Data status: crossed test not read; continue to disable reading during development of Phase 5

## 1. Conclusions and design principles

The Phase 3/4 has given sufficient clarity to the boundaries: global sharing is important and complete isolation can result in loss being moved; larger, fixed-segment integration and sub-classification can result in negative migration; and independent seed ensemble has significant and widespread differential returns.

Evidence available  Results  Phase 5 Meaning
|---|---:|---|
E1 relative best single task RF  AUROC ZXQ0QZ  not returned to a task-by-task model; must keep shared across task
E3b task-balanced Factorized MMOE `0.8148 ± 0.0033`  Freezed baseline  for all single model candidates
E6 Hard Group  ZXQ0QZ  Interdict global path or search hard task grouping
E7 H2-Kk resocial apter ZXQ0QZ Small private path is feasible but there is a need to stabilize evidence of conflict
E5 FAMO single-seed ZXQ0QZ  not to use dynamically moving direction
E9 CNN / E12 auxiliary  ZXQ0QZ / `−0.0318`  not prioritized for expansion; metadata for conditionality rather than auxiliary predictions
E11 Fixed global/H2 Integration  World-6 ZXQ0QZ  Fixed average without weak branch
E15 5-seed E.E.R.D.

Thus, the formal master assumption is to retain only three:

1. ** Sorting target mismatch**: BCE does not directly optimize task-macro AUROC;
2. ** Local parameter conflict**: a small number of stable conflict layers require a very small diversion, but global path retention;
3. ** Tier task structure not visibly used**: task is ZXQ0QZ and head should be partially poold rather than fully independent.

The gradient surgery and the hypernetwork were not the first major experiments, but only after the diagnosis or the main experiment had provided positive evidence.

## Freezing agreements and uniform determinations

### 2.1 Data and Comparison Agreement

- Read only ZXQ0QZ: 24 tasks, 6,766 days.
- Fixed 3-old plain-grouped OOF; split seen for ZXQ0QZ.
- Develops fixed ZXQ0QZ; final confirmation only adds ZXQ1QZ.
- Each candidate is matched to the same olds, same seeds, and E3b that reruns from the training budget, and does not replace the match made baseline with historical averages.
- Except for algorithms themselves, the embedding, etc., width, dropot, epochs, task-balanced sampling and date entropy are the same as E3b/E15.
- E24 does not read the match test before E24; ultimately only one-time reading is allowed by the only frozen model against E15.

### 2.2 Reporting items

The main indicators are the mean task AUROC; the key sub-indicators are the meaning task AUPRC, World-6 AUROC, World-task AUROC.

1. Matching margin by task and 24 tasks/minus;
2. task-paired bootstrap 95% CI;
3. Average change per H2 macro;
4. Three seed mean values, SD values and each seed;
5. Number of parameters, training time, peaks and mechanism diagnosis.

### 2.3 Promotion and cessation rules

Three seed single model candidates relative to match e3b must be satisfied simultaneously:

1. mean task AUROC `≥ +0.0030`;
2. means task AUPRC does not drop more than ZXQ0QZ;
3. The first 6 AUROC does not drop over ZXQ0QZ;
4. (a) any H2 group AUROC does not decline more than `0.0100`;
5. At least three of the 14/24 tasks seed average AUROC;
6. AUROC gains 95% of the bottom CI `> -0.0010`.

Seed single filter is only used to decide whether to complete three seeds: AUROC must not be less than Matched E3b ZXQ0QZ and AUPRC, World-6 cannot be significantly different. One failure stops the pre-registration configuration; no width, rank, loss weights or re-selected scores.

## Reordered E16–E24 Experiment

No.  Experiment  punctuation  execution status
|---|---|---|---|
E16
E17  Task-wise ranking lost  High: Pair data + AUROC master indicator  Main experiment A
E18 Recon-lite rehabilitation high: keep soft correction of global share
E19  Hierarchical Tissue-H2 Head  High: Direct use of the cross-tier of task  Main experiment C
E20PCGrad: only validating a stable direction conflict  execution of conditions dependent on E16
E21  CAGrad : Independent alternative to PCGrad  Implementation of conditions dependent on E16
E22  RotoGrad  Medium and low: need to be accompanied by an imbalance of direction and scale  executed under conditions, dependent on E16
E23  FiLM / low-rank hyper-head  medium height: metadata conditioned  executed, dependent on E18 or E19
E24, freeze winner 5-seed confirmed.

### E16:E3b: hierarchical primary mission gradient conflict audit

Purpose: Use gradient surgery or private apperter only when supported by evidence, to avoid miscalculating small samples with acrimony.

Achieved:

- Calculates the gradient for 24 master task separately; does not include H2/tissue auxiliary loss.
- Take several fixed bits each morning/medium/end of each old, seed training, calculate cosine after the accumulated average gradient, avoiding single catch noise.
- Layer output peptide embedding, share encoder, task-pair cosine for each extert, gate, negative cosine scale, gradient parameters and parameter ratios.
- The three aggregate conflict matrixes, task, H2, tissue, and the relevance of different fold/seed/epoch matrices, are exported simultaneously.

Stable Conflict Layer Definition: In at least 2/3 olds, 2/3 seeds and at least two training phases, both are at 25% before negative cosine intensity and the conflict matrix relevance of the adjacent audit phase ZXQ0QZ. E16 selects the exclusive structure without having to use the help-out tab and does not participate in model rankings.

### E17:E3b + task-wise ranking loss

Purpose: To bring the training objectives closer to what means task AUROC without changing the reasoning structure of E3b and sharing it with the whole world.

AUROC measures the order of all positive and negative samples in the same task, so pre-registered two fixed melts:

- **E17a: metted-pair ranking**. Calculated loss of positive and negative samples for the same original ZXQ0QZ only;
- **E17b: task-wise all-pair ranking**. In each task-balanced catch, two or more of the same list is calculated to be lost.

For task (t), use:

\[
L_{\mathrm{rank}}^{(t)}=
\frac{1}{N_t^+N_t^-}
\sum_{i=1}^{N_t^+}\sum_{j=1}^{N_t^-}
\log\left(1+\exp\left[-(s_i^+-s_j^-)/\tau\right]\right).
\]

Equal rights to task and add to BCE:

\[
L=L_{\mathrm{BCE}}+\lambda L_{\mathrm{rank}}.
\]

Fixed ZXQ0QZ, `tau=1.0`. The two types of decomposition are constructed only within the littlesplit, together with the task, and cannot cross the old or the task. Run the E17a/E17b of the same seed first; only the higher AUROC enters the three seed. The selection rule is fixed as a single seed mean task AUROC, and the calculation of the faster E17a is less than `0.0010`. It must be reported that AUPRC, MCC, Brier score; AUROC upgrades but AUPRC/MCC significantly deteriorates cannot be promoted.

### E18: Recon-lite Stabilizing the low conflict layer

Condition: E16 finds at least one stable layer of conflict. Otherwise, you can skip E18 explicitly and cannot manually assign H2-Kk or other task to OOF by observing it.

Purpose: To test whether the partial soft correction implied by E7 can be generalized without repeating E6 hard group errors.

Achieved:

- Inserts only the apperter; global E3b path in the E16 defined layer.
- Bottleneck rank fixed `8`, output layer zero initialized, initialized `0` and applied L2 contraction to aadapter.
- **E18-control matched aadapt**; all task shares a adapter.
- Do again **E18-H2 apper**; four H2 aadapter each, do not create 24 sets of task apperter.

Only E18-H2 is interpreted as a benefit being made subject to H2 conditionality if it is better than E18-control and Matched E3b; if it is better than E3b, the conclusion is that only "additional bound capacity is useful". E18-H2 failure cannot extend to a Tissue apperter, per-task apperter or rank search.

### E19:Hierarchical tissue–H2 head + low-rank interaction

Purpose: A `tissue × H2` structure with a visible expression of task to partially pool instead of 24 completely independent head.

Use:

\[
z_{t,h}(x)=z_0(x)+z_t^{\mathrm{tissue}}(x)+z_h^{\mathrm{H2}}(x)+z_{t,h}^{\mathrm{int}}(x).
\]

The focus is created by a two-linear low-level combination of tissue/H2 embedding, with rank fixed to `4`. All off-the-counter elements are zero-initialized and scaled to zero; the effect of interactivity is stronger than the H2 and the primary effect of tissue. To avoid the cross-off between global, H2 and tissue, zero mean/centralisation constraints are implemented:

\[
\sum_h z_h^{\mathrm{H2}}=0,\qquad
\sum_t z_t^{\mathrm{tissue}}=0.
\]

The parameters for logit fractions and small samples are reported; the parameters for intermission explosion or small samples tabk monopolization are judged to be failures.

### E20:E3b + PCGrad

Condition: E16 confirms a stable direction conflict; E18/E19 is not required to succeed. PCGrad and CAGrad are independent alternatives and are not pre-conditions for CAGrad with the phrase "PCGrad does not fail".

Apply PCGrad;heads and the rest of the E16 defined SCC only. The tabs of each step are generated by tracking seed fixed. After the filter has been passed, the list of seeds is added three; the projection ratio, the project gradient parameters and the training time are recorded. No project sequence or tabs are selected for retry after failure.

### E21:E3b + CAGrad

Condition: E16 confirms a stable direction conflict. Pre-registered `c=0.2`, only for stabilization layers of the conflict base task guides; the rest of the training protocols are the same as E20.

PCGrad and CAGrad retain the higher mean task AUROC if both pass the three seed threshold; keep the faster training and simpler if the difference is less than `0.0010`.

### E22:E3b + RotoGrad

Condition: E16 also shows a stable direction conflict, a balance of at least three times the mission gradient paradigm and a recovery over the period, and a single perceived signal from E20 or E21 does not pass the full threshold.

The experiment only tested whether a joint correction of the Direction+Standard is necessary. If the single seed does not meet the relative match of match E3b ZXQ0QZ AUROC and the key sub-indicators are poor, it stops; it does not extend to Nash-MTL. Nash-MTL remains a candidate for the future independence phase, not the automatic experiment of the current round.

### E23:task-conditioned FiLM / low-rank hyper-head

Condition: E18-H2 or E19 at least one meets the single perceived non-poor threshold.

Purpose: To test the validity of metadata conditionality and to distinguish strictly from the E12 metadata subsidiary classification.

- Enter only tissue/H2 embedding; small networks generate FiLM ZXQ0QZ or low-rank head delta.
- Not generating complete encoder; total parameter increment does not exceed 15% of E3b.
- ZXQ0QZ, `beta/delta=0` were initialized and the training starting point was therefore equal to E3b.
- Select only the E18 or E19 that is shown as a valid modem; do not search for FiLM, adapter and head simultaneously.

### E24: Freezing winner 5-seed confirmed regular integration with pre-registered

The most two candidates for E24 are: the winning variant of E17 and the best MTL candidate for E18-E23 seed through the unified threshold. If no new candidate passes, E15 maintains the final model, Phase 5 ends.

Implementation:

1. Freezing candidate structures, hyper-parameters, training rounds and adding new seeds `20260707/08`; preparing candidates for election, etc., probability etc., without removing members.
2. Candidates 5-seed esmble first compare directly with E15; the threshold is used to become the final model.
3. If both candidates pass, the only pre-registered ZXQ0QZ task-wise percentile-rankfosion is implemented; the predictive correlation threshold is not used to determine integration and the weight is not searched.
4. The Facion relative best single ensemble must be: AUROC ZXQ0QZ, AUPRC, World-6 does not drop more than `0.0010`; otherwise the best entice must be retained.
5. E24 then freezes the Phase 5. Fixed test only confirms the only winner and freeze E15 once, and cannot re-select the model based on test.

## 4. Implementing dependency, sequencing and arithmetic controls

```text
E16 gradient audit
 E18 Recon-lite apter
 ♪ E20 PCGrad (Intrusionally Clashing)
 E21 CAGRAD (in case of a stable direction conflict)
       E22 RotoGrad (when there is another measure of stability)

E17 ranking loss ──────────────────────────────┐
E23 hyper-head
                                                   └─> E24 5-seed confirmation
```

Recommended actual order:

1. E16;
2. E17, E19, and E18 when E16 conditions are met;
3. E20/E21 Only when a conflict is established to stabilize the main mission gradient;
4. E22/E23 as conditional extension;
5. E24.

The official master assumption is E17, E18, E19. The gradient method adds up to one winner of three seeds; the total number of new candidates entering E24 is not more than two, limiting multiple comparative deviations of the same OOF benchmark.

E3b single saw 3-fold cost 1:E16 about `0.3–0.6`; E17/E18/E19 about `1–1.3`; E20/E21/E22 about ZXQ2QZ for a task gradient; E23 about `1.1–1.4`; and two new seed about `0.67` for each E24 candidate. All operations are recorded as wall-lock, peaks and failure reasons.

## 5. Products and naming

Suggested scripts:

- `scripts/run_mousepmhc_phase5_e16_gradient_audit.py`
- `scripts/run_mousepmhc_phase5_e17_taskwise_ranking_mmoe_oof.py`
- `scripts/run_mousepmhc_phase5_e18_recon_lora_adapters_oof.py`
- `scripts/run_mousepmhc_phase5_e19_hierarchical_heads_oof.py`
- `scripts/run_mousepmhc_phase5_e20_pcgrad_mmoe_oof.py`
- `scripts/run_mousepmhc_phase5_e21_cagrad_mmoe_oof.py`
- `scripts/run_mousepmhc_phase5_e22_rotograd_mmoe_oof.py`
- `scripts/run_mousepmhc_phase5_e23_task_conditioned_hyperhead_oof.py`
- `scripts/run_mousepmhc_phase5_e24_five_seed_confirmation.py`

The results catalogue is consolidated as follows:

```text
Reults/mousePMHC_page5_esn.
```

At least one formal experiment will keep OOF prections, per-task metrics, submary, state, metadata, matched base and mechanism diagnostics. Metadata must record ZXQ0QZ, gitcommittee, complete CLI, environment, hold/seed, parameter volume, training time, promotion thresholds and final determination.

## 6. Documentation

- Yu et al., *Gradient Surgery for Multi-Task Learning*, NeurIPS 2020(PCGrad):https://papers.nips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html
- Liu et al., *Conflict-Averse Gradient Descent for Multi-task Learning*, NeurIPS 2021(CAGrad):https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html
- Javaloy and Valera, *RotoGrad: Gradient Homogenization in Multitask Learning*, ICLR 2022:https://openreview.net/forum?id=T8wHz4rnuGL
- Shi et al., *Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning*, ICLR 2023:https://openreview.net/forum?id=ivwZO-HnzG_
- Navon et al., *Multi-Task Learning as a Bargaining Game*, ICML 2022(Nash-MTL):https://proceedings.mlr.press/v162/navon22a.html

Documentation provides only arithmetical candidates; the local OOF evidence priority for Phase 3/4. All conclusions are limited to the current balance of 24-task, ZXQ0-XZ, Pair-grouped OOF benchmark and cannot be extrapolated to the conclusion of the sixed-test, peptide-disjoint, protein-disjoint, unseen-H2 or the foreign forces.
