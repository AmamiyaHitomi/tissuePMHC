# Phase 3 Roadmap: Mouse MHC-I organises multitask learning

## Current freeze (2026-07-12)

- Benchmark: ZXQ0QZ, fixed 100 days per task; 24 items-H2 taskes, 13 items, 4 H2 responses.
- Train: 6,766 fairs (13,532 rows); task 124 - 470 train fairs, median 264.5.
- E0 and E1 have completed 3-old plain-grouped data-only OOF on this data set; both are `test_data_read=false`.
- All E2+ results, script assumptions and performance thresholds for the old `>500`, 5-task are historical records and are not allowed for current comparisons.
- Current single seed filter: E2 H2-grouped share, E4 Factorized PLE-lite, E5 FAMO all stopped; right to task-balanced MMoE control in E5 became E3b, entering 3-seed confirmed.

## Baseline completed

OOF means task AUROC
|---|---|---:|---|
E0  BLOSUM62 Random Forest (best of five traditional candidates)  0.7530  Single task hard baseline
E1  Full sharing of peptide encoder + 24 task heads  0.8057  Shared baseline

E1 gains relative to E0 are significant H2 heterogeneity: 12 tasks for H2-Db were fully upgraded, with an average of ZXQ0QZ AUROC; the average group gains for H2-Kb and H2-Kk were ZXQ1QZ and `-0.0106` respectively. This means that the core issue for the next phase is not to expand the network, but to share selectively on H2 and avoid moving across negative H2.

## Unified OOF protocol

- Develops and candidates filters only to read `data/mousePMHC/mousePMHC_train.csv.gz`; fixed test can only be read once by the final frozen candidate.
- First round: 3-old plain-grouped OOF, standard seen ZXQ0QZ, training seed ZXQ1QZ.
- After the first round: supplemented seeds ZXQ0QZ, `20260706`, reporting 3-seed OOF mean and standard deviations.
- Main indicator: Mean task AUROC; also reporting AUPRC, World-6 task AUROC, each H2 group means AUROC, every task AUROC, parameter quantity and training time.
- First round successful: at least `+0.005` relative to E1, mean task AUROC; average AUROC of any H2 group is no less than E1 ZXQ1QZ; most-6 task means AUROC does not drop more than `0.010`.
- Disables the use of " no task shall decline above 0.020 " as the hard threshold: 24 low-resource missions, where the single task OOF is too volatile; the full report of each task difference and the whiteted bootstrap CI is still required.

## E2-E7: Current official route

Method  Minimum achievement and problems  First round of decision making
|---|---|---|---|
E2 **H2-grouped hard share**  each H2 small peptide encoder; shares encoder with H2 tasks each tab reserve head. Checks directly whether "cross-H2 sharing" is a negative migration source.
E3 Factorized MMOE-lite**3 small share extensions; Gate entered peptide, tessue, H2; 24 task heads; record date entry and expert user. E2 executable
E4 Factorized PLE-lite**  global, H2, tessue rootes; no large per-task expert, only very small task recovery if necessary.  Implementation only when E3 cannot protect Kb/ Kk
E5  FAMO**  dynamic task weight on E3 MMOE backup, first 5 epoch etc. warm-up.  completed single seed, no better than equal control, stop
E6 **TAG-guided grouping**  to migrate task step by step by task identification task groups to be trained compared to the grouping of the H2 a priori.  code completed; E3b still has a stable negative migration in Kk group, so perform
E7  Recon-style partial share**  H2 gradient conflict in each train-only seen-old audit shared peptide encoder; global MMOE share of E3b for pre-declared H2-Kk plus pre-declared H2 & k & restual aptter, (optional ZXQ0 XZ audit option).  E6 hard grouping failed but Kk was almost flat and thus became final target candidate; code completed

E3b uses E5 internal equivalent task-balanced MMOE control, defaults seeds 20260704/05/06; accessed test is not allowed until confirmed.

## Method of not being formally presented as the first round

- PCGrad, CAGrad, Nash-MTL: 24 task-by-task gradients are high; only to be considered for subsequent diagnosis with 4 H2 group gradients.
- Auto-Lambda: Additional meta-validation is required; the smallest task is 124 train days, further severing the validation would result in significant loss of training data.
- Complete Cross-Stitch, AdaShare: 24 private network or structure searches are over-costed for parameters and searches under the current sample size.

## Documentation

- H2 Group and follow-up automatic grouping: [TAG, NeuriIPS 2021] (https://proceedings.neurips.cc/paper/2021/hash/e77910ebb93b511588557806310f78f1-Abstract.html).
- Dynamic task weight: [FOMO, NeuriIPS 2023] (https://proceedings.neurips.cc/paper_files/paper/2023/hash/b2fe1ee8d936ac08dd26f2ff58986c8f-Abstract-Conference.html).
- Conflict layer splitting: [Recon, ICLR 2023] (https://iclr.cc/virtual/2023/poster/11669).
- Selective parameter sharing: [AdaShare, Neurips 2020] (https://papers.nips.cc/paper_files/paper/2020/hash/634841a6831464b64c072c8510c7f35c-Abstract.html).

## Status of realization and results

- Data audit: `results/mousePMHC_phase3_data_audit/`.
- E0:`scripts/run_mousepmhc_phase3_e0_oof.py`.
- E1:`scripts/run_mousepmhc_phase3_e1_oof.py`.
- E1 Supplementary seeds 20260705/06: `scripts/run_mousepmhc_phase3_e1_additional_seeds_oof.py`; output to `results/mousePMHC_phase3_e1_oof_additional_seeds/`, not overwrite seed 20260704.
- E2:`scripts/run_mousepmhc_phase3_e2_h2_grouped_oof.py`.
- E3: ZXQ0QZ; formal Min200 output `results/mousePMHC_phase3_e3_factorized_mmoe_min200_oof/`.
- E4: ZXQ0QZ; formal Min200 output `results/mousePMHC_phase3_e4_factorized_ple_min200_oof/`.
- E5: `scripts/run_mousepmhc_phase3_e5_famo_mmoe_oof.py`; formal Min200 output `results/mousePMHC_phase3_e5_famo_mmoe_min200_oof/`.
- E3b: ZXQ0QZ; 3-seed confirmation of the winning right in E5 to task-balanced MMoE, output `results/mousePMHC_phase3_e3b_task_balanced_mmoe_min200_oof/`.
- `scripts/run_mousepmhc_phase3_e2_structured_ple_oof.py` is an old 5-task history realized and cannot run directly.
