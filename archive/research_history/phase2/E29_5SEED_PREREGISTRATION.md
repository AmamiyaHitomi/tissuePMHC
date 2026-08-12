# E29 5-seed incremental extension pre-registration

Pre-registration date: 2026-07-11
Status: pre-registered; OOF threshold adopted only once to perform a fixed 5-bed test assessment

## Objective

Tests whether OOF performance continues to improve after E29 Multi-kernel CNN E14a has been expanded from 3 seeds to 5 seeds. The experiment is the last performance extension on the standard split; regardless of the result, the model or integration based on the split is stopped after completion.

## Fixed members

Membership: 202,600,704, 202,705, 202,2706.
Additional members: 202,2707, 202,2708.
Bans the removal of individual seed after the result is generated, changing to a 4-seed subset or selecting members based on test.

## Fixed model and training configuration

Full reuse E29 3-seed configuration: position-preserving multi-multi-kernelconf1d peptide encoder; kernel size 2, 3, 5; kennel 32 channels; edding dimension 16; hidden dimension 128; dropot 0.2; 25 epochs; AdamW; leaving ratte 0.001; weight decay 0001; tissue/HLA auxiliary l'ossight 0.1; batt size 512.

Only new trainings are available at seeds 20260707 and 20260708. Three of the OF and test projections have been re-used directly and without repeating training.

## OOF Decision-making Rules

First, two new seeds run exactly the same 3-old fair-grouped OOF as E293-seed. Compare the five seed OOF projections to the already fixed E293-seed OOF.

Read the two new full-train test models that have been projected and trained is only allowed if the following three are met at the same time:

```text
5-seed OOF mean AUROC - 3-seed OOF mean AUROC >= 0.0010
5-seed OOF worst-10 mean AUROC - 3-seed OOF worst-10 mean AUROC >= -0.0010
5-seed OOF mean AUPRC - 3-seed OOF mean AUPRC >= -0.0005
```

When any condition fails, the experiment stops at OOF, E29 3-seed to continue as the official master result and does not read or generate the new test forecast for the seed.

## Test Policy

OOF only trained two new Seed complete training set models. It was then merged with three already-seed fixed test projections to calculate E295-seed mean once. Models, weights, members, thresholds or integration rules cannot be modified according to test.

The official main comparison is E295-seed means relative to E293-seed mean; while maintaining a fixed comparison with E175-seed. After this evaluation, there is no further need to add seed or to select models on the standard split.

## Results of implementation (recorded after pre-registration)

Two new seeds have completed training as pre-registered configurations. The 5-said OOF means AUROC gains of 0.00191, the world-10 means AUROC gains of 0.000298 and the mean AUPRC gains of 0.000242; all three thresholds were passed, thus allowing and completing a fixed full-train 5-set test assessment.

In official test, E29 5-seed means the AUROC 0.837, mean AUPPC 0.8259, mean integration 0.7588, mean mCC 0.5175, world-10 means AUROC 0.767. The AUROC gain relative to E29 3-seed is 0.00316. This result does not trigger selection, integration weight adjustment or further standard split entry.
