# Mouse  Human Migration Report

Update: 2026-07-28
Status: Migration code completed; formal training not implemented.

Supplementary note: Verification confirmation is missing for the Human/Mouse list IDENTENTICAL-fold tables three matched controls.
Not one-hot blog response, no supporting library, with backup
Branch not running on freeze mode E33 peptide-disjoint folds. Added
ZXQ0QZ and the Tri-Mechanical entrance
`run_strict_three_all_in_one.py`; the supplemental experiment does not read the match test.

## 1. Number of methodologies

The human 157-tasktable 5 methodology pool contains 28 de-weighting methods. The paper also recorded the capacitity-matched
The three contrasts between MHz-only, MHz and NetMHCpan EL, so the entire human paper is currently available.
There are **31 independent methods**.

Reference mouse Phase 3-6, peptide-disjoint artitecture exteriors and issue5
general-pMHC controls, 31 methods:

- 15 tested in the mice project;
- 16 not yet tested;
- 16 gaps are covered by 12 separate entry points in this directory.

For a complete methodological audit see `METHOD_INVENTORY.md`.

## 2. Principle of non-repetition of relocation

Rats have been tested to map by method rather than by human experimental numbering. For example:

- Human MHC-grouped for use H2-grouped E2;
- (b) Human MMOE for use E3/E3b;
- (a) Human FAMO for mouse E5;
- Human fixed global/MHCfosion match Mouse E11;
- Human air running towards mouse E17;
- Human auxiliary subvision to mouse E12;
- (a) transitory baselines corresponding to mode E0;
- MSC-only/MHCflury/NetMHCpan has been completed by issue5.

These methods are not repeated in the new directory. Mapping only indicates that the methodology has been tested and does not differentiate between the backbone or the protocol.
Explained completely Matched Comparison.

## 3. New migration content

Entry  Number of new human methods
|---|---:|---|
| `run_neural_single_conditioned.py` | 2 | neural single-task, conditioned tissue/H2 |
ZXQ0QZ  1  H2 pseudo-security; waiting for a true sequence file
ZXQ0QZ  1  H2 ID + pseudo-security; waiting for a true sequence file
| `run_tissue_grouping.py` | 1 | tissue-grouped hard sharing |
| `run_selective_grouping.py` | 1 | validation-selected global/H2 |
| `run_adaptive_soft_ensemble.py` | 2 | validation-delta, validation-softmax |
| `run_cagrad.py` | 1 | CAGrad |
ZXQ0QZ 24x256 and 6x128 MMOE
| `run_dbmtl.py` | 1 | DB-MTL |
ZXQ0QZ22 2
| `run_mlp_dual_seed_ensemble.py` | 1 | row-level seed ensemble |
| `run_tissuepmhc_full.py` | 1 | full multi-kernel dual-branch TissuePMHC |
** Total** ** 16**  12 entrances **

## 4. Full TissePMMHC ' s decision

Mouse Chance 4 E9 tested multi-kernel CNN shared headers but not TissePMHC
Full group: multi-kernel global auxiliary branch, MHC-specific plain branch, task-wise
The rank foundation and multi-sealed approach. Therefore, the complete TissePMHC is still counted as an incomplete method.

`run_tissuepmhc_full.py` repeats frozen mode E15 OFPLANATIONs as already used base base line, and then uses the same type of information as the same.
and write the corresponding name to the transfer control. It still executes the OOF record; if the date fails, it will be replaced by a new name.
The original runner would refuse to enter the fix test and avoid bypassing the established leak-proof rules.

## H2 pseudo-security boundary

The warehouse currently has no source-verified H2-Db/H2-Kb/H2-Kd/H2-Kk pseudo-file.
Two relevant entrances have been migrated with codes and parameters, but must be provided before they are officially operational
ZXQ0QZ. The adaptor stops when the file is missing and cannot use HLA
Replaces a sequence, an empty sequence or an artificially constructed sequence.

## 6. Paths and recoverability

All new experiments read:

```text
data/mousePMHC/mousePMHC_train.csv.gz
data/mousePMHC/mousePMHC_test.csv.gz
```

All new results are written:

```text
results/mousePMHC_human_method_transfer/<experiment>/
```

Write every operation to `transfer_contract.json`, recording human method names, source runner, mouse
Train/test, Seed, hyperparameter and output position. Path audits refuse to write output to history mouse or human
Results catalogue.

## All-in-one entrance

New `run_all_in_one.py`, call all 12 separate entry points in a dependent sequence; experiments are still in place
Independent subdirectories, without combining results. The entrances support breakpoints and return numbers by start, end, and return.
and state to write to `all_in_one_status.json`.

H2 pseudo-equality files missing, default only two related experiments are recorded as
ZXQ0QZ, which does not block the remaining 10 entrances.
Compulsory inspection; `--continue-on-error` can continue with an independent experiment after a particular experiment has failed.
By this code migration.
