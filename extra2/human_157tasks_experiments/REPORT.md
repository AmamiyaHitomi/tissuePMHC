# Human 44-task experimental migration report to 157-task benchmark

Update: 2026-07-28
Status: ** The migration code has been completed, formal training has not been performed, no new performance values have been generated or filled in. **

## 1. Objectives

Table 5 Only currently available in Shared Heads, Auxiliary Dual Branch, MLP current history
Four methods. To match the number of methods for Table 4, Table 5 is required at least
10 methods. Compatibility experiment for 44-task projects was achieved in earlier stages of this directory migration, 157-task at Phase 7
On the standandard fix-test benchmark re-training and evaluation.

All methods used to check `n_tasks=157`, data path and migration condition are retained.
10 Methods are not cut off by performance, negative outcomes are not removed, and the best performance method is not selected.

## 2. Data, codes and output locations

The migration code is located in:

```text
extra2/human_157tasks_experiments/
```

Fixed data are:

```text
data/tissuePMHC_phase7_min200/tissuePMHC_phase7_min200_train.csv.gz
data/tissuePMHC_phase7_min200/tissuePMHC_phase7_min200_test.csv.gz
```

The default result position remains unchanged:

```text
results/tissuePMHC_phase7_min200_migrated_44tasks/
```

This directory is with the original 44-task ZXQ0QZ and the existing Phase 7
ZXQ0QZ separates. Each experiment is written into its own subdirectories.

Public adaptor `migrate_44tasks_to_157tasks.py` responsible for:

1. Positioning project root directory, original ZXQ0QZ and Phase 7 data;
2. Switch the old Runner track/test parameter to 157-task data;
3. Redirect all result paths to the independent migration root directory;
4. Check and reject any path still pointing to the original 44-task result;
5. Write to `migration_contract.json`, recording source scriptt, data, Seed, parameters and output;
6. Validate upstream reliance only from this 157-task migration.

15 public experimental documents are separate operational portals; they share the security logic, but only trigger each time
An experiment.

Also provide `run_all_in_one.py` as a fixed table-5 main entrance.
`migrate_44tasks_to_157tasks.py --preset table5` select identical 15 entries in order of dependency
Run in a series and not allow the switch to `all` preset containing extra experiments. The main entrance defaults to skip already existing contract
and complete experiment with submary and auto-retry with only conject-only interruption experiments.

The old pseudo-equality table for E4/E4b covers only 44-task HLA and is not directly used for 157-task data.
Adaptor is now located from the warehouse at the present NetMHCpan-style 34-bit point
Builds the Migration table, writing the `_derived_inputs` subdirectories for the migration root directory. It does not modify or mix
Original 44-task pseudo-sequience table.

E10, E10b, E11 Mission-by-task indicators for E9 CAGrad are used only for post-operation external comparisons and are not involved in model training,
This is marked as optional. `run_all_in_one.py` default omitted E9 and continued E10-E14;
ZXQ0QZ visible recovery E9. Jumping-related external comparison tables do not contain CAGrad lines, but each
The training of candidate models, task-by-task indicators and aggregate indicators are not affected.

## 3. Scope of the relocation

♪ The independent entrance ♪
|---|---|---|
| `run_e0_traditional_157tasks.py` | E0 | five traditional per-task baselines |
| `run_e1_e2_e3_neural_157tasks.py` | E1/E2/E3 | single-task neural, shared heads, conditioned model |
| `run_e4_hla_pseudoseq_157tasks.py` | E4 | HLA pseudo-sequence |
| `run_e4b_hla_hybrid_157tasks.py` | E4b | HLA ID + pseudo-sequence |
| `run_e5_famo_157tasks.py` | E5 | FAMO |
| `run_e6_task_grouping_157tasks.py` | E6 | HLA/tissue grouped hard sharing |
| `run_e7_selective_grouping_157tasks.py` | E7 | selective grouping |
| `run_e8_soft_ensemble_157tasks.py` | E8 | global/HLA soft ensemble |
| `run_e9_cagrad_157tasks.py` | E9 | CAGrad |
| `run_e10_mmoe_157tasks.py` | E10 | MMoE |
| `run_e10b_mmoe_tuning_157tasks.py` | E10b | expanded MMoE configurations |
| `run_e11_dbmtl_157tasks.py` | E11 | DB-MTL |
| `run_e12_pair_ranking_157tasks.py` | E12 | BCE + pair ranking |
| `run_e13_auxiliary_157tasks.py` | E13 | tissue/HLA auxiliary supervision |
| `run_e14_auxiliary_soft_157tasks.py` | E14 | auxiliary global + HLA soft ensemble |

E17, E26 and E29 have 157-task results for Phase 7 and therefore do not re-mobilize. E30/E31 uses different uses
OOF generalization protocol, not a sandard fixed-testtable 5 migration.

## 4. Dependence and operational order

E0  E1/E2/E3  E4b  E5 E6  E7  E8  E9
E10 → E10b → E11 → E12 → E13 → E14.

Main dependency:

- E6, E7, E8 Dependence on E1/E2/E3 task-by-task results;
- E9 Dependence E1/E2/E3, E5, E7 and E8;
- E10 Dependence E1/E2/E3, E5, E7, E8 and E9;
- E10b Dependence E1/E2/E3, E8, E9 and E10;
- E11 Dependence E1/E2/E3, E8, E9, E10 and E10b;
- E12 Dependence E1/E2/E3, E8, E10 and E11;
- E13 Dependence E1/E2/E3, E8, E10 and E12;
- E14 Dependence E1/E2/E3, E8 and E13.

The downstream entrance does not automatically rerun depend on you.
Full PowerShell command is available at ZXQ0QZ.

## 5. Table 5 Summary rules

ZXQ0QZ Read only the root directory of the migration result. Default requires at least 10 different
Migration method; stop and advise to continue the experiment when insufficient, and output all rows over and over. The summary results should not be mixed into the original
44-task indicator.

Official paper Table 5 may only add rows that meet the following conditions:

1. `n_tasks=157`;
2. Train/test to Page 7 Min200 data;
3. (a) `migration_contract.json` which exists and is checked;
4. Source metrics from the independent migration results directory;
5. Indicators and aggregation are clearly stated in the table.

The 157-task values of any relocation model are not foreseen in this report until formal training and review are completed.
