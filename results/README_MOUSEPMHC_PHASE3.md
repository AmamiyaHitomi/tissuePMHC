# mousePMHC Phase 3 Result Naming Convention

All mouse Phase 3 results use `mousePMHC_phase3_<experiment>` and are kept completely separate from the human `tissuePMHC_<experiment>` results.

## Current Validity

On 2026-07-12, Phase 3 was rolled back to the pair-count selection stage using `min_pairs > 200`. The E2+, E14a, and E29 directories from the old five-task benchmark with the `>500` threshold are retained only for historical traceability and must not be used as current results. The E0/E1 directories are retained, but their existing OOF results also correspond only to the old `>500` data; E0/E1 must be rerun after the new benchmark is established. See `phase3/ROADMAP_PHASE_3.md` for the detailed rules.

| Directory | Type | Official result? |
|---|---|---|
| `mousePMHC_phase3_data_audit/` | Fixed data audit | Yes |
| `mousePMHC_phase3_e0_oof/` | E0 traditional-model train-only OOF | Yes |
| `mousePMHC_phase3_e1_oof/` | mousePMHC E1 shared-task-head MTL train-only OOF | Yes |

Subsequent experiments must follow the same convention. Directories containing `_smoke` must not be included in official performance tables or cross-species conclusions.

The mouse E1 method was derived from human `tissuePMHC E2`, but it is independently numbered E1 in the mouse research track. Historical E14a/E29 results are retained only as records of completed exploration; their mouse execution scripts are no longer retained, and they are not part of the current official workflow.
