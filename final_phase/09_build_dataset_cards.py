#!/usr/bin/env python3
"""Build draft dataset card, reproducibility statement and ethics statement."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common import EXPERIMENTS, OUTPUT_ROOT, ensure_output, read_train


def dataset_section(species: str) -> str:
    experiment = EXPERIMENTS[species]
    train = read_train(experiment)
    return f"""## {species.capitalize()} benchmark

- Tasks: {train.task_name.nunique()}
- Tissues: {train.target_tissue.nunique()}
- MHC restrictions: {train.mhc_restriction.nunique()}
- Training pairs/rows: {train.pair_id.nunique():,}/{len(train):,}
- Unique peptides: {train.peptide_sequence.nunique():,}
- Unique parent UniProt IDs: {train.molecule_parent_uniprot_id.nunique():,}
- Peptide length: {int(train.peptide_sequence.str.len().min())}–{int(train.peptide_sequence.str.len().max())}
- Labels: one reported target-tissue peptide and one same-MHC/same-parent-protein matched negative per pair.
- Primary scope: previously observed tissue–MHC tasks.
- Strict scope: connected-component peptide-disjoint OOF; not protein-, study-, tissue- or MHC-disjoint.
"""


def optional_summary(path: Path) -> str:
    if not path.is_file():
        return "Not yet generated."
    return "\n\n" + pd.read_csv(path).to_markdown(index=False)


def main() -> None:
    output = ensure_output("09_dataset_cards")
    pairacc = OUTPUT_ROOT / "01_pairacc/pairacc_summary.csv"
    protein = OUTPUT_ROOT / "03_parent_protein_overlap/protein_overlap_summary.csv"
    provenance = OUTPUT_ROOT / "04_provenance_feasibility/provenance_feasibility.json"
    provenance_text = "Not yet generated."
    if provenance.is_file():
        payload = json.loads(provenance.read_text(encoding="utf-8"))
        provenance_text = json.dumps(payload, indent=2, ensure_ascii=False)

    dataset_card = f"""# TissuePMHC benchmark dataset card (draft)

## Purpose

The benchmarks support research on tissue-conditioned MHC-I peptide presentation preference. They are paired, balanced research benchmarks and are not estimates of natural peptide prevalence.

{dataset_section('human')}

{dataset_section('mouse')}

## Evaluation protocols

Standard evaluation is pair-disjoint but permits peptide and parent-protein reuse across tasks/folds. Strict evaluation groups all pairs connected by a shared peptide into one fold. The strict result therefore measures seen-task/unseen-peptide robustness only.

## Known limitations

- Negatives are constructed matched comparators, not experimentally confirmed biological non-presenters.
- AUPRC and threshold metrics on balanced pairs do not represent natural prevalence.
- Peptide-disjoint evaluation still permits parent-protein, study, platform and motif-family overlap.
- Existing fixed tests are internal project splits, not external cohorts.
- The models are not validated for clinical decision-making.

## Pair-ranking summary
{optional_summary(pairacc)}

## Parent-protein overlap summary
{optional_summary(protein)}

## Provenance feasibility audit

```json
{provenance_text}
```
"""
    reproducibility = """# Reproducibility statement (draft)

All reported strict results are generated from deterministic connected-component manifests. Each outer fold is trained only on fitting components. Predictions, member seeds, fold assignments, configurations, timing and data hashes are retained in the repository outputs. Task-macro metrics are computed after concatenating each task's outer-fold predictions. Ensemble members are not treated as independent statistical replicates.

The strict models were structurally frozen before metric inspection, but the work was not registered in an external preregistration service. This should be described as a frozen robustness evaluation rather than a prospective clinical validation. Exact peak GPU memory was not recorded during the completed runs and must be reported as unavailable unless a representative profiling rerun is performed.
"""
    ethics = """# Ethics and intended-use statement (draft)

This project uses public immunopeptidomics-derived records and protein/peptide annotations. It does not establish clinical validity, treatment benefit, immunogenicity or safety. Outputs are intended for computational benchmarking and hypothesis prioritization. They must not be used as stand-alone evidence for patient selection, vaccine design, therapeutic dosing or diagnostic decisions.

Dataset construction can inherit publication, tissue-sampling, allele-frequency, assay-platform and reporting biases. Performance differences across tissues or MHC groups must not be interpreted as population-level biological differences without targeted validation. Users should preserve source attribution and comply with the licenses and terms of all upstream databases and external predictors.
"""
    (output / "DATASET_CARD.md").write_text(dataset_card, encoding="utf-8")
    (output / "REPRODUCIBILITY_STATEMENT.md").write_text(reproducibility, encoding="utf-8")
    (output / "ETHICS_AND_INTENDED_USE.md").write_text(ethics, encoding="utf-8")
    print(f"wrote dataset/reproducibility/ethics drafts to {output}")


if __name__ == "__main__":
    main()
