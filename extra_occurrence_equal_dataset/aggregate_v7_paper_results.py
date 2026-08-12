#!/usr/bin/env python3
"""Aggregate Human v7 paper values from occurrence-equal results only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "results"
RERUN = RESULTS / "v7_full_rerun"
OUT = RERUN / "paper_results"
TRAIN = (ROOT / "data" / "humanPMHC_occurence_equal_dataset" / "humanPMHC_train.csv.gz").resolve()
TEST = (ROOT / "data" / "humanPMHC_occurence_equal_dataset" / "humanPMHC_test.csv.gz").resolve()
HLA_PSEUDO = (ROOT / "data" / "processed" / "hla_pseudo_sequences_occurrence_equal.csv").resolve()
SEEDS = [20260704, 20260705, 20260706]
N_TASKS = 77


# paper label, source relative to RESULTS, model key, aggregation family
SPECS = [
    ("Neural single-task MLP", "v7_full_rerun/neural_single_conditioned", "neural_single_task", "mean"),
    ("Shared encoder with task heads", "e2_shared_heads", "e2_shared_heads", "mean"),
    ("Conditioned tissue and HLA IDs", "v7_full_rerun/neural_single_conditioned", "conditioned_tissue_hla", "mean"),
    ("HLA pseudo-sequence conditioned", "v7_full_rerun/hla_pseudoseq", "conditioned_hla_pseudoseq", "mean"),
    ("HLA ID + pseudo-sequence hybrid", "v7_full_rerun/hla_hybrid", "conditioned_hla_hybrid", "mean"),
    ("FAMO shared heads", "v7_full_rerun/famo", "e5_famo", "mean"),
    ("HLA-grouped hard sharing", "v7_full_rerun/task_grouping", "hla_grouped", "mean"),
    ("Tissue-grouped hard sharing", "v7_full_rerun/task_grouping", "tissue_grouped", "mean"),
    ("Selective global/HLA grouping", "v7_full_rerun/selective_grouping", "e7_selective_hla_or_global", "mean"),
    ("Fixed global/HLA dual-branch average", "v7_full_rerun/adaptive_soft_ensemble", "e8a_fixed_average", "mean"),
    ("Validation-delta clipped ensemble", "v7_full_rerun/adaptive_soft_ensemble", "e8b_validation_delta_clipped", "mean"),
    ("Validation-softmax ensemble", "v7_full_rerun/adaptive_soft_ensemble", "e8c_validation_softmax", "mean"),
    ("MMoE", "v7_full_rerun/mmoe", "e10_mmoe", "mean"),
    ("MMoE, 4 experts x width 256", "v7_full_rerun/mmoe_tuning", "e10b_4experts_256", "mean"),
    ("MMoE, 6 experts x width 128", "v7_full_rerun/mmoe_tuning", "e10b_6experts_128", "mean"),
    ("DB-MTL shared heads", "v7_full_rerun/dbmtl", "e11_dbmtl", "mean"),
    ("Pair-ranking objective", "v7_full_rerun/pair_ranking", "e12_pair_ranking", "mean"),
    ("Tissue/HLA auxiliary supervision", "v7_full_rerun/auxiliary_tasks", "e13_aux_tissue_hla", "mean"),
    ("Auxiliary-global + plain-HLA dual branch", "e14a_auxiliary_dual_branch", "e14a_auxiliary_dual_branch", "mean"),
    ("Auxiliary-global + auxiliary-HLA dual branch", "v7_full_rerun/auxiliary_soft", "e14b_global_aux_hla_aux", "mean"),
    ("MLP dual-branch rank average", "v7_full_rerun/mlp_dual_seed_ensemble", "e17_3seed_rank_average", "aggregate"),
    ("Full multi-kernel TissuePMHC", "e29_multikernel_cnn", "e29_multikernel_cnn", "mean"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_contracts() -> list[dict[str, str]]:
    records = []
    expected_train, expected_test = str(TRAIN), str(TEST)
    for directory in sorted(path for path in RERUN.iterdir() if path.is_dir()):
        contract_path = directory / "human_run_contract.json"
        if not contract_path.is_file():
            continue
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if str(Path(contract["train"]).resolve()) != expected_train or str(Path(contract["test"]).resolve()) != expected_test:
            raise ValueError(f"Wrong Human dataset in {contract_path}")
        supplied_seeds = contract["arguments"].get("seeds")
        if supplied_seeds is not None and list(map(int, supplied_seeds)) != SEEDS:
            raise ValueError(f"Wrong seed coverage in {contract_path}: {supplied_seeds}")
        records.append({"experiment": directory.name, "contract": str(contract_path.resolve())})
    return records


def read_spec(label: str, relative: str, model: str, family: str) -> pd.DataFrame:
    path = RESULTS / relative / "per_task_metrics.csv"
    frame = pd.read_csv(path)
    selected = frame[frame["model"] == model].copy()
    if selected.empty:
        raise ValueError(f"Model {model} missing from {path}")
    selected["paper_model"] = label
    selected["aggregation_family"] = family
    selected["source_file"] = str(path.resolve())
    if family == "mean":
        observed = sorted(selected["seed"].astype(int).unique().tolist())
        counts = selected.groupby("seed").size().tolist()
        if observed != SEEDS or counts != [N_TASKS] * 3:
            raise ValueError(f"Coverage mismatch for {label}: seeds={observed}, counts={counts}")
    elif len(selected) != N_TASKS:
        raise ValueError(f"Aggregate coverage mismatch for {label}: {len(selected)}")
    return selected


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, model_rows in frame.groupby("paper_model", sort=False):
        family = model_rows["aggregation_family"].iloc[0]
        per_task = model_rows.groupby(["target_tissue", "mhc_restriction"], dropna=False, as_index=False)[
            ["auroc", "auprc", "accuracy", "mcc"]
        ].mean()
        per_seed = model_rows.groupby("seed")[["auroc", "auprc", "accuracy", "mcc"]].mean()
        rows.append({
            "paper_model": label,
            "n_tasks": len(per_task),
            "aggregation": "row-level prediction aggregate" if family == "aggregate" else "mean of seed-level metrics",
            "mean_auroc": per_seed["auroc"].mean(),
            "sd_auroc": per_seed["auroc"].std(ddof=1) if len(per_seed) > 1 else np.nan,
            "median_auroc": model_rows.groupby("seed")["auroc"].median().mean(),
            "mean_auprc": per_seed["auprc"].mean(),
            "sd_auprc": per_seed["auprc"].std(ddof=1) if len(per_seed) > 1 else np.nan,
            "mean_accuracy": per_seed["accuracy"].mean(),
            "mean_mcc": per_seed["mcc"].mean(),
            "worst10_mean_auroc": per_task.nsmallest(10, "auroc")["auroc"].mean(),
            "source_file": model_rows["source_file"].iloc[0],
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    validated = validate_contracts()
    architecture = pd.concat([read_spec(*spec) for spec in SPECS], ignore_index=True, sort=False)
    architecture.to_csv(OUT / "paper_architecture_per_task.csv", index=False)
    summary = summarize(architecture)
    summary.to_csv(OUT / "paper_architecture_summary.csv", index=False)
    provenance = {
        "policy": "Only Human occurrence-equal data/results are eligible; legacy results are forbidden.",
        "train": str(TRAIN), "test": str(TEST), "train_sha256": sha256(TRAIN), "test_sha256": sha256(TEST),
        "hla_pseudo_sequences": str(HLA_PSEUDO), "hla_pseudo_sequences_sha256": sha256(HLA_PSEUDO),
        "seeds": SEEDS, "n_tasks": N_TASKS, "validated_contracts": validated,
        "sources": summary[["paper_model", "source_file"]].to_dict("records"),
    }
    (OUT / "paper_results_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary[["paper_model", "mean_auroc", "sd_auroc", "mean_auprc", "worst10_mean_auroc"]].to_string(index=False))
    print(f"wrote: {OUT}")


if __name__ == "__main__":
    main()
