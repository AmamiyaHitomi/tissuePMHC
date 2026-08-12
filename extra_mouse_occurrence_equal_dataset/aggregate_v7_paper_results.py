#!/usr/bin/env python3
"""Build the Mouse v7 paper result source from occurrence-equal runs only."""

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
DATA_DIR = ROOT / "data" / "mousePMHC_occurence_equal_dataset"
TRAIN = (DATA_DIR / "mousePMHC_train.csv.gz").resolve()
TEST = (DATA_DIR / "mousePMHC_test.csv.gz").resolve()
H2_PSEUDO = (ROOT / "data" / "processed" / "h2_pseudo_sequences.csv").resolve()
H2_PROVENANCE = HERE / "h2_pseudo_sequence_provenance.json"
SEEDS = [20260704, 20260705, 20260706]


# paper label, source relative to RESULTS, model key, family
SPECS = [
    ("Neural single-task MLP", "v7_full_rerun/neural_single_conditioned", "neural_single_task", "architecture"),
    ("Tissue-grouped hard sharing", "v7_full_rerun/tissue_grouping", "tissue_grouped", "architecture"),
    ("Conditioned tissue and H2 IDs", "v7_full_rerun/neural_single_conditioned", "conditioned_tissue_hla", "architecture"),
    ("H2 pseudo-sequence conditioned", "v7_full_rerun/h2_pseudoseq", "conditioned_hla_pseudoseq", "architecture"),
    ("H2 ID + pseudo-sequence hybrid", "v7_full_rerun/h2_hybrid", "conditioned_hla_hybrid", "architecture"),
    ("Selective global/H2 grouping", "v7_full_rerun/selective_grouping", "e7_selective_hla_or_global", "architecture"),
    ("DB-MTL shared heads", "v7_full_rerun/dbmtl", "e11_dbmtl", "architecture"),
    ("MMoE, 4 experts x width 256", "v7_full_rerun/mmoe_tuning", "e10b_4experts_256", "architecture"),
    ("MMoE, 6 experts x width 128", "v7_full_rerun/mmoe_tuning", "e10b_6experts_128", "architecture"),
    ("Validation-softmax dual-branch ensemble", "v7_full_rerun/adaptive_soft_ensemble", "e8c_validation_softmax", "architecture"),
    ("Shared encoder with task heads", "e2_shared_heads", "e2_shared_heads", "architecture"),
    ("Validation-delta clipped dual-branch ensemble", "v7_full_rerun/adaptive_soft_ensemble", "e8b_validation_delta_clipped", "architecture"),
    ("Fixed global/H2 dual-branch average", "v7_full_rerun/adaptive_soft_ensemble", "e8a_fixed_average", "architecture"),
    ("Auxiliary-global + auxiliary-H2 dual branch", "v7_full_rerun/auxiliary_soft", "e14b_global_aux_hla_aux", "architecture"),
    ("Auxiliary-global + plain-H2 dual branch", "v7_full_rerun/auxiliary_soft", "e14a_global_aux_hla_plain", "architecture"),
    ("MLP dual-branch rank average", "v7_full_rerun/mlp_dual_seed_ensemble", "e17_3seed_rank_average", "architecture_aggregate"),
    ("Full multi-kernel TissuePMHC", "e29_multikernel_cnn", "e29_multikernel_cnn", "architecture"),
    ("CAGrad shared heads", "v7_full_rerun/cagrad", "e9_e2_cagrad", "additional"),
    ("Task-balanced shared heads", "v7_full_rerun/cagrad", "e2_task_balanced", "additional"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_contracts() -> list[dict[str, object]]:
    expected_train, expected_test = str(TRAIN), str(TEST)
    records = []
    for directory in sorted(path for path in RERUN.iterdir() if path.is_dir()):
        contract_path = directory / "transfer_contract.json"
        if not contract_path.is_file():
            continue
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if str(Path(contract["train"]).resolve()) != expected_train:
            raise ValueError(f"Wrong training dataset in {contract_path}")
        if str(Path(contract["test"]).resolve()) != expected_test:
            raise ValueError(f"Wrong test dataset in {contract_path}")
        seeds = contract["arguments"].get("seeds")
        if seeds is not None and list(map(int, seeds)) != SEEDS:
            raise ValueError(f"Wrong seeds in {contract_path}: {seeds}")
        if directory.name in {"h2_pseudoseq", "h2_hybrid"}:
            supplied = Path(contract["arguments"]["pseudo_sequences"]).resolve()
            if supplied != H2_PSEUDO:
                raise ValueError(f"Wrong H2 pseudo-sequence input in {contract_path}: {supplied}")
        records.append({"experiment": directory.name, "contract": str(contract_path.resolve()), "status": "validated"})
    return records


def read_spec(label: str, relative: str, model: str, family: str) -> pd.DataFrame:
    path = RESULTS / relative / "per_task_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    selected = frame[frame["model"] == model].copy()
    if selected.empty:
        raise ValueError(f"Model {model} not found in {path}")
    selected["paper_model"] = label
    selected["family"] = family
    selected["source_file"] = str(path.resolve())
    if family != "architecture_aggregate":
        observed = sorted(selected["seed"].astype(int).unique().tolist())
        if observed != SEEDS:
            raise ValueError(f"Seed coverage for {label}: {observed}")
        counts = selected.groupby("seed").size().tolist()
        if counts != [11, 11, 11]:
            raise ValueError(f"Task coverage for {label}: {counts}")
    elif len(selected) != 11:
        raise ValueError(f"Aggregate task coverage for {label}: {len(selected)}")
    return selected


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, model_rows in frame.groupby("paper_model", sort=False):
        family = model_rows["family"].iloc[0]
        per_task = model_rows.groupby(["target_tissue", "mhc_restriction"], as_index=False)[["auroc", "auprc", "accuracy", "mcc"]].mean()
        per_seed = model_rows.groupby("seed")[["auroc", "auprc", "accuracy", "mcc"]].mean()
        seed_medians = model_rows.groupby("seed")["auroc"].median()
        worst5 = float(per_task.nsmallest(5, "auroc")["auroc"].mean())
        row = {
            "paper_model": label,
            "family": family,
            "n_tasks": len(per_task),
            "n_independent_fits": 3 if family != "architecture_aggregate" else 3,
            "aggregation": "row-level prediction aggregate" if family == "architecture_aggregate" else "mean of seed-level metrics",
            "mean_auroc": per_seed["auroc"].mean(),
            "sd_auroc": per_seed["auroc"].std(ddof=1) if len(per_seed) > 1 else np.nan,
            "median_auroc": seed_medians.mean(),
            "mean_auprc": per_seed["auprc"].mean(),
            "sd_auprc": per_seed["auprc"].std(ddof=1) if len(per_seed) > 1 else np.nan,
            "mean_accuracy": per_seed["accuracy"].mean(),
            "mean_mcc": per_seed["mcc"].mean(),
            "worst5_mean_auroc": worst5,
            "source_file": model_rows["source_file"].iloc[0],
        }
        if "pair_accuracy" in model_rows and model_rows["pair_accuracy"].notna().any():
            row["mean_pair_accuracy"] = model_rows.groupby("seed")["pair_accuracy"].mean().mean()
        else:
            row["mean_pair_accuracy"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def controls() -> tuple[pd.DataFrame, pd.DataFrame]:
    trained = pd.read_csv(RESULTS / "e0_traditional" / "per_task_metrics.csv")
    trained["paper_model"] = trained["model"]
    trained["family"] = "traditional"
    trained["source_file"] = str((RESULTS / "e0_traditional" / "per_task_metrics.csv").resolve())
    external = pd.read_csv(RESULTS / "external_predictors" / "per_task_metrics.csv")
    external = external[external["is_primary_mode"]].copy()
    external["seed"] = 0
    external["paper_model"] = external["model"]
    external["family"] = "external"
    external["source_file"] = str((RESULTS / "external_predictors" / "per_task_metrics.csv").resolve())
    return trained, external


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    validated = validate_contracts()
    frames = [read_spec(*spec) for spec in SPECS]
    architecture = pd.concat(frames, ignore_index=True, sort=False)
    architecture.to_csv(OUT / "paper_architecture_per_task.csv", index=False)
    summary = summarize(architecture)
    summary.to_csv(OUT / "paper_architecture_summary.csv", index=False)

    traditional, external = controls()
    summarize(traditional).to_csv(OUT / "paper_traditional_summary.csv", index=False)
    ext_summary = external.groupby(["paper_model", "family"], as_index=False).agg(
        n_tasks=("auroc", "size"), mean_auroc=("auroc", "mean"), mean_auprc=("auprc", "mean"),
        mean_pair_accuracy=("pair_accuracy", "mean"), worst5_mean_auroc=("auroc", lambda x: x.nsmallest(5).mean()),
        source_file=("source_file", "first"),
    )
    ext_summary.to_csv(OUT / "paper_external_summary.csv", index=False)

    main_model = architecture[architecture["paper_model"] == "Full multi-kernel TissuePMHC"].copy()
    task_mean = main_model.groupby(["target_tissue", "mhc_restriction"], as_index=False)[["auroc", "auprc"]].mean()
    task_ranked = task_mean.sort_values("auroc").reset_index(drop=True)
    task_ranked.to_csv(OUT / "paper_tissue_h2_task_summary.csv", index=False)
    task_mean.groupby("mhc_restriction", as_index=False)[["auroc", "auprc"]].mean().to_csv(OUT / "paper_h2_summary.csv", index=False)
    task_mean.groupby("target_tissue", as_index=False)[["auroc", "auprc"]].mean().to_csv(OUT / "paper_tissue_summary.csv", index=False)
    pd.concat([task_ranked.head(5).assign(group="lowest_5"), task_ranked.tail(5).assign(group="highest_5")]).to_csv(
        OUT / "paper_task_extremes.csv", index=False
    )

    provenance = {
        "policy": "Only occurrence-equal Mouse data and results are eligible; no legacy results directory is read.",
        "train": str(TRAIN), "test": str(TEST),
        "train_sha256": sha256(TRAIN), "test_sha256": sha256(TEST),
        "h2_pseudo_sequences": str(H2_PSEUDO),
        "h2_pseudo_sequences_sha256": sha256(H2_PSEUDO),
        "h2_pseudo_sequence_provenance": json.loads(H2_PROVENANCE.read_text(encoding="utf-8")),
        "seeds": SEEDS, "n_tasks": 11,
        "validated_transfer_contracts": validated,
        "reused_basic_configuration": str((HERE / "common.py").resolve()),
        "reused_basic_configuration_check": "TRAIN_PATH and TEST_PATH are fixed to data/mousePMHC_occurence_equal_dataset; each retained model has seeds 20260704--20260706 and 11 tasks per seed",
        "architecture_sources": summary[["paper_model", "source_file"]].to_dict("records"),
        "traditional_source": str((RESULTS / "e0_traditional" / "per_task_metrics.csv").resolve()),
        "external_source": str((RESULTS / "external_predictors" / "per_task_metrics.csv").resolve()),
    }
    (OUT / "paper_results_provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary[["paper_model", "mean_auroc", "sd_auroc", "mean_auprc", "worst5_mean_auroc"]].to_string(index=False))
    print(f"wrote: {OUT}")


if __name__ == "__main__":
    main()
