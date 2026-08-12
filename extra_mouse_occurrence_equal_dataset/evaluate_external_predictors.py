"""Import and evaluate frozen external scores on mouse occurrence-equal test."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import common
from _runner import run_callable


QUERY_DIR = common.EXTERNAL_ROOT / "queries"
RAW_DIR = common.EXTERNAL_ROOT / "raw_outputs"
CACHE_DIR = common.EXTERNAL_ROOT / "score_cache"
OUTPUT_DIR = common.RESULTS_ROOT / "external_predictors"
MAIN_MODES = {
    "mhcflurry_2.2.1::presentation_score",
    "netmhcpan_4.1b::ba_rank",
    "netmhcpan_4.1b::el_rank",
}


def pair_accuracy(frame: pd.DataFrame) -> float:
    wide = frame.pivot(index="pair_id", columns="label", values="score")
    if list(wide.columns) != [0, 1] or wide.isna().any().any():
        raise ValueError("Pair accuracy requires one scored positive and negative per pair.")
    return float((wide[1] > wide[0]).mean())


def complete_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    status = frame.groupby("pair_id")["score"].agg(["size", "count"])
    keep = status.index[status["size"].eq(2) & status["count"].eq(2)]
    return frame[frame["pair_id"].isin(keep)].copy()


def evaluate_task(task: pd.DataFrame) -> dict[str, float]:
    labels = task["label"].to_numpy(dtype=np.int8)
    scores = task["score"].to_numpy(dtype=np.float64)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "pair_accuracy": pair_accuracy(task),
    }


def main() -> None:
    query_path = QUERY_DIR / "mouse_test_unique_peptide_h2.csv.gz"
    mhcflurry_raw = RAW_DIR / "mouse_test_mhcflurry.csv"
    netmhcpan_manifest = QUERY_DIR / "mouse_test_netmhcpan_manifest.csv"
    required = [query_path, mhcflurry_raw, netmhcpan_manifest]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("External inputs/outputs are incomplete:\n" + "\n".join(missing))

    common.enable_project_package_imports()
    from extra1.issue5.import_scores import import_mhcflurry, import_netmhcpan

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    mhcflurry_cache = CACHE_DIR / "mouse_test_mhcflurry.csv.gz"
    netmhcpan_cache = CACHE_DIR / "mouse_test_netmhcpan.csv.gz"
    import_mhcflurry(query_path, mhcflurry_raw, mhcflurry_cache, "2.2.1")
    import_netmhcpan(query_path, netmhcpan_manifest, netmhcpan_cache, "4.1b")

    test = pd.read_csv(common.TEST_PATH, keep_default_na=False)
    queries = pd.read_csv(query_path)
    test = test.merge(
        queries[["query_id", "peptide_sequence", "mhc_restriction"]],
        on=["peptide_sequence", "mhc_restriction"],
        how="left",
        validate="many_to_one",
    )
    if test["query_id"].isna().any():
        raise ValueError("Query table does not cover every mouse test row.")
    caches = pd.concat(
        [pd.read_csv(mhcflurry_cache), pd.read_csv(netmhcpan_cache)], ignore_index=True
    )

    prediction_parts = []
    task_rows = []
    coverage_rows = []
    summary_rows = []
    for (predictor, mode), cache in caches.groupby(["predictor", "scoring_mode"], sort=True):
        model = f"{predictor}::{mode}"
        scored = test[
            [
                "sample_id",
                "pair_id",
                "target_tissue",
                "mhc_restriction",
                "peptide_sequence",
                "label",
                "query_id",
            ]
        ].merge(
            cache[["query_id", "score", "is_supported", "missing_reason"]],
            on="query_id",
            how="left",
            validate="many_to_one",
        )
        scored.insert(0, "model", model)
        scored["is_supported"] = scored["is_supported"].fillna(False).astype(bool)
        scored["missing_reason"] = scored["missing_reason"].fillna("query_absent_from_cache")
        prediction_parts.append(scored)
        covered = complete_pairs(scored)
        per_task_model = []
        for (tissue, mhc), task in covered.groupby(
            ["target_tissue", "mhc_restriction"], sort=True
        ):
            row = {
                "model": model,
                "is_primary_mode": model in MAIN_MODES,
                "target_tissue": tissue,
                "mhc_restriction": mhc,
                "test_rows": int(len(task)),
                "complete_pairs": int(task["pair_id"].nunique()),
                **evaluate_task(task),
            }
            per_task_model.append(row)
            task_rows.append(row)

        total_pairs = int(scored["pair_id"].nunique())
        covered_pairs = int(covered["pair_id"].nunique())
        full_tasks = int(
            covered.groupby(["target_tissue", "mhc_restriction"])["pair_id"]
            .nunique().eq(50).sum()
        )
        coverage_rows.append(
            {
                "model": model,
                "test_rows": int(len(scored)),
                "scored_rows": int(scored["score"].notna().sum()),
                "row_coverage": float(scored["score"].notna().mean()),
                "test_pairs": total_pairs,
                "complete_pairs": covered_pairs,
                "complete_pair_coverage": covered_pairs / total_pairs,
                "tasks_with_complete_pairs": int(
                    covered.groupby(["target_tissue", "mhc_restriction"]).ngroups
                ),
                "fully_covered_tasks": full_tasks,
            }
        )
        if per_task_model:
            task_frame = pd.DataFrame(per_task_model)
            worst_n = min(10, len(task_frame))
            summary_rows.append(
                {
                    "model": model,
                    "is_primary_mode": model in MAIN_MODES,
                    "n_tasks": int(len(task_frame)),
                    "mean_task_auroc": float(task_frame["auroc"].mean()),
                    "mean_task_auprc": float(task_frame["auprc"].mean()),
                    "mean_task_pair_accuracy": float(task_frame["pair_accuracy"].mean()),
                    "worst_10_mean_auroc": float(
                        task_frame.nsmallest(worst_n, "auroc")["auroc"].mean()
                    ),
                    "complete_pairs": covered_pairs,
                    "complete_pair_coverage": covered_pairs / total_pairs,
                }
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(prediction_parts, ignore_index=True).to_csv(
        OUTPUT_DIR / "test_predictions.csv.gz", index=False
    )
    pd.DataFrame(task_rows).to_csv(OUTPUT_DIR / "per_task_metrics.csv", index=False)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["is_primary_mode", "mean_task_auroc"], ascending=[False, False]
    )
    summary.to_csv(OUTPUT_DIR / "summary_metrics.csv", index=False)
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(OUTPUT_DIR / "coverage_audit.csv", index=False)
    (OUTPUT_DIR / "run_settings.json").write_text(
        json.dumps(
            {
                "purpose": "Frozen external predictor fixed-test evaluation",
                "test": str(common.TEST_PATH),
                "split": "supplied standard test; no same-label split",
                "mhcflurry_version": "2.2.1",
                "mhcflurry_flanks": "disabled",
                "netmhcpan_version": "4.1b",
                "score_direction": "higher is stronger after cache transformation",
                "coverage_policy": "metrics use only complete scored pairs",
                "primary_modes": sorted(MAIN_MODES),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    primary_coverage = coverage[coverage["model"].isin(MAIN_MODES)]
    if len(primary_coverage) != 3 or not primary_coverage["complete_pair_coverage"].eq(1.0).all():
        raise ValueError("Primary external modes do not have 100% complete-pair coverage.")
    print(summary[[
        "model", "n_tasks", "mean_task_auroc", "mean_task_auprc",
        "mean_task_pair_accuracy", "complete_pair_coverage"
    ]].to_string(index=False), flush=True)


run_callable("evaluate_external_predictors.py", main)
