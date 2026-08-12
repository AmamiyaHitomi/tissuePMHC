#!/usr/bin/env python3
"""Import and evaluate NetMHCpan/MHCflurry scores on premium fixed test."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import common


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
        raise ValueError("PairAcc requires one scored positive and negative per pair.")
    return float((wide[1] > wide[0]).mean())


def complete_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    status = frame.groupby("pair_id")["score"].agg(["size", "count"])
    keep = status.index[(status["size"] == 2) & (status["count"] == 2)]
    return frame[frame["pair_id"].isin(keep)].copy()


def evaluate_task(task: pd.DataFrame) -> dict[str, float]:
    y = task["label"].to_numpy(dtype=np.int8)
    score = task["score"].to_numpy(dtype=np.float64)
    return {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "pair_accuracy": pair_accuracy(task),
    }


def main() -> None:
    query_path = QUERY_DIR / "premium_test_unique_peptide_hla.csv.gz"
    mhcflurry_raw = RAW_DIR / "premium_test_mhcflurry.csv"
    netmhcpan_manifest = QUERY_DIR / "premium_test_netmhcpan_manifest.csv"
    required = [query_path, mhcflurry_raw, netmhcpan_manifest]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "External predictor inputs/outputs are incomplete:\n" + "\n".join(missing)
        )

    common.enable_project_package_imports()
    from extra1.issue5.import_scores import import_mhcflurry, import_netmhcpan

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    mhcflurry_cache = CACHE_DIR / "premium_test_mhcflurry.csv.gz"
    netmhcpan_cache = CACHE_DIR / "premium_test_netmhcpan.csv.gz"
    import_mhcflurry(query_path, mhcflurry_raw, mhcflurry_cache, "2.2.1")
    import_netmhcpan(
        query_path,
        netmhcpan_manifest,
        netmhcpan_cache,
        "4.1b",
    )

    test = pd.read_csv(common.TEST_PATH)
    queries = pd.read_csv(query_path)
    test = test.merge(
        queries[
            ["query_id", "peptide_sequence", "mhc_restriction"]
        ],
        on=["peptide_sequence", "mhc_restriction"],
        how="left",
        validate="many_to_one",
    )
    if test["query_id"].isna().any():
        raise ValueError("Query table does not cover every premium test row.")

    caches = pd.concat(
        [pd.read_csv(mhcflurry_cache), pd.read_csv(netmhcpan_cache)],
        ignore_index=True,
    )
    prediction_parts: list[pd.DataFrame] = []
    task_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for (predictor, mode), cache in caches.groupby(
        ["predictor", "scoring_mode"], sort=True
    ):
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
        scored["missing_reason"] = scored["missing_reason"].fillna(
            "query_absent_from_cache"
        )
        prediction_parts.append(scored)

        covered = complete_pairs(scored)
        per_task_model: list[dict[str, object]] = []
        for (tissue, hla), task in covered.groupby(
            ["target_tissue", "mhc_restriction"], sort=True
        ):
            metrics = evaluate_task(task)
            row = {
                "model": model,
                "is_primary_mode": model in MAIN_MODES,
                "target_tissue": tissue,
                "mhc_restriction": hla,
                "test_rows": int(len(task)),
                "complete_pairs": int(task["pair_id"].nunique()),
                **metrics,
            }
            per_task_model.append(row)
            task_rows.append(row)

        row_coverage = float(scored["score"].notna().mean())
        total_pairs = int(scored["pair_id"].nunique())
        covered_pairs = int(covered["pair_id"].nunique())
        full_tasks = int(
            covered.groupby(["target_tissue", "mhc_restriction"])["pair_id"]
            .nunique()
            .eq(50)
            .sum()
        )
        coverage_rows.append(
            {
                "model": model,
                "test_rows": int(len(scored)),
                "scored_rows": int(scored["score"].notna().sum()),
                "row_coverage": row_coverage,
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
                    "mean_task_pair_accuracy": float(
                        task_frame["pair_accuracy"].mean()
                    ),
                    "worst_10_mean_auroc": float(
                        task_frame.nsmallest(worst_n, "auroc")["auroc"].mean()
                    ),
                    "complete_pairs": covered_pairs,
                    "complete_pair_coverage": covered_pairs / total_pairs,
                }
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    predictions.to_csv(OUTPUT_DIR / "test_predictions.csv.gz", index=False)
    pd.DataFrame(task_rows).to_csv(OUTPUT_DIR / "per_task_metrics.csv", index=False)
    pd.DataFrame(summary_rows).sort_values(
        ["is_primary_mode", "mean_task_auroc"],
        ascending=[False, False],
    ).to_csv(OUTPUT_DIR / "summary_metrics.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(
        OUTPUT_DIR / "coverage_audit.csv", index=False
    )
    (OUTPUT_DIR / "run_settings.json").write_text(
        json.dumps(
            {
                "purpose": "Frozen external predictor fixed-test evaluation",
                "test": str(common.TEST_PATH),
                "query_file": str(query_path),
                "mhcflurry_version": "2.2.1",
                "mhcflurry_flanks": "disabled",
                "netmhcpan_version": "4.1b",
                "score_direction": "all cached scores are transformed so higher is stronger",
                "coverage_policy": "metrics use only pairs with both rows scored",
                "primary_modes": sorted(MAIN_MODES),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote: {OUTPUT_DIR / 'summary_metrics.csv'}", flush=True)
    print(
        pd.DataFrame(summary_rows)
        .sort_values("mean_task_auroc", ascending=False)
        [
            [
                "model",
                "n_tasks",
                "mean_task_auroc",
                "mean_task_auprc",
                "mean_task_pair_accuracy",
                "complete_pair_coverage",
            ]
        ]
        .to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()

