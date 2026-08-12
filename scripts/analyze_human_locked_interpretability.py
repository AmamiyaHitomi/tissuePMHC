#!/usr/bin/env python3
"""Compare locked-auxiliary and no-auxiliary SHAP/ISM explanations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "human_locked_interpretability_v1"


def shap_vector(root: Path, branch: str) -> pd.DataFrame:
    frame = pd.read_csv(root / "shap" / "shap_residue_position_summary.csv.gz", keep_default_na=False)
    source = frame[
        (frame.seed.astype(str) == "seed_mean")
        & (frame.branch == branch)
        & (frame.scope_type == "overall")
        & (frame.scope_value == "ALL")
        & (frame.label_group == "all")
    ][["position", "amino_acid", "mean_shap", "mean_abs_shap"]]
    if len(source) != 180:
        raise ValueError(f"Expected 180 SHAP cells for {branch}, observed {len(source)}")
    return source.sort_values(["position", "amino_acid"]).reset_index(drop=True)


def ism_frame(root: Path) -> pd.DataFrame:
    path = root / "ism" / "mutation_effects_three_seed_mean.csv.gz"
    frame = pd.read_csv(path, keep_default_na=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    locked = args.input / "locked_auxiliary"
    noaux = args.input / "no_auxiliary"
    if not all((root / "ism" / "metadata.json").is_file() for root in (locked, noaux)):
        raise RuntimeError("Locked and no-auxiliary interpretability runs must both be complete")

    rows: list[dict[str, object]] = []
    for branch in ("global_aux", "hla_plain", "descriptive_branch_consensus"):
        left, right = shap_vector(locked, branch), shap_vector(noaux, branch)
        if not left[["position", "amino_acid"]].equals(right[["position", "amino_acid"]]):
            raise AssertionError("SHAP cell alignment failed")
        for value in ("mean_shap", "mean_abs_shap"):
            correlation = spearmanr(left[value], right[value]).statistic
            rows.append({
                "method": "SHAP",
                "branch": branch,
                "quantity": value,
                "spearman_locked_vs_noaux": float(correlation),
                "mean_absolute_difference": float(np.mean(np.abs(left[value] - right[value]))),
                "observations": len(left),
            })

    locked_ism, noaux_ism = ism_frame(locked), ism_frame(noaux)
    keys = ["sample_id", "position", "mutant_amino_acid"]
    merged = locked_ism.merge(noaux_ism, on=keys, suffixes=("_locked", "_noaux"), validate="one_to_one")
    for quantity in (
        "delta_logit_global", "delta_logit_hla", "delta_rank_fusion",
        "classification_support_loss_fusion",
    ):
        left = merged[f"{quantity}_locked"]
        right = merged[f"{quantity}_noaux"]
        rows.append({
            "method": "ISM",
            "branch": "rank_fusion" if "fusion" in quantity else quantity.split("_")[-1],
            "quantity": quantity,
            "spearman_locked_vs_noaux": float(spearmanr(left, right).statistic),
            "mean_absolute_difference": float(np.mean(np.abs(left - right))),
            "observations": len(merged),
        })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(args.input / "locked_vs_noaux_explanation_consistency.csv", index=False)

    position_rows: list[pd.DataFrame] = []
    for condition, frame in (("locked_auxiliary", locked_ism), ("no_auxiliary", noaux_ism)):
        item = frame.groupby("position", as_index=False).agg(
            mean_abs_delta_rank_fusion=("delta_rank_fusion", lambda values: float(np.mean(np.abs(values)))),
            mean_classification_support_loss=("classification_support_loss_fusion", "mean"),
        )
        item.insert(0, "condition", condition)
        item["sensitivity_rank"] = item.mean_abs_delta_rank_fusion.rank(ascending=False, method="min").astype(int)
        position_rows.append(item)
    positions = pd.concat(position_rows, ignore_index=True)
    positions.to_csv(args.input / "locked_vs_noaux_position_ranking.csv", index=False)
    (args.input / "comparison_summary.json").write_text(json.dumps({
        "selection_use": False,
        "shap_consensus_spearman": float(comparison.loc[
            (comparison.method == "SHAP")
            & (comparison.branch == "descriptive_branch_consensus")
            & (comparison.quantity == "mean_shap"),
            "spearman_locked_vs_noaux",
        ].iloc[0]),
        "ism_rank_fusion_spearman": float(comparison.loc[
            (comparison.method == "ISM") & (comparison.quantity == "delta_rank_fusion"),
            "spearman_locked_vs_noaux",
        ].iloc[0]),
    }, indent=2), encoding="utf-8")
    print(f"[WROTE] {args.input}", flush=True)


if __name__ == "__main__":
    main()
