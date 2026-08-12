#!/usr/bin/env python3
"""Counterfactual same-HLA, different-tissue ISM comparison for E29.

For every HLA represented by at least two tissue tasks, this analysis fixes a
peptide sequence and a single-amino-acid substitution, then changes only the
tissue--HLA task head.  It reuses the frozen three-seed checkpoints and the
same reproducibly selected test-peptide panel as the formal Expected-IG/ISM
analysis.  Tissue dispersion is compared with training-seed dispersion so
that deterministic but unstable task-head differences are not overinterpreted.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_occurrence_equal_e29_ism as ism_run  # noqa: E402
import run_occurrence_equal_e29_shap as shap_run  # noqa: E402


base = shap_run.base
DEFAULT_SEEDS = shap_run.DEFAULT_SEEDS
UNIT_COLUMNS = [
    "peptide_sequence", "position", "original_amino_acid", "mutant_amino_acid",
]
METRICS = {
    "global_logit": "delta_logit_global",
    "hla_logit": "delta_logit_hla",
    "rank_fusion": "delta_rank_fusion",
}


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 1.0
    for rank_index in range(len(values) - 1, -1, -1):
        original_index = order[rank_index]
        rank = rank_index + 1
        running = min(running, values[original_index] * len(values) / rank)
        adjusted[original_index] = running
    return np.clip(adjusted, 0.0, 1.0)


def bootstrap_mean_difference(
    tissue: np.ndarray,
    seed_noise: np.ndarray,
    random_seed: int,
    replicates: int,
) -> tuple[float, float]:
    tissue = np.asarray(tissue, dtype=float)
    seed_noise = np.asarray(seed_noise, dtype=float)
    difference = tissue - seed_noise
    if len(difference) == 0:
        return math.nan, math.nan
    if len(difference) == 1 or replicates <= 0:
        return float(difference[0]), float(difference[0])
    rng = np.random.default_rng(random_seed)
    means = np.empty(replicates, dtype=float)
    chunk_size = min(200, replicates)
    written = 0
    while written < replicates:
        take = min(chunk_size, replicates - written)
        indices = rng.integers(0, len(difference), size=(take, len(difference)))
        means[written : written + take] = difference[indices].mean(axis=1)
        written += take
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def safe_hla_name(hla: str) -> str:
    return hla.replace("*", "_").replace(":", "_").replace("/", "_")


def select_panel(
    test: pd.DataFrame,
    pairs_per_task: int,
    sample_seed: int,
    smoke: bool,
    smoke_peptides: int,
) -> tuple[pd.DataFrame, list[str]]:
    selected_parts = []
    for task_name, group in test.groupby("task_name", sort=True):
        selected_parts.append(
            shap_run.select_complete_pairs(
                group,
                pairs_per_task,
                shap_run.stable_seed(task_name, sample_seed),
            )
        )
    selected = pd.concat(selected_parts, ignore_index=True)
    tissue_counts = test.groupby("mhc_restriction")["task_name"].nunique()
    eligible_hlas = sorted(tissue_counts[tissue_counts >= 2].index.astype(str))
    if smoke:
        eligible_hlas = eligible_hlas[:2]
    selected = selected[selected["mhc_restriction"].isin(eligible_hlas)].copy()
    panel = (
        selected[["mhc_restriction", "peptide_sequence"]]
        .drop_duplicates()
        .sort_values(["mhc_restriction", "peptide_sequence"])
        .reset_index(drop=True)
    )
    if smoke:
        panel = panel.groupby("mhc_restriction", as_index=False, group_keys=False).head(smoke_peptides)
    return panel, eligible_hlas


def build_context_frame(
    hla: str,
    peptides: pd.DataFrame,
    tasks: pd.DataFrame,
) -> pd.DataFrame:
    left = peptides[["peptide_sequence"]].drop_duplicates().assign(_join=1)
    right = tasks[["target_tissue", "mhc_restriction", "task_name"]].drop_duplicates().assign(_join=1)
    context = left.merge(right, on="_join", how="inner").drop(columns="_join")
    context = context.sort_values(["task_name", "peptide_sequence"]).reset_index(drop=True)
    context.insert(0, "sample_id", [f"cross_context_{safe_hla_name(hla)}_{i:06d}" for i in range(len(context))])
    context["pair_id"] = context["sample_id"]
    context["label"] = 1
    return context


def insertion_percentile_ranks(
    reference: pd.DataFrame,
    mutation_rows: pd.DataFrame,
    score_column: str,
    value_column: str,
) -> np.ndarray:
    """Rank a counterfactual candidate after inserting it into a fixed task cohort."""
    output = np.empty(len(mutation_rows), dtype=np.float64)
    references = {
        task: np.sort(group[score_column].to_numpy(dtype=float))
        for task, group in reference.groupby("task_name", sort=False)
    }
    for task_name, indices in mutation_rows.groupby("task_name", sort=False).indices.items():
        indices = np.asarray(indices, dtype=np.int64)
        values = mutation_rows.iloc[indices][value_column].to_numpy(dtype=float)
        scores = references[task_name]
        left = np.searchsorted(scores, values, side="left")
        right = np.searchsorted(scores, values, side="right")
        average_inserted_rank = (left + right + 2.0) / 2.0
        output[indices] = average_inserted_rank / (len(scores) + 1.0)
    return output


def combine_context_branches(
    global_effects: pd.DataFrame,
    hla_effects: pd.DataFrame,
    reference: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    mutation_keys = [
        *ism_run.KEY_COLUMNS,
        "position", "original_amino_acid", "mutant_amino_acid",
    ]
    global_small = global_effects[mutation_keys + [
        "original_logit", "mutant_logit", "original_probability", "mutant_probability",
    ]].rename(columns={
        "original_logit": "original_logit_global",
        "mutant_logit": "mutant_logit_global",
        "original_probability": "original_probability_global",
        "mutant_probability": "mutant_probability_global",
    })
    hla_small = hla_effects[mutation_keys + [
        "original_logit", "mutant_logit", "original_probability", "mutant_probability",
    ]].rename(columns={
        "original_logit": "original_logit_hla",
        "mutant_logit": "mutant_logit_hla",
        "original_probability": "original_probability_hla",
        "mutant_probability": "mutant_probability_hla",
    })
    merged = global_small.merge(hla_small, on=mutation_keys, validate="one_to_one")
    merged["delta_logit_global"] = merged["mutant_logit_global"] - merged["original_logit_global"]
    merged["delta_logit_hla"] = merged["mutant_logit_hla"] - merged["original_logit_hla"]
    original_rank_global = insertion_percentile_ranks(
        reference, merged, "global_score", "original_probability_global",
    )
    mutant_rank_global = insertion_percentile_ranks(
        reference, merged, "global_score", "mutant_probability_global",
    )
    original_rank_hla = insertion_percentile_ranks(
        reference, merged, "hla_score", "original_probability_hla",
    )
    mutant_rank_hla = insertion_percentile_ranks(
        reference, merged, "hla_score", "mutant_probability_hla",
    )
    merged["delta_rank_fusion"] = 0.5 * (
        mutant_rank_global + mutant_rank_hla - original_rank_global - original_rank_hla
    )
    compact = merged[[
        "target_tissue", "mhc_restriction", "task_name", *UNIT_COLUMNS,
        "delta_logit_global", "delta_logit_hla", "delta_rank_fusion",
    ]].copy()
    compact.insert(0, "seed", seed)
    return compact.sort_values(["task_name", *UNIT_COLUMNS]).reset_index(drop=True)


def mean_pairwise_absolute(matrix: np.ndarray) -> np.ndarray:
    pair_values = [np.abs(matrix[:, a] - matrix[:, b]) for a, b in combinations(range(matrix.shape[1]), 2)]
    return np.mean(np.column_stack(pair_values), axis=1)


def hla_dispersion_tables(
    hla: str,
    effects: pd.DataFrame,
    bootstrap_replicates: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    task_names = sorted(effects["task_name"].unique())
    seeds = sorted(pd.to_numeric(effects["seed"]).astype(int).unique())
    unit_index = effects[UNIT_COLUMNS].drop_duplicates().sort_values(UNIT_COLUMNS)
    peptide_rows: list[pd.DataFrame] = []
    position_rows: list[dict[str, Any]] = []
    hla_record: dict[str, Any] = {
        "mhc_restriction": hla,
        "n_tissues": len(task_names),
        "n_tissue_pairs": math.comb(len(task_names), 2),
        "n_peptides": int(effects["peptide_sequence"].nunique()),
        "n_mutation_units": len(unit_index),
    }
    for metric_name, value_column in METRICS.items():
        ensemble = effects.groupby(["task_name", *UNIT_COLUMNS], as_index=False)[value_column].mean()
        tissue_matrix = ensemble.pivot(index=UNIT_COLUMNS, columns="task_name", values=value_column).reindex(unit_index.set_index(UNIT_COLUMNS).index)
        if tissue_matrix.isna().any().any():
            raise AssertionError(f"Incomplete tissue grid for {hla}, metric={metric_name}")
        tissue_dispersion = mean_pairwise_absolute(tissue_matrix.to_numpy(dtype=float))

        seed_differences = []
        for task_name in task_names:
            task = effects[effects["task_name"] == task_name]
            seed_matrix = task.pivot(index=UNIT_COLUMNS, columns="seed", values=value_column).reindex(unit_index.set_index(UNIT_COLUMNS).index)
            if seed_matrix.isna().any().any():
                raise AssertionError(f"Incomplete seed grid for {hla}/{task_name}, metric={metric_name}")
            seed_differences.append(mean_pairwise_absolute(seed_matrix.to_numpy(dtype=float)))
        seed_dispersion = np.mean(np.column_stack(seed_differences), axis=1)

        unit = unit_index.copy()
        unit["tissue_dispersion"] = tissue_dispersion
        unit["seed_dispersion"] = seed_dispersion
        by_peptide = unit.groupby("peptide_sequence", as_index=False).agg(
            tissue_dispersion=("tissue_dispersion", "mean"),
            seed_dispersion=("seed_dispersion", "mean"),
        )
        by_peptide.insert(0, "metric", metric_name)
        by_peptide.insert(0, "mhc_restriction", hla)
        peptide_rows.append(by_peptide)

        try:
            statistic, p_value = wilcoxon(
                by_peptide["tissue_dispersion"],
                by_peptide["seed_dispersion"],
                alternative="greater",
                zero_method="wilcox",
            )
        except ValueError:
            statistic, p_value = math.nan, 1.0
        low, high = bootstrap_mean_difference(
            by_peptide["tissue_dispersion"].to_numpy(),
            by_peptide["seed_dispersion"].to_numpy(),
            random_seed + shap_run.stable_seed(f"{hla}|{metric_name}", random_seed),
            bootstrap_replicates,
        )
        tissue_mean = float(by_peptide["tissue_dispersion"].mean())
        seed_mean = float(by_peptide["seed_dispersion"].mean())
        prefix = metric_name
        hla_record[f"{prefix}_tissue_dispersion"] = tissue_mean
        hla_record[f"{prefix}_seed_dispersion"] = seed_mean
        hla_record[f"{prefix}_dispersion_ratio"] = tissue_mean / seed_mean if seed_mean > 0 else math.nan
        hla_record[f"{prefix}_mean_difference"] = tissue_mean - seed_mean
        hla_record[f"{prefix}_bootstrap_95_ci_low"] = low
        hla_record[f"{prefix}_bootstrap_95_ci_high"] = high
        hla_record[f"{prefix}_wilcoxon_statistic"] = float(statistic)
        hla_record[f"{prefix}_p_value_one_sided"] = float(p_value)

        unit_position = unit.groupby("position", as_index=False).agg(
            tissue_dispersion=("tissue_dispersion", "mean"),
            seed_dispersion=("seed_dispersion", "mean"),
            n_peptides=("peptide_sequence", "nunique"),
            n_mutation_units=("peptide_sequence", "size"),
        )
        for row in unit_position.itertuples(index=False):
            position_rows.append({
                "mhc_restriction": hla,
                "metric": metric_name,
                "position": int(row.position),
                "n_peptides": int(row.n_peptides),
                "n_mutation_units": int(row.n_mutation_units),
                "tissue_dispersion": float(row.tissue_dispersion),
                "seed_dispersion": float(row.seed_dispersion),
                "dispersion_ratio": float(row.tissue_dispersion / row.seed_dispersion) if row.seed_dispersion > 0 else math.nan,
            })
    return pd.DataFrame([hla_record]), pd.concat(peptide_rows, ignore_index=True), pd.DataFrame(position_rows)


def task_pair_tables(hla: str, effects: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensemble = effects.groupby(["target_tissue", "task_name", *UNIT_COLUMNS], as_index=False).agg(
        **{value: (value, "mean") for value in METRICS.values()}
    )
    pair_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    task_metadata = ensemble[["target_tissue", "task_name"]].drop_duplicates().sort_values("task_name")
    seeds = sorted(pd.to_numeric(effects["seed"]).astype(int).unique())
    for left_row, right_row in combinations(task_metadata.itertuples(index=False), 2):
        left = ensemble[ensemble["task_name"] == left_row.task_name]
        right = ensemble[ensemble["task_name"] == right_row.task_name]
        aligned = left.merge(right, on=UNIT_COLUMNS, validate="one_to_one", suffixes=("_a", "_b"))
        record: dict[str, Any] = {
            "mhc_restriction": hla,
            "tissue_a": left_row.target_tissue,
            "tissue_b": right_row.target_tissue,
            "task_a": left_row.task_name,
            "task_b": right_row.task_name,
            "n_mutation_units": len(aligned),
        }
        for metric_name, value_column in METRICS.items():
            a = aligned[f"{value_column}_a"].to_numpy(dtype=float)
            b = aligned[f"{value_column}_b"].to_numpy(dtype=float)
            rho = spearmanr(a, b).statistic
            record[f"{metric_name}_spearman_r"] = float(rho)
            record[f"{metric_name}_mean_abs_difference"] = float(np.mean(np.abs(a - b)))
            nonzero = (np.abs(a) > 1e-12) | (np.abs(b) > 1e-12)
            record[f"{metric_name}_sign_disagreement"] = float(np.mean(np.sign(a[nonzero]) != np.sign(b[nonzero])))

            contrasts: dict[int, np.ndarray] = {}
            for seed in seeds:
                seed_effect = effects[pd.to_numeric(effects["seed"]).astype(int) == seed]
                seed_a = seed_effect[seed_effect["task_name"] == left_row.task_name][UNIT_COLUMNS + [value_column]]
                seed_b = seed_effect[seed_effect["task_name"] == right_row.task_name][UNIT_COLUMNS + [value_column]]
                seed_aligned = seed_a.merge(seed_b, on=UNIT_COLUMNS, validate="one_to_one", suffixes=("_a", "_b"))
                contrasts[seed] = (
                    seed_aligned[f"{value_column}_a"].to_numpy(dtype=float)
                    - seed_aligned[f"{value_column}_b"].to_numpy(dtype=float)
                )
            contrast_correlations = []
            for seed_a, seed_b in combinations(seeds, 2):
                contrast_rho = spearmanr(contrasts[seed_a], contrasts[seed_b]).statistic
                contrast_correlations.append(float(contrast_rho))
                stability_rows.append({
                    "mhc_restriction": hla,
                    "tissue_a": left_row.target_tissue,
                    "tissue_b": right_row.target_tissue,
                    "metric": metric_name,
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "n_mutation_units": len(contrasts[seed_a]),
                    "contrast_spearman_r": float(contrast_rho),
                })
            record[f"{metric_name}_median_seed_contrast_spearman_r"] = float(np.median(contrast_correlations))
        pair_rows.append(record)
    return pd.DataFrame(pair_rows), pd.DataFrame(stability_rows)


def bootstrap_dispersion_ratios(
    peptide_summary: pd.DataFrame,
    replicates: int,
    random_seed: int,
) -> pd.DataFrame:
    """Paired peptide bootstrap intervals for the tissue-to-seed dispersion ratio."""
    chosen = peptide_summary[peptide_summary["metric"] == "rank_fusion"]
    rows = []
    for hla_index, (hla, group) in enumerate(chosen.groupby("mhc_restriction", sort=True)):
        tissue = group["tissue_dispersion"].to_numpy(dtype=float)
        seed = group["seed_dispersion"].to_numpy(dtype=float)
        point = float(tissue.mean() / seed.mean())
        rng = np.random.default_rng(random_seed + hla_index)
        ratios = np.empty(replicates, dtype=float)
        written = 0
        while written < replicates:
            take = min(200, replicates - written)
            indices = rng.integers(0, len(group), size=(take, len(group)))
            ratios[written : written + take] = (
                tissue[indices].mean(axis=1) / seed[indices].mean(axis=1)
            )
            written += take
        low, high = np.quantile(ratios, [0.025, 0.975])
        rows.append({
            "mhc_restriction": hla,
            "rank_fusion_dispersion_ratio": point,
            "ratio_ci_low": float(low),
            "ratio_ci_high": float(high),
            "n_peptides": len(group),
        })
    return pd.DataFrame(rows)


def hla_a0201_fingerprint(raw_dir: Path) -> pd.DataFrame:
    """Return tissue-by-position deviations for the representative HLA-A*02:01 panel."""
    path = raw_dir / "cross_tissue_effects_HLA-A_02_01.csv.gz"
    effects = pd.read_csv(
        path,
        usecols=["seed", "target_tissue", *UNIT_COLUMNS, "delta_rank_fusion"],
        keep_default_na=False,
    )
    ensemble = effects.groupby(
        ["target_tissue", *UNIT_COLUMNS], as_index=False
    )["delta_rank_fusion"].mean()
    sensitivity = (
        ensemble.assign(abs_effect=ensemble["delta_rank_fusion"].abs())
        .groupby(["target_tissue", "position"], as_index=False)["abs_effect"].mean()
    )
    matrix = sensitivity.pivot(index="target_tissue", columns="position", values="abs_effect")
    matrix = matrix.drop(index="NA", errors="ignore").reindex(columns=range(1, 10))
    tissue_order = [
        "blood", "bone", "brain", "breast", "lung", "lymph node", "lymphoid", "ovary", "uterine cervix",
    ]
    matrix = matrix.reindex([tissue for tissue in tissue_order if tissue in matrix.index])
    return matrix.subtract(matrix.mean(axis=0), axis="columns")


def create_figure(
    ratio_intervals: pd.DataFrame,
    raw_dir: Path,
    output_dir: Path,
) -> None:
    locus_colors = {"A": "#4C78A8", "B": "#F58518", "C": "#54A24B"}
    ordered = ratio_intervals.sort_values("rank_fusion_dispersion_ratio").reset_index(drop=True)
    log_ratio = np.log2(ordered["rank_fusion_dispersion_ratio"].to_numpy(dtype=float))
    low = np.log2(ordered["ratio_ci_low"].to_numpy(dtype=float))
    high = np.log2(ordered["ratio_ci_high"].to_numpy(dtype=float))
    y = np.arange(len(ordered))
    fig, axes = plt.subplots(
        2, 1, figsize=(7.4, 8.8), constrained_layout=True,
        gridspec_kw={"height_ratios": [1.65, 1]},
    )
    ax = axes[0]
    for index, row in ordered.iterrows():
        locus = row.mhc_restriction.split("*")[0].replace("HLA-", "")
        color = locus_colors[locus]
        ax.hlines(index, 0, log_ratio[index], color=color, alpha=0.55, linewidth=1.3)
        ax.errorbar(
            log_ratio[index], index,
            xerr=[[log_ratio[index] - low[index]], [high[index] - log_ratio[index]]],
            fmt="none", ecolor=color, elinewidth=1.15, capsize=2.2, zorder=2,
        )
        ax.scatter(log_ratio[index], index, s=26, color=color, edgecolor="white", linewidth=0.45, zorder=3)
    ax.axvline(0, color="#333333", linestyle="--", linewidth=1)
    ax.set_yticks(y, ordered["mhc_restriction"])
    ax.set_xlabel(r"$\log_2$(tissue dispersion / seed dispersion)")
    ax.set_title("Tissue-task perturbation differences exceed seed variation for every HLA")
    ax.tick_params(axis="y", labelsize=7.0)
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.6)
    ax.set_axisbelow(True)
    handles = [
        plt.Line2D([], [], marker="o", color="w", markerfacecolor=color, markersize=6, label=f"HLA-{locus}")
        for locus, color in locus_colors.items()
    ]
    ax.legend(handles=handles, frameon=False, ncol=3, loc="lower right", fontsize=8)

    fingerprint = hla_a0201_fingerprint(raw_dir)
    ax = axes[1]
    maximum = float(np.nanmax(np.abs(fingerprint.to_numpy(dtype=float))))
    image = ax.imshow(
        fingerprint.to_numpy(dtype=float), aspect="auto", cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum),
    )
    ax.set_xticks(np.arange(9), [f"P{position}" for position in range(1, 10)])
    ax.set_yticks(np.arange(len(fingerprint.index)), fingerprint.index)
    ax.set_title("HLA-A*02:01 tissue perturbation fingerprint")
    ax.set_ylabel("Tissue task")
    for label in ax.get_xticklabels():
        if label.get_text() in {"P2", "P3", "P9"}:
            label.set_fontweight("bold")
    for row_index in range(fingerprint.shape[0]):
        for col_index in range(fingerprint.shape[1]):
            value = fingerprint.iat[row_index, col_index]
            color = "white" if abs(value) > maximum * 0.55 else "#222222"
            ax.text(col_index, row_index, f"{value:+.3f}", ha="center", va="center", fontsize=6.6, color=color)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    colorbar.set_label("Deviation from positional mean absolute effect", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"06_cross_tissue_ism_comparison.{suffix}", dpi=320, bbox_inches="tight")
    plt.close(fig)


def write_report(
    output_dir: Path,
    hla_summary: pd.DataFrame,
    task_pairs: pd.DataFrame,
    position_summary: pd.DataFrame,
    elapsed: float,
) -> None:
    fusion_positions = (
        position_summary[position_summary["metric"] == "rank_fusion"]
        .groupby("position", as_index=False)
        .agg(tissue_dispersion=("tissue_dispersion", "mean"), seed_dispersion=("seed_dispersion", "mean"))
        .sort_values("tissue_dispersion", ascending=False)
    )
    significant = int((hla_summary["rank_fusion_fdr_bh"] < 0.05).sum())
    ratio_above_one = int((hla_summary["rank_fusion_dispersion_ratio"] > 1.0).sum())
    eligible_tasks = int(hla_summary["n_tissues"].sum())
    ensemble_context_mutations = int(
        (hla_summary["n_mutation_units"] * hla_summary["n_tissues"]).sum()
    )
    lines = [
        "# Same-HLA, different-tissue ISM comparison",
        "",
        "## Design",
        "",
        f"- Eligible HLA alleles: {len(hla_summary)}.",
        f"- Eligible tissue--HLA tasks: {eligible_tasks}.",
        f"- Tissue-task pairs: {len(task_pairs)}.",
        f"- Unique peptide panels across HLA alleles: {int(hla_summary['n_peptides'].sum())} (counted within HLA).",
        f"- Counterfactual context--mutation rows: {ensemble_context_mutations:,} per ensemble and {ensemble_context_mutations * 3:,} across three seeds.",
        "- Each peptide, position, and amino-acid replacement was scored under every represented tissue task for the same HLA.",
        "- Tissue dispersion is the mean absolute pairwise difference between three-seed mean perturbation effects across tissue tasks.",
        "- Seed dispersion is the corresponding mean absolute pairwise difference across the three fitted seeds within a tissue task.",
        "",
        "## Main results",
        "",
        f"- Median HLA-level tissue/seed dispersion ratio for final rank fusion: {hla_summary['rank_fusion_dispersion_ratio'].median():.4f}.",
        f"- HLA alleles with ratio > 1: {ratio_above_one}/{len(hla_summary)}.",
        f"- HLA alleles with one-sided peptide-level Wilcoxon BH q < 0.05: {significant}/{len(hla_summary)}.",
        f"- Median tissue-pair Spearman correlation of final mutation effects: {task_pairs['rank_fusion_spearman_r'].median():.4f}.",
        f"- Median tissue-pair mean absolute difference in final mutation effects: {task_pairs['rank_fusion_mean_abs_difference'].median():.6f}.",
        f"- Median cross-seed reproducibility of tissue-contrast vectors: {task_pairs['rank_fusion_median_seed_contrast_spearman_r'].median():.4f}.",
        "- Positions with the largest equal-HLA mean tissue dispersion: "
        + ", ".join(f"P{int(row.position)} ({row.tissue_dispersion:.5f})" for row in fusion_positions.head(3).itertuples())
        + ".",
        "",
        "## Interpretation boundary",
        "",
        "The calculation establishes fitted-model task dependence while holding peptide sequence and HLA fixed. It does not identify a causal tissue-processing mechanism, and differences that are not reproducible across training seeds should not be treated as stable tissue effects.",
        "",
        f"Total elapsed time: {elapsed:.3f} seconds.",
        "",
    ]
    (output_dir / "CROSS_TISSUE_ISM_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run(cli: argparse.Namespace) -> None:
    torch, nn, _, _ = base.require_torch()
    device = shap_run.resolve_device(cli.device, torch)
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = cli.output_dir / "cross_tissue_raw_by_hla"
    raw_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train, test, mappings, peptide_length = shap_run.validate_and_prepare(cli.train, cli.test, 0)
    if peptide_length != 9:
        raise ValueError(f"Expected 9-mer peptides, observed {peptide_length}")
    if cli.seeds != DEFAULT_SEEDS:
        raise ValueError(f"This analysis uses the frozen seeds {DEFAULT_SEEDS}")
    panel, eligible_hlas = select_panel(
        test, cli.pairs_per_task, cli.sample_seed, cli.smoke, cli.smoke_peptides,
    )
    branch_predictions = pd.read_csv(cli.branch_predictions, keep_default_na=False)
    if "task_name" not in branch_predictions.columns:
        branch_predictions = branch_predictions.merge(
            test[["sample_id", "task_name"]], on="sample_id", how="left", validate="many_to_one",
        )
    args = SimpleNamespace(**shap_run.FROZEN_PARAMETERS)
    global_models: dict[int, tuple[Any, dict[str, Any]]] = {}
    for seed in cli.seeds:
        seed_dir = cli.checkpoint_root / f"seed_{seed}"
        global_models[seed] = shap_run.load_model(
            torch, nn, args, seed_dir / "global_aux.pt", peptide_length,
            len(mappings["tissue_to_id"]), len(mappings["hla_to_id"]), device,
        )

    timing_rows: list[dict[str, Any]] = []
    hla_summaries: list[pd.DataFrame] = []
    peptide_summaries: list[pd.DataFrame] = []
    position_summaries: list[pd.DataFrame] = []
    pair_summaries: list[pd.DataFrame] = []
    stability_summaries: list[pd.DataFrame] = []
    print(
        f"[CROSS-TISSUE ISM START] eligible_hlas={len(eligible_hlas)} panel_peptides={len(panel)} "
        f"seeds={cli.seeds} device={device}",
        flush=True,
    )
    for hla_index, hla in enumerate(eligible_hlas, start=1):
        tasks = test[test["mhc_restriction"] == hla][
            ["target_tissue", "mhc_restriction", "task_name"]
        ].drop_duplicates()
        peptides = panel[panel["mhc_restriction"] == hla]
        context = build_context_frame(hla, peptides, tasks)
        seed_parts: list[pd.DataFrame] = []
        for seed in cli.seeds:
            seed_started = time.perf_counter()
            reference = branch_predictions[
                pd.to_numeric(branch_predictions["seed"], errors="coerce") == seed
            ].copy()
            global_model, global_payload = global_models[seed]
            global_effects = ism_run.score_branch(
                torch, global_model, global_payload["task_to_id"], context,
                peptide_length, device, cli.inference_batch_size,
            )
            checkpoint = (
                cli.checkpoint_root / f"seed_{seed}" /
                f"hla_plain__{safe_hla_name(hla)}.pt"
            )
            hla_model, hla_payload = shap_run.load_model(
                torch, nn, args, checkpoint, peptide_length,
                len(mappings["tissue_to_id"]), len(mappings["hla_to_id"]), device,
            )
            hla_effects = ism_run.score_branch(
                torch, hla_model, hla_payload["task_to_id"], context,
                peptide_length, device, cli.inference_batch_size,
            )
            combined = combine_context_branches(global_effects, hla_effects, reference, seed)
            seed_parts.append(combined)
            del hla_model, global_effects, hla_effects, combined
            if device == "cuda":
                torch.cuda.empty_cache()
            seed_elapsed = time.perf_counter() - seed_started
            timing_rows.append({
                "seed": seed, "mhc_restriction": hla,
                "stage": "cross_tissue_ism_scoring", "seconds": seed_elapsed,
            })
            print(
                f"[SEED CROSS-TISSUE TIME] hla={hla} seed={seed} seconds={seed_elapsed:.3f} "
                f"contexts={len(context)} mutation_rows={len(seed_parts[-1])}",
                flush=True,
            )
        hla_effects_all = pd.concat(seed_parts, ignore_index=True)
        hla_effects_all.to_csv(
            raw_dir / f"cross_tissue_effects_{safe_hla_name(hla)}.csv.gz",
            index=False, compression="gzip",
        )
        hla_summary, peptide_summary, position_summary = hla_dispersion_tables(
            hla, hla_effects_all, cli.bootstrap_replicates, cli.sample_seed,
        )
        task_pairs, contrast_stability = task_pair_tables(hla, hla_effects_all)
        hla_summaries.append(hla_summary)
        peptide_summaries.append(peptide_summary)
        position_summaries.append(position_summary)
        pair_summaries.append(task_pairs)
        stability_summaries.append(contrast_stability)
        print(
            f"[HLA COMPLETE] {hla_index}/{len(eligible_hlas)} hla={hla} "
            f"tissues={len(tasks)} peptides={len(peptides)}",
            flush=True,
        )

    hla_summary = pd.concat(hla_summaries, ignore_index=True)
    for metric_name in METRICS:
        hla_summary[f"{metric_name}_fdr_bh"] = bh_adjust(
            hla_summary[f"{metric_name}_p_value_one_sided"].to_numpy(dtype=float)
        )
    peptide_summary = pd.concat(peptide_summaries, ignore_index=True)
    position_summary = pd.concat(position_summaries, ignore_index=True)
    task_pairs = pd.concat(pair_summaries, ignore_index=True)
    contrast_stability = pd.concat(stability_summaries, ignore_index=True)
    hla_summary.to_csv(cli.output_dir / "cross_tissue_hla_summary.csv", index=False)
    peptide_summary.to_csv(cli.output_dir / "cross_tissue_peptide_dispersion.csv.gz", index=False, compression="gzip")
    position_summary.to_csv(cli.output_dir / "cross_tissue_position_summary.csv", index=False)
    task_pairs.to_csv(cli.output_dir / "cross_tissue_task_pair_summary.csv", index=False)
    contrast_stability.to_csv(cli.output_dir / "cross_tissue_seed_contrast_stability.csv", index=False)
    ratio_intervals = bootstrap_dispersion_ratios(
        peptide_summary, cli.bootstrap_replicates, cli.sample_seed,
    )
    ratio_intervals.to_csv(cli.output_dir / "cross_tissue_hla_ratio_ci.csv", index=False)
    create_figure(ratio_intervals, raw_dir, cli.output_dir)
    elapsed = time.perf_counter() - started
    timing_rows.append({"seed": "all", "mhc_restriction": "ALL", "stage": "total", "seconds": elapsed})
    pd.DataFrame(timing_rows).to_csv(cli.output_dir / "timing_results.csv", index=False)
    write_report(cli.output_dir, hla_summary, task_pairs, position_summary, elapsed)
    metadata = {
        "analysis": "same-HLA different-tissue counterfactual single-amino-acid ISM",
        "train": str(cli.train.resolve()),
        "test": str(cli.test.resolve()),
        "checkpoint_root": str(cli.checkpoint_root.resolve()),
        "branch_predictions": str(cli.branch_predictions.resolve()),
        "seeds": cli.seeds,
        "device": device,
        "eligible_hlas": len(eligible_hlas),
        "eligible_tasks": int(test[test["mhc_restriction"].isin(eligible_hlas)]["task_name"].nunique()),
        "hla_counted_unique_peptides": int(hla_summary["n_peptides"].sum()),
        "tissue_task_pairs": len(task_pairs),
        "pairs_per_task_source_panel": cli.pairs_per_task,
        "sample_seed": cli.sample_seed,
        "bootstrap_replicates": cli.bootstrap_replicates,
        "rank_effect_definition": (
            "for a fixed peptide and substitution, insert original and mutant probabilities "
            "separately into each target tissue-task fixed-test reference distribution; average "
            "global and HLA-branch percentile-rank shifts"
        ),
        "tissue_dispersion_definition": (
            "mean absolute pairwise difference across same-HLA tissue tasks after averaging each effect across seeds"
        ),
        "seed_dispersion_definition": (
            "mean absolute pairwise difference across seeds within each tissue task, averaged across tasks"
        ),
        "versions": {
            "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
        },
        "elapsed_seconds": elapsed,
    }
    (cli.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"[TOTAL TIME] seconds={elapsed:.3f}", flush=True)
    print(f"[WROTE] {cli.output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train", type=Path,
        default=PROJECT_ROOT / "data/humanPMHC_occurence_equal_dataset/humanPMHC_train.csv.gz",
    )
    parser.add_argument(
        "--test", type=Path,
        default=PROJECT_ROOT / "data/humanPMHC_occurence_equal_dataset/humanPMHC_test.csv.gz",
    )
    parser.add_argument(
        "--checkpoint-root", type=Path,
        default=PROJECT_ROOT / "extra_occurrence_equal_dataset/results/e29_tuned_shap/checkpoints",
    )
    parser.add_argument(
        "--branch-predictions", type=Path,
        default=PROJECT_ROOT / "extra_occurrence_equal_dataset/results/e29_tuned_shap_expected_ig/branch_predictions.csv.gz",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "extra_occurrence_equal_dataset/results/e29_tuned_ism_cross_tissue",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--pairs-per-task", type=int, default=20)
    parser.add_argument("--sample-seed", type=int, default=20260805)
    parser.add_argument("--inference-batch-size", type=int, default=8192)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-peptides", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
