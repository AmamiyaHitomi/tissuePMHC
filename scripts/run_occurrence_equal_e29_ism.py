#!/usr/bin/env python3
"""Exhaustive single-amino-acid ISM validation for occurrence-equal E29.

The script reuses the frozen three-seed checkpoints produced by the formal
Expected-IG/SHAP analysis.  Every residue of each selected nine-mer is replaced
with each of the other 19 canonical amino acids.  Effects are reported on both
branch logits and on E29's final within-task percentile-rank fusion score.
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
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_occurrence_equal_e29_shap as shap_run  # noqa: E402


base = shap_run.base
AMINO_ACIDS = list(base.AMINO_ACIDS)
AA_INDICES = np.asarray([base.AA_TO_INDEX[aa] for aa in AMINO_ACIDS], dtype=np.int64)
DEFAULT_SEEDS = shap_run.DEFAULT_SEEDS
KEY_COLUMNS = [
    "sample_id", "pair_id", "label", "target_tissue", "mhc_restriction",
    "task_name", "peptide_sequence",
]


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


def bootstrap_mean_ci(values: np.ndarray, seed: int, n_bootstrap: int) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return math.nan, math.nan
    if len(clean) == 1 or n_bootstrap <= 0:
        return float(clean[0]), float(clean[0])
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=float)
    chunk = min(200, n_bootstrap)
    written = 0
    while written < n_bootstrap:
        take = min(chunk, n_bootstrap - written)
        indices = rng.integers(0, len(clean), size=(take, len(clean)))
        means[written : written + take] = clean[indices].mean(axis=1)
        written += take
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def make_mutation_index(
    frame: pd.DataFrame, peptide_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    encoded = base.encode_peptides(frame["peptide_sequence"], peptide_length).astype(np.int64, copy=True)
    rows_per_sample = peptide_length * (len(AMINO_ACIDS) - 1)
    sample_index = np.repeat(np.arange(len(frame), dtype=np.int64), rows_per_sample)
    position = np.tile(
        np.repeat(np.arange(peptide_length, dtype=np.int64), len(AMINO_ACIDS) - 1), len(frame)
    )
    mutant_index = np.empty(len(sample_index), dtype=np.int64)
    cursor = 0
    for sequence_ids in encoded:
        for original_index in sequence_ids:
            alternatives = AA_INDICES[AA_INDICES != original_index]
            if len(alternatives) != len(AMINO_ACIDS) - 1:
                raise ValueError(f"Non-canonical peptide token encountered: {original_index}")
            mutant_index[cursor : cursor + len(alternatives)] = alternatives
            cursor += len(alternatives)
    mutated = encoded[sample_index].copy()
    mutated[np.arange(len(mutated)), position] = mutant_index
    return encoded, mutated, sample_index, position, mutant_index


def predict_logits(
    torch: Any,
    model: Any,
    peptide_ids: np.ndarray,
    task_ids: np.ndarray,
    device: str,
    batch_size: int,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(peptide_ids), batch_size):
            stop = min(start + batch_size, len(peptide_ids))
            peptides = torch.as_tensor(peptide_ids[start:stop], dtype=torch.long, device=device)
            tasks = torch.as_tensor(task_ids[start:stop].copy(), dtype=torch.long, device=device)
            outputs.append(model(peptides, tasks).detach().cpu().numpy().astype(np.float64))
    return np.concatenate(outputs)


def score_branch(
    torch: Any,
    model: Any,
    task_to_id: dict[str, int],
    frame: pd.DataFrame,
    peptide_length: int,
    device: str,
    batch_size: int,
) -> pd.DataFrame:
    encoded, mutated, sample_index, position, mutant_index = make_mutation_index(frame, peptide_length)
    mapped_tasks = frame["task_name"].map(task_to_id)
    if mapped_tasks.isna().any():
        missing = sorted(frame.loc[mapped_tasks.isna(), "task_name"].unique())
        raise ValueError(f"Checkpoint task map misses: {missing}")
    original_tasks = mapped_tasks.to_numpy(dtype=np.int64)
    original_logits = predict_logits(torch, model, encoded, original_tasks, device, batch_size)
    mutant_logits = predict_logits(
        torch, model, mutated, original_tasks[sample_index], device, batch_size,
    )
    metadata = frame.iloc[sample_index][KEY_COLUMNS].reset_index(drop=True)
    result = metadata.copy()
    result["position"] = position + 1
    result["original_amino_acid"] = [AMINO_ACIDS[AA_INDICES.tolist().index(x)] for x in encoded[sample_index, position]]
    index_to_aa = {index: aa for aa, index in base.AA_TO_INDEX.items()}
    result["mutant_amino_acid"] = [index_to_aa[int(index)] for index in mutant_index]
    result["original_logit"] = original_logits[sample_index]
    result["mutant_logit"] = mutant_logits
    result["delta_logit"] = mutant_logits - original_logits[sample_index]
    result["original_probability"] = 1.0 / (1.0 + np.exp(-original_logits[sample_index]))
    result["mutant_probability"] = 1.0 / (1.0 + np.exp(-mutant_logits))
    result["delta_probability"] = result["mutant_probability"] - result["original_probability"]
    return result


def replacement_percentile_ranks(
    reference: pd.DataFrame,
    mutation_rows: pd.DataFrame,
    score_column: str,
    mutant_column: str,
) -> np.ndarray:
    output = np.empty(len(mutation_rows), dtype=np.float64)
    reference_by_task = {task: group for task, group in reference.groupby("task_name", sort=False)}
    for task_name, indices in mutation_rows.groupby("task_name", sort=False).indices.items():
        indices = np.asarray(indices, dtype=np.int64)
        task_reference = reference_by_task[task_name]
        score_by_sample = task_reference.set_index("sample_id")[score_column]
        original = mutation_rows.iloc[indices]["sample_id"].map(score_by_sample).to_numpy(dtype=float)
        mutant = mutation_rows.iloc[indices][mutant_column].to_numpy(dtype=float)
        sorted_scores = np.sort(task_reference[score_column].to_numpy(dtype=float))
        left = np.searchsorted(sorted_scores, mutant, side="left")
        right = np.searchsorted(sorted_scores, mutant, side="right")
        less_other = left - (original < mutant).astype(np.int64)
        equal_other = (right - left) - (original == mutant).astype(np.int64)
        average_rank = less_other + (equal_other + 2.0) / 2.0
        output[indices] = average_rank / len(sorted_scores)
    return output


def combine_branches(
    global_effects: pd.DataFrame,
    hla_effects: pd.DataFrame,
    reference: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    mutation_keys = [*KEY_COLUMNS, "position", "original_amino_acid", "mutant_amino_acid"]
    global_renamed = global_effects.rename(columns={
        column: f"{column}_global" for column in (
            "original_logit", "mutant_logit", "delta_logit", "original_probability",
            "mutant_probability", "delta_probability",
        )
    })
    hla_renamed = hla_effects.rename(columns={
        column: f"{column}_hla" for column in (
            "original_logit", "mutant_logit", "delta_logit", "original_probability",
            "mutant_probability", "delta_probability",
        )
    })
    merged = global_renamed.merge(hla_renamed, on=mutation_keys, how="inner", validate="one_to_one")
    if len(merged) != len(global_effects) or len(merged) != len(hla_effects):
        raise AssertionError("Global and HLA mutation grids do not align")
    merged.insert(0, "seed", seed)
    merged["mutant_rank_global"] = replacement_percentile_ranks(
        reference, merged, "global_score", "mutant_probability_global"
    )
    merged["mutant_rank_hla"] = replacement_percentile_ranks(
        reference, merged, "hla_score", "mutant_probability_hla"
    )
    original_reference = reference.set_index("sample_id")
    merged["original_rank_fusion"] = merged["sample_id"].map(original_reference["fused_score"])
    merged["mutant_rank_fusion"] = 0.5 * (merged["mutant_rank_global"] + merged["mutant_rank_hla"])
    merged["delta_rank_fusion"] = merged["mutant_rank_fusion"] - merged["original_rank_fusion"]
    direction = 2.0 * merged["label"].to_numpy(dtype=float) - 1.0
    merged["classification_support_loss_global"] = -direction * merged["delta_logit_global"]
    merged["classification_support_loss_hla"] = -direction * merged["delta_logit_hla"]
    merged["classification_support_loss_fusion"] = -direction * merged["delta_rank_fusion"]

    selected_original = merged.drop_duplicates("sample_id")
    audit = {}
    for branch in ("global", "hla"):
        expected = selected_original["sample_id"].map(original_reference[f"{branch}_score"]).to_numpy(dtype=float)
        actual = selected_original[f"original_probability_{branch}"].to_numpy(dtype=float)
        audit[f"max_abs_original_probability_error_{branch}"] = float(np.max(np.abs(expected - actual)))
    reconstructed_fused = 0.5 * (
        reference.groupby("task_name", sort=False)["global_score"].rank(method="average", pct=True)
        + reference.groupby("task_name", sort=False)["hla_score"].rank(method="average", pct=True)
    )
    audit["max_abs_reference_fusion_error"] = float(
        np.max(np.abs(reconstructed_fused.to_numpy() - reference["fused_score"].to_numpy()))
    )
    return merged, audit


def seed_position_matrix(effect: pd.DataFrame, value: str) -> pd.Series:
    return effect.groupby(
        ["position", "original_amino_acid", "mutant_amino_acid"], sort=True
    )[value].mean()


def stability_analysis(seed_effects: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = {
        "global_logit": "delta_logit_global",
        "hla_logit": "delta_logit_hla",
        "rank_fusion": "delta_rank_fusion",
    }
    for metric_name, column in metrics.items():
        vectors = {seed: seed_position_matrix(effect, column) for seed, effect in seed_effects.items()}
        for seed_a, seed_b in combinations(sorted(vectors), 2):
            aligned = pd.concat([vectors[seed_a], vectors[seed_b]], axis=1).dropna()
            rho = spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1]).statistic
            sign_agreement = np.mean(np.sign(aligned.iloc[:, 0]) == np.sign(aligned.iloc[:, 1]))
            rows.append({
                "metric": metric_name, "seed_a": seed_a, "seed_b": seed_b,
                "n_position_substitutions": len(aligned), "spearman_r": float(rho),
                "sign_agreement": float(sign_agreement),
            })
    return pd.DataFrame(rows, columns=[
        "metric", "seed_a", "seed_b", "n_position_substitutions",
        "spearman_r", "sign_agreement",
    ])


def ensemble_effects(seed_effects: dict[int, pd.DataFrame]) -> pd.DataFrame:
    numeric = [
        "original_logit_global", "mutant_logit_global", "delta_logit_global",
        "original_probability_global", "mutant_probability_global", "delta_probability_global",
        "original_logit_hla", "mutant_logit_hla", "delta_logit_hla",
        "original_probability_hla", "mutant_probability_hla", "delta_probability_hla",
        "original_rank_fusion", "mutant_rank_global", "mutant_rank_hla",
        "mutant_rank_fusion", "delta_rank_fusion", "classification_support_loss_global",
        "classification_support_loss_hla", "classification_support_loss_fusion",
    ]
    first_seed = sorted(seed_effects)[0]
    result = seed_effects[first_seed].drop(columns=["seed", *numeric]).copy()
    for column in numeric:
        result[column] = np.mean(
            [seed_effects[seed][column].to_numpy(dtype=float) for seed in sorted(seed_effects)], axis=0
        )
    result.insert(0, "seed", "three_seed_mean" if len(seed_effects) == 3 else "seed_mean")
    return result


def sample_position_table(effect: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "sample_id", "pair_id", "label", "target_tissue", "mhc_restriction",
        "task_name", "peptide_sequence", "position", "original_amino_acid",
    ]
    result = effect.groupby(group_columns, as_index=False).agg(
        mean_delta_logit_global=("delta_logit_global", "mean"),
        mean_abs_delta_logit_global=("delta_logit_global", lambda x: float(np.mean(np.abs(x)))),
        mean_delta_logit_hla=("delta_logit_hla", "mean"),
        mean_abs_delta_logit_hla=("delta_logit_hla", lambda x: float(np.mean(np.abs(x)))),
        mean_delta_rank_fusion=("delta_rank_fusion", "mean"),
        mean_abs_delta_rank_fusion=("delta_rank_fusion", lambda x: float(np.mean(np.abs(x)))),
        mean_support_loss_global=("classification_support_loss_global", "mean"),
        mean_support_loss_hla=("classification_support_loss_hla", "mean"),
        mean_support_loss_fusion=("classification_support_loss_fusion", "mean"),
        strongest_score_decrease=("delta_rank_fusion", "min"),
        strongest_score_increase=("delta_rank_fusion", "max"),
    )
    return result


def position_summary(sample_position: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, str, pd.DataFrame]] = [("overall", "ALL", sample_position)]
    for column, scope_type in (
        ("mhc_restriction", "hla"), ("target_tissue", "tissue"), ("task_name", "task")
    ):
        scopes.extend((scope_type, str(value), group) for value, group in sample_position.groupby(column))
    metrics = [
        "mean_abs_delta_logit_global", "mean_abs_delta_logit_hla", "mean_abs_delta_rank_fusion",
        "mean_support_loss_global", "mean_support_loss_hla", "mean_support_loss_fusion",
    ]
    for scope_type, scope_value, scope in scopes:
        for label_group, label in (("all", None), ("positive", 1), ("negative", 0)):
            chosen = scope if label is None else scope[scope["label"] == label]
            for position, group in chosen.groupby("position", sort=True):
                record: dict[str, Any] = {
                    "scope_type": scope_type, "scope_value": scope_value,
                    "label_group": label_group, "position": int(position),
                    "n_samples": int(group["sample_id"].nunique()),
                }
                for metric in metrics:
                    record[f"{metric}_mean"] = float(group[metric].mean())
                    record[f"{metric}_median"] = float(group[metric].median())
                    record[f"{metric}_sd"] = float(group[metric].std(ddof=1))
                rows.append(record)
    return pd.DataFrame(rows)


def substitution_summary(effect: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    scopes: list[tuple[str, str, pd.DataFrame]] = [("overall", "ALL", effect)]
    scopes.extend(("hla", str(value), group) for value, group in effect.groupby("mhc_restriction"))
    for scope_type, scope_value, scope in scopes:
        for label_group, label in (("all", None), ("positive", 1), ("negative", 0)):
            chosen = scope if label is None else scope[scope["label"] == label]
            summary = chosen.groupby(
                ["position", "original_amino_acid", "mutant_amino_acid"], as_index=False
            ).agg(
                n_mutations=("sample_id", "size"),
                mean_delta_logit_global=("delta_logit_global", "mean"),
                mean_delta_logit_hla=("delta_logit_hla", "mean"),
                mean_delta_rank_fusion=("delta_rank_fusion", "mean"),
                mean_support_loss_fusion=("classification_support_loss_fusion", "mean"),
            )
            summary.insert(0, "label_group", label_group)
            summary.insert(0, "scope_value", scope_value)
            summary.insert(0, "scope_type", scope_type)
            frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def anchor_tests(sample_position: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    metrics = [
        "mean_abs_delta_logit_global", "mean_abs_delta_logit_hla", "mean_abs_delta_rank_fusion",
    ]
    rows: list[dict[str, Any]] = []
    for label_group, label in (("all", None), ("positive", 1), ("negative", 0)):
        chosen = sample_position if label is None else sample_position[sample_position["label"] == label]
        for metric in metrics:
            pivot = chosen.pivot(index="sample_id", columns="position", values=metric).dropna()
            anchor = pivot[[2, 9]].mean(axis=1).to_numpy(dtype=float)
            non_anchor = pivot[[position for position in range(1, 10) if position not in (2, 9)]].mean(axis=1).to_numpy(dtype=float)
            difference = anchor - non_anchor
            try:
                statistic, p_value = wilcoxon(difference, alternative="greater", zero_method="wilcox")
            except ValueError:
                statistic, p_value = math.nan, 1.0
            low, high = bootstrap_mean_ci(difference, seed + len(rows), n_bootstrap)
            rows.append({
                "label_group": label_group, "metric": metric, "n_samples": len(difference),
                "anchor_mean": float(anchor.mean()), "non_anchor_mean": float(non_anchor.mean()),
                "paired_mean_difference": float(difference.mean()),
                "paired_median_difference": float(np.median(difference)),
                "bootstrap_95_ci_low": low, "bootstrap_95_ci_high": high,
                "wilcoxon_statistic": float(statistic), "p_value_one_sided": float(p_value),
            })
    result = pd.DataFrame(rows)
    result["fdr_bh"] = bh_adjust(result["p_value_one_sided"].to_numpy())
    return result


def paired_position_analysis(sample_position: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["pair_id", "target_tissue", "mhc_restriction", "task_name", "position"]
    metrics = ["mean_support_loss_global", "mean_support_loss_hla", "mean_support_loss_fusion"]
    positive = sample_position[sample_position["label"] == 1][keys + ["sample_id", *metrics]].copy()
    negative = sample_position[sample_position["label"] == 0][keys + ["sample_id", *metrics]].copy()
    paired = positive.merge(negative, on=keys, validate="one_to_one", suffixes=("_positive", "_negative"))
    paired = paired.rename(columns={"sample_id_positive": "positive_sample_id", "sample_id_negative": "negative_sample_id"})
    for metric in metrics:
        paired[f"positive_minus_negative_{metric}"] = paired[f"{metric}_positive"] - paired[f"{metric}_negative"]
    summary = paired.groupby("position", as_index=False).agg(
        n_pairs=("pair_id", "size"),
        mean_positive_support_global=("mean_support_loss_global_positive", "mean"),
        mean_negative_support_global=("mean_support_loss_global_negative", "mean"),
        mean_positive_support_hla=("mean_support_loss_hla_positive", "mean"),
        mean_negative_support_hla=("mean_support_loss_hla_negative", "mean"),
        mean_positive_support_fusion=("mean_support_loss_fusion_positive", "mean"),
        mean_negative_support_fusion=("mean_support_loss_fusion_negative", "mean"),
        mean_positive_minus_negative_fusion=("positive_minus_negative_mean_support_loss_fusion", "mean"),
    )
    return paired, summary


def shap_concordance(
    seed_effects: dict[int, pd.DataFrame], shap_observed_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed = pd.read_csv(shap_observed_path, keep_default_na=False)
    long_parts = []
    for seed, effect in seed_effects.items():
        for branch, delta_column in (("global_aux", "delta_logit_global"), ("hla_plain", "delta_logit_hla")):
            position_loss = effect.groupby(["sample_id", "position"], as_index=False)[delta_column].mean()
            position_loss["mean_ism_mutation_loss"] = -position_loss[delta_column]
            position_loss["seed"] = seed
            position_loss["branch"] = branch
            long_parts.append(position_loss[["seed", "branch", "sample_id", "position", "mean_ism_mutation_loss"]])
    ism_long = pd.concat(long_parts, ignore_index=True)
    shap_parts = []
    peptide_length = 9
    for position in range(1, peptide_length + 1):
        part = observed[["seed", "branch", "sample_id", "label", "task_name", f"position_{position}_shap"]].copy()
        part["position"] = position
        part = part.rename(columns={f"position_{position}_shap": "observed_residue_shap"})
        shap_parts.append(part)
    shap_long = pd.concat(shap_parts, ignore_index=True)
    shap_long["seed"] = pd.to_numeric(shap_long["seed"], errors="raise").astype(int)
    merged = shap_long.merge(ism_long, on=["seed", "branch", "sample_id", "position"], validate="one_to_one")
    rows: list[dict[str, Any]] = []
    for (seed, branch), group in merged.groupby(["seed", "branch"], sort=True):
        for scope, chosen in [("all_positions", group), *[(f"P{p}", group[group["position"] == p]) for p in range(1, 10)]]:
            rho = spearmanr(chosen["observed_residue_shap"], chosen["mean_ism_mutation_loss"]).statistic
            rows.append({
                "seed": seed, "branch": branch, "scope": scope, "n_points": len(chosen),
                "spearman_r": float(rho),
            })
    return merged, pd.DataFrame(rows)


def save_figure(fig: Any, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def create_figures(
    effect: pd.DataFrame,
    sample_position: pd.DataFrame,
    paired_summary: pd.DataFrame,
    shap_merged: pd.DataFrame,
    output_dir: Path,
) -> None:
    overall = effect.groupby(["mutant_amino_acid", "position"])["delta_rank_fusion"].mean().unstack("position")
    overall = overall.reindex(index=AMINO_ACIDS, columns=range(1, 10))
    vmax = max(float(np.nanpercentile(np.abs(overall.to_numpy()), 98)), 1e-8)
    fig, ax = plt.subplots(figsize=(8.4, 5.5))
    image = ax.imshow(overall, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(9), labels=range(1, 10))
    ax.set_yticks(range(20), labels=AMINO_ACIDS)
    ax.set_xlabel("Peptide position")
    ax.set_ylabel("Mutant amino acid")
    ax.set_title("Three-seed E29 ISM: mean change in final rank-fusion score")
    fig.colorbar(image, ax=ax, label="Mutant minus original score")
    fig.tight_layout()
    save_figure(fig, output_dir, "01_overall_ism_mutant_heatmap")

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), constrained_layout=True)
    plot_specs = [
        ("mean_abs_delta_logit_global", "Global-auxiliary", "Mean |Δ logit|"),
        ("mean_abs_delta_logit_hla", "HLA-specific", "Mean |Δ logit|"),
        ("mean_abs_delta_rank_fusion", "Final rank fusion", "Mean |Δ rank score|"),
    ]
    for ax, (metric, title, ylabel) in zip(axes, plot_specs):
        for label, color in ((1, "#d62728"), (0, "#1f77b4")):
            grouped = sample_position[sample_position["label"] == label].groupby("position")[metric]
            mean = grouped.mean()
            sem = grouped.sem()
            ax.plot(mean.index, mean.values, marker="o", color=color, label="positive" if label else "negative")
            ax.fill_between(mean.index, mean - 1.96 * sem, mean + 1.96 * sem, color=color, alpha=0.15)
        ax.set_xticks(range(1, 10))
        ax.set_xlabel("Peptide position")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.axvline(2, color="grey", linestyle="--", linewidth=0.8)
        ax.axvline(9, color="grey", linestyle="--", linewidth=0.8)
    axes[0].legend(frameon=False)
    save_figure(fig, output_dir, "02_position_sensitivity")

    top_hlas = sample_position.groupby("mhc_restriction")["task_name"].nunique().sort_values(ascending=False).head(6).index
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True, sharex=True)
    for ax, hla in zip(axes.flat, top_hlas):
        group = sample_position[sample_position["mhc_restriction"] == hla]
        line = group.groupby("position")["mean_abs_delta_rank_fusion"].mean()
        ax.plot(line.index, line.values, marker="o")
        ax.set_title(str(hla))
        ax.set_xticks(range(1, 10))
        ax.set_xlabel("Position")
        ax.set_ylabel("Mean |Δ rank score|")
    save_figure(fig, output_dir, "03_top_hla_position_sensitivity")

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(paired_summary["position"], paired_summary["mean_positive_support_fusion"], marker="o", label="Positive peptide")
    ax.plot(paired_summary["position"], paired_summary["mean_negative_support_fusion"], marker="o", label="Matched negative peptide")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(range(1, 10))
    ax.set_xlabel("Peptide position")
    ax.set_ylabel("Mean classification-support loss after mutation")
    ax.set_title("Pair-matched ISM validation")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir, "04_paired_position_validation")

    consensus = shap_merged.groupby(["branch", "sample_id", "position"], as_index=False).agg(
        observed_residue_shap=("observed_residue_shap", "mean"),
        mean_ism_mutation_loss=("mean_ism_mutation_loss", "mean"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
    for ax, branch in zip(axes, ("global_aux", "hla_plain")):
        chosen = consensus[consensus["branch"] == branch]
        image = ax.hexbin(
            chosen["observed_residue_shap"], chosen["mean_ism_mutation_loss"],
            gridsize=45, mincnt=1, bins="log", cmap="viridis",
        )
        rho = spearmanr(chosen["observed_residue_shap"], chosen["mean_ism_mutation_loss"]).statistic
        ax.set_xlabel("Expected-IG observed-residue attribution")
        ax.set_ylabel("Mean ISM mutation loss (logit)")
        ax.set_title(f"{branch}: Spearman ρ={rho:.3f}")
        fig.colorbar(image, ax=ax, label="log10(count)")
    save_figure(fig, output_dir, "05_shap_ism_concordance")


def write_report(
    cli: argparse.Namespace,
    effect: pd.DataFrame,
    sample_position: pd.DataFrame,
    stability: pd.DataFrame,
    anchor: pd.DataFrame,
    paired_summary: pd.DataFrame,
    concordance: pd.DataFrame,
    audits: pd.DataFrame,
    elapsed: float,
) -> None:
    positive_positions = sample_position[sample_position["label"] == 1].groupby("position")["mean_support_loss_fusion"].mean().sort_values(ascending=False)
    sensitivity_positions = sample_position.groupby("position")["mean_abs_delta_rank_fusion"].mean().sort_values(ascending=False)
    strongest_losses = effect.nsmallest(10, "delta_rank_fusion")
    strongest_gains = effect.nlargest(10, "delta_rank_fusion")
    stability_median = (
        stability.groupby("metric")["spearman_r"].median()
        if not stability.empty else pd.Series(dtype=float)
    )
    concordance_overall = concordance[concordance["scope"] == "all_positions"].groupby("branch")["spearman_r"].median()
    anchor_fusion = anchor[(anchor["label_group"] == "all") & (anchor["metric"] == "mean_abs_delta_rank_fusion")].iloc[0]
    anchor_fdr_text = (
        "<1e-300" if float(anchor_fusion["fdr_bh"]) == 0.0
        else f"{float(anchor_fusion['fdr_bh']):.3g}"
    )
    paired_top = paired_summary.sort_values("mean_positive_support_fusion", ascending=False).head(3)

    lines = [
        "#occurence-equal tisuePHC: E29 ISM Disturbation Validation Report",
        "",
        "# Analyse design",
        "",
        "- Model: official three-said E29 checkpoints, including the global-auxiliary and HLA-specific branches.",
        f"- Sample: taken per mission:{cli.pairs_per_task}Full positive or negative, total.{sample_position['sample_id'].nunique()}Barclay.",
        "- Disturbation: each of the 9-mer points is replaced by the remaining 19 standard amino acids.",
        f"- Total mutant scores:{len(effect):,}An average of three-seseed monoamino acid replacement.",
        "- Effect measure: the branch logit change and the final recalculated percentage change in task by reference distribution for fixed tests.",
        "- Definition of mutant fractions: minus original platinum fractions; negative values represent mutation reduction model projections.",
        "",
        "# Quality control",
        "",
        f"- The greatest absolute error in the probability of the original branch of the pyridium and the preservation of the forecast:{audits[['max_abs_original_probability_error_global', 'max_abs_original_probability_error_hla']].to_numpy().max():.3g}.",
        f"- Recalculate the maximum absolute error in saving results by rank fusion:{audits['max_abs_reference_fusion_error'].max():.3g}.",
        f"- Three-seed position - replacement effect Spearman median: global={stability_median.get('global_logit', math.nan):.4f},HLA={stability_median.get('hla_logit', math.nan):.4f},fusion={stability_median.get('rank_fusion', math.nan):.4f}.",
        f"- Total running time:{elapsed:.3f}Seconds.",
        "",
        "# The main result",
        "",
        "# Position sensitivity",
        "",
        "- The most sensitive position of the final integration score for the entire sample is, in order of precedence:" + ", ".join(f"P{int(p)}" for p in sensitivity_positions.index[:5]) + ".",
        "- Among the positive peptide, the most significant disruptions to the positive predictions are, in descending order:" + ", ".join(f"P{int(p)}" for p in positive_positions.index[:5]) + ".",
        f"- Average P2/P9 sensitivity less remaining position{anchor_fusion['paired_mean_difference']:.6f},95% bootstrap CI [{anchor_fusion['bootstrap_95_ci_low']:.6f}, {anchor_fusion['bootstrap_95_ci_high']:.6f}♪ One-sided Wilcoxon FDR ♪{anchor_fdr_text}.",
        "- In the positive-negative-comparison analysis, the three positions where the most significant losses in the positive platinum classification are:" + ", ".join(f"P{int(row.position)}" for row in paired_top.itertuples()) + ".",
        "",
        "# Consistency with Expected-IG/SHAP",
        "",
        f"- Average mutation loss of ISM by sample in relation to Expected-IG observed-residetation median: global={concordance_overall.get('global_aux', math.nan):.4f},HLA={concordance_overall.get('hla_plain', math.nan):.4f}.",
        "- The two measures are not identical: ISM compares the original pyromium to all single-point mutagenics relative to the background of the mission; therefore the medium-high correlation constitutes independent disturbance evidence, but does not require an equal value.",
        "",
        "# The strongest drop mutation (final rank Fusion)",
        "",
        "The sample is the original position of the sample, which is the replacement of the lang fasion.",
        "|---|---|---:|:---:|---:|---|---|",
    ]
    for row in strongest_losses.itertuples():
        lines.append(f"|{row.sample_id}|{row.peptide_sequence}|{row.position}|{row.original_amino_acid}→{row.mutant_amino_acid}|{row.delta_rank_fusion:.6f}|{row.target_tissue}|{row.mhc_restriction}|")
    lines.extend([
        "",
        "# The most powerful ascending mutation (final rank Fusion)",
        "",
        "The sample is the original position of the sample, which is the replacement of the lang fasion.",
        "|---|---|---:|:---:|---:|---|---|",
    ])
    for row in strongest_gains.itertuples():
        lines.append(f"|{row.sample_id}|{row.peptide_sequence}|{row.position}|{row.original_amino_acid}→{row.mutant_amino_acid}|{row.delta_rank_fusion:.6f}|{row.target_tissue}|{row.mhc_restriction}|")
    lines.extend([
        "",
        "# Explain the boundary",
        "",
        "1. ISM has demonstrated that the model relies on the calculation of single point replacement and is not equivalent to internal causal effects.",
        "2. The mutant thorium may not be in the training distribution; the extreme effects shall be reviewed in conjunction with the known HLA motif, the combined experiments or the external forecaster.",
        "The final rank effect is fixed by the original test task fractions and replaces the original sample with the mutant; it is not a re-entry of the entire test set into a combined mutation.",
        "4. P2/P9 is a classic anchor position assumption for cross-HLA aggregation, and the equivalent genetic speciality model should be based on the HLA stratification result.",
        "5. This analysis follows the same sample of the SHAP for each mission test Pair; the conclusion represents the sample explained and should not be extrapolated to the full potential of platinum.",
        "",
    ])
    (cli.output_dir / "ISM_ANALYSIS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run(cli: argparse.Namespace) -> None:
    torch, nn, _, _ = base.require_torch()
    device = shap_run.resolve_device(cli.device, torch)
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train, test, mappings, peptide_length = shap_run.validate_and_prepare(cli.train, cli.test, 0)
    if peptide_length != 9:
        raise ValueError(f"Formal ISM expects 9-mers, observed length={peptide_length}")
    seeds = cli.seeds[:1] if cli.smoke else cli.seeds
    if not cli.smoke and seeds != DEFAULT_SEEDS:
        raise ValueError(f"Formal analysis uses frozen seeds {DEFAULT_SEEDS}")
    for path in (cli.checkpoint_root,):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if not cli.branch_predictions.is_file():
        raise FileNotFoundError(cli.branch_predictions)
    if not cli.shap_observed.is_file():
        raise FileNotFoundError(cli.shap_observed)

    selected_parts = []
    for task_name, group in test.groupby("task_name", sort=True):
        selected_parts.append(shap_run.select_complete_pairs(
            group, cli.pairs_per_task,
            shap_run.stable_seed(task_name, cli.sample_seed),
        ))
    selected = pd.concat(selected_parts, ignore_index=True).sort_values("sample_id").reset_index(drop=True)
    if cli.smoke:
        smoke_parts = []
        for _, group in selected.groupby("task_name", sort=True):
            first_pair = sorted(group["pair_id"].unique(), key=str)[0]
            smoke_parts.append(group[group["pair_id"] == first_pair])
            if sum(len(part) for part in smoke_parts) >= cli.smoke_rows:
                break
        selected = pd.concat(smoke_parts, ignore_index=True)
    branch_predictions = pd.read_csv(cli.branch_predictions, keep_default_na=False)
    if "task_name" not in branch_predictions.columns:
        branch_predictions = branch_predictions.merge(
            test[["sample_id", "task_name"]], on="sample_id", how="left", validate="many_to_one"
        )
        if branch_predictions["task_name"].isna().any():
            raise ValueError("Could not recover task_name for every saved branch prediction")
    args = shap_run.model_args(cli)
    seed_effects: dict[int, pd.DataFrame] = {}
    timing_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    print(
        f"[ISM START] train={len(train)} test={len(test)} selected={len(selected)} "
        f"tasks={selected['task_name'].nunique()} seeds={seeds} device={device}", flush=True,
    )

    for seed in seeds:
        seed_started = time.perf_counter()
        seed_dir = cli.checkpoint_root / f"seed_{seed}"
        reference = branch_predictions[pd.to_numeric(branch_predictions["seed"], errors="coerce") == seed].copy()
        global_model, global_payload = shap_run.load_model(
            torch, nn, args, seed_dir / "global_aux.pt", peptide_length,
            len(mappings["tissue_to_id"]), len(mappings["hla_to_id"]), device,
        )
        global_effects = score_branch(
            torch, global_model, global_payload["task_to_id"], selected,
            peptide_length, device, cli.inference_batch_size,
        )
        del global_model
        if device == "cuda":
            torch.cuda.empty_cache()

        hla_parts = []
        hlas = sorted(selected["mhc_restriction"].unique())
        for hla_index, hla in enumerate(hlas, start=1):
            safe_hla = hla.replace("*", "_").replace(":", "_").replace("/", "_")
            checkpoint = seed_dir / f"hla_plain__{safe_hla}.pt"
            model, payload = shap_run.load_model(
                torch, nn, args, checkpoint, peptide_length,
                len(mappings["tissue_to_id"]), len(mappings["hla_to_id"]), device,
            )
            subset = selected[selected["mhc_restriction"] == hla].copy()
            hla_parts.append(score_branch(
                torch, model, payload["task_to_id"], subset,
                peptide_length, device, cli.inference_batch_size,
            ))
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            print(f"[ISM HLA] seed={seed} {hla_index}/{len(hlas)} hla={hla} rows={len(subset)}", flush=True)
        hla_effects = pd.concat(hla_parts, ignore_index=True)
        combined, audit = combine_branches(global_effects, hla_effects, reference, seed)
        combined = combined.sort_values(
            ["sample_id", "position", "mutant_amino_acid"]
        ).reset_index(drop=True)
        output_path = cli.output_dir / f"mutation_effects_seed_{seed}.csv.gz"
        combined.to_csv(output_path, index=False, compression="gzip")
        seed_effects[seed] = combined
        audit_rows.append({"seed": seed, **audit, "mutation_rows": len(combined)})
        seed_elapsed = time.perf_counter() - seed_started
        timing_rows.append({"seed": seed, "stage": "ism_all_branches_and_rank_fusion", "seconds": seed_elapsed})
        print(f"[SEED ISM TIME] seed={seed} seconds={seed_elapsed:.3f} rows={len(combined)}", flush=True)

    stability = stability_analysis(seed_effects)
    ensemble = ensemble_effects(seed_effects)
    ensemble.to_csv(cli.output_dir / "mutation_effects_three_seed_mean.csv.gz", index=False, compression="gzip")
    sample_position = sample_position_table(ensemble)
    positions = position_summary(sample_position)
    substitutions = substitution_summary(ensemble)
    anchor = anchor_tests(sample_position, cli.bootstrap_replicates, cli.sample_seed)
    paired, paired_summary = paired_position_analysis(sample_position)
    shap_merged, concordance = shap_concordance(seed_effects, cli.shap_observed)
    audits = pd.DataFrame(audit_rows)

    sample_position.to_csv(cli.output_dir / "sample_position_sensitivity.csv.gz", index=False, compression="gzip")
    positions.to_csv(cli.output_dir / "position_sensitivity_summary.csv.gz", index=False, compression="gzip")
    substitutions.to_csv(cli.output_dir / "substitution_summary.csv.gz", index=False, compression="gzip")
    anchor.to_csv(cli.output_dir / "anchor_position_tests.csv", index=False)
    paired.to_csv(cli.output_dir / "paired_position_ism.csv.gz", index=False, compression="gzip")
    paired_summary.to_csv(cli.output_dir / "paired_position_ism_summary.csv", index=False)
    stability.to_csv(cli.output_dir / "seed_stability.csv", index=False)
    shap_merged.to_csv(cli.output_dir / "shap_ism_sample_position.csv.gz", index=False, compression="gzip")
    concordance.to_csv(cli.output_dir / "shap_ism_concordance.csv", index=False)
    audits.to_csv(cli.output_dir / "prediction_reproduction_audit.csv", index=False)
    ensemble.nsmallest(100, "delta_rank_fusion").to_csv(cli.output_dir / "top_score_decreasing_mutations.csv", index=False)
    ensemble.nlargest(100, "delta_rank_fusion").to_csv(cli.output_dir / "top_score_increasing_mutations.csv", index=False)
    create_figures(ensemble, sample_position, paired_summary, shap_merged, cli.output_dir)

    elapsed = time.perf_counter() - started
    timing_rows.append({"seed": "all", "stage": "total", "seconds": elapsed})
    pd.DataFrame(timing_rows).to_csv(cli.output_dir / "timing_results.csv", index=False)
    write_report(
        cli, ensemble, sample_position, stability, anchor, paired_summary,
        concordance, audits, elapsed,
    )
    metadata = {
        "analysis": "exhaustive single-amino-acid ISM validation of occurrence-equal E29",
        "train": str(cli.train.resolve()), "test": str(cli.test.resolve()),
        "checkpoint_root": str(cli.checkpoint_root.resolve()),
        "branch_predictions": str(cli.branch_predictions.resolve()),
        "shap_observed": str(cli.shap_observed.resolve()),
        "seeds": seeds, "device": device, "train_rows": len(train), "test_rows": len(test),
        "selected_rows": len(selected), "selected_pairs": selected["pair_id"].nunique(),
        "selected_tasks": selected["task_name"].nunique(), "peptide_length": peptide_length,
        "mutations_per_peptide": peptide_length * 19, "ensemble_mutation_rows": len(ensemble),
        "pairs_per_task": cli.pairs_per_task, "sample_seed": cli.sample_seed,
        "bootstrap_replicates": cli.bootstrap_replicates,
        "rank_effect_definition": (
            "replace the original sample by one mutant within the fixed complete test-task "
            "reference distribution, recompute each branch percentile rank, then average"
        ),
        "versions": {
            "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
        },
        "elapsed_seconds": elapsed,
    }
    (cli.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
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
        "--shap-observed", type=Path,
        default=PROJECT_ROOT / "extra_occurrence_equal_dataset/results/e29_tuned_shap_expected_ig/sample_observed_shap.csv.gz",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "extra_occurrence_equal_dataset/results/e29_tuned_ism",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--aux-weight", type=float, default=0.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--pairs-per-task", type=int, default=20)
    parser.add_argument("--sample-seed", type=int, default=20260805)
    parser.add_argument("--inference-batch-size", type=int, default=8192)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-rows", type=int, default=4)
    parser.add_argument("--smoke-epochs", type=int, default=1, help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
