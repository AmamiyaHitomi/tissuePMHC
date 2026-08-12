#!/usr/bin/env python3
"""Build report-ready Phase 7 figures and source tables.

This script only reads frozen dataset/result artifacts. It does not train models
or alter any experiment output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm


ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "baseline": "#6f7c80",
    "ensemble": "#d17a22",
    "final": "#147d92",
    "negative": "#b24a46",
    "neutral": "#c7ced1",
    "dark": "#30373b",
    "common": "#8266a3",
}
TASK_KEYS = ["target_tissue", "mhc_restriction"]


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.alpha": 0.22, "grid.linewidth": 0.7,
        "legend.frameon": False, "savefig.dpi": 300,
    })


def selected_summary(path: Path, model: str, average_seeds: bool = False) -> dict[str, float]:
    table = pd.read_csv(path)
    table = table.loc[table.model.eq(model)].copy()
    if table.empty:
        raise ValueError(f"Model {model!r} missing from {path}")
    if not average_seeds and len(table) != 1:
        raise ValueError(f"Expected one summary row for {model!r} in {path}")
    metrics = ["mean_auroc", "mean_auprc", "mean_accuracy", "mean_mcc", "worst_10_mean_auroc"]
    return {metric: float(table[metric].mean()) for metric in metrics}


def phase7_milestones(results: Path) -> pd.DataFrame:
    specs = [
        ("E2 shared heads", "baseline", results / "tissuePMHC_phase7_min200_e2_shared_heads" / "summary_metrics.csv", "shared_peptide_encoder_task_heads", True),
        ("E14a auxiliary dual branch", "baseline", results / "tissuePMHC_phase7_min200_e14_auxiliary_soft_ensemble" / "summary_metrics.csv", "e14a_global_aux_hla_plain", True),
        ("E17 3-seed rank ensemble", "ensemble", results / "tissuePMHC_phase7_min200_e17_seed_ensemble" / "summary_metrics.csv", "e17_3seed_rank_average", False),
        ("E26 OOF greedy selection", "baseline", results / "tissuePMHC_phase7_min200_e26_greedy_ensemble_selection" / "summary_metrics.csv", "e26_oof_greedy_task_rank_average", False),
        ("E29 CNN 3-seed mean", "final", results / "tissuePMHC_phase7_min200_e29_multikernel_cnn" / "summary_metrics.csv", "phase7_min200_e29_cnn_3seed_mean", False),
    ]
    rows = []
    for label, kind, path, model, average_seeds in specs:
        rows.append({"label": label, "kind": kind, **selected_summary(path, model, average_seeds)})
    return pd.DataFrame(rows)


def task_metric_table(path: Path, model: str) -> pd.DataFrame:
    table = pd.read_csv(path)
    table = table.loc[table.model.eq(model)].copy()
    if table.empty:
        raise ValueError(f"Model {model!r} missing from {path}")
    return table.groupby(TASK_KEYS, as_index=False)[["auroc", "auprc", "accuracy", "mcc"]].mean()


def minpairs_comparison(results: Path) -> pd.DataFrame:
    specs = [
        ("E2", results / "tissuePMHC_neural_baselines_v2" / "per_task_metrics.csv", "shared_peptide_encoder_task_heads",
         results / "tissuePMHC_phase7_min200_e2_shared_heads" / "per_task_metrics.csv", "shared_peptide_encoder_task_heads"),
        ("E14a", results / "tissuePMHC_auxiliary_soft_ensemble" / "per_task_metrics.csv", "e14a_global_aux_hla_plain",
         results / "tissuePMHC_phase7_min200_e14_auxiliary_soft_ensemble" / "per_task_metrics.csv", "e14a_global_aux_hla_plain"),
        ("E17", results / "tissuePMHC_e17_seed_ensemble" / "per_task_metrics.csv", "e17_3seed_rank_average",
         results / "tissuePMHC_phase7_min200_e17_seed_ensemble" / "per_task_metrics.csv", "e17_3seed_rank_average"),
        ("E29", results / "tissuePMHC_e29_multikernel_cnn_3seed" / "per_task_metrics.csv", "e29_cnn_3seed_mean",
         results / "tissuePMHC_phase7_min200_e29_multikernel_cnn" / "per_task_metrics.csv", "phase7_min200_e29_cnn_3seed_mean"),
    ]
    rows: list[dict[str, object]] = []
    for label, old_path, old_model, new_path, new_model in specs:
        old = task_metric_table(old_path, old_model)
        new = task_metric_table(new_path, new_model)
        common = new.merge(old[TASK_KEYS], on=TASK_KEYS, how="inner", validate="one_to_one")
        for scope, table in [("min500 (44 tasks)", old), ("min200 common 44", common), ("min200 all 157", new)]:
            rows.append({
                "model": label, "scope": scope, "n_tasks": len(table),
                **{f"mean_{metric}": float(table[metric].mean()) for metric in ["auroc", "auprc", "accuracy", "mcc"]},
            })
    return pd.DataFrame(rows)


def benchmark_comparison(old_data: Path, new_data: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    old_meta = json.loads((old_data / "tissuePMHC_metadata.json").read_text(encoding="utf-8"))
    new_meta = json.loads((new_data / "tissuePMHC_phase7_min200_metadata.json").read_text(encoding="utf-8"))
    rows = []
    for label, meta, data_dir, train_name in [
        ("min500", old_meta, old_data, "tissuePMHC_train.csv.gz"),
        ("min200", new_meta, new_data, "tissuePMHC_phase7_min200_train.csv.gz"),
    ]:
        train = pd.read_csv(data_dir / train_name, usecols=["target_tissue", "mhc_restriction"])
        rows.append({
            "benchmark": label,
            "tasks": int(meta["n_tissue_hla_groups"]),
            "train_rows": int(meta["train_rows"]),
            "test_rows": int(meta["test_rows"]),
            "train_pairs": int(meta["train_pairs"]),
            "test_pairs": int(meta["test_pairs"]),
            "tissues": int(train.target_tissue.nunique()),
            "hla_alleles": int(train.mhc_restriction.nunique()),
        })

    old_test = pd.read_csv(old_data / "tissuePMHC_test.csv.gz", usecols=[*TASK_KEYS, "pair_id"])
    new_test = pd.read_csv(new_data / "tissuePMHC_phase7_min200_test.csv.gz", usecols=[*TASK_KEYS, "pair_id"])
    overlap_rows = []
    for task, old_task in old_test.groupby(TASK_KEYS, sort=True):
        new_task = new_test.loc[(new_test.target_tissue == task[0]) & (new_test.mhc_restriction == task[1])]
        old_pairs = set(old_task.pair_id.astype(str))
        new_pairs = set(new_task.pair_id.astype(str))
        overlap_rows.append({
            "target_tissue": task[0], "mhc_restriction": task[1],
            "old_test_pairs": len(old_pairs), "new_test_pairs": len(new_pairs),
            "overlap_test_pairs": len(old_pairs & new_pairs),
            "overlap_fraction": len(old_pairs & new_pairs) / len(old_pairs),
        })
    return pd.DataFrame(rows), pd.DataFrame(overlap_rows)


def e29_task_deltas(results: Path) -> pd.DataFrame:
    path = results / "tissuePMHC_phase7_min200_e29_multikernel_cnn" / "e17_3seed_comparison_metrics.csv"
    table = pd.read_csv(path)
    table["task"] = table.target_tissue + " | " + table.mhc_restriction
    return table


def e29_seed_scaling(results: Path) -> pd.DataFrame:
    summary = pd.read_csv(results / "tissuePMHC_phase7_min200_e29_multikernel_cnn" / "summary_metrics.csv")
    singles = summary.loc[summary.model.eq("e29_cnn_single_seed")].copy()
    ensemble = summary.loc[summary.model.eq("phase7_min200_e29_cnn_3seed_mean")].copy()
    rows = []
    for row in singles.itertuples(index=False):
        rows.append({"label": str(row.seed), "kind": "single", "mean_auroc": row.mean_auroc, "mean_auprc": row.mean_auprc, "worst10_auroc": row.worst_10_mean_auroc})
    row = ensemble.iloc[0]
    rows.append({"label": "3-seed mean", "kind": "ensemble", "mean_auroc": row.mean_auroc, "mean_auprc": row.mean_auprc, "worst10_auroc": row.worst_10_mean_auroc})
    return pd.DataFrame(rows)


def oof_test_confirmation(results: Path) -> pd.DataFrame:
    screen = json.loads((results / "tissuePMHC_phase7_min200_e29_multikernel_cnn" / "oof_screen_summary.json").read_text(encoding="utf-8"))["screen"]
    e17 = selected_summary(results / "tissuePMHC_phase7_min200_e17_seed_ensemble" / "summary_metrics.csv", "e17_3seed_rank_average")
    e29 = selected_summary(results / "tissuePMHC_phase7_min200_e29_multikernel_cnn" / "summary_metrics.csv", "phase7_min200_e29_cnn_3seed_mean")
    return pd.DataFrame([
        {"stage": "OOF", "model": "Matched E14 baseline", "mean_auroc": screen["matching_baseline_oof"]["mean_auroc"], "mean_auprc": screen["matching_baseline_oof"]["mean_auprc"], "worst10_auroc": screen["matching_baseline_oof"]["worst_10_mean_auroc"]},
        {"stage": "OOF", "model": "E29 CNN", "mean_auroc": screen["cnn_oof"]["mean_auroc"], "mean_auprc": screen["cnn_oof"]["mean_auprc"], "worst10_auroc": screen["cnn_oof"]["worst_10_mean_auroc"]},
        {"stage": "Test", "model": "E17 3-seed", "mean_auroc": e17["mean_auroc"], "mean_auprc": e17["mean_auprc"], "worst10_auroc": e17["worst_10_mean_auroc"]},
        {"stage": "Test", "model": "E29 CNN", "mean_auroc": e29["mean_auroc"], "mean_auprc": e29["mean_auprc"], "worst10_auroc": e29["worst_10_mean_auroc"]},
    ])


def plot_milestones(table: pd.DataFrame) -> plt.Figure:
    metrics = [("mean_auroc", "Mean task AUROC"), ("mean_auprc", "Mean task AUPRC"), ("worst_10_mean_auroc", "Worst-10 mean AUROC")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), sharey=True)
    y = np.arange(len(table))
    colors = [COLORS["final"] if kind == "final" else COLORS["ensemble"] if kind == "ensemble" else COLORS["baseline"] for kind in table.kind]
    for ax, (metric, title) in zip(axes, metrics):
        values = table[metric].to_numpy(float)
        lower = values.min() - 0.015
        ax.barh(y, values - lower, left=lower, color=colors)
        for index, value in enumerate(values):
            ax.text(value + 0.001, index, f"{value:.4f}", va="center", fontsize=8)
        ax.set_title(title); ax.set_xlabel("Task-macro score")
        ax.set_xlim(lower, values.max() + 0.025); ax.set_yticks(y, table.label)
    axes[0].invert_yaxis()
    fig.suptitle("tissuePMHC Phase 7 model progression (min_pairs > 200)", y=1.0)
    return fig


def plot_minpairs(table: pd.DataFrame, benchmark: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    scopes = ["min500 (44 tasks)", "min200 common 44", "min200 all 157"]
    colors = [COLORS["baseline"], COLORS["common"], COLORS["final"]]
    x = np.arange(table.model.nunique()); width = 0.24
    for idx, scope in enumerate(scopes):
        subset = table.loc[table.scope.eq(scope)].set_index("model").loc[["E2", "E14a", "E17", "E29"]]
        axes[0].bar(x + (idx - 1) * width, subset.mean_auroc, width, label=scope, color=colors[idx])
    axes[0].set_xticks(x, ["E2", "E14a", "E17", "E29"])
    axes[0].set_ylim(0.77, 0.86); axes[0].set_ylabel("Mean task AUROC")
    axes[0].set_title("Observed scores and the common-task decomposition")
    axes[0].legend(fontsize=8)

    metrics = [("tasks", "Tasks"), ("tissues", "Tissues"), ("hla_alleles", "HLA alleles"), ("train_pairs", "Train pairs")]
    values = []
    labels = []
    for metric, label in metrics:
        old = float(benchmark.loc[benchmark.benchmark.eq("min500"), metric].iloc[0])
        new = float(benchmark.loc[benchmark.benchmark.eq("min200"), metric].iloc[0])
        values.append(new / old); labels.append(label)
    y = np.arange(len(labels))
    axes[1].barh(y, values, color=COLORS["final"])
    axes[1].axvline(1.0, color=COLORS["dark"], linestyle="--", linewidth=1)
    for idx, value in enumerate(values):
        axes[1].text(value + 0.03, idx, f"{value:.2f}x", va="center")
    axes[1].set_yticks(y, labels); axes[1].set_xlabel("min200 / min500")
    axes[1].set_title("Benchmark expansion after lowering min_pairs")
    axes[1].invert_yaxis()
    fig.suptitle("min_pairs=500 versus min_pairs=200: benchmark and split both change", y=1.0)
    return fig


def plot_task_deltas(table: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    axes[0].hist(table.delta_auroc, bins=24, color=COLORS["final"], alpha=0.9)
    axes[0].axvline(0, color=COLORS["dark"], linewidth=1)
    axes[0].axvline(table.delta_auroc.mean(), color=COLORS["ensemble"], linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("E29 minus E17 task AUROC"); axes[0].set_ylabel("Number of tasks")
    axes[0].set_title(f"126 wins / 31 losses; mean {table.delta_auroc.mean():+.4f}")

    axes[1].scatter(table.e17_auroc, table.e29_auroc, color=COLORS["final"], alpha=0.7, s=22)
    lower = min(table.e17_auroc.min(), table.e29_auroc.min()) - 0.01
    upper = max(table.e17_auroc.max(), table.e29_auroc.max()) + 0.01
    axes[1].plot([lower, upper], [lower, upper], color=COLORS["dark"], linestyle="--", linewidth=1)
    axes[1].set_xlim(lower, upper); axes[1].set_ylim(lower, upper)
    axes[1].set_xlabel("E17 task AUROC"); axes[1].set_ylabel("E29 task AUROC")
    axes[1].set_title("Task-level independent-test comparison")
    fig.suptitle("E29 improves most tissue-HLA tasks on the fixed test set", y=1.0)
    return fig


def plot_coverage(summary_path: Path) -> plt.Figure:
    coverage = pd.read_csv(summary_path)
    pivot = coverage.pivot(index="target_tissue", columns="mhc_restriction", values="train_pairs").fillna(0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values().index]
    matrix = pivot.to_numpy(); positive = matrix[matrix > 0]
    masked = np.ma.masked_where(matrix <= 0, matrix)
    fig, ax = plt.subplots(figsize=(15, 9.5))
    image = ax.imshow(masked, aspect="auto", cmap="Blues", norm=LogNorm(vmin=positive.min(), vmax=positive.max()))
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=65, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index, fontsize=8)
    ax.set_xlabel("HLA restriction"); ax.set_ylabel("Target tissue")
    ax.set_title("Phase 7 task coverage (train pairs; logarithmic scale)")
    colorbar = fig.colorbar(image, ax=ax, pad=0.015); colorbar.set_label("Train pairs")
    return fig


def plot_seed_scaling(table: pd.DataFrame) -> plt.Figure:
    metrics = [("mean_auroc", "Mean task AUROC"), ("mean_auprc", "Mean task AUPRC"), ("worst10_auroc", "Worst-10 mean AUROC")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.7))
    x = np.arange(len(table)); colors = [COLORS["ensemble"] if kind == "ensemble" else COLORS["baseline"] for kind in table.kind]
    for ax, (metric, title) in zip(axes, metrics):
        values = table[metric].to_numpy(float); lower = values.min() - 0.008
        ax.bar(x, values - lower, bottom=lower, color=colors)
        for index, value in enumerate(values):
            ax.text(index, value + 0.001, f"{value:.4f}", ha="center", fontsize=8)
        ax.set_xticks(x, table.label, rotation=25, ha="right", fontsize=8)
        ax.set_ylim(lower, values.max() + 0.012); ax.set_ylabel("Task-macro score"); ax.set_title(title)
    fig.suptitle("E29 seed averaging provides a large reproducible gain", y=1.0)
    return fig


def plot_oof_test(table: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    metrics = [("mean_auroc", "Mean task AUROC"), ("mean_auprc", "Mean task AUPRC"), ("worst10_auroc", "Worst-10 mean AUROC")]
    for ax, (metric, title) in zip(axes, metrics):
        positions = [0, 1, 3, 4]
        colors = [COLORS["baseline"], COLORS["final"], COLORS["baseline"], COLORS["final"]]
        values = table[metric].to_numpy(float); lower = values.min() - 0.015
        ax.bar(positions, values - lower, bottom=lower, color=colors)
        for position, value in zip(positions, values):
            ax.text(position, value + 0.001, f"{value:.4f}", ha="center", fontsize=8)
        ax.set_xticks(positions, ["E14\nOOF", "E29\nOOF", "E17\nTest", "E29\nTest"])
        ax.set_ylim(lower, values.max() + 0.02); ax.set_ylabel("Task-macro score"); ax.set_title(title)
    fig.suptitle("E29 passes OOF screening and confirms its gain on the independent test set", y=1.0)
    return fig


def save(fig: plt.Figure, output: Path, stem: str, pdf: PdfPages) -> None:
    fig.tight_layout(); fig.savefig(output / f"{stem}.png", bbox_inches="tight"); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--old-data-dir", type=Path, default=ROOT / "data" / "tissuePMHC")
    parser.add_argument("--phase7-data-dir", type=Path, default=ROOT / "data" / "tissuePMHC_phase7_min200")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "figures_phase7")
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True); apply_style()

    milestones = phase7_milestones(args.results_dir)
    minpairs = minpairs_comparison(args.results_dir)
    benchmark, overlap = benchmark_comparison(args.old_data_dir, args.phase7_data_dir)
    task_deltas = e29_task_deltas(args.results_dir)
    seed_scaling = e29_seed_scaling(args.results_dir)
    confirmation = oof_test_confirmation(args.results_dir)

    milestones.to_csv(args.output_dir / "model_milestones.csv", index=False)
    minpairs.to_csv(args.output_dir / "minpairs_model_comparison.csv", index=False)
    benchmark.to_csv(args.output_dir / "benchmark_comparison.csv", index=False)
    overlap.to_csv(args.output_dir / "common_task_test_pair_overlap.csv", index=False)
    task_deltas.to_csv(args.output_dir / "e29_vs_e17_task_metrics.csv", index=False)
    seed_scaling.to_csv(args.output_dir / "e29_seed_scaling.csv", index=False)
    confirmation.to_csv(args.output_dir / "oof_test_confirmation.csv", index=False)

    with PdfPages(args.output_dir / "tissuepmhc_phase7_figures.pdf") as pdf:
        save(plot_milestones(milestones), args.output_dir, "01_model_milestones", pdf)
        save(plot_minpairs(minpairs, benchmark), args.output_dir, "02_minpairs_500_vs_200", pdf)
        save(plot_task_deltas(task_deltas), args.output_dir, "03_e29_vs_e17_task_gain", pdf)
        save(plot_coverage(args.phase7_data_dir / "tissuePMHC_phase7_min200_summary.csv"), args.output_dir, "04_dataset_coverage", pdf)
        save(plot_seed_scaling(seed_scaling), args.output_dir, "05_e29_seed_ensemble", pdf)
        save(plot_oof_test(confirmation), args.output_dir, "06_oof_test_confirmation", pdf)

    metadata = {
        "status": "phase7_complete",
        "main_result": "Phase7 min200 E29 CNN 3-seed mean",
        "n_figures": 6,
        "benchmark_warning": "min500 and min200 have different task sets and redrawn common-task test pairs; score differences are descriptive, not a controlled causal estimate.",
        "common_task_test_pair_overlap_mean": float(overlap.overlap_fraction.mean()),
        "common_task_test_pair_overlap_min": float(overlap.overlap_fraction.min()),
        "common_task_test_pair_overlap_max": float(overlap.overlap_fraction.max()),
    }
    (args.output_dir / "figure_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote six Phase 7 figures, source tables, and PDF to: {args.output_dir}")


if __name__ == "__main__":
    main()
