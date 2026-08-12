#!/usr/bin/env python3
"""Create publication-style visual summaries for all premium benchmark results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "extra_premium" / "results"
OUTPUT = RESULTS / "visualization"
TRAIN = ROOT / "data" / "humanPMHC_premium" / "humanPMHC_train.csv.gz"
TEST = ROOT / "data" / "humanPMHC_premium" / "humanPMHC_test.csv.gz"
OLD_EXTERNAL = (
    ROOT
    / "results"
    / "issue5_general_pmhc"
    / "external_evaluation"
    / "summary_metrics.csv"
)

COLORS = {
    "traditional": "#7A8594",
    "external_mhcflurry": "#D89000",
    "external_netmhcpan": "#8A63A8",
    "e2": "#3978B5",
    "e14": "#238B82",
    "e29": "#D85843",
    "reference": "#AEB6C2",
    "positive": "#238B82",
    "negative": "#D85843",
}

DISPLAY = {
    "blosum62_logistic_regression": "E0 · BLOSUM LR",
    "onehot_logistic_regression": "E0 · One-hot LR",
    "blosum62_random_forest": "E0 · Random forest",
    "blosum62_extra_trees": "E0 · Extra Trees",
    "blosum62_hist_gradient_boosting": "E0 · HistGradientBoost",
    "e2_shared_heads": "E2 · Shared heads",
    "e14a_auxiliary_dual_branch": "E14a · Auxiliary dual branch",
    "e29_multikernel_cnn": "E29 · Multi-kernel CNN",
    "mhcflurry_2.2.1::affinity_nm": "MHCflurry · Affinity",
    "mhcflurry_2.2.1::affinity_percentile": "MHCflurry · Affinity percentile",
    "mhcflurry_2.2.1::presentation_score": "MHCflurry · Presentation",
    "netmhcpan_4.1b::ba_rank": "NetMHCpan · BA rank",
    "netmhcpan_4.1b::el_rank": "NetMHCpan · EL rank",
    "netmhcpan_4.1b::el_score": "NetMHCpan · EL score",
}

PRIMARY = [
    "netmhcpan_4.1b::ba_rank",
    "netmhcpan_4.1b::el_rank",
    "mhcflurry_2.2.1::presentation_score",
    "blosum62_random_forest",
    "e2_shared_heads",
    "e14a_auxiliary_dual_branch",
    "e29_multikernel_cnn",
]


def model_color(model: str) -> str:
    if model == "e29_multikernel_cnn":
        return COLORS["e29"]
    if model == "e14a_auxiliary_dual_branch":
        return COLORS["e14"]
    if model == "e2_shared_heads":
        return COLORS["e2"]
    if model.startswith("mhcflurry"):
        return COLORS["external_mhcflurry"]
    if model.startswith("netmhcpan"):
        return COLORS["external_netmhcpan"]
    return COLORS["traditional"]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": "#AEB6C2",
            "axes.linewidth": 0.8,
            "xtick.color": "#4B5563",
            "ytick.color": "#374151",
            "text.color": "#1F2937",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axis(ax: plt.Axes, grid_axis: str | None = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#E5E7EB", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
    )


def load_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries: list[pd.DataFrame] = []
    tasks: list[pd.DataFrame] = []
    predictions: list[pd.DataFrame] = []
    for directory in sorted(path for path in RESULTS.iterdir() if path.is_dir()):
        summary_path = directory / "summary_metrics.csv"
        task_path = directory / "per_task_metrics.csv"
        prediction_candidates = [
            directory / "test_predictions.csv",
            directory / "test_predictions.csv.gz",
        ]
        if summary_path.is_file():
            frame = pd.read_csv(summary_path)
            frame["result_group"] = directory.name
            summaries.append(frame)
        if task_path.is_file():
            tasks.append(pd.read_csv(task_path))
        for prediction_path in prediction_candidates:
            if prediction_path.is_file():
                frame = pd.read_csv(prediction_path)
                required = {"model", "sample_id", "pair_id", "label", "score"}
                if required <= set(frame.columns):
                    predictions.append(frame[list(required)])
                break
    if not summaries or not tasks or not predictions:
        raise FileNotFoundError("Complete premium summary/task/prediction results were not found.")
    summary = pd.concat(summaries, ignore_index=True, sort=False)
    per_task = pd.concat(tasks, ignore_index=True, sort=False)
    prediction = pd.concat(predictions, ignore_index=True, sort=False)
    expected = set(DISPLAY)
    missing = sorted(expected - set(summary["model"]))
    if missing:
        raise ValueError(f"Summary results are missing model(s): {missing}")
    return summary, per_task, prediction


def bootstrap_stage_gains(
    per_task: pd.DataFrame, iterations: int = 20000
) -> pd.DataFrame:
    wide = per_task.pivot(
        index=["target_tissue", "mhc_restriction"],
        columns="model",
        values="auroc",
    )
    comparisons = [
        ("E2 − E0 RF", "e2_shared_heads", "blosum62_random_forest"),
        (
            "E14a − E2",
            "e14a_auxiliary_dual_branch",
            "e2_shared_heads",
        ),
        (
            "E29 − E14a",
            "e29_multikernel_cnn",
            "e14a_auxiliary_dual_branch",
        ),
    ]
    rng = np.random.default_rng(20260730)
    rows = []
    for label, newer, older in comparisons:
        difference = (wide[newer] - wide[older]).to_numpy(dtype=float)
        sampled = rng.choice(
            difference,
            size=(iterations, len(difference)),
            replace=True,
        ).mean(axis=1)
        rows.append(
            {
                "comparison": label,
                "newer": newer,
                "older": older,
                "mean_delta": float(difference.mean()),
                "median_delta": float(np.median(difference)),
                "ci_low": float(np.quantile(sampled, 0.025)),
                "ci_high": float(np.quantile(sampled, 0.975)),
                "wins": int((difference > 0).sum()),
                "ties": int((difference == 0).sum()),
                "losses": int((difference < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def original_stage_values() -> pd.DataFrame:
    e0 = pd.read_csv(ROOT / "results/tissuePMHC_baselines/summary_metrics.csv")
    e2 = pd.read_csv(ROOT / "results/tissuePMHC_neural_baselines_v2/summary_metrics.csv")
    e14 = pd.read_csv(
        ROOT / "results/tissuePMHC_auxiliary_soft_ensemble/summary_metrics.csv"
    )
    e29 = pd.read_csv(
        ROOT / "results/tissuePMHC_e29_multikernel_cnn_3seed/summary_metrics.csv"
    )
    rows = [
        {
            "stage": "E0 best",
            "dataset": "Original 44-task",
            "auroc": float(e0["mean_auroc"].max()),
        },
        {
            "stage": "E2",
            "dataset": "Original 44-task",
            "auroc": float(
                e2[
                    e2["experiment_name"].eq(
                        "E2_shared_peptide_encoder_task_heads"
                    )
                    & e2["seed"].eq(20260704)
                ]["mean_auroc"].iloc[0]
            ),
        },
        {
            "stage": "E14a",
            "dataset": "Original 44-task",
            "auroc": float(
                e14[
                    e14["model"].eq("e14a_global_aux_hla_plain")
                    & e14["seed"].eq(20260704)
                ]["mean_auroc"].iloc[0]
            ),
        },
        {
            "stage": "E29",
            "dataset": "Original 44-task",
            "auroc": float(
                e29[
                    e29["model"].eq("e29_cnn_single_seed")
                    & e29["seed"].eq(20260704)
                ]["mean_auroc"].iloc[0]
            ),
        },
    ]
    return pd.DataFrame(rows)


def premium_stage_values(summary: pd.DataFrame) -> pd.DataFrame:
    e0 = summary[
        summary["result_group"].eq("e0_traditional")
    ].sort_values("mean_task_auroc", ascending=False).iloc[0]
    mapping = [
        ("E0 best", e0),
        (
            "E2",
            summary[summary["model"].eq("e2_shared_heads")].iloc[0],
        ),
        (
            "E14a",
            summary[
                summary["model"].eq("e14a_auxiliary_dual_branch")
            ].iloc[0],
        ),
        (
            "E29",
            summary[summary["model"].eq("e29_multikernel_cnn")].iloc[0],
        ),
    ]
    return pd.DataFrame(
        [
            {
                "stage": stage,
                "dataset": "Premium 75-task",
                "auroc": float(row["mean_task_auroc"]),
            }
            for stage, row in mapping
        ]
    )


def pair_diagnostics(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(
        TRAIN,
        usecols=["peptide_sequence", "mhc_restriction"],
    )
    test = pd.read_csv(
        TEST,
        usecols=[
            "sample_id",
            "pair_id",
            "label",
            "peptide_sequence",
            "mhc_restriction",
            "other_tissue_presentation_count",
        ],
    )
    seen_peptides = set(train["peptide_sequence"])
    seen_hla_peptides = set(
        zip(train["mhc_restriction"], train["peptide_sequence"], strict=True)
    )
    test["peptide_seen"] = test["peptide_sequence"].isin(seen_peptides)
    test["hla_peptide_seen"] = [
        item in seen_hla_peptides
        for item in zip(
            test["mhc_restriction"],
            test["peptide_sequence"],
            strict=True,
        )
    ]
    pair_meta = test.groupby("pair_id", sort=False).agg(
        peptide_seen_count=("peptide_seen", "sum"),
        hla_peptide_seen_count=("hla_peptide_seen", "sum"),
        other_tissue_count=("other_tissue_presentation_count", "first"),
    )
    working = predictions.merge(
        pair_meta,
        left_on="pair_id",
        right_index=True,
        how="left",
        validate="many_to_one",
    )
    working["other_count_group"] = working["other_tissue_count"].map(
        lambda value: "1" if value == 1 else ("2" if value == 2 else "3+")
    )
    working["peptide_seen_group"] = working["peptide_seen_count"].map(
        {0: "0 seen", 1: "1 seen", 2: "2 seen"}
    )
    rows = []
    for group_column in ["other_count_group", "peptide_seen_group"]:
        for (model, group_name), group in working.groupby(
            ["model", group_column], sort=True
        ):
            pair_scores = group.pivot(
                index="pair_id", columns="label", values="score"
            )
            rows.append(
                {
                    "dimension": group_column,
                    "group": group_name,
                    "model": model,
                    "pairs": int(len(pair_scores)),
                    "pair_accuracy": float(
                        (pair_scores[1] > pair_scores[0]).mean()
                    ),
                }
            )
    diagnostic = pd.DataFrame(rows)
    pair_counts = pair_meta.reset_index()
    return diagnostic, pair_counts, test


def save_figure(
    fig: plt.Figure,
    stem: str,
    multipage: PdfPages,
) -> None:
    fig.savefig(OUTPUT / f"{stem}.png", dpi=320)
    fig.savefig(OUTPUT / f"{stem}.pdf")
    multipage.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def figure_overview(
    summary: pd.DataFrame,
    per_task: pd.DataFrame,
    stage_gains: pd.DataFrame,
    stage_values: pd.DataFrame,
    multipage: PdfPages,
) -> None:
    ordered = summary.sort_values("mean_task_auroc", ascending=True).copy()
    fig = plt.figure(figsize=(15, 10.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1.2, 1])
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, 0])
    ax4 = fig.add_subplot(grid[1, 1])

    y = np.arange(len(ordered))
    for metric, marker, offset, label in [
        ("mean_task_auroc", "o", 0.16, "AUROC"),
        ("mean_task_auprc", "s", 0.00, "AUPRC"),
        ("mean_task_pair_accuracy", "^", -0.16, "PairAcc"),
    ]:
        ax1.scatter(
            ordered[metric],
            y + offset,
            marker=marker,
            s=38,
            color=[model_color(model) for model in ordered["model"]],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
            label=label,
        )
    ax1.set_yticks(y, [DISPLAY[model] for model in ordered["model"]])
    ax1.set_xlim(0.59, 0.72)
    ax1.set_xlabel("Task-macro score")
    ax1.set_title("All models, all reported ranking metrics")
    ax1.legend(frameon=False, ncols=3, loc="lower right")
    clean_axis(ax1)
    panel_label(ax1, "A")

    stages = ["E0 best", "E2", "E14a", "E29"]
    x = np.arange(len(stages))
    for dataset, color, marker in [
        ("Original 44-task", "#6B7280", "o"),
        ("Premium 75-task", COLORS["e29"], "s"),
    ]:
        subset = stage_values[stage_values["dataset"].eq(dataset)].set_index(
            "stage"
        ).loc[stages]
        ax2.plot(
            x,
            subset["auroc"],
            color=color,
            marker=marker,
            linewidth=2.2,
            markersize=7,
            label=dataset,
        )
        for index, value in enumerate(subset["auroc"]):
            ax2.text(
                index,
                value + 0.006,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
            )
    ax2.set_xticks(x, stages)
    ax2.set_ylim(0.64, 0.85)
    ax2.set_ylabel("Mean task AUROC")
    ax2.set_title("Stage progression is preserved but compressed")
    ax2.legend(frameon=False, loc="lower right")
    clean_axis(ax2, "y")
    panel_label(ax2, "B")

    gain_y = np.arange(len(stage_gains))[::-1]
    for position, row in zip(gain_y, stage_gains.itertuples(), strict=True):
        color = (
            COLORS["positive"]
            if row.ci_low > 0
            else COLORS["reference"]
        )
        ax3.errorbar(
            row.mean_delta,
            position,
            xerr=[
                [row.mean_delta - row.ci_low],
                [row.ci_high - row.mean_delta],
            ],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=4,
            markersize=7,
            linewidth=2,
        )
        ax3.text(
            row.ci_high + 0.002,
            position,
            f"{row.wins}W / {row.losses}L",
            va="center",
            fontsize=8,
        )
    ax3.axvline(0, color="#4B5563", linewidth=1)
    ax3.set_yticks(gain_y, stage_gains["comparison"])
    ax3.set_xlim(-0.012, 0.031)
    ax3.set_xlabel("Paired mean AUROC gain (task-bootstrap 95% interval)")
    ax3.set_title("Only the E14a step is clearly stable in one seed")
    clean_axis(ax3)
    panel_label(ax3, "C")

    primary_summary = summary[summary["model"].isin(PRIMARY)].copy()
    for row in primary_summary.itertuples():
        ax4.scatter(
            row.mean_task_auroc,
            row.worst_10_mean_auroc,
            s=85,
            color=model_color(row.model),
            marker="o" if "external" not in row.result_group else "D",
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        label_offsets = {
            "blosum62_random_forest": (5, 7),
            "e2_shared_heads": (5, -9),
        }
        ax4.annotate(
            DISPLAY[row.model],
            (row.mean_task_auroc, row.worst_10_mean_auroc),
            xytext=label_offsets.get(row.model, (5, 4)),
            textcoords="offset points",
            fontsize=7.5,
        )
    ax4.set_xlabel("Mean task AUROC")
    ax4.set_ylabel("Worst-10 mean AUROC")
    ax4.set_title("Average performance versus tail robustness")
    clean_axis(ax4)
    panel_label(ax4, "D")

    fig.suptitle(
        "humanPMHC premium benchmark · unified performance overview",
        fontsize=16,
        fontweight="bold",
    )
    save_figure(fig, "01_performance_overview", multipage)


def figure_task_heatmap(
    per_task: pd.DataFrame,
    multipage: PdfPages,
) -> None:
    working = per_task[per_task["model"].isin(PRIMARY)].copy()
    wide = working.pivot(
        index=["target_tissue", "mhc_restriction"],
        columns="model",
        values="auroc",
    )
    order_frame = wide.reset_index()
    order_frame["e29_order"] = order_frame["e29_multikernel_cnn"]
    order_frame = order_frame.sort_values(
        ["target_tissue", "e29_order"],
        ascending=[True, False],
    )
    task_index = pd.MultiIndex.from_frame(
        order_frame[["target_tissue", "mhc_restriction"]]
    )
    wide = wide.loc[task_index, PRIMARY]

    fig, ax = plt.subplots(figsize=(12.5, 18.5), constrained_layout=True)
    image = ax.imshow(
        wide.to_numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap="RdYlBu",
        vmin=0.42,
        vmax=0.92,
    )
    ax.set_xticks(
        np.arange(len(PRIMARY)),
        [DISPLAY[model] for model in PRIMARY],
        rotation=28,
        ha="right",
    )
    labels = [f"{tissue} · {hla}" for tissue, hla in wide.index]
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=6.5)
    tissues = [item[0] for item in wide.index]
    for index in range(1, len(tissues)):
        if tissues[index] != tissues[index - 1]:
            ax.axhline(index - 0.5, color="white", linewidth=1.6)
    ax.set_title(
        "Task-level AUROC heatmap · shared difficulty and model complementarity",
        pad=14,
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.022, pad=0.02)
    colorbar.set_label("AUROC")
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    save_figure(fig, "02_task_auroc_heatmap", multipage)

    wide.reset_index().to_csv(OUTPUT / "source_task_auroc_primary.csv", index=False)


def figure_data_diagnostics(
    per_task: pd.DataFrame,
    pair_diagnostic: pd.DataFrame,
    multipage: PdfPages,
) -> None:
    internal = [
        "blosum62_random_forest",
        "e2_shared_heads",
        "e14a_auxiliary_dual_branch",
        "e29_multikernel_cnn",
    ]
    selected = [
        "blosum62_random_forest",
        "e2_shared_heads",
        "e14a_auxiliary_dual_branch",
        "e29_multikernel_cnn",
        "mhcflurry_2.2.1::presentation_score",
        "netmhcpan_4.1b::el_rank",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.flat

    e29_tasks = per_task[per_task["model"].eq("e29_multikernel_cnn")][
        ["target_tissue", "mhc_restriction", "train_rows"]
    ].copy()
    e29_tasks["quartile"] = pd.qcut(
        e29_tasks["train_rows"],
        4,
        labels=["Q1 smallest", "Q2", "Q3", "Q4 largest"],
    )
    task_with_size = per_task.merge(
        e29_tasks,
        on=["target_tissue", "mhc_restriction"],
        how="left",
        suffixes=("", "_e29"),
    )
    quartiles = ["Q1 smallest", "Q2", "Q3", "Q4 largest"]
    for model in internal:
        values = (
            task_with_size[task_with_size["model"].eq(model)]
            .groupby("quartile", observed=True)["auroc"]
            .mean()
            .reindex(quartiles)
        )
        ax1.plot(
            quartiles,
            values,
            marker="o",
            linewidth=2,
            color=model_color(model),
            label=DISPLAY[model],
        )
    ax1.set_ylim(0.61, 0.77)
    ax1.set_ylabel("Mean task AUROC")
    ax1.set_title("More task-specific training data consistently helps")
    ax1.legend(frameon=False, fontsize=8)
    clean_axis(ax1, "y")
    panel_label(ax1, "A")

    other = pair_diagnostic[
        pair_diagnostic["dimension"].eq("other_count_group")
        & pair_diagnostic["model"].isin(selected)
    ]
    for model in selected:
        values = (
            other[other["model"].eq(model)]
            .set_index("group")
            .reindex(["1", "2", "3+"])
        )
        ax2.plot(
            ["1", "2", "3+"],
            values["pair_accuracy"],
            marker="o",
            linewidth=2,
            color=model_color(model),
            label=DISPLAY[model],
        )
    ax2.axhline(0.5, color="#9CA3AF", linestyle="--", linewidth=1)
    ax2.set_ylim(0.54, 0.74)
    ax2.set_xlabel("Other-tissue presentation count")
    ax2.set_ylabel("PairAcc")
    ax2.set_title("Cross-tissue promiscuity erodes the separable signal")
    ax2.legend(frameon=False, fontsize=7, ncols=2)
    clean_axis(ax2, "y")
    panel_label(ax2, "B")

    seen = pair_diagnostic[
        pair_diagnostic["dimension"].eq("peptide_seen_group")
        & pair_diagnostic["model"].isin(selected)
    ]
    for model in selected:
        values = (
            seen[seen["model"].eq(model)]
            .set_index("group")
            .reindex(["0 seen", "1 seen", "2 seen"])
        )
        ax3.plot(
            ["0 seen", "1 seen", "2 seen"],
            values["pair_accuracy"],
            marker="o",
            linewidth=2,
            color=model_color(model),
            label=DISPLAY[model],
        )
    ax3.set_ylim(0.61, 0.79)
    ax3.set_xlabel("Peptides in each test pair already seen in premium train")
    ax3.set_ylabel("PairAcc")
    ax3.set_title("Internal models gain most when both peptides were seen")
    clean_axis(ax3, "y")
    panel_label(ax3, "C")

    wide = per_task.pivot(
        index=["target_tissue", "mhc_restriction"],
        columns="model",
        values="auroc",
    )
    size = e29_tasks.set_index(["target_tissue", "mhc_restriction"])[
        "train_rows"
    ]
    point_size = 20 + 90 * (
        np.log1p(size) - np.log1p(size).min()
    ) / (np.log1p(size).max() - np.log1p(size).min())
    ax4.scatter(
        wide["e14a_auxiliary_dual_branch"],
        wide["e29_multikernel_cnn"],
        s=point_size,
        c=np.where(
            wide["e29_multikernel_cnn"]
            >= wide["e14a_auxiliary_dual_branch"],
            COLORS["positive"],
            COLORS["negative"],
        ),
        alpha=0.72,
        edgecolor="white",
        linewidth=0.5,
    )
    ax4.plot([0.4, 0.95], [0.4, 0.95], color="#6B7280", linewidth=1)
    difference = (
        wide["e29_multikernel_cnn"]
        - wide["e14a_auxiliary_dual_branch"]
    )
    for task in list(difference.nlargest(2).index) + list(
        difference.nsmallest(2).index
    ):
        ax4.annotate(
            f"{task[0]} · {task[1]}",
            (
                wide.loc[task, "e14a_auxiliary_dual_branch"],
                wide.loc[task, "e29_multikernel_cnn"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
        )
    ax4.set_xlim(0.42, 0.94)
    ax4.set_ylim(0.42, 0.94)
    ax4.set_xlabel("E14a task AUROC")
    ax4.set_ylabel("E29 task AUROC")
    ax4.set_title("E29's small mean gain hides large task-level reversals")
    clean_axis(ax4)
    panel_label(ax4, "D")

    fig.suptitle(
        "Data and split diagnostics",
        fontsize=16,
        fontweight="bold",
    )
    save_figure(fig, "03_data_split_diagnostics", multipage)


def figure_external(
    summary: pd.DataFrame,
    per_task: pd.DataFrame,
    pair_diagnostic: pd.DataFrame,
    multipage: PdfPages,
) -> None:
    external = summary[
        summary["result_group"].eq("external_predictors")
    ].sort_values("mean_task_auroc")
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.flat

    y = np.arange(len(external))
    ax1.barh(
        y,
        external["mean_task_auroc"] - 0.5,
        left=0.5,
        color=[model_color(model) for model in external["model"]],
        height=0.62,
    )
    ax1.set_yticks(y, [DISPLAY[model] for model in external["model"]])
    ax1.set_xlim(0.60, 0.68)
    ax1.set_xlabel("Mean task AUROC")
    ax1.set_title("Presentation/elution scores outperform affinity-only scores")
    for index, value in enumerate(external["mean_task_auroc"]):
        ax1.text(value + 0.001, index, f"{value:.3f}", va="center", fontsize=8)
    clean_axis(ax1)
    panel_label(ax1, "A")

    main_external = [
        "mhcflurry_2.2.1::presentation_score",
        "netmhcpan_4.1b::el_rank",
        "netmhcpan_4.1b::ba_rank",
    ]
    other = pair_diagnostic[
        pair_diagnostic["dimension"].eq("other_count_group")
        & pair_diagnostic["model"].isin(main_external)
    ]
    for model in main_external:
        values = (
            other[other["model"].eq(model)]
            .set_index("group")
            .reindex(["1", "2", "3+"])
        )
        ax2.plot(
            ["1", "2", "3+"],
            values["pair_accuracy"],
            marker="o",
            linewidth=2.4,
            color=model_color(model),
            linestyle="-" if "mhcflurry" in model else "--",
            label=DISPLAY[model],
        )
    ax2.set_ylim(0.53, 0.71)
    ax2.set_xlabel("Other-tissue presentation count")
    ax2.set_ylabel("PairAcc")
    ax2.set_title("General pMHC signal fails on multi-tissue ambiguity")
    ax2.legend(frameon=False, fontsize=8)
    clean_axis(ax2, "y")
    panel_label(ax2, "B")

    wide = per_task.pivot(
        index=["target_tissue", "mhc_restriction"],
        columns="model",
        values="auroc",
    )
    x = wide["mhcflurry_2.2.1::presentation_score"]
    y_values = wide["e29_multikernel_cnn"]
    ax3.scatter(
        x,
        y_values,
        color=np.where(
            y_values >= x,
            COLORS["e29"],
            COLORS["external_mhcflurry"],
        ),
        alpha=0.76,
        edgecolor="white",
        linewidth=0.5,
        s=46,
    )
    ax3.plot([0.42, 0.95], [0.42, 0.95], color="#6B7280", linewidth=1)
    ax3.set_xlim(0.42, 0.93)
    ax3.set_ylim(0.42, 0.93)
    ax3.set_xlabel("MHCflurry presentation task AUROC")
    ax3.set_ylabel("E29 task AUROC")
    ax3.set_title("MHCflurry still wins 30/75 tasks")
    clean_axis(ax3)
    panel_label(ax3, "C")

    old = pd.read_csv(OLD_EXTERNAL)
    old = old[
        old["species"].eq("human")
        & old["protocol"].eq("standard_fixed_test")
        & old["evaluated_model"].isin(main_external)
    ][["evaluated_model", "mean_task_auroc"]].rename(
        columns={
            "evaluated_model": "model",
            "mean_task_auroc": "original_phase7",
        }
    )
    premium = external[
        external["model"].isin(main_external)
    ][["model", "mean_task_auroc"]].rename(
        columns={"mean_task_auroc": "premium"}
    )
    comparison = old.merge(premium, on="model", validate="one_to_one")
    comparison = comparison.sort_values("original_phase7")
    for index, row in enumerate(comparison.itertuples()):
        ax4.plot(
            [row.premium, row.original_phase7],
            [index, index],
            color="#CBD5E1",
            linewidth=3,
        )
        ax4.scatter(
            row.original_phase7,
            index,
            color="#6B7280",
            s=65,
            label="Original Phase 7" if index == 0 else None,
            zorder=3,
        )
        ax4.scatter(
            row.premium,
            index,
            color=model_color(row.model),
            s=65,
            label="Premium" if index == 0 else None,
            zorder=3,
        )
    ax4.set_yticks(
        np.arange(len(comparison)),
        [DISPLAY[model] for model in comparison["model"]],
    )
    ax4.set_xlim(0.61, 0.70)
    ax4.set_xlabel("Mean task AUROC")
    ax4.set_title("All frozen external controls also drop on premium")
    ax4.legend(frameon=False, fontsize=8)
    clean_axis(ax4)
    panel_label(ax4, "D")

    fig.suptitle(
        "Why frozen external predictors score low",
        fontsize=16,
        fontweight="bold",
    )
    save_figure(fig, "04_external_predictor_analysis", multipage)


def write_data_audit(pair_counts: pd.DataFrame) -> pd.DataFrame:
    specifications = [
        (
            "premium",
            TRAIN,
            TEST,
        ),
        (
            "original_44task",
            ROOT / "data/tissuePMHC/tissuePMHC_train.csv.gz",
            ROOT / "data/tissuePMHC/tissuePMHC_test.csv.gz",
        ),
        (
            "phase7_157task",
            ROOT
            / "data/tissuePMHC_phase7_min200"
            / "tissuePMHC_phase7_min200_train.csv.gz",
            ROOT
            / "data/tissuePMHC_phase7_min200"
            / "tissuePMHC_phase7_min200_test.csv.gz",
        ),
    ]
    rows = []
    for dataset, train_path, test_path in specifications:
        columns = [
            "target_tissue",
            "mhc_restriction",
            "peptide_sequence",
            "molecule_parent_uniprot_id",
        ]
        train = pd.read_csv(train_path, usecols=columns)
        test = pd.read_csv(test_path, usecols=columns)
        train_group_pairs = (
            train.groupby(["target_tissue", "mhc_restriction"]).size() / 2
        )
        record = {
            "dataset": dataset,
            "tasks": int(len(train_group_pairs)),
            "train_pairs": int(len(train) / 2),
            "median_train_pairs_per_task": float(train_group_pairs.median()),
            "q1_train_pairs_per_task": float(train_group_pairs.quantile(0.25)),
            "test_pairs_per_task": float(
                (test.groupby(["target_tissue", "mhc_restriction"]).size() / 2)
                .median()
            ),
        }
        for name, columns_for_overlap in [
            ("peptide", ["peptide_sequence"]),
            ("hla_peptide", ["mhc_restriction", "peptide_sequence"]),
            ("protein", ["molecule_parent_uniprot_id"]),
        ]:
            train_entities = set(
                map(
                    tuple,
                    train[columns_for_overlap].astype(str).to_numpy().tolist(),
                )
            )
            test_entities = set(
                map(
                    tuple,
                    test[columns_for_overlap].astype(str).to_numpy().tolist(),
                )
            )
            record[f"test_{name}_overlap"] = len(
                train_entities & test_entities
            ) / len(test_entities)
        rows.append(record)
    audit = pd.DataFrame(rows)
    audit.to_csv(OUTPUT / "source_dataset_split_audit.csv", index=False)
    pair_counts.to_csv(OUTPUT / "source_premium_pair_diagnostics.csv", index=False)
    return audit


def main() -> None:
    configure_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary, per_task, predictions = load_results()
    stage_gains = bootstrap_stage_gains(per_task)
    stage_values = pd.concat(
        [original_stage_values(), premium_stage_values(summary)],
        ignore_index=True,
    )
    pair_diagnostic, pair_counts, _ = pair_diagnostics(predictions)
    audit = write_data_audit(pair_counts)

    summary.sort_values("mean_task_auroc", ascending=False).to_csv(
        OUTPUT / "source_model_summary_all.csv",
        index=False,
    )
    stage_gains.to_csv(OUTPUT / "source_stage_gain_bootstrap.csv", index=False)
    stage_values.to_csv(OUTPUT / "source_stage_progression.csv", index=False)
    pair_diagnostic.to_csv(
        OUTPUT / "source_pair_accuracy_diagnostics.csv",
        index=False,
    )

    combined_path = OUTPUT / "premium_results_all_figures.pdf"
    with PdfPages(combined_path) as multipage:
        figure_overview(
            summary,
            per_task,
            stage_gains,
            stage_values,
            multipage,
        )
        figure_task_heatmap(per_task, multipage)
        figure_data_diagnostics(per_task, pair_diagnostic, multipage)
        figure_external(summary, per_task, pair_diagnostic, multipage)

    metadata = {
        "figures": [
            "01_performance_overview",
            "02_task_auroc_heatmap",
            "03_data_split_diagnostics",
            "04_external_predictor_analysis",
        ],
        "combined_pdf": str(combined_path),
        "n_models": int(summary["model"].nunique()),
        "n_tasks": int(
            per_task[per_task["model"].eq("e29_multikernel_cnn")].shape[0]
        ),
        "bootstrap_iterations": 20000,
        "primary_models": PRIMARY,
        "data_audit": audit.to_dict(orient="records"),
    }
    (OUTPUT / "visualization_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote figures and source tables to: {OUTPUT}", flush=True)
    print(f"combined PDF: {combined_path}", flush=True)


if __name__ == "__main__":
    main()
