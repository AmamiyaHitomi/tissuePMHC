#!/usr/bin/env python3
"""Build the publication version of Figure 5 from frozen task metrics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "extra_occurrence_equal_dataset" / "results"
TUNED_RESULTS = (
    ROOT
    / "extra_occurrence_equal_dataset"
    / "adjusting"
    / "results"
    / "e29_final_test"
)
OUTPUT = ROOT / "paper" / "tissuePMHC_latex_v9" / "figures"
TASK_KEYS = ["target_tissue", "mhc_restriction"]

COLORS = {
    "shared": "#7A8793",
    "guided": "#4C78A8",
    "two_view": "#6C5B9B",
    "tissuepmhc": "#008A91",
    "negative": "#C45A52",
    "ink": "#263238",
    "grid": "#D8DEE3",
}


def load_task_aurocs(
    relative_path: str,
    model: str,
    average_runs: bool,
    base: Path = RESULTS,
) -> pd.Series:
    table = pd.read_csv(base / relative_path, keep_default_na=False)
    table = table.loc[table["model"].eq(model), [*TASK_KEYS, "auroc"]]
    # The tuned contract writes the unspecified tissue as the literal "NA",
    # whereas the frozen comparator uses an empty string.
    table["target_tissue"] = table["target_tissue"].replace("NA", "")
    if average_runs:
        table = table.groupby(TASK_KEYS, as_index=False, dropna=False)["auroc"].mean()
    if len(table) != 77:
        raise ValueError(
            f"Expected 77 task AUROCs for {model}, found {len(table)}"
        )
    return table.set_index(TASK_KEYS)["auroc"].sort_index()


def half_violin(
    ax: plt.Axes,
    values: np.ndarray,
    position: float,
    color: str,
    width: float = 0.56,
) -> None:
    violin = ax.violinplot(
        values,
        positions=[position],
        vert=False,
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method=0.25,
    )
    body = violin["bodies"][0]
    body.set_facecolor(color)
    body.set_edgecolor(color)
    body.set_linewidth(0.8)
    body.set_alpha(0.22)
    for path in body.get_paths():
        vertices = path.vertices
        vertices[:, 1] = np.maximum(vertices[:, 1], position)


def raincloud_row(
    ax: plt.Axes,
    values: np.ndarray,
    position: float,
    color: str,
    rng: np.random.Generator,
    point_colors: np.ndarray | None = None,
) -> None:
    half_violin(ax, values, position, color)
    jitter = rng.uniform(0.10, 0.27, size=len(values))
    colors = point_colors if point_colors is not None else color
    ax.scatter(
        values,
        position - jitter,
        s=10,
        c=colors,
        alpha=0.48,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )
    ax.boxplot(
        values,
        positions=[position],
        vert=False,
        widths=0.105,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
        boxprops={"facecolor": "white", "edgecolor": color, "linewidth": 1.15},
        medianprops={"color": COLORS["ink"], "linewidth": 1.4},
        whiskerprops={"color": color, "linewidth": 1.0},
        capprops={"color": color, "linewidth": 1.0},
    )
    ax.scatter(
        [values.mean()],
        [position],
        marker="D",
        s=30,
        facecolor=color,
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
    )


def main() -> None:
    series = {
        "Shared encoder": load_task_aurocs(
            "e2_shared_heads/per_task_metrics.csv",
            "e2_shared_heads",
            True,
        ),
        "Guided dual branch": load_task_aurocs(
            "e14a_auxiliary_dual_branch/per_task_metrics.csv",
            "e14a_auxiliary_dual_branch",
            True,
        ),
        "Two-view MLP": load_task_aurocs(
            "v7_full_rerun/mlp_dual_seed_ensemble/per_task_metrics.csv",
            "e17_3seed_rank_average",
            False,
        ),
        "TissuePMHC": load_task_aurocs(
            "all_seed_per_task_metrics.csv",
            "e29_tuned",
            True,
            TUNED_RESULTS,
        ),
    }
    palette = [
        COLORS["shared"],
        COLORS["guided"],
        COLORS["two_view"],
        COLORS["tissuepmhc"],
    ]

    plt.rcParams.update(
        {
            # The manuscript uses newtxtext/newtxmath. Times New Roman is the
            # closest locally available system font and keeps the external
            # figure visually consistent with the LaTeX-rendered PGF figures.
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, (ax_distribution, ax_delta) = plt.subplots(
        1,
        2,
        figsize=(7.15, 3.25),
        gridspec_kw={"width_ratios": [1.55, 1.0]},
    )
    rng = np.random.default_rng(20260728)

    positions = np.arange(4, 0, -1, dtype=float)
    for position, values, color in zip(positions, series.values(), palette):
        raincloud_row(
            ax_distribution,
            values.to_numpy(float),
            position,
            color,
            rng,
        )

    ax_distribution.set_yticks(positions, list(series))
    ax_distribution.set_xlim(0.58, 1.005)
    ax_distribution.set_ylim(0.55, 4.45)
    ax_distribution.set_xticks(np.arange(0.6, 1.01, 0.1))
    ax_distribution.set_xlabel("Per-combination AUROC")
    ax_distribution.set_title("A  Performance distributions", loc="left")
    ax_distribution.grid(axis="x", color=COLORS["grid"], linewidth=0.65)
    ax_distribution.tick_params(axis="y", length=0)
    for position, values, color in zip(positions, series.values(), palette):
        ax_distribution.text(
            0.998,
            position + 0.18,
            f"mean {values.mean():.4f}",
            ha="right",
            va="center",
            fontsize=7.3,
            color=color,
        )

    delta = (series["TissuePMHC"] - series["Two-view MLP"]).to_numpy(float)
    delta_colors = np.where(
        delta >= 0, COLORS["tissuepmhc"], COLORS["negative"]
    )
    raincloud_row(
        ax_delta,
        delta,
        1.0,
        COLORS["tissuepmhc"],
        rng,
        point_colors=delta_colors,
    )
    ax_delta.axvline(
        0,
        color=COLORS["ink"],
        linewidth=0.9,
        linestyle=(0, (3, 2)),
        zorder=1,
    )

    bootstrap_rng = np.random.default_rng(20260728)
    bootstrap_means = bootstrap_rng.choice(
        delta, size=(10_000, len(delta)), replace=True
    ).mean(axis=1)
    ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
    mean_delta = float(delta.mean())
    ax_delta.plot(
        [ci_low, ci_high],
        [1.31, 1.31],
        color=COLORS["ink"],
        linewidth=2.0,
        solid_capstyle="round",
        zorder=4,
    )
    ax_delta.scatter(
        [mean_delta],
        [1.31],
        marker="D",
        s=34,
        facecolor=COLORS["tissuepmhc"],
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
    )
    ax_delta.text(
        0.086,
        1.54,
        f"mean {mean_delta:+.4f}\n95% CI [{ci_low:+.4f}, {ci_high:+.4f}]",
        ha="right",
        va="top",
        fontsize=7.5,
        color=COLORS["ink"],
    )
    ax_delta.text(
        -0.075,
        0.59,
        f"{np.sum(delta < 0)} declined",
        ha="left",
        va="bottom",
        fontsize=7.5,
        color=COLORS["negative"],
    )
    ax_delta.text(
        0.086,
        0.59,
        f"{np.sum(delta > 0)} improved",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=COLORS["tissuepmhc"],
    )
    ax_delta.set_xlim(-0.08, 0.09)
    ax_delta.set_ylim(0.55, 1.62)
    ax_delta.set_yticks([])
    ax_delta.set_xlabel("TissuePMHC - two-view MLP AUROC")
    ax_delta.set_title("B  Paired combination-level improvement", loc="left")
    ax_delta.grid(axis="x", color=COLORS["grid"], linewidth=0.65)
    ax_delta.spines["left"].set_visible(False)

    fig.subplots_adjust(
        left=0.16, right=0.985, top=0.89, bottom=0.17, wspace=0.28
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(
            OUTPUT / f"figure5_human_benchmark_raincloud.{suffix}",
            dpi=450,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
