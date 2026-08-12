"""Redraw main-text Figures 2 and 3 at native single-column size."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parents[1]
FIGURES = PAPER / "figures"

HUMAN_METRICS = (
    ROOT
    / "extra_occurrence_equal_dataset"
    / "adjusting"
    / "results"
    / "e29_final_test"
    / "all_seed_per_task_metrics.csv"
)
MOUSE_METRICS = (
    ROOT
    / "extra_mouse_occurrence_equal_dataset"
    / "adjusting"
    / "results"
    / "final_per_task_metrics.csv"
)
ATTRIBUTIONS = (
    ROOT
    / "extra_occurrence_equal_dataset"
    / "results"
    / "e29_tuned_shap_expected_ig"
    / "pair_position_shap_summary.csv.gz"
)

NAVY = "#17324D"
BLUE = "#3E78B8"
CORAL = "#D96B61"
TEAL = "#178F8F"
GOLD = "#D39B2A"
GRID = "#D9E2EA"
ROW = "#F3F6F9"


plt.rcParams.update(
    {
        # The manuscript currently compiles in Computer Modern Roman.  Let
        # LaTeX typeset every figure label so glyph shapes and weights match.
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage[T1]{fontenc}",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 7.5,
        "axes.labelsize": 8.0,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "axes.edgecolor": "#8291A0",
        "axes.linewidth": 0.65,
        "text.color": NAVY,
        "axes.labelcolor": NAVY,
        "xtick.color": NAVY,
        "ytick.color": NAVY,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(FIGURES / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def task_means(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    frame["target_tissue"] = frame["target_tissue"].replace(
        {"": "NA/unspecified", "NA": "NA/unspecified"}
    )
    return (
        frame.groupby(["target_tissue", "mhc_restriction"], as_index=False)
        .agg(auroc=("auroc", "mean"))
    )


def compact_raincloud_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    title: str,
    color: str,
    seed: int,
) -> None:
    summary = frame.groupby("target_tissue")["auroc"].mean().sort_values()
    order = summary.index.tolist()
    rng = np.random.default_rng(seed)

    for column, tissue in enumerate(order):
        values = frame.loc[frame["target_tissue"] == tissue, "auroc"].to_numpy()
        # Half-density ``cloud`` above the row.  With fewer than three tasks a
        # KDE would imply unsupported distributional detail, so only rain is
        # shown for those tissues.
        if len(values) >= 3 and np.ptp(values) > 1e-8:
            padding = max(0.012, 0.18 * np.ptp(values))
            density_y = np.linspace(
                max(0.48, values.min() - padding),
                min(0.985, values.max() + padding),
                160,
            )
            density = gaussian_kde(values, bw_method=0.48)(density_y)
            density = density / density.max() * 0.28
            ax.fill_betweenx(
                density_y,
                column,
                column + density,
                color=color,
                alpha=0.18,
                linewidth=0,
                zorder=1,
            )
            ax.plot(
                column + density,
                density_y,
                color=color,
                alpha=0.48,
                lw=0.7,
                zorder=2,
            )

        jitter = rng.uniform(-0.20, -0.075, len(values)) if len(values) > 1 else np.zeros(1)
        if len(values) > 1:
            ax.plot(
                [column - 0.035, column - 0.035],
                [values.min(), values.max()],
                color=color,
                lw=0.85,
                alpha=0.38,
                solid_capstyle="round",
                zorder=2,
            )
        ax.scatter(
            column + jitter,
            values,
            s=12,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            alpha=0.72,
            zorder=3,
        )
        mean = float(values.mean())
        ax.scatter(
            column - 0.035,
            mean,
            marker="D",
            s=18,
            color=NAVY,
            edgecolor="white",
            linewidth=0.55,
            zorder=5,
        )

    labels = [tissue.replace('_', ' ').title() for tissue in order]
    labels = [label.replace("Na/Unspecified", "NA/unspecified") for label in labels]
    ax.set_xticks(range(len(order)), labels, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.55, len(order) - 0.45)
    ax.set_ylim(0.47, 0.985)
    ax.set_yticks(np.arange(0.5, 1.0, 0.1))
    ax.axhline(0.5, color="#8796A5", lw=0.85, ls=(0, (3, 2)), zorder=1)
    ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.75, zorder=0)
    ax.set_title(title, loc="left", pad=4)
    ax.tick_params(axis="x", length=0, pad=2, labelsize=5.3)
    ax.tick_params(axis="y", length=2.0, labelsize=6.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def figure2() -> None:
    human = task_means(HUMAN_METRICS)
    mouse = task_means(MOUSE_METRICS)
    fig = plt.figure(figsize=(3.35, 3.02))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.55, 1], hspace=1.18)
    human_ax = fig.add_subplot(grid[0])
    # Tissue inventories differ by species, so the categorical x axes must
    # remain independent even though their AUROC y scales are identical.
    mouse_ax = fig.add_subplot(grid[1])
    compact_raincloud_panel(
        human_ax, human, r"\textbf{A}\quad Human: 77 tissue--HLA tasks", BLUE, 20260704
    )
    compact_raincloud_panel(
        mouse_ax, mouse, r"\textbf{B}\quad Mouse: 11 tissue--H-2 tasks", CORAL, 20260705
    )
    human_ax.set_ylabel("Task AUROC", labelpad=2)
    mouse_ax.set_ylabel("Task AUROC", labelpad=2)
    # A compact direct legend avoids an extra legend box in the narrow column.
    human_ax.fill_between([], [], [], color=BLUE, alpha=0.18, label=r"Density ($n\geq3$)")
    human_ax.scatter([], [], s=12, color=BLUE, alpha=0.72, label="Task")
    human_ax.scatter([], [], s=18, marker="D", color=NAVY, label="Tissue mean")
    human_ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.50, -0.50),
        frameon=False,
        fontsize=5.5,
        ncol=3,
        handletextpad=0.35,
        columnspacing=0.65,
        borderaxespad=0.25,
    )
    fig.subplots_adjust(left=0.13, right=0.99, top=0.97, bottom=0.16)
    save(fig, "figure6_tissue_colored_heterogeneity")


def figure3() -> None:
    frame = pd.read_csv(ATTRIBUTIONS)
    frame = frame[
        (frame["scope_type"] == "overall")
        & (frame["scope_value"] == "ALL")
        & (frame["seed"].astype(str).isin(["three_seed_mean", "seed_mean"]))
    ]
    fig, ax = plt.subplots(figsize=(3.42, 2.72))
    ax.axvspan(1.72, 2.28, color="#F2E7C8", alpha=0.65, zorder=0)
    ax.axvspan(2.72, 3.28, color="#F2E7C8", alpha=0.36, zorder=0)
    ax.axvspan(8.72, 9.28, color="#F2E7C8", alpha=0.65, zorder=0)
    values = {
        branch: frame[frame["branch"] == branch]
        .sort_values("position")
        .set_index("position")["mean_positive_minus_negative_shap"]
        for branch in ("global_aux", "hla_plain", "descriptive_branch_consensus")
    }
    positions = np.arange(1, 10)
    global_values = values["global_aux"].reindex(positions).to_numpy()
    hla_values = values["hla_plain"].reindex(positions).to_numpy()
    consensus_values = values["descriptive_branch_consensus"].reindex(positions).to_numpy()

    # A compact range-and-consensus display: the pale stem gives magnitude,
    # the vertical segment gives between-pathway spread, and the diamond gives
    # the descriptive branch consensus.
    ax.vlines(positions, 0, consensus_values, color="#B8C7D5", lw=1.05, zorder=1)
    ax.vlines(
        positions,
        np.minimum(global_values, hla_values),
        np.maximum(global_values, hla_values),
        color="#667B8F",
        lw=1.7,
        zorder=2,
    )
    ax.scatter(
        positions - 0.075,
        global_values,
        s=22,
        color=BLUE,
        edgecolor="white",
        linewidth=0.45,
        label="Global pathway",
        zorder=4,
    )
    ax.scatter(
        positions + 0.075,
        hla_values,
        s=22,
        marker="s",
        color=CORAL,
        edgecolor="white",
        linewidth=0.45,
        label="HLA-specific pathway",
        zorder=4,
    )
    ax.plot(
        positions,
        consensus_values,
        color=NAVY,
        lw=0.9,
        alpha=0.72,
        zorder=3,
    )
    ax.scatter(
        positions,
        consensus_values,
        s=31,
        marker="D",
        color=NAVY,
        edgecolor="white",
        linewidth=0.55,
        label="Branch consensus",
        zorder=5,
    )

    consensus = values["descriptive_branch_consensus"]
    for position in (2, 3, 9):
        value = float(consensus.loc[position])
        ax.annotate(
            f"P{position}",
            (position, value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=6.8,
            fontweight="bold",
            color=NAVY,
        )

    ax.set_xlim(0.75, 9.25)
    ax.set_ylim(0.0, 0.81)
    ax.set_xticks(range(1, 10))
    ax.set_yticks(np.arange(0.0, 0.81, 0.2))
    ax.set_xlabel("Peptide position")
    ax.set_ylabel(r"Attribution difference")
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=6.6,
        handlelength=2.1,
        labelspacing=0.35,
        borderaxespad=0.25,
    )
    fig.subplots_adjust(left=0.16, right=0.985, top=0.97, bottom=0.19)
    save(fig, "figure8_shap_paired_position")


if __name__ == "__main__":
    figure2()
    figure3()
