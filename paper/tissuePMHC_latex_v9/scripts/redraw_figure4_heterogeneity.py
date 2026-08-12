"""Redraw the tissue heterogeneity figure at native single-column size.

The Human task values were recovered from the original vector figure and
checked against the tissue counts and means reported in the Supplementary
tables. Mouse values are the task-level values reported in the Supplementary
table. The resulting layout is designed for a journal column rather than
being a reduced double-column graphic.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "figure6_tissue_colored_heterogeneity_redrawn"

NAVY = "#17324D"
HUMAN_BLUE = "#3B6FB6"
MOUSE_CORAL = "#D95F59"
GRID = "#D8E0E8"
ROW = "#F4F7FA"


HUMAN = {
    "Brain": [0.6987, 0.6077, 0.8070, 0.8121],
    "NA/unspecified": [0.7384],
    "Thymus": [0.7407],
    "Lung": [0.6581, 0.8036, 0.8419, 0.6599, 0.6911, 0.8167, 0.7620],
    "Spleen": [0.7765],
    "Lymphoid": [
        0.8525, 0.7215, 0.6779, 0.8244, 0.7548, 0.7688, 0.7992, 0.8993,
        0.9165, 0.6641, 0.6927, 0.7675, 0.7681, 0.8022, 0.8745, 0.8311,
        0.8091, 0.7798, 0.8271, 0.9480, 0.9359, 0.8561, 0.9263,
    ],
    "Ovary": [0.8129],
    "Blood": [
        0.7591, 0.8350, 0.6358, 0.8161, 0.6221, 0.8790, 0.8091, 0.7565,
        0.7306, 0.8427, 0.9247, 0.9122, 0.9372, 0.7279, 0.7215, 0.7101,
        0.8308, 0.8344, 0.9053, 0.8873, 0.9455, 0.8317, 0.9427, 0.8879,
        0.8589, 0.9439,
    ],
    "Uterine cervix": [0.8704, 0.7681, 0.7846, 0.8955, 0.8453],
    "Bone": [0.8627],
    "Lymph node": [0.8463, 0.8996, 0.7778, 0.8964, 0.9484],
    "Kidney": [0.8769],
    "Breast": [0.9640],
}

MOUSE = {
    "Colon": [0.5016],
    "Liver": [0.6754, 0.7999, 0.9113, 0.9180],
    "Skin": [0.7821, 0.8760, 0.9180],
    "Spleen": [0.8461, 0.8693, 0.9161],
}


def draw_panel(ax, data, panel_title, color, seed):
    ordered = sorted(data, key=lambda name: np.mean(data[name]))
    rng = np.random.default_rng(seed)

    for row, tissue in enumerate(ordered):
        if row % 2 == 0:
            ax.axhspan(row - 0.5, row + 0.5, color=ROW, zorder=0)
        values = np.asarray(data[tissue])
        jitter = rng.uniform(-0.16, 0.16, len(values)) if len(values) > 1 else np.zeros(1)
        ax.scatter(
            values,
            row + jitter,
            s=25,
            color=color,
            edgecolor="white",
            linewidth=0.55,
            alpha=0.82,
            zorder=3,
        )
        mean = float(values.mean())
        ax.plot([mean, mean], [row - 0.25, row + 0.25], color=NAVY, lw=1.5, zorder=4)
        ax.scatter(mean, row, marker="D", s=22, color=NAVY, edgecolor="white", linewidth=0.45, zorder=5)

    labels = [f"{name}  ({len(data[name])})" for name in ordered]
    ax.set_yticks(range(len(ordered)), labels)
    ax.set_ylim(len(ordered) - 0.45, -0.55)
    ax.set_xlim(0.48, 0.985)
    ax.set_xticks(np.arange(0.5, 1.01, 0.1))
    ax.axvline(0.5, color="#8796A5", lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax.grid(axis="x", color=GRID, lw=0.65, zorder=0)
    ax.set_title(panel_title, loc="left", fontsize=9.2, fontweight="bold", color=NAVY, pad=5)
    ax.tick_params(axis="x", labelsize=7.2, length=2.5, colors=NAVY)
    ax.tick_params(axis="y", labelsize=7.0, length=0, pad=3, colors=NAVY)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#9AA8B5")
    ax.spines["bottom"].set_linewidth(0.7)


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(3.45, 6.55))
    grid = fig.add_gridspec(2, 1, height_ratios=[3.15, 1.05], hspace=0.28)
    ax_h = fig.add_subplot(grid[0])
    ax_m = fig.add_subplot(grid[1], sharex=ax_h)

    draw_panel(ax_h, HUMAN, "A  Human · 77 tissue–HLA tasks", HUMAN_BLUE, 20260704)
    draw_panel(ax_m, MOUSE, "B  Mouse · 11 tissue–H2 tasks", MOUSE_CORAL, 20260705)
    ax_h.tick_params(axis="x", labelbottom=False)
    ax_m.set_xlabel("Task AUROC (three-seed mean)", fontsize=7.8, color=NAVY, labelpad=4)

    fig.subplots_adjust(left=0.35, right=0.985, top=0.985, bottom=0.075)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(OUT.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


if __name__ == "__main__":
    main()