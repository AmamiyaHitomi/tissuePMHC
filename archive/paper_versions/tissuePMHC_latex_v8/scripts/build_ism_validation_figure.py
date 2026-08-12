#!/usr/bin/env python3
"""Build the paper-ready ISM validation figure from formal result tables."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
from scipy.stats import spearmanr

def manuscript_font_family() -> str:
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        bundle_data = Path(local_appdata) / "TectonicProject" / "Tectonic" / "bundles" / "data"
        for regular in bundle_data.glob("*/TeXGyreTermesX-Regular.otf"):
            for font_path in regular.parent.glob("TeXGyreTermesX-*.otf"):
                font_manager.fontManager.addfont(font_path)
            return font_manager.FontProperties(fname=regular).get_name()
    return "Times New Roman"


plt.rcParams.update(
    {
        "font.family": manuscript_font_family(),
        "font.size": 13.0,
        "axes.titlesize": 15.0,
        "axes.labelsize": 14.0,
        "xtick.labelsize": 12.5,
        "ytick.labelsize": 12.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


PAPER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PAPER_ROOT.parents[1]
RESULT_ROOT = PROJECT_ROOT / "extra_occurrence_equal_dataset/results/e29_tuned_ism"
FIGURE_ROOT = PAPER_ROOT / "figures"


def main() -> None:
    position = pd.read_csv(RESULT_ROOT / "sample_position_sensitivity.csv.gz")
    concordance = pd.read_csv(RESULT_ROOT / "shap_ism_sample_position.csv.gz")
    consensus = concordance.groupby(
        ["branch", "sample_id", "position"], as_index=False
    ).agg(
        observed_residue_shap=("observed_residue_shap", "mean"),
        mean_ism_mutation_loss=("mean_ism_mutation_loss", "mean"),
    )

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.8), constrained_layout=True)
    ax = axes[0]
    for label, name, color in ((1, "Recorded positive", "#c62828"), (0, "Matched pseudo-negative", "#1565c0")):
        grouped = position[position["label"] == label].groupby("position")["mean_abs_delta_rank_fusion"]
        mean, sem = grouped.mean(), grouped.sem()
        ax.plot(mean.index, mean.values, marker="o", linewidth=2, color=color, label=name)
        ax.fill_between(mean.index, mean - 1.96 * sem, mean + 1.96 * sem, color=color, alpha=0.16)
    ax.axvline(2, color="0.5", linestyle="--", linewidth=0.9)
    ax.axvline(9, color="0.5", linestyle="--", linewidth=0.9)
    ax.set_xticks(range(1, 10))
    ax.set_xlabel("Peptide position")
    ax.set_ylabel(r"Mean $|\Delta|$ final rank-fusion score")
    ax.set_title("(a) Single-residue sensitivity")
    ax.legend(frameon=False, fontsize=12.0, loc="upper right")

    for ax, branch, title in zip(
        axes[1:],
        ("global_aux", "hla_plain"),
        ("(b) Global auxiliary branch", "(c) HLA-specific branch"),
    ):
        chosen = consensus[consensus["branch"] == branch]
        rho = spearmanr(
            chosen["observed_residue_shap"], chosen["mean_ism_mutation_loss"]
        ).statistic
        image = ax.hexbin(
            chosen["observed_residue_shap"], chosen["mean_ism_mutation_loss"],
            gridsize=38, mincnt=1, bins="log", cmap="viridis",
        )
        ax.set_xlabel("Expected-IG observed-residue attribution")
        ax.set_ylabel("Mean ISM mutation loss (logit)")
        ax.set_title(f"{title}\nSpearman $\\rho={rho:.3f}$")
        colorbar = fig.colorbar(image, ax=ax, label="log count", fraction=0.047, pad=0.025)
        colorbar.ax.tick_params(labelsize=12.0)
        colorbar.set_label("log count", fontsize=13.0)

    for suffix in ("pdf", "png"):
        kwargs = {"dpi": 300} if suffix == "png" else {}
        fig.savefig(
            FIGURE_ROOT / f"figure10_ism_validation.{suffix}",
            bbox_inches="tight", **kwargs,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
