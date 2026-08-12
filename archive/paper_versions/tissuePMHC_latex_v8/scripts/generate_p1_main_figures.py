"""Generate the revised P1 main-text figures from occurrence-equal results.

Outputs are written to ``paper/tissuePMHC_latex_v7/figures``.  The script
also writes a compact JSON audit containing the plotted occurrence totals and
the task-wise tuning statistics used in Figure 7.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from matplotlib.ticker import FuncFormatter, StrMethodFormatter
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, wilcoxon


ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "paper" / "tissuePMHC_latex_v8"
FIGURES = PAPER / "figures"

HUMAN_DATA = ROOT / "data" / "humanPMHC_occurence_equal_dataset"
MOUSE_DATA = ROOT / "data" / "mousePMHC_occurence_equal_dataset"
HUMAN_RESULTS = ROOT / "extra_occurrence_equal_dataset"
MOUSE_RESULTS = ROOT / "extra_mouse_occurrence_equal_dataset"

NAVY = "#17324D"
BLUE = "#3B6FB6"
TEAL = "#138A8A"
CORAL = "#D95F59"
GOLD = "#D8A126"
LIGHT = "#EEF3F8"
GRID = "#D5DCE5"


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.edgecolor": NAVY,
        "axes.linewidth": 0.8,
        "xtick.color": NAVY,
        "ytick.color": NAVY,
        "text.color": NAVY,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def _save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(
        FIGURES / f"{stem}.png",
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)


def _clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _manuscript_font_family() -> str:
    """Register the newtxtext font used by the manuscript when available."""
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        bundle_data = (
            Path(local_appdata)
            / "TectonicProject"
            / "Tectonic"
            / "bundles"
            / "data"
        )
        for regular in bundle_data.glob("*/TeXGyreTermesX-Regular.otf"):
            for font_path in regular.parent.glob("TeXGyreTermesX-*.otf"):
                font_manager.fontManager.addfont(font_path)
            return font_manager.FontProperties(fname=regular).get_name()
    return "Times New Roman"


def figure1_lung_mechanism() -> None:
    # Keep this local to Figure 1 so the quantitative figures are unchanged.
    figure_font = _manuscript_font_family()
    base = mpimg.imread(FIGURES / "figure1_lung_mechanism_base_v2.png")
    fig = plt.figure(figsize=(12.0, 6.8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(base, extent=[0, 1, 0, 1], aspect="auto")
    ax.set_axis_off()

    # Opaque layout zones keep generated biological artwork and exact text
    # physically separated.  This is intentionally not an overlay layout.
    ax.add_patch(Rectangle((0, 0.885), 1, 0.115, facecolor="white", edgecolor="none", zorder=3))
    ax.add_line(plt.Line2D([0.035, 0.965], [0.887, 0.887], color="#D7E1EA", lw=1.0, zorder=4))
    ax.text(
        0.035,
        0.956,
        "How a lung cell processes and displays an MHC-I peptide",
        fontsize=16.5,
        fontfamily=figure_font,
        fontweight="bold",
        va="top",
        color=NAVY,
        zorder=5,
    )
    ax.text(
        0.037,
        0.910,
        "Representative Human context  •  six biological layers  •  canonical genes shown for orientation",
        fontsize=9.0,
        fontfamily=figure_font,
        va="top",
        color="#526579",
        zorder=5,
    )
    ax.text(
        0.963,
        0.934,
        "LUNG  |  7 TISSUE–HLA TASKS",
        ha="right",
        va="center",
        fontsize=7.8,
        fontfamily=figure_font,
        fontweight="bold",
        color=BLUE,
        bbox=dict(boxstyle="round,pad=0.42", fc="#EDF4FB", ec="#AFC5DC", lw=0.8),
        zorder=5,
    )

    panel = FancyBboxPatch(
        (0.025, 0.085),
        0.325,
        0.765,
        boxstyle="round,pad=0.010,rounding_size=0.012",
        facecolor="#F7FAFC",
        edgecolor="#C7D4E0",
        linewidth=1.0,
        zorder=3,
    )
    ax.add_patch(panel)
    ax.text(
        0.047,
        0.823,
        "BIOLOGICAL LAYERS",
        fontsize=8.0,
        fontfamily=figure_font,
        fontweight="bold",
        color="#6A7D90",
        zorder=5,
    )

    callouts = [
        ("1", "Source abundance & lung cell state", "EPCAM  SFTPA1  SFTPB  SCGB1A1", CORAL),
        ("2", "Proteasomal cleavage", "PSMB8  PSMB9  PSMB10", "#6F6DB2"),
        ("3", "Transport into the ER", "TAP1  TAP2", TEAL),
        ("4", "ER peptide trimming", "ERAP1  ERAP2", "#7B5AA6"),
        ("5", "Peptide-loading complex", "TAPBP  CALR  PDIA3", GOLD),
        ("6", "Surface MHC-I display", "HLA-A/B/C  B2M", BLUE),
    ]
    y_positions = np.linspace(0.755, 0.145, len(callouts))

    for (num, layer, genes, accent), y in zip(callouts, y_positions):
        card = FancyBboxPatch(
            (0.043, y - 0.047),
            0.286,
            0.091,
            boxstyle="round,pad=0.006,rounding_size=0.008",
            facecolor="white",
            edgecolor="#D2DCE6",
            linewidth=0.75,
            zorder=4,
        )
        ax.add_patch(card)
        ax.add_patch(Rectangle((0.043, y - 0.047), 0.006, 0.091, facecolor=accent, edgecolor="none", zorder=5))
        ax.text(
            0.071,
            y,
            num,
            ha="center",
            va="center",
            fontsize=9.2,
            fontfamily=figure_font,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="circle,pad=0.31", fc=accent, ec="white", lw=0.9),
            zorder=6,
        )
        ax.text(
            0.098,
            y + 0.014,
            layer,
            fontsize=8.7,
            fontfamily=figure_font,
            fontweight="bold",
            va="center",
            color=NAVY,
            zorder=6,
        )
        ax.text(
            0.099,
            y - 0.020,
            genes,
            fontsize=7.15,
            fontfamily=figure_font,
            va="center",
            color="#53697E",
            zorder=6,
        )

    # Put molecular names beside the structures they identify. Short leader
    # lines anchor every label while compact white boxes preserve legibility.
    def molecule_label(text, xy, xytext, align="center"):
        ax.annotate(
            text,
            xy=xy,
            xytext=xytext,
            ha=align,
            va="center",
            fontsize=7.3,
            fontfamily=figure_font,
            fontweight="semibold",
            color="#111111",
            linespacing=1.12,
            bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="#B8C6D4", lw=0.65, alpha=0.94),
            arrowprops=dict(arrowstyle="-", color="#566A7E", lw=0.8, shrinkA=2, shrinkB=3),
            zorder=8,
        )

    molecule_label("EPCAM · SFTPA1 · SFTPB · SCGB1A1", (0.558, 0.730), (0.545, 0.830))
    molecule_label("PSMB8 · PSMB9 · PSMB10", (0.704, 0.721), (0.704, 0.817))
    molecule_label("Peptide fragments", (0.801, 0.670), (0.830, 0.747))
    molecule_label("TAP1 · TAP2", (0.652, 0.545), (0.603, 0.589), align="right")
    molecule_label("ERAP1 · ERAP2", (0.492, 0.575), (0.438, 0.515), align="left")
    molecule_label("TAPBP · CALR · PDIA3", (0.617, 0.640), (0.604, 0.484))
    molecule_label("HLA-A/B/C · B2M", (0.778, 0.607), (0.790, 0.519))
    molecule_label("Surface MHC-I", (0.892, 0.488), (0.845, 0.574), align="right")

    ax.text(
        0.675,
        0.047,
        "SOURCE  →  CLEAVAGE  →  ER TRANSPORT  →  TRIMMING  →  LOADING  →  SURFACE DISPLAY",
        ha="center",
        fontsize=8.4,
        fontfamily=figure_font,
        fontweight="bold",
        color=NAVY,
        bbox=dict(boxstyle="round,pad=0.46", fc="white", ec="#B7C6D6", lw=0.8, alpha=0.98),
        zorder=6,
    )
    _save(fig, "figure1_lung_presentation_pathway")


def figure1_add_structure_labels() -> None:
    """Add direct, non-overlapping molecular labels to the final Figure 1 art."""
    figure_font = _manuscript_font_family()
    base = mpimg.imread(FIGURES / "figure1_lung_presentation_pathway_v2.png")
    height, width = base.shape[:2]
    fig = plt.figure(figsize=(12.0, 12.0 * height / width))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(base, extent=[0, 1, 0, 1], aspect="auto")
    ax.set_axis_off()

    def label(text, xy, xytext, align="center"):
        ax.annotate(
            text,
            xy=xy,
            xytext=xytext,
            ha=align,
            va="center",
            fontsize=9.2,
            fontfamily=figure_font,
            fontweight="semibold",
            color="#263D70",
            path_effects=[path_effects.withStroke(linewidth=2.4, foreground="white", alpha=0.95)],
            arrowprops=dict(arrowstyle="-", color="#6677A0", lw=1.0, shrinkA=2, shrinkB=3),
            zorder=8,
        )

    label("EPCAM · SFTPA1 · SFTPB · SCGB1A1", (0.505, 0.765), (0.505, 0.852))
    label("PSMB8 · PSMB9 · PSMB10", (0.681, 0.749), (0.681, 0.837))
    label("Peptide fragments", (0.796, 0.724), (0.825, 0.792))
    label("TAP1 · TAP2", (0.610, 0.617), (0.566, 0.670), align="right")
    label("ERAP1 · ERAP2", (0.405, 0.493), (0.350, 0.390), align="left")
    label("TAPBP · CALR · PDIA3", (0.548, 0.418), (0.670, 0.535))
    label("HLA-A/B/C · B2M", (0.758, 0.405), (0.758, 0.491))
    label("Surface MHC-I", (0.897, 0.563), (0.844, 0.640), align="right")

    _save(fig, "figure1_lung_presentation_pathway_v3")


def _read_full_dataset(data_dir: Path, prefix: str) -> pd.DataFrame:
    parts = []
    for split in ("train", "test"):
        path = data_dir / f"{prefix}_{split}.csv.gz"
        part = pd.read_csv(
            path,
            usecols=["label", "target_tissue", "presentation_tissue_count"],
            keep_default_na=False,
        )
        parts.append(part)
    data = pd.concat(parts, ignore_index=True)
    data["target_tissue"] = data["target_tissue"].replace({"": "NA/unspecified", "NA": "NA/unspecified"})
    data["target_tissue"] = data["target_tissue"].str.replace("_", " ").str.title()
    data["target_tissue"] = data["target_tissue"].replace({"Na/Unspecified": "NA/unspecified"})
    data["label"] = pd.to_numeric(data["label"])
    data["presentation_tissue_count"] = pd.to_numeric(data["presentation_tissue_count"])
    return data


def _occurrence_totals(data: pd.DataFrame, species: str) -> pd.DataFrame:
    out = (
        data.groupby(["target_tissue", "label"], as_index=False)["presentation_tissue_count"]
        .sum()
        .pivot(index="target_tissue", columns="label", values="presentation_tissue_count")
        .reset_index()
        .rename(columns={0: "label0", 1: "label1"})
    )
    out["difference"] = out["label1"] - out["label0"]
    out["species"] = species
    return out.sort_values("label0")


def figure2_occurrence_balance() -> list[dict[str, object]]:
    # Match the TeX Gyre Termes/NewTX face used by the manuscript body.
    figure_font = _manuscript_font_family()
    with plt.rc_context({"font.family": figure_font, "font.size": 10.5}):
        return _figure2_occurrence_balance_impl()


def _figure2_occurrence_balance_impl() -> list[dict[str, object]]:
    human = _occurrence_totals(_read_full_dataset(HUMAN_DATA, "humanPMHC"), "Human")
    mouse = _occurrence_totals(_read_full_dataset(MOUSE_DATA, "mousePMHC"), "Mouse")

    # A mirrored bar chart makes exact balance visible as bilateral symmetry.
    # The blue/orange pair is colour-blind safe and remains distinct in print.
    label0_color = "#3B6FB6"
    label1_color = "#D55E00"
    text_color = "#20242A"
    muted = "#5F6670"

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.25, 6.7),
        sharex=False,
        gridspec_kw={"height_ratios": [13, 4]},
        facecolor="white",
    )
    fig.subplots_adjust(left=0.17, right=0.965, top=0.85, bottom=0.13, hspace=0.48)

    for ax, frame, panel, title in zip(
        axes,
        [human, mouse],
        ["A", "B"],
        ["Human (13 tissues)", "Mouse (4 tissues)"],
    ):
        frame = frame.sort_values("label0").reset_index(drop=True)
        y = np.arange(len(frame))
        values0 = frame["label0"].to_numpy()
        values1 = frame["label1"].to_numpy()
        limit = float(max(values0.max(), values1.max())) * 1.22
        ax.set_facecolor("white")
        ax.barh(
            y,
            -values0,
            height=0.62,
            color=label0_color,
            edgecolor="white",
            linewidth=0.55,
            zorder=2,
        )
        ax.barh(
            y,
            values1,
            height=0.62,
            color=label1_color,
            edgecolor="white",
            linewidth=0.55,
            zorder=2,
        )
        offset = limit * 0.018
        for yi, value0, value1 in zip(y, values0, values1):
            ax.text(
                -value0 - offset,
                yi,
                f"{int(value0):,}",
                va="center",
                ha="right",
                fontsize=9.0,
                color=muted,
            )
            ax.annotate(
                f"{int(value1):,}",
                xy=(value1 + offset, yi),
                va="center",
                ha="left",
                fontsize=9.0,
                color=muted,
            )
        ax.set_yticks(y, frame["target_tissue"])
        ax.set_xlim(-limit, limit)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{abs(value):,.0f}"))
        ax.grid(axis="x", color="#DDE1E6", linewidth=0.5, zorder=0)
        ax.axvline(0, color="#33383F", linewidth=0.7, zorder=3)
        ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", pad=10, color=text_color)
        ax.text(
            -0.115,
            1.035,
            panel,
            transform=ax.transAxes,
            fontsize=13.0,
            fontweight="bold",
            ha="left",
            va="bottom",
            color=text_color,
        )
        ax.tick_params(axis="y", length=0, pad=7, labelsize=10.3, colors=text_color)
        ax.tick_params(axis="x", which="major", length=3.0, width=0.65, labelsize=10.0, colors=text_color)
        ax.spines["bottom"].set_color("#555555")
        ax.spines["bottom"].set_linewidth(0.7)
        _clean_axes(ax)

    fig.supxlabel(
        "Summed tissue occurrences across benchmark rows",
        fontsize=11.0,
        y=0.04,
        color=text_color,
    )

    legend_handles = [
        Patch(facecolor=label0_color, edgecolor="white", linewidth=0.5,
              label="Pseudo-negative (label 0)"),
        Patch(facecolor=label1_color, edgecolor="white", linewidth=0.5,
              label="Recorded positive (label 1)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.56, 0.972),
        ncol=2,
        frameon=False,
        fontsize=10.5,
        handletextpad=0.5,
        columnspacing=1.6,
    )
    _save(fig, "figure2_occurrence_balance_by_tissue")
    return pd.concat([human, mouse], ignore_index=True).to_dict(orient="records")


def _human_architecture_summary() -> pd.DataFrame:
    summary = pd.read_csv(
        HUMAN_RESULTS / "results" / "v7_full_rerun" / "paper_results" / "paper_architecture_summary.csv"
    )
    summary.loc[summary["paper_model"] == "Full multi-kernel TissuePMHC", "paper_model"] = "TissuePMHC (untuned)"
    tuned_seeds = pd.read_csv(HUMAN_RESULTS / "adjusting" / "results" / "e29_final_test" / "per_seed_summary.csv")
    tuned = pd.DataFrame(
        [
            {
                "paper_model": "TissuePMHC (tuned)",
                "mean_auroc": tuned_seeds["mean_task_auroc"].mean(),
                "sd_auroc": tuned_seeds["mean_task_auroc"].std(ddof=1),
                "mean_auprc": tuned_seeds["mean_task_auprc"].mean(),
                "sd_auprc": tuned_seeds["mean_task_auprc"].std(ddof=1),
            }
        ]
    )
    return pd.concat([summary, tuned], ignore_index=True).sort_values("mean_auroc")


def figure5_architecture_survey() -> None:
    # NewTX uses TeX Gyre Termes; use the same face in the standalone PDF.
    figure_font = _manuscript_font_family()
    with plt.rc_context({
        "font.family": figure_font,
        "font.size": 11.0,
        "axes.titlesize": 13.5,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.0,
    }):
        _figure5_architecture_survey_impl()


def _figure5_architecture_survey_impl() -> None:
    frame = _human_architecture_summary().reset_index(drop=True)
    y = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(8.3, 10.2))
    fig.subplots_adjust(left=0.36, right=0.98, top=0.90, bottom=0.09)
    ax.axvspan(0.80, 0.82, color="#F5E7E5", alpha=0.65, zorder=0)
    ax.axvline(0.80, color="#9AA8B5", lw=0.9, ls=(0, (3, 3)), zorder=1)
    ax.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.85, zorder=0)

    for i, row in frame.iterrows():
        is_tuned = row["paper_model"] == "TissuePMHC (tuned)"
        color = CORAL if is_tuned else BLUE
        ax.plot([row["mean_auroc"], row["mean_auprc"]], [i, i], color=NAVY, lw=1.8, alpha=0.65, zorder=2)
        if pd.notna(row.get("sd_auroc")):
            ax.errorbar(row["mean_auroc"], i, xerr=row["sd_auroc"], fmt="none", ecolor=color, elinewidth=1.5, capsize=2.5, zorder=3)
        if pd.notna(row.get("sd_auprc")):
            ax.errorbar(row["mean_auprc"], i, xerr=row["sd_auprc"], fmt="none", ecolor=TEAL, elinewidth=1.35, capsize=2.5, zorder=3)
        ax.scatter(row["mean_auroc"], i, s=58 if is_tuned else 39, color=color, edgecolor="white", linewidth=0.8, alpha=0.95, zorder=4)
        ax.scatter(row["mean_auprc"], i, s=48 if is_tuned else 33, marker="D", color=GOLD if is_tuned else TEAL, edgecolor="white", linewidth=0.7, alpha=0.95, zorder=4)

    ax.set_yticks(y, frame["paper_model"])
    for label, model in zip(ax.get_yticklabels(), frame["paper_model"]):
        label.set_fontsize(9.0)
        label.set_color("black")
        if model == "TissuePMHC (tuned)":
            label.set_fontweight("bold")
    ax.set_xlim(0.67, 0.825)
    ax.tick_params(axis="x", colors="black")
    ax.set_xlabel("Task-macro fixed-test performance across 77 tissue–HLA tasks")
    fig.suptitle(
        "Human architecture survey on the occurrence-matched fixed test",
        fontsize=14.0, fontweight="bold", color="black", y=0.982,
    )
    fig.text(
        0.67, 0.942, "All 23 methods from the main result table",
        fontsize=9.5, color="black", ha="center",
    )
    ax.set_title("A  Human: 23 surveyed architectures", loc="left", fontweight="bold", pad=12, color="black")
    ax.scatter([], [], s=38, color=BLUE, label="AUROC (mean ± seed SD)")
    ax.scatter([], [], s=32, marker="D", color=TEAL, label="AUPRC (mean ± seed SD)")
    legend = ax.legend(loc="lower right", frameon=False, fontsize=9.5, labelspacing=0.65)
    ax.xaxis.label.set_color("black")
    for text in legend.get_texts():
        text.set_color("black")
    _clean_axes(ax)
    _save(fig, "figure5_human_architecture_survey")


def _task_means(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, keep_default_na=False)
    data["target_tissue"] = data["target_tissue"].replace({"": "NA/unspecified", "NA": "NA/unspecified"})
    return (
        data.groupby(["target_tissue", "mhc_restriction"], as_index=False)
        .agg(auroc=("auroc", "mean"), auprc=("auprc", "mean"))
    )


def _tissue_palette(tissues: list[str]) -> dict[str, str]:
    colors = list(plt.get_cmap("tab20").colors)
    return {tissue: colors[i % len(colors)] for i, tissue in enumerate(sorted(set(tissues)))}


def _jittered_tissue_panel(ax: plt.Axes, frame: pd.DataFrame, title: str, palette: dict[str, str], seed: int) -> None:
    summary = frame.groupby("target_tissue")["auroc"].mean().sort_values()
    order = summary.index.tolist()
    positions = {t: i for i, t in enumerate(order)}
    rng = np.random.default_rng(seed)
    for tissue in order:
        sub = frame[frame["target_tissue"] == tissue]
        x = positions[tissue] + rng.uniform(-0.18, 0.18, len(sub))
        ax.scatter(x, sub["auroc"], s=31, color=palette[tissue], edgecolor="white", linewidth=0.55, alpha=0.92, zorder=3)
        ax.plot([positions[tissue] - 0.23, positions[tissue] + 0.23], [summary[tissue], summary[tissue]], color=NAVY, lw=2.0, zorder=4)
    ax.axhline(0.5, color="#8D9BA8", ls=(0, (4, 3)), lw=1.0)
    display_labels = ["NA/unspecified" if t == "NA/unspecified" else t.replace("_", " ").title() for t in order]
    ax.set_xticks(range(len(order)), display_labels, rotation=42, ha="right")
    ax.set_ylim(0.46, 0.99)
    ax.set_ylabel("Task AUROC (three-seed mean)", color="black")
    ax.set_title(title, loc="left", fontweight="bold", color="black")
    ax.tick_params(axis="both", colors="black")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color("black")
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.8)
    _clean_axes(ax)


def figure6_tissue_heterogeneity() -> None:
    figure_font = _manuscript_font_family()
    with plt.rc_context({"font.family": figure_font}):
        _figure6_tissue_heterogeneity_impl()


def _figure6_tissue_heterogeneity_impl() -> None:
    human = _task_means(HUMAN_RESULTS / "adjusting" / "results" / "e29_final_test" / "all_seed_per_task_metrics.csv")
    mouse = _task_means(MOUSE_RESULTS / "adjusting" / "results" / "final_per_task_metrics.csv")
    palette = _tissue_palette(human["target_tissue"].tolist() + mouse["target_tissue"].tolist())

    fig = plt.figure(figsize=(9.2, 7.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.45, 1], hspace=0.52)
    _jittered_tissue_panel(fig.add_subplot(gs[0]), human, "A  Human: 77 tissue–HLA tasks", palette, 20260704)
    _jittered_tissue_panel(fig.add_subplot(gs[1]), mouse, "B  Mouse: 11 tissue–H2 tasks", palette, 20260705)
    fig.suptitle("Fixed-test performance is heterogeneous within and between tissues", fontsize=14, fontweight="bold", color="black", y=1.01)
    fig.text(0.5, 0.965, "Point color encodes tissue only; navy bars are tissue means", ha="center", fontsize=8.5, color="black")
    _save(fig, "figure6_tissue_colored_heterogeneity")


def _paired_tuning_deltas(tuned_path: Path, untuned_path: Path) -> pd.DataFrame:
    tuned = _task_means(tuned_path).rename(columns={"auroc": "tuned_auroc", "auprc": "tuned_auprc"})
    untuned = _task_means(untuned_path).rename(columns={"auroc": "untuned_auroc", "auprc": "untuned_auprc"})
    merged = tuned.merge(untuned, on=["target_tissue", "mhc_restriction"], validate="one_to_one")
    merged["delta_auroc"] = merged["tuned_auroc"] - merged["untuned_auroc"]
    return merged


def _bootstrap_mean_ci(values: np.ndarray, seed: int, n_boot: int = 10_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def _holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def figure7_tuning_effect() -> list[dict[str, object]]:
    figure_font = _manuscript_font_family()
    with plt.rc_context({"font.family": figure_font}):
        return _figure7_tuning_effect_impl()


def _figure7_tuning_effect_impl() -> list[dict[str, object]]:
    human = _paired_tuning_deltas(
        HUMAN_RESULTS / "adjusting" / "results" / "e29_final_test" / "all_seed_per_task_metrics.csv",
        HUMAN_RESULTS / "results" / "e29_multikernel_cnn" / "per_task_metrics.csv",
    )
    mouse = _paired_tuning_deltas(
        MOUSE_RESULTS / "adjusting" / "results" / "final_per_task_metrics.csv",
        MOUSE_RESULTS / "results" / "e29_multikernel_cnn" / "per_task_metrics.csv",
    )

    frames = [("Human", human, BLUE, 20260704), ("Mouse", mouse, CORAL, 20260705)]
    raw_p = [float(wilcoxon(frame["delta_auroc"], alternative="two-sided").pvalue) for _, frame, _, _ in frames]
    adjusted_p = _holm_adjust(raw_p)
    stats = []

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.25), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.80, bottom=0.18, wspace=0.18)
    x_limits = (-0.16, 0.12)

    for panel_index, (ax, (species, frame, color, seed), p_adj) in enumerate(
        zip(axes, frames, adjusted_p)
    ):
        values = frame["delta_auroc"].to_numpy()
        mean = float(values.mean())
        lo, hi = _bootstrap_mean_ci(values, seed)
        improved = int((values > 0).sum())
        declined = int((values < 0).sum())
        tied = int((values == 0).sum())

        ax.set_xlim(*x_limits)
        ax.set_ylim(0.0, 1.02)
        ax.axvline(0, color="black", lw=1.3, ls=(0, (4, 3)), zorder=1)
        ax.grid(axis="x", color=GRID, linewidth=0.75, alpha=0.9, zorder=0)

        density_x = np.linspace(max(x_limits[0], values.min() - 0.02), min(x_limits[1], values.max() + 0.02), 300)
        density = gaussian_kde(values)(density_x)
        density = density / density.max() * 0.26
        density_base = 0.48
        ax.fill_between(density_x, density_base, density_base + density, color=TEAL, alpha=0.22, zorder=2)
        ax.plot(density_x, density_base + density, color=TEAL, alpha=0.50, lw=1.2, zorder=3)

        box = ax.boxplot(
            values, vert=False, positions=[0.50], widths=0.16,
            patch_artist=True, showfliers=False,
            medianprops={"color": NAVY, "linewidth": 1.8},
            boxprops={"facecolor": "white", "edgecolor": TEAL, "linewidth": 1.8},
            whiskerprops={"color": TEAL, "linewidth": 1.5},
            capprops={"color": TEAL, "linewidth": 1.5},
        )

        rng = np.random.default_rng(seed)
        jitter = rng.uniform(-0.075, 0.075, len(values))
        point_colors = np.where(values > 0, TEAL, np.where(values < 0, CORAL, NAVY))
        ax.scatter(
            values, 0.27 + jitter, s=28 if species == "Human" else 42,
            c=point_colors, edgecolor="white", linewidth=0.45,
            alpha=0.68 if species == "Human" else 0.82, zorder=4,
        )
        ax.errorbar(
            mean, 0.77, xerr=[[mean - lo], [hi - mean]], fmt="D",
            color=NAVY, mfc=TEAL, mec="white", mew=0.8,
            ms=8, capsize=4, lw=1.7, zorder=6,
        )

        ax.text(
            0.995, 0.985,
            f"mean {mean:+.4f}\n95% CI [{lo:+.4f}, {hi:+.4f}]\nHolm $p={p_adj:.4g}$",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=12.5, color="#263746", linespacing=1.12,
        )
        ax.text(
            0.03, 0.06, f"{declined} declined",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=11.5, color=CORAL,
        )
        right_text = f"{improved} improved" + (f"; {tied} tied" if tied else "")
        ax.text(
            0.97, 0.06, right_text,
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=11.5, color=TEAL,
        )
        ax.set_title(
            f"{chr(65 + panel_index)}  {species}: {len(values)} tasks",
            loc="left", fontsize=13.5, fontweight="bold", color="black", pad=10,
        )
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=12.0, colors="black")
        _clean_axes(ax)

        stats.append(
            {
                "species": species,
                "n_tasks": int(len(values)),
                "mean_delta_auroc": mean,
                "median_delta_auroc": float(np.median(values)),
                "bootstrap_95_ci": [lo, hi],
                "wilcoxon_raw_p": raw_p[panel_index],
                "wilcoxon_holm_p": p_adj,
                "n_improved": improved,
                "n_declined": declined,
                "n_tied": tied,
            }
        )

    fig.suptitle(
        "Training-only tuning changes fixed-test performance differently across tasks",
        fontsize=15.5, fontweight="bold", color="black", y=0.965,
    )
    fig.text(
        0.5, 0.885,
        "Rainclouds show complete task distributions; diamonds show means and task-bootstrap 95% intervals",
        ha="center", fontsize=11.5, color="black",
    )
    fig.supxlabel("Task-wise AUROC change: tuned − untuned TissuePMHC", fontsize=13.5, color="black", y=0.06)
    _save(fig, "figure7_tuned_vs_untuned_task_deltas")
    return stats


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure1_lung_mechanism()
    figure1_add_structure_labels()
    occurrence = figure2_occurrence_balance()
    figure5_architecture_survey()
    figure6_tissue_heterogeneity()
    tuning_stats = figure7_tuning_effect()
    audit = {
        "figure2_occurrence_totals": occurrence,
        "figure7_tuning_statistics": tuning_stats,
        "sources": {
            "human_data": str(HUMAN_DATA),
            "mouse_data": str(MOUSE_DATA),
            "human_tuned": str(HUMAN_RESULTS / "adjusting" / "results" / "e29_final_test"),
            "mouse_tuned": str(MOUSE_RESULTS / "adjusting" / "results"),
        },
    }
    (FIGURES / "p1_main_figure_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
