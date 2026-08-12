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
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from matplotlib.ticker import FuncFormatter, StrMethodFormatter
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "paper" / "tissuePMHC_latex_v7"
FIGURES = PAPER / "figures"

HUMAN_DATA = ROOT / "data" / "humanPMHC_occurence_equal_dataset"
MOUSE_DATA = ROOT / "data" / "mousePMHC_occurence_equal_dataset"
HUMAN_RESULTS = ROOT / "extra_occurence_equal_dataset"
MOUSE_RESULTS = ROOT / "extra_mouse_occurence_equal_dataset"

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
    # Use a standard journal figure face locally without changing the other
    # main-text plots generated by this script.
    with plt.rc_context({"font.family": "Arial", "font.size": 8.0}):
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
        figsize=(7.6, 6.15),
        sharex=False,
        gridspec_kw={"height_ratios": [13, 4]},
        facecolor="white",
    )
    fig.subplots_adjust(left=0.16, right=0.965, top=0.865, bottom=0.12, hspace=0.44)

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
                fontsize=6.7,
                color=muted,
            )
            ax.annotate(
                f"{int(value1):,}",
                xy=(value1 + offset, yi),
                va="center",
                ha="left",
                fontsize=6.7,
                color=muted,
            )
        ax.set_yticks(y, frame["target_tissue"])
        ax.set_xlim(-limit, limit)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{abs(value):,.0f}"))
        ax.grid(axis="x", color="#DDE1E6", linewidth=0.5, zorder=0)
        ax.axvline(0, color="#33383F", linewidth=0.7, zorder=3)
        ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold", pad=8, color=text_color)
        ax.text(
            -0.115,
            1.035,
            panel,
            transform=ax.transAxes,
            fontsize=10.0,
            fontweight="bold",
            ha="left",
            va="bottom",
            color=text_color,
        )
        ax.tick_params(axis="y", length=0, pad=7, labelsize=7.7, colors=text_color)
        ax.tick_params(axis="x", which="major", length=3.0, width=0.65, labelsize=7.4, colors=text_color)
        ax.spines["bottom"].set_color("#555555")
        ax.spines["bottom"].set_linewidth(0.7)
        _clean_axes(ax)

    fig.supxlabel(
        "Summed tissue occurrences across benchmark rows",
        fontsize=8.2,
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
        fontsize=7.8,
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
    frame = _human_architecture_summary().reset_index(drop=True)
    y = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(8.3, 9.4))
    ax.axvspan(0.80, 0.82, color="#F5E7E5", alpha=0.65, zorder=0)
    ax.axvline(0.80, color="#9AA8B5", lw=0.9, ls=(0, (3, 3)), zorder=1)
    ax.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.85, zorder=0)

    for i, row in frame.iterrows():
        is_tuned = row["paper_model"] == "TissuePMHC (tuned)"
        color = CORAL if is_tuned else BLUE
        ax.plot([row["mean_auroc"], row["mean_auprc"]], [i, i], color="#BAC6D2", lw=1.0, zorder=2)
        if pd.notna(row.get("sd_auroc")):
            ax.errorbar(row["mean_auroc"], i, xerr=row["sd_auroc"], fmt="none", ecolor=color, elinewidth=1.2, capsize=2, zorder=3)
        if pd.notna(row.get("sd_auprc")):
            ax.errorbar(row["mean_auprc"], i, xerr=row["sd_auprc"], fmt="none", ecolor=TEAL, elinewidth=1.0, capsize=2, zorder=3)
        ax.scatter(row["mean_auroc"], i, s=48 if is_tuned else 31, color=color, edgecolor="white", linewidth=0.7, zorder=4)
        ax.scatter(row["mean_auprc"], i, s=38 if is_tuned else 25, marker="D", color=GOLD if is_tuned else TEAL, edgecolor="white", linewidth=0.6, zorder=4)

    ax.set_yticks(y, frame["paper_model"])
    for label, model in zip(ax.get_yticklabels(), frame["paper_model"]):
        label.set_fontsize(7.5)
        if model == "TissuePMHC (tuned)":
            label.set_fontweight("bold")
            label.set_color(CORAL)
    ax.set_xlim(0.67, 0.825)
    ax.set_xlabel("Task-macro fixed-test performance across 77 tissue–HLA tasks")
    ax.set_title("Human architecture survey on the occurrence-matched fixed test", loc="left", fontweight="bold", pad=11)
    ax.text(0.671, len(frame) - 0.1, "All 23 methods from the main result table", fontsize=8, color="#526579", va="bottom")
    ax.scatter([], [], s=38, color=BLUE, label="AUROC (mean ± seed SD)")
    ax.scatter([], [], s=32, marker="D", color=TEAL, label="AUPRC (mean ± seed SD)")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
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
    ax.set_ylabel("Task AUROC (three-seed mean)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.8)
    _clean_axes(ax)


def figure6_tissue_heterogeneity() -> None:
    human = _task_means(HUMAN_RESULTS / "adjusting" / "results" / "e29_final_test" / "all_seed_per_task_metrics.csv")
    mouse = _task_means(MOUSE_RESULTS / "adjusting" / "results" / "final_per_task_metrics.csv")
    palette = _tissue_palette(human["target_tissue"].tolist() + mouse["target_tissue"].tolist())

    fig = plt.figure(figsize=(9.2, 7.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.45, 1], hspace=0.52)
    _jittered_tissue_panel(fig.add_subplot(gs[0]), human, "A  Human: 77 tissue–HLA tasks", palette, 20260704)
    _jittered_tissue_panel(fig.add_subplot(gs[1]), mouse, "B  Mouse: 11 tissue–H2 tasks", palette, 20260705)
    fig.suptitle("Fixed-test performance is heterogeneous within and between tissues", fontsize=14, fontweight="bold", y=1.01)
    fig.text(0.5, 0.965, "Point color encodes tissue only; navy bars are tissue means", ha="center", fontsize=8.5, color="#526579")
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

    fig, ax = plt.subplots(figsize=(8.3, 4.2))
    ax.axvline(0, color="#7E8E9D", lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.8)
    for y, ((species, frame, color, seed), p_adj) in enumerate(zip(frames, adjusted_p)):
        values = frame["delta_auroc"].to_numpy()
        rng = np.random.default_rng(seed)
        jitter = rng.uniform(-0.13, 0.13, len(values))
        violin = ax.violinplot(values, positions=[y], vert=False, widths=0.62, showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor("none")
            body.set_alpha(0.17)
        ax.scatter(values, y + jitter, s=24, color=color, edgecolor="white", linewidth=0.45, alpha=0.82, zorder=3)
        mean = float(values.mean())
        lo, hi = _bootstrap_mean_ci(values, seed)
        ax.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]], fmt="D", color=NAVY, mfc="white", mec=NAVY, ms=6, capsize=4, lw=1.7, zorder=5)
        ax.text(
            ax.get_xlim()[1] if False else max(0.13, values.max() + 0.008),
            y,
            f"mean {mean:+.4f}  |  95% CI [{lo:+.4f}, {hi:+.4f}]  |  Holm p={p_adj:.4g}",
            va="center",
            fontsize=7.6,
            color="#43596E",
        )
        stats.append(
            {
                "species": species,
                "n_tasks": int(len(values)),
                "mean_delta_auroc": mean,
                "median_delta_auroc": float(np.median(values)),
                "bootstrap_95_ci": [lo, hi],
                "wilcoxon_raw_p": raw_p[y],
                "wilcoxon_holm_p": p_adj,
                "n_improved": int((values > 0).sum()),
                "n_declined": int((values < 0).sum()),
                "n_tied": int((values == 0).sum()),
            }
        )

    ax.set_yticks([0, 1], ["Human (77 tasks)", "Mouse (11 tasks)"])
    ax.set_xlim(-0.16, 0.29)
    ax.set_xlabel("Task-wise AUROC change: tuned − untuned TissuePMHC")
    ax.set_title("Training-only tuning changes fixed-test performance differently across tasks", loc="left", fontweight="bold", pad=28)
    ax.text(
        0.0,
        1.015,
        "Diamonds show mean changes with 10,000-task-bootstrap 95% intervals",
        transform=ax.transAxes,
        fontsize=8,
        color="#526579",
        va="bottom",
    )
    _clean_axes(ax)
    _save(fig, "figure7_tuned_vs_untuned_task_deltas")
    return stats


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure1_lung_mechanism()
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
