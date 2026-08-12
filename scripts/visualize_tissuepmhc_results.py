"""Create publication-ready figures for the current tissuePMHC results.

The script reads frozen datasets and experiment outputs only; it never trains a
model.  The figure set follows the Phase 2 narrative through the preregistered
E29 five-seed result and explicitly shows both performance and benchmark scope.
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
    "e17": "#d17a22",
    "e29": "#147d92",
    "negative": "#b24a46",
    "neutral": "#c7ced1",
    "dark": "#30373b",
}

# Deliberately selected milestones, not every attempted experiment.  Negative
# results remain represented by the leakage-safe E27 stacker.
MILESTONES = [
    ("E0 one-hot logistic", "tissuePMHC_baselines", "onehot_logistic_regression"),
    ("E2 shared task heads", "tissuePMHC_neural_baselines_v2", "shared_peptide_encoder_task_heads"),
    ("E8a soft ensemble", "tissuePMHC_soft_ensemble", "e8a_fixed_average"),
    ("E14a auxiliary ensemble", "tissuePMHC_auxiliary_soft_ensemble", "e14a_global_aux_hla_plain"),
    ("E15 task-rank fusion", "tissuePMHC_e15_fusion_ablation", "e15_task_rank_average"),
    ("E17 3-seed ensemble", "tissuePMHC_e17_seed_ensemble", "e17_3seed_rank_average"),
    ("E27 OOF stacker", "tissuePMHC_e27_stacked_generalization", "e27_oof_rank_logistic_stacker"),
    ("E17 5-seed ensemble", "tissuePMHC_e17_seed_ensemble", "e17_5seed_rank_average"),
    ("E29 CNN 3-seed", "tissuePMHC_e29_multikernel_cnn_3seed", "e29_cnn_3seed_mean"),
    ("E29 CNN 5-seed (final)", "tissuePMHC_e29_multikernel_cnn_5seed", "e29_cnn_5seed_mean"),
]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "savefig.dpi": 300,
        }
    )


def read_model(results_root: Path, folder: str, model: str) -> pd.DataFrame:
    path = results_root / folder / "summary_metrics.csv"
    table = pd.read_csv(path)
    table = table.loc[table["model"].eq(model)].copy()
    if table.empty:
        raise ValueError(f"Model {model!r} is absent from {path}")
    return table


def collect_milestones(results_root: Path) -> pd.DataFrame:
    rows = []
    for label, folder, model in MILESTONES:
        data = read_model(results_root, folder, model)
        if "worst_10_mean_auroc" in data.columns:
            worst_10 = data["worst_10_mean_auroc"].mean()
        else:
            per_task = pd.read_csv(results_root / folder / "per_task_metrics.csv")
            per_task = per_task.loc[per_task["model"].eq(model)]
            worst_10 = per_task.nsmallest(10, "auroc")["auroc"].mean()
        rows.append(
            {
                "label": label,
                "mean_auroc": data["mean_auroc"].mean(),
                "mean_auprc": data["mean_auprc"].mean(),
                "worst_10_mean_auroc": worst_10,
                "n_rows": len(data),
            }
        )
    return pd.DataFrame(rows)


def milestone_color(label: str) -> str:
    if label.startswith("E29"):
        return COLORS["e29"]
    if label.startswith("E17"):
        return COLORS["e17"]
    if label.startswith("E27"):
        return COLORS["negative"]
    return COLORS["baseline"]


def plot_model_milestones(summary: pd.DataFrame) -> plt.Figure:
    # Keep one stable research ordering across all facets: strongest mean AUROC
    # at the top, weakest at the bottom.
    summary = summary.sort_values("mean_auroc", ascending=False).reset_index(drop=True)
    metrics = [
        ("mean_auroc", "Mean AUROC"),
        ("mean_auprc", "Mean AUPRC"),
        ("worst_10_mean_auroc", "Worst-10 mean AUROC"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.6), sharey=True)
    y = np.arange(len(summary))
    colors = [milestone_color(label) for label in summary["label"]]
    for ax, (metric, title) in zip(axes, metrics):
        values = summary[metric].to_numpy()
        e0_value = float(summary.loc[summary["label"].eq("E0 one-hot logistic"), metric].iloc[0])
        lower = values.min() - 0.012
        ax.barh(y, values - lower, left=lower, color=colors, alpha=0.92)
        ax.axvline(e0_value, color=COLORS["dark"], linestyle="--", linewidth=1.1, zorder=3)
        ax.set_title(title)
        ax.set_xlabel("Task-macro score")
        ax.set_xlim(lower, min(1.0, values.max() + 0.025))
        ax.set_yticks(y, summary["label"])
        for idx, value in enumerate(values):
            ax.text(value + 0.001, idx, f"{value:.4f}", va="center", fontsize=8)
    axes[0].set_ylabel("Research milestone")
    axes[0].invert_yaxis()
    handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=COLORS[key],
                   markeredgecolor="none", markersize=8, label=label)
        for key, label in [("baseline", "Earlier milestone"), ("e17", "Seed ensemble"),
                           ("e29", "Multi-kernel CNN"), ("negative", "OOF negative result")]
    ]
    handles.append(
        plt.Line2D([0], [0], color=COLORS["dark"], linestyle="--", linewidth=1.1, label="E0 reference")
    )
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.045), ncol=5, fontsize=8)
    fig.suptitle("Phase 1-2 model progression: E29 is the frozen standard-split endpoint", y=1.0)
    return fig


def read_e29_task_comparison(results_root: Path) -> pd.DataFrame:
    data = pd.read_csv(
        results_root / "tissuePMHC_e29_multikernel_cnn_5seed" / "e17_5seed_comparison_metrics.csv"
    )
    data["task"] = data["target_tissue"] + " | " + data["mhc_restriction"]
    return data


def plot_e29_task_gain(paired: pd.DataFrame) -> plt.Figure:
    ordered = paired.sort_values("delta_auroc").reset_index(drop=True)
    colors = np.where(ordered["delta_auroc"] >= 0, COLORS["e29"], COLORS["negative"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 9), gridspec_kw={"width_ratios": [1.45, 1]})
    y = np.arange(len(ordered))
    axes[0].barh(y, ordered["delta_auroc"], color=colors)
    axes[0].axvline(0, color=COLORS["dark"], linewidth=0.9)
    axes[0].set_yticks(y, ordered["task"], fontsize=7)
    axes[0].set_xlabel("E29 - E17 task AUROC")
    axes[0].set_title(
        f"AUROC: {(ordered.delta_auroc > 0).sum()} wins / {(ordered.delta_auroc < 0).sum()} losses; "
        f"mean {ordered.delta_auroc.mean():+.4f}"
    )

    metrics = [("delta_auroc", "AUROC"), ("delta_auprc", "AUPRC"),
               ("delta_accuracy", "Accuracy"), ("delta_mcc", "MCC")]
    positions = np.arange(len(metrics))
    for x, (column, label) in zip(positions, metrics):
        vals = paired[column].dropna().to_numpy()
        parts = axes[1].violinplot(vals, positions=[x], widths=0.72, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(COLORS["e29"])
            body.set_edgecolor("none")
            body.set_alpha(0.28)
        jitter = np.linspace(-0.11, 0.11, len(vals))
        axes[1].scatter(x + jitter, np.sort(vals), s=11, color=COLORS["e29"], alpha=0.65)
        axes[1].plot([x - 0.18, x + 0.18], [np.mean(vals)] * 2, color=COLORS["dark"], lw=2)
        axes[1].text(x, vals.max() + 0.006, f"mean {np.mean(vals):+.3f}", ha="center", fontsize=8)
    axes[1].axhline(0, color=COLORS["dark"], linewidth=0.9)
    axes[1].set_xticks(positions, [label for _, label in metrics])
    axes[1].set_ylabel("E29 - E17 task-level change")
    axes[1].set_title("Gain distribution across 44 tissue-HLA tasks")
    return fig


def plot_dataset_coverage(data_dir: Path) -> plt.Figure:
    data = pd.read_csv(data_dir / "tissuePMHC_summary.csv")
    count_column = next(
        (name for name in ["n_train", "train_rows", "n_samples", "count"] if name in data.columns), None
    )
    if count_column is None:
        numeric = data.select_dtypes(include="number").columns.tolist()
        if not numeric:
            raise ValueError("No numeric task-size column found in tissuePMHC_summary.csv")
        count_column = numeric[0]
    pivot = data.pivot_table(
        index="target_tissue", columns="mhc_restriction", values=count_column, aggfunc="sum", fill_value=0
    )
    pivot = pivot.loc[pivot.sum(axis=1).sort_values().index]
    masked = np.ma.masked_where(pivot.to_numpy() <= 0, pivot.to_numpy())
    positive = masked.compressed()
    fig, ax = plt.subplots(figsize=(12, max(5.5, 0.48 * len(pivot))))
    image = ax.imshow(
        masked,
        aspect="auto",
        cmap="Blues",
        norm=LogNorm(vmin=max(1, positive.min()), vmax=positive.max()),
    )
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_xlabel("HLA restriction")
    ax.set_ylabel("Target tissue")
    ax.set_title(f"Dataset task coverage ({count_column}; logarithmic color scale)")
    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            value = pivot.iat[row, col]
            if value > 0:
                ax.text(col, row, f"{int(value):,}", ha="center", va="center", fontsize=6.5,
                        color="white" if value > np.sqrt(positive.min() * positive.max()) else COLORS["dark"])
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label(count_column)
    return fig


def seed_scaling_table(results_root: Path) -> pd.DataFrame:
    e17 = pd.read_csv(results_root / "tissuePMHC_e17_seed_ensemble" / "summary_metrics.csv")
    e29_3 = pd.read_csv(results_root / "tissuePMHC_e29_multikernel_cnn_3seed" / "summary_metrics.csv")
    e29_5 = pd.read_csv(results_root / "tissuePMHC_e29_multikernel_cnn_5seed" / "summary_metrics.csv")
    rows = []
    for family, seeds, frame, model in [
        ("E17", 3, e17, "e17_3seed_rank_average"),
        ("E17", 5, e17, "e17_5seed_rank_average"),
        ("E29", 3, e29_3, "e29_cnn_3seed_mean"),
        ("E29", 5, e29_5, "e29_cnn_5seed_mean"),
    ]:
        row = frame.loc[frame.model.eq(model)].iloc[0]
        rows.append({"family": family, "seeds": seeds, "mean_auroc": row.mean_auroc,
                     "mean_auprc": row.mean_auprc, "worst_10_mean_auroc": row.worst_10_mean_auroc})
    return pd.DataFrame(rows)


def plot_seed_scaling(scaling: pd.DataFrame) -> plt.Figure:
    metrics = [("mean_auroc", "Mean AUROC"), ("mean_auprc", "Mean AUPRC"),
               ("worst_10_mean_auroc", "Worst-10 mean AUROC")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    for ax, (metric, title) in zip(axes, metrics):
        for family, color, marker in [("E17", COLORS["e17"], "o"), ("E29", COLORS["e29"], "s")]:
            subset = scaling.loc[scaling.family.eq(family)].sort_values("seeds")
            ax.plot(subset.seeds, subset[metric], color=color, marker=marker, linewidth=2, label=family)
            for _, row in subset.iterrows():
                # Push edge-point labels inward so they do not collide with the
                # y-axis on the left or clip against the right plot boundary.
                offset_x = 8 if row.seeds == 3 else -8
                align = "left" if row.seeds == 3 else "right"
                ax.annotate(f"{row[metric]:.4f}", (row.seeds, row[metric]), xytext=(offset_x, 7),
                            textcoords="offset points", ha=align, fontsize=8)
        ax.set_xticks([3, 5], ["3 seeds", "5 seeds"])
        ax.set_title(title)
        ax.set_ylabel("Task-macro score")
        values = scaling[metric]
        ax.set_ylim(values.min() - 0.006, values.max() + 0.009)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.suptitle("Seed averaging helps both encoders; E29 retains a larger representation gain")
    return fig


def benchmark_scope(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["sample_id", "peptide_sequence", "molecule_parent_uniprot_id"]
    train = pd.read_csv(data_dir / "tissuePMHC_train.csv.gz", usecols=columns[1:])
    test = pd.read_csv(data_dir / "tissuePMHC_test.csv.gz", usecols=columns)
    peptide_seen = test.peptide_sequence.isin(set(train.peptide_sequence.dropna()))
    protein_seen = test.molecule_parent_uniprot_id.isin(set(train.molecule_parent_uniprot_id.dropna()))
    overlap = pd.DataFrame(
        {
            "entity": ["Peptide", "Parent UniProt"],
            "seen_fraction": [peptide_seen.mean(), protein_seen.mean()],
            "unseen_fraction": [1 - peptide_seen.mean(), 1 - protein_seen.mean()],
        }
    )
    # These audited subset values are documented in PHASE_2_REPORT_zh.md, section 14.4.
    subset = pd.DataFrame(
        {
            "subset": ["Seen peptide", "Unseen peptide"],
            "E17": [0.83966, 0.73761],
            "E29": [0.85328, 0.74199],
        }
    )
    return overlap, subset


def plot_generalization_boundary(overlap: pd.DataFrame, subset: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(overlap))
    axes[0].bar(x, overlap.seen_fraction * 100, color=COLORS["baseline"], label="Seen in train")
    axes[0].bar(x, overlap.unseen_fraction * 100, bottom=overlap.seen_fraction * 100,
                color=COLORS["neutral"], label="Unseen in train")
    axes[0].set_xticks(x, overlap.entity)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Share of standard test rows (%)")
    axes[0].set_title("Closed-set overlap with training data")
    for idx, value in enumerate(overlap.seen_fraction * 100):
        axes[0].text(idx, value / 2, f"{value:.2f}% seen", ha="center", va="center", color="white")
        axes[0].text(idx, value + (100 - value) / 2, f"{100-value:.2f}% unseen",
                     ha="center", va="center", color=COLORS["dark"], fontsize=8)

    width = 0.34
    x = np.arange(len(subset))
    axes[1].bar(x - width / 2, subset.E17, width, color=COLORS["e17"], label="E17 5-seed")
    axes[1].bar(x + width / 2, subset.E29, width, color=COLORS["e29"], label="E29 5-seed")
    axes[1].set_xticks(x, subset.subset)
    axes[1].set_ylim(0.68, 0.89)
    axes[1].set_ylabel("Task-macro AUROC")
    axes[1].set_title("Performance drops on unseen peptides")
    for idx, row in subset.iterrows():
        axes[1].text(idx - width / 2, row.E17 + 0.004, f"{row.E17:.3f}", ha="center", fontsize=8)
        axes[1].text(idx + width / 2, row.E29 + 0.004, f"{row.E29:.3f}", ha="center", fontsize=8)
        bracket_y = max(row.E17, row.E29) + 0.012
        axes[1].plot([idx - width / 2, idx + width / 2], [bracket_y, bracket_y],
                     color=COLORS["dark"], linewidth=0.8)
        axes[1].text(idx, bracket_y + 0.004, f"delta {row.E29-row.E17:+.3f}",
                     ha="center", va="bottom", fontsize=8)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    return fig


def save(fig: plt.Figure, output_dir: Path, stem: str, pdf: PdfPages) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "tissuePMHC")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "figures_phase2_final")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    milestones = collect_milestones(args.results_dir)
    paired = read_e29_task_comparison(args.results_dir)
    scaling = seed_scaling_table(args.results_dir)
    overlap, subset = benchmark_scope(args.data_dir)
    milestones.to_csv(args.output_dir / "model_milestones.csv", index=False)
    paired.to_csv(args.output_dir / "e29_vs_e17_task_metrics.csv", index=False)
    scaling.to_csv(args.output_dir / "seed_scaling.csv", index=False)
    overlap.to_csv(args.output_dir / "standard_split_overlap.csv", index=False)
    subset.to_csv(args.output_dir / "seen_unseen_peptide_auroc.csv", index=False)

    with PdfPages(args.output_dir / "tissuepmhc_phase2_final_figures.pdf") as pdf:
        save(plot_model_milestones(milestones), args.output_dir, "01_model_milestones", pdf)
        save(plot_e29_task_gain(paired), args.output_dir, "02_e29_vs_e17_task_gain", pdf)
        save(plot_dataset_coverage(args.data_dir), args.output_dir, "03_dataset_coverage", pdf)
        save(plot_seed_scaling(scaling), args.output_dir, "04_seed_ensemble_scaling", pdf)
        save(plot_generalization_boundary(overlap, subset), args.output_dir, "05_generalization_boundary", pdf)

    metadata = {
        "status": "phase2_final_standard_split",
        "main_result": "E29 CNN 5-seed",
        "n_figures": 5,
        "benchmark_warning": "Closed-set standard split; not peptide/protein/HLA-disjoint external validation.",
    }
    (args.output_dir / "figure_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote five figures, source tables, and one PDF to: {args.output_dir}")


if __name__ == "__main__":
    main()
