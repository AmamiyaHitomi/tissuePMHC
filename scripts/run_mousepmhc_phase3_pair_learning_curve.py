#!/usr/bin/env python3
"""Estimate a mousePMHC minimum-pairs threshold using train-only learning curves.

At each pair-count point, the script selects every tissue-H2 task with at least
that many available pairs, then repeatedly subsamples the same number of pairs
from each selected task. It evaluates E0 one-hot logistic, E0 BLOSUM62
ExtraTrees, and mousePMHC E1 shared task heads with identical pair-grouped OOF
folds. Thus both per-task sample size and the number of jointly trained tasks
change across points. The shared-task-head implementation source is human
tissuePMHC E2. The fixed mouse test set is never accepted or read.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_tissuepmhc_baselines as human_e0
import run_tissuepmhc_e26_all_in_one as fold_utils
import run_tissuepmhc_neural_baselines_v2 as human_e2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "mousePMHC_phase3_pair_learning_curve"
MODELS = ["e0_onehot_logistic", "e0_blosum62_extra_trees", "e1_shared_task_heads"]
MODEL_STYLE = {
    "e0_blosum62_extra_trees": ("E0 BLOSUM62 ExtraTrees", "#2673B8", "o"),
    "e0_onehot_logistic": ("E0 one-hot logistic", "#E18A20", "s"),
    "e1_shared_task_heads": ("mousePMHC E1 shared task heads", "#2D9A68", "^"),
}


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def read_pairs(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    frame["label"] = frame.label.astype(int)
    frame["task_name"] = frame.target_tissue + "||" + frame.mhc_restriction
    if not frame.mhc_restriction.str.startswith("H2-").all():
        raise ValueError("Learning-curve input contains a non-H2 restriction.")
    pair_check = frame.groupby("pair_id").agg(rows=("label", "size"), labels=("label", "nunique"))
    if not ((pair_check.rows == 2) & (pair_check.labels == 2)).all():
        raise ValueError("Every pair_id must contain one positive and one negative row.")
    return frame


def subsample_tasks(frame: pd.DataFrame, tasks: list[str], pairs_per_task: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    selected: list[pd.DataFrame] = []
    for task_name in tasks:
        task = frame[frame.task_name == task_name]
        pair_ids = np.asarray(sorted(task.pair_id.unique()))
        if len(pair_ids) < pairs_per_task:
            raise ValueError(f"{task_name} has only {len(pair_ids)} pairs, requested {pairs_per_task}.")
        keep = set(rng.choice(pair_ids, size=pairs_per_task, replace=False))
        selected.append(task[task.pair_id.isin(keep)].copy())
    return pd.concat(selected, ignore_index=True)


def evaluate_predictions(frame: pd.DataFrame, score: np.ndarray, model: str, pairs_per_task: int, repeat_seed: int) -> list[dict[str, Any]]:
    working = frame[["target_tissue", "mhc_restriction", "label"]].copy()
    working["score"] = score
    rows: list[dict[str, Any]] = []
    for (tissue, mhc), task in working.groupby(["target_tissue", "mhc_restriction"], sort=True):
        rows.append({"experiment_name": EXPERIMENT, "pairs_per_task": pairs_per_task,
            "repeat_seed": repeat_seed, "model": model, "target_tissue": tissue,
            "mhc_restriction": mhc, "rows": len(task),
            **human_e0.evaluate(task.label.to_numpy(), task.score.to_numpy())})
    return rows


def run_e0(sample: pd.DataFrame, assignments: pd.Series, args: argparse.Namespace, pairs_per_task: int, repeat_seed: int) -> list[dict[str, Any]]:
    predictions = {"e0_onehot_logistic": np.empty(len(sample)), "e0_blosum62_extra_trees": np.empty(len(sample))}
    specifications = {
        "e0_onehot_logistic": "onehot_logistic_regression",
        "e0_blosum62_extra_trees": "blosum62_extra_trees",
    }
    for fold in range(args.oof_folds):
        fit, held = sample[assignments != fold], sample[assignments == fold]
        for output_name, human_name in specifications.items():
            encoder, _ = human_e0.get_models(repeat_seed)[human_name]
            for task_name, held_task in held.groupby("task_name", sort=True):
                fit_task = fit[fit.task_name == task_name]
                _, estimator = human_e0.get_models(repeat_seed)[human_name]
                estimator.fit(encoder(fit_task.peptide_sequence.tolist()), fit_task.label.to_numpy())
                positions = sample.index.get_indexer(held_task.index)
                predictions[output_name][positions] = human_e0.predict_scores(estimator, encoder(held_task.peptide_sequence.tolist()))
    rows: list[dict[str, Any]] = []
    for model, scores in predictions.items():
        rows.extend(evaluate_predictions(sample, scores, model, pairs_per_task, repeat_seed))
    return rows


def e1_arrays(frame: pd.DataFrame, peptide_length: int) -> list[np.ndarray]:
    return [human_e2.encode_peptides(frame.peptide_sequence, peptide_length).copy(),
        frame.task_id.to_numpy(dtype=np.int64, copy=True), frame.label.to_numpy(dtype=np.int64, copy=True)]


def run_e1(sample: pd.DataFrame, assignments: pd.Series, args: argparse.Namespace, pairs_per_task: int, repeat_seed: int) -> list[dict[str, Any]]:
    torch, nn, DataLoader, TensorDataset = human_e2.require_torch()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    prepared, _, mappings = human_e2.add_task_columns(sample, sample.copy())
    assignments = assignments.reindex(sample.index)
    predictions = np.empty(len(prepared))
    _, SharedTaskHeadsModel, _ = human_e2.define_models(nn)
    peptide_length = 9
    for fold in range(args.oof_folds):
        fit, held = prepared[assignments.to_numpy() != fold], prepared[assignments.to_numpy() == fold]
        human_e2.set_seed(repeat_seed, torch)
        model = SharedTaskHeadsModel(peptide_length, len(mappings["tasks"]), args.embedding_dim, args.hidden_dim, args.dropout).to(device)
        loader = human_e2.build_loader(torch, DataLoader, TensorDataset, e1_arrays(fit, peptide_length), args.batch_size, True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        human_e2.train_binary_model(torch, model, loader, optimizer, nn.BCEWithLogitsLoss(), device, "task_heads", args.epochs)
        held_loader = human_e2.build_loader(torch, DataLoader, TensorDataset, e1_arrays(held, peptide_length), args.batch_size, False)
        labels, scores = human_e2.predict_scores(torch, model, held_loader, device, "task_heads")
        if not np.array_equal(labels, held.label.to_numpy()):
            raise AssertionError("mousePMHC E1 learning-curve labels are misaligned.")
        predictions[np.flatnonzero(assignments.to_numpy() == fold)] = scores
    return evaluate_predictions(prepared, predictions, "e1_shared_task_heads", pairs_per_task, repeat_seed)


def summarize(rows: pd.DataFrame, args: argparse.Namespace, coverage: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    repeat = rows.groupby(["pairs_per_task", "repeat_seed", "model"], sort=True).auroc.mean().rename("mean_task_auroc").reset_index()
    summary = repeat.groupby(["pairs_per_task", "model"], sort=True).mean_task_auroc.agg(["mean", "std", "min", "max"]).reset_index()
    summary = summary.rename(columns={"mean": "mean_auroc", "std": "repeat_sd_auroc", "min": "min_repeat_auroc", "max": "max_repeat_auroc"})
    summary["gain_to_next_size"] = np.nan
    for model, indices in summary.groupby("model").groups.items():
        ordered = summary.loc[indices].sort_values("pairs_per_task")
        summary.loc[ordered.index, "gain_to_next_size"] = ordered.mean_auroc.shift(-1).to_numpy() - ordered.mean_auroc.to_numpy()
    selected = None
    ordered_sizes = sorted(summary.pairs_per_task.unique())
    # A single small next-step gain is not evidence of a plateau: a noisy
    # learning curve can flatten locally and then rise again. Require at least
    # two larger sizes, stable increments throughout the remaining curve, and
    # no cumulative drift beyond the same tolerance.
    for size in ordered_sizes[:-2]:
        block = summary[summary.pairs_per_task == size]
        coverage_row = coverage[coverage.min_pairs == size].iloc[0]
        stable_tail = True
        for model in MODELS:
            tail = summary[(summary.model == model) & (summary.pairs_per_task >= size)].sort_values("pairs_per_task")
            increments = tail.mean_auroc.diff().dropna().abs()
            cumulative = (tail.mean_auroc - tail.mean_auroc.iloc[0]).abs().max()
            if (increments.gt(args.maximum_absolute_gain_to_next).any() or
                cumulative > args.maximum_absolute_gain_to_next or
                tail.repeat_sd_auroc.fillna(np.inf).gt(args.maximum_repeat_sd).any()):
                stable_tail = False
                break
        if (len(block) == len(MODELS) and coverage_row.n_tasks >= args.minimum_tasks and
            stable_tail):
            selected = int(size); break
    recommendation = {
        "selected_min_pairs": selected,
        "selection_rule": {"minimum_tasks": args.minimum_tasks, "maximum_repeat_sd": args.maximum_repeat_sd,
            "maximum_absolute_gain_to_next": args.maximum_absolute_gain_to_next,
            "minimum_larger_sizes": 2, "maximum_cumulative_change_after_selection": args.maximum_absolute_gain_to_next,
            "required_models": MODELS},
        "interpretation": "No value means the tested range did not reach the pre-registered stability rule." if selected is None else
            f"Provisional fixed-cohort minimum: {selected} pairs per task. Confirm it on every newly admitted task; retain min500 separately for strict human alignment.",
    }
    return summary, recommendation


def render_figure(args: argparse.Namespace) -> None:
    """Create the learning-curve figure without relying on a separate script."""
    summary_path = args.output_dir / "mousePMHC_phase3_pair_learning_curve_summary.csv"
    coverage_path = args.output_dir / "mousePMHC_phase3_pair_threshold_coverage.csv"
    summary, coverage = pd.read_csv(summary_path), pd.read_csv(coverage_path)
    args.figure_output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.dpi": 150, "savefig.dpi": 300})
    fig, (ax, ax_coverage) = plt.subplots(
        2, 1, figsize=(8.2, 7.1), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.15], "hspace": 0.12}, layout="constrained")
    for model, (label, color, marker) in MODEL_STYLE.items():
        data = summary[summary.model == model].sort_values("pairs_per_task")
        ax.errorbar(data.pairs_per_task, data.mean_auroc, yerr=data.repeat_sd_auroc,
                    color=color, marker=marker, markersize=5.5, linewidth=2,
                    elinewidth=1, capsize=3, label=label)
        last = data.iloc[-1]
        ax.annotate(f"{last.mean_auroc:.3f}", (last.pairs_per_task, last.mean_auroc),
                    xytext=(7, 0), textcoords="offset points", va="center", color=color)
    lower = max(0.0, min(0.5, float((summary.mean_auroc - summary.repeat_sd_auroc.fillna(0)).min()) - 0.01))
    upper = min(1.0, max(0.75, float((summary.mean_auroc + summary.repeat_sd_auroc.fillna(0)).max()) + 0.01))
    ax.set_ylim(lower, upper)
    ax.set_ylabel("Mean task AUROC")
    ax.set_title("mousePMHC Phase 3 learning curves: all eligible tasks per threshold", pad=44)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.7, alpha=0.75)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.012),
              ncol=3, fontsize=9, columnspacing=1.2, handlelength=2.2)
    ax.text(0.995, 0.015, "Error bars: +/- 1 SD across repeated subsamples",
            transform=ax.transAxes, ha="right", va="bottom", color="#666666", fontsize=8.5)
    ax_coverage.plot(coverage.min_pairs, coverage.n_tasks, color="#7655B5", marker="D", markersize=5, linewidth=2)
    for row in coverage.itertuples(index=False):
        ax_coverage.annotate(str(row.n_tasks), (row.min_pairs, row.n_tasks), xytext=(0, 7),
                              textcoords="offset points", ha="center", color="#7655B5", fontsize=9)
    ax_coverage.set_ylabel("Eligible tasks")
    ax_coverage.set_xlabel("Pairs per tissue-H2 task")
    ax_coverage.set_ylim(0, int(coverage.n_tasks.max()) + 8)
    ax_coverage.set_xticks(coverage.min_pairs)
    ax_coverage.grid(axis="y", color="#D8D8D8", linewidth=0.7, alpha=0.75)
    png = args.figure_output_dir / "01_mousePMHC_phase3_pair_learning_curve.png"
    pdf = args.figure_output_dir / "01_mousePMHC_phase3_pair_learning_curve.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    (args.figure_output_dir / "figure_metadata.json").write_text(json.dumps({
        "figure": "mousePMHC Phase 3 pair learning curve", "summary_source": str(summary_path),
        "coverage_source": str(coverage_path),
        "cohort_policy": "Each point includes every tissue-H2 task eligible at that threshold; every included task is subsampled to the plotted pair count.",
        "outputs": [str(png), str(pdf)],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote: {png}")
    print(f"wrote: {pdf}")


def run(args: argparse.Namespace) -> None:
    if sorted(set(args.sample_sizes)) != args.sample_sizes:
        raise ValueError("--sample-sizes must be unique and increasing.")
    if args.summarize_only:
        detail_path = args.output_dir / "mousePMHC_phase3_pair_learning_curve_per_task.csv"
        coverage_path = args.output_dir / "mousePMHC_phase3_pair_threshold_coverage.csv"
        detail, coverage = pd.read_csv(detail_path), pd.read_csv(coverage_path)
        summary, recommendation = summarize(detail, args, coverage)
        summary.to_csv(args.output_dir / "mousePMHC_phase3_pair_learning_curve_summary.csv", index=False)
        metadata_path = args.output_dir / "mousePMHC_phase3_pair_learning_curve_recommendation.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["recommendation"] = recommendation
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        render_figure(args)
        print(summary.to_string(index=False)); print(json.dumps(recommendation, ensure_ascii=False, indent=2))
        return
    frame = read_pairs(args.input)
    pair_counts = frame.groupby("task_name").pair_id.nunique()
    eligible_tasks_by_size: dict[int, list[str]] = {}
    coverage_rows: list[dict[str, int]] = []
    for size in args.sample_sizes:
        eligible_tasks = sorted(pair_counts[pair_counts >= size].index)
        if args.max_tasks:
            eligible_tasks = eligible_tasks[:args.max_tasks]
        if len(eligible_tasks) < 2:
            raise ValueError(f"At least two eligible tasks are required at pairs_per_task={size}.")
        eligible_tasks_by_size[size] = eligible_tasks
        eligible_frame = frame[frame.task_name.isin(eligible_tasks)]
        coverage_rows.append({"min_pairs": size, "n_tasks": len(eligible_tasks),
            "n_tissues": int(eligible_frame.target_tissue.nunique()),
            "n_mhc": int(eligible_frame.mhc_restriction.nunique())})
    coverage = pd.DataFrame(coverage_rows)
    all_rows: list[dict[str, Any]] = []
    for size in args.sample_sizes:
        eligible_tasks = eligible_tasks_by_size[size]
        for repeat_seed in args.repeat_seeds:
            seed_started = time.perf_counter()
            sample = subsample_tasks(frame, eligible_tasks, size, repeat_seed)
            assignments = fold_utils.make_pair_grouped_folds(sample, args.oof_folds, repeat_seed)
            print(f"learning curve pairs={size} repeat={repeat_seed} tasks={len(eligible_tasks)}", flush=True)
            all_rows.extend(run_e0(sample, assignments, args, size, repeat_seed))
            all_rows.extend(run_e1(sample, assignments, args, size, repeat_seed))
            elapsed = time.perf_counter() - seed_started
            print(f"completed pairs={size} repeat={repeat_seed} elapsed={elapsed / 60:.1f} min ({elapsed:.1f} s)", flush=True)
    detail = pd.DataFrame(all_rows)
    summary, recommendation = summarize(detail, args, coverage)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_dir / "mousePMHC_phase3_pair_learning_curve_per_task.csv", index=False)
    summary.to_csv(args.output_dir / "mousePMHC_phase3_pair_learning_curve_summary.csv", index=False)
    coverage.to_csv(args.output_dir / "mousePMHC_phase3_pair_threshold_coverage.csv", index=False)
    metadata = {"experiment_name": EXPERIMENT, "test_data_read": False, "input": str(args.input),
        "cohort_policy": "At each pair-count point, include every task with at least that many available pairs; subsample every included task to exactly that many pairs.",
        "eligible_tasks_by_pairs_per_task": {str(size): tasks for size, tasks in eligible_tasks_by_size.items()},
        "sample_sizes": args.sample_sizes, "repeat_seeds": args.repeat_seeds,
        "oof_folds": args.oof_folds, "epochs": args.epochs, "models": MODELS, "recommendation": recommendation}
    (args.output_dir / "mousePMHC_phase3_pair_learning_curve_recommendation.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    render_figure(args)
    print(summary.to_string(index=False)); print(json.dumps(recommendation, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=project_path("data/processed/iedb_mouse_tissue_specificity_pairs.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=project_path("results/mousePMHC_phase3_pair_learning_curve"))
    parser.add_argument("--figure-output-dir", type=Path, default=project_path("results/figures_phase3"),
                        help="Directory for the figure generated automatically after the learning curve.")
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=[100, 150, 200, 250, 300, 350, 400])
    parser.add_argument("--repeat-seeds", nargs="+", type=int, default=[20260721, 20260722, 20260723],
                        help="Three repeated subsamples for screening; extend only the final candidate threshold if confirmation is needed.")
    parser.add_argument("--oof-folds", type=int, default=3); parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=25); parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3); parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=16); parser.add_argument("--hidden-dim", type=int, default=128); parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--minimum-tasks", type=int, default=10); parser.add_argument("--maximum-repeat-sd", type=float, default=0.03)
    parser.add_argument("--maximum-absolute-gain-to-next", type=float, default=0.01)
    parser.add_argument("--max-tasks", type=int, default=0,
                        help="Optional cap on eligible tasks per point for smoke tests; default 0 uses every eligible task.")
    parser.add_argument("--summarize-only", action="store_true", help="Recompute summaries and recommendation from existing detailed results.")
    return parser.parse_args()


if __name__ == "__main__": run(parse_args())
