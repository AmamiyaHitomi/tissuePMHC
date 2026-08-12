#!/usr/bin/env python3
"""Run Phase 4 E11: E8 global branch plus a plain H2-grouped OOF branch.

The global branch is the frozen E8 three-seed probability ensemble.  This
script trains only the missing plain H2-grouped branch using the Phase 3 E2
architecture on matched seeds and pair-grouped folds.  Both branches are
converted to percentile ranks within each tissue-H2 task, then combined using
the predeclared fixed 0.5/0.5 rank average.  Probability averaging is saved as
a fixed ablation.  No fixed test data are opened or scored.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_mousepmhc_phase3_e2_h2_grouped_oof as h2_model
import run_tissuepmhc_e26_all_in_one as folds
import run_tissuepmhc_neural_baselines_v2 as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "mousePMHC_phase4_e11_global_h2_soft_fusion_oof"
GLOBAL_CANDIDATE = "mousePMHC_phase4_e8_e3b_3seed_probability_mean"
H2_CANDIDATE = "mousePMHC_phase4_e11_plain_h2_grouped_3seed_probability_mean"
RANK_FUSION_CANDIDATE = "mousePMHC_phase4_e11_e8_global_h2_rank_average"
PROBABILITY_FUSION_CANDIDATE = "mousePMHC_phase4_e11_e8_global_h2_probability_average"
KEYS = ["sample_id", "target_tissue", "mhc_restriction", "label"]
METRICS = ["accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "mcc"]


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def format_duration(seconds: float) -> str:
    minutes, remainder = divmod(seconds, 60.0)
    return f"{int(minutes)}m {remainder:04.1f}s" if minutes >= 1 else f"{remainder:.1f}s"


def validate_train(frame: pd.DataFrame) -> None:
    h2_model.validate_input(frame)
    if frame.groupby("task_name").hla_id.nunique().ne(1).any():
        raise AssertionError("E11 requires every tissue-H2 task to have exactly one H2 restriction.")


def train_h2_model(args: argparse.Namespace, torch: Any, nn: Any, DataLoader: Any, TensorDataset: Any,
                   fitting: pd.DataFrame, peptide_length: int, n_tasks: int, n_h2: int,
                   device: str, seed: int, fold: int) -> Any:
    model = h2_model.define_model(nn, peptide_length, n_tasks, n_h2, args.embedding_dim, args.hidden_dim, args.dropout).to(device)
    loader = base.build_loader(torch, DataLoader, TensorDataset, h2_model.arrays(fitting, peptide_length), args.batch_size, True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_function = nn.BCEWithLogitsLoss()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        losses: list[float] = []
        for peptide_ids, task_ids, h2_ids, labels in loader:
            peptide_ids, task_ids, h2_ids, labels = (item.to(device) for item in (peptide_ids, task_ids, h2_ids, labels))
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(peptide_ids, task_ids, h2_ids), labels.float())
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(
            f"E11 H2 seed={seed} fold={fold + 1}/{args.oof_folds} epoch={epoch}/{args.epochs} "
            f"bce={np.mean(losses):.5f} elapsed={format_duration(time.perf_counter() - epoch_started)}",
            flush=True,
        )
    return model


def predict_h2(torch: Any, DataLoader: Any, TensorDataset: Any, model: Any, held_out: pd.DataFrame,
               peptide_length: int, batch_size: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    loader = base.build_loader(torch, DataLoader, TensorDataset, h2_model.arrays(held_out, peptide_length), batch_size, False)
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for peptide_ids, task_ids, h2_ids, batch_labels in loader:
            logits = model(peptide_ids.to(device), task_ids.to(device), h2_ids.to(device))
            labels.append(batch_labels.numpy())
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(labels).astype(np.int64), np.concatenate(scores).astype(float)


def h2_seed_metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    for (seed, tissue, h2), task in predictions.groupby(["seed", "target_tissue", "mhc_restriction"], sort=True):
        records.append({
            "experiment_name": EXPERIMENT,
            "candidate": H2_CANDIDATE,
            "seed": int(seed),
            "target_tissue": tissue,
            "mhc_restriction": h2,
            "oof_rows": len(task),
            **base.evaluate(task.label.to_numpy(dtype=int), task.score.to_numpy(dtype=float)),
        })
    per_task = pd.DataFrame(records)
    summary_rows: list[dict[str, object]] = []
    for seed, tasks in per_task.groupby("seed", sort=True):
        row: dict[str, object] = {"experiment_name": EXPERIMENT, "candidate": H2_CANDIDATE, "seed": int(seed)}
        for metric in METRICS:
            row[f"mean_task_{metric}"] = float(tasks[metric].mean())
        row["worst_task_auroc"] = float(tasks.auroc.min())
        row["worst6_task_auroc"] = float(tasks.nsmallest(6, "auroc").auroc.mean())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    stability_rows: list[dict[str, object]] = []
    for metric in [column for column in summary if column.startswith("mean_task_") or column.startswith("worst")]:
        stability_rows.append({
            "experiment_name": EXPERIMENT,
            "candidate": H2_CANDIDATE,
            "metric": metric,
            "n_independent_seeds": len(summary),
            "seed_mean": float(summary[metric].mean()),
            "seed_sd": float(summary[metric].std(ddof=1)) if len(summary) > 1 else float("nan"),
            "seed_min": float(summary[metric].min()),
            "seed_max": float(summary[metric].max()),
        })
    return per_task, summary, pd.DataFrame(stability_rows)


def load_global_predictions(path: Path, expected_keys: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"split", "candidate", *KEYS, "score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"E11 global source is missing columns: {sorted(missing)}")
    global_predictions = frame[(frame.split == "oof") & (frame.candidate == GLOBAL_CANDIDATE)].copy()
    if global_predictions.duplicated("sample_id").any():
        raise AssertionError("E11 E8 global source has duplicate sample_id predictions.")
    aligned = expected_keys.merge(
        global_predictions[KEYS + ["score"]], on=KEYS, how="outer", indicator=True, validate="one_to_one"
    )
    if len(aligned) != len(expected_keys) or not (aligned._merge == "both").all():
        raise AssertionError("E11 global E8 OOF predictions do not align exactly with H2 branch rows.")
    return aligned.drop(columns="_merge").rename(columns={"score": "global_score"})


def aggregate_h2_predictions(seed_predictions: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
    aggregate = seed_predictions.groupby(KEYS, as_index=False).agg(
        h2_score=("score", "mean"),
        h2_probability_std_across_seeds=("score", lambda values: float(np.std(values, ddof=0))),
        n_h2_members=("seed", "nunique"),
    )
    if not (aggregate.n_h2_members == len(seeds)).all():
        raise AssertionError("E11 H2 ensemble does not contain every requested seed for every sample.")
    return aggregate


def task_ranks(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby(["target_tissue", "mhc_restriction"], sort=False)[column].rank(method="average", pct=True)


def build_candidates(global_predictions: pd.DataFrame, h2_aggregate: pd.DataFrame) -> pd.DataFrame:
    merged = global_predictions.merge(h2_aggregate, on=KEYS, how="inner", validate="one_to_one")
    if len(merged) != len(global_predictions):
        raise AssertionError("E11 lost rows while aligning global and H2 ensemble predictions.")
    merged["global_rank"] = task_ranks(merged, "global_score")
    merged["h2_rank"] = task_ranks(merged, "h2_score")
    merged["rank_fusion_score"] = 0.5 * (merged.global_rank + merged.h2_rank)
    merged["probability_fusion_score"] = 0.5 * (merged.global_score + merged.h2_score)
    common = merged[KEYS + ["h2_probability_std_across_seeds", "n_h2_members"]].copy()
    outputs: list[pd.DataFrame] = []
    for candidate, score_column in [
        (GLOBAL_CANDIDATE, "global_score"),
        (H2_CANDIDATE, "h2_score"),
        (RANK_FUSION_CANDIDATE, "rank_fusion_score"),
        (PROBABILITY_FUSION_CANDIDATE, "probability_fusion_score"),
    ]:
        result = common.copy()
        result.insert(0, "split", "oof")
        result.insert(1, "candidate", candidate)
        result["score"] = merged[score_column]
        outputs.append(result)
    return pd.concat(outputs, ignore_index=True)[
        ["split", "candidate", *KEYS, "score", "h2_probability_std_across_seeds", "n_h2_members"]
    ]


def candidate_metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    for (candidate, tissue, h2), task in predictions.groupby(["candidate", "target_tissue", "mhc_restriction"], sort=True):
        records.append({
            "experiment_name": EXPERIMENT,
            "candidate": candidate,
            "target_tissue": tissue,
            "mhc_restriction": h2,
            "oof_rows": len(task),
            **base.evaluate(task.label.to_numpy(dtype=int), task.score.to_numpy(dtype=float)),
        })
    per_task = pd.DataFrame(records)
    rows: list[dict[str, object]] = []
    for candidate, tasks in per_task.groupby("candidate", sort=True):
        row: dict[str, object] = {"experiment_name": EXPERIMENT, "candidate": candidate}
        for metric in METRICS:
            row[f"mean_task_{metric}"] = float(tasks[metric].mean())
        row["worst_task_auroc"] = float(tasks.auroc.min())
        row["worst6_task_auroc"] = float(tasks.nsmallest(6, "auroc").auroc.mean())
        rows.append(row)
    return per_task, pd.DataFrame(rows)


def comparison_to_global(summary: pd.DataFrame) -> pd.DataFrame:
    baseline = summary[summary.candidate == GLOBAL_CANDIDATE]
    if len(baseline) != 1:
        raise AssertionError("E11 requires exactly one E8 global baseline summary.")
    baseline_row = baseline.iloc[0]
    rows: list[dict[str, object]] = []
    for _, candidate in summary.iterrows():
        if candidate.candidate == GLOBAL_CANDIDATE:
            continue
        row: dict[str, object] = {"experiment_name": EXPERIMENT, "candidate": candidate.candidate, "baseline": GLOBAL_CANDIDATE}
        for metric in [column for column in summary.columns if column.startswith("mean_task_") or column.startswith("worst")]:
            row[f"baseline_{metric}"] = float(baseline_row[metric])
            row[metric] = float(candidate[metric])
            row[f"delta_{metric}"] = float(candidate[metric] - baseline_row[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    total_started = time.perf_counter()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("E11 seeds must be unique.")
    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    raw = base.read_dataset(args.train)
    train, _, mappings = base.add_task_columns(raw, raw.copy())
    validate_train(train)
    peptide_length = int(train.peptide_sequence.str.len().iloc[0])
    assignments = folds.make_pair_grouped_folds(train, args.oof_folds, args.oof_split_seed)
    parts: list[pd.DataFrame] = []
    parameter_count: int | None = None
    for seed in args.seeds:
        seed_started = time.perf_counter()
        for fold in range(args.oof_folds):
            fold_started = time.perf_counter()
            fitting, held_out = train[assignments != fold].copy(), train[assignments == fold].copy()
            base.set_seed(int(seed), torch)
            print(
                f"E11 H2 seed={seed} fold={fold + 1}/{args.oof_folds} start "
                f"fit_rows={len(fitting)} holdout_rows={len(held_out)} device={device}",
                flush=True,
            )
            model = train_h2_model(args, torch, nn, DataLoader, TensorDataset, fitting, peptide_length,
                                   len(mappings["tasks"]), len(mappings["hla_to_id"]), device, int(seed), fold)
            if parameter_count is None:
                parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
            labels, scores = predict_h2(torch, DataLoader, TensorDataset, model, held_out, peptide_length, args.batch_size, device)
            if not np.array_equal(labels, held_out.label.to_numpy(dtype=np.int64)):
                raise AssertionError("E11 H2 OOF labels are not aligned with held-out rows.")
            output = held_out[[*KEYS[:-1], "label"]].copy()
            output.insert(0, "split", "oof")
            output.insert(1, "candidate", H2_CANDIDATE)
            output.insert(2, "seed", int(seed))
            output["score"] = scores
            parts.append(output[["split", "candidate", "seed", *KEYS, "score"]])
            print(
                f"E11 H2 seed={seed} fold={fold + 1}/{args.oof_folds} complete "
                f"elapsed={format_duration(time.perf_counter() - fold_started)}",
                flush=True,
            )
        print(f"E11 H2 seed={seed} complete elapsed={format_duration(time.perf_counter() - seed_started)}", flush=True)
    seed_predictions = pd.concat(parts, ignore_index=True)
    if len(seed_predictions) != len(train) * len(args.seeds) or seed_predictions.duplicated(["seed", "sample_id"]).any():
        raise AssertionError("E11 H2 OOF predictions must cover every training row exactly once per seed.")
    h2_per_task, h2_summary, h2_stability = h2_seed_metric_tables(seed_predictions)
    h2_aggregate = aggregate_h2_predictions(seed_predictions, args.seeds)
    expected_keys = h2_aggregate[KEYS].copy()
    global_predictions = load_global_predictions(args.global_predictions, expected_keys)
    candidate_predictions = build_candidates(global_predictions, h2_aggregate)
    candidate_per_task, candidate_summary = candidate_metric_tables(candidate_predictions)
    comparison = comparison_to_global(candidate_summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_predictions.to_csv(args.output_dir / "mousePMHC_phase4_e11_h2_seed_oof_predictions.csv", index=False)
    h2_per_task.to_csv(args.output_dir / "mousePMHC_phase4_e11_h2_seed_oof_per_task_metrics.csv", index=False)
    h2_summary.to_csv(args.output_dir / "mousePMHC_phase4_e11_h2_seed_oof_summary_metrics.csv", index=False)
    h2_stability.to_csv(args.output_dir / "mousePMHC_phase4_e11_h2_seed_oof_stability_metrics.csv", index=False)
    candidate_predictions.to_csv(args.output_dir / "mousePMHC_phase4_e11_oof_predictions.csv", index=False)
    candidate_per_task.to_csv(args.output_dir / "mousePMHC_phase4_e11_oof_per_task_metrics.csv", index=False)
    candidate_summary.to_csv(args.output_dir / "mousePMHC_phase4_e11_oof_summary_metrics.csv", index=False)
    comparison.to_csv(args.output_dir / "mousePMHC_phase4_e11_oof_comparison.csv", index=False)
    metadata = {
        "experiment_name": EXPERIMENT,
        "test_data_read": False,
        "global_source": str(args.global_predictions),
        "global_candidate": GLOBAL_CANDIDATE,
        "h2_branch_candidate": H2_CANDIDATE,
        "rank_fusion_candidate": RANK_FUSION_CANDIDATE,
        "probability_fusion_candidate": PROBABILITY_FUSION_CANDIDATE,
        "fusion": "within tissue-H2 percentile rank, then 0.5 * global + 0.5 * H2; probability mean retained as fixed ablation",
        "train": str(args.train),
        "n_rows": len(train),
        "n_pairs": int(train.pair_id.nunique()),
        "n_tasks": len(mappings["tasks"]),
        "n_h2": len(mappings["hla_to_id"]),
        "seeds": args.seeds,
        "oof_folds": args.oof_folds,
        "oof_split_seed": args.oof_split_seed,
        "device": device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "max_grad_norm": args.max_grad_norm,
        "h2_branch_parameter_count": parameter_count,
        "h2_branch_architecture": "one Flatten-MLP peptide encoder per H2 restriction and task-specific linear heads",
        "selection_policy": "fixed equal branch weights; no per-task weight search, no fixed-test access",
    }
    (args.output_dir / "mousePMHC_phase4_e11_oof_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("E11 candidate OOF summary", flush=True)
    print(candidate_summary.to_string(index=False), flush=True)
    print("E11 comparison to frozen E8 global branch", flush=True)
    print(comparison.to_string(index=False), flush=True)
    print(f"E11 total complete elapsed={format_duration(time.perf_counter() - total_started)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=project_path("data/mousePMHC/mousePMHC_train.csv.gz"))
    parser.add_argument(
        "--global-predictions", type=Path,
        default=project_path("results/mousePMHC_phase4_e8_e3b_seed_ensemble_oof/mousePMHC_phase4_e8_oof_predictions.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=project_path("results/mousePMHC_phase4_e11_global_h2_soft_fusion_oof"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260704, 20260705, 20260706])
    parser.add_argument("--oof-folds", type=int, default=3)
    parser.add_argument("--oof-split-seed", type=int, default=20260711)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
