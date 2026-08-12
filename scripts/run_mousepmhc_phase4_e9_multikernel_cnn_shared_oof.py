#!/usr/bin/env python3
"""Run Phase 4 E9: multi-kernel CNN shared task heads on mousePMHC train-only OOF.

E9 is the isolated encoder-transfer experiment.  It keeps the E1/human-E2
shared task heads, sample-level BCE, AdamW, pair-grouped OOF folds, and
training schedule.  Only the peptide encoder changes from Flatten-MLP to a
position-preserving multi-kernel Conv1d encoder (kernels 2, 3, and 5).

The fixed test file is never opened.  Each epoch, seed-fold, seed, and the
complete run report elapsed time to the terminal; no timing output file is
written.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_tissuepmhc_e26_all_in_one as folds
import run_tissuepmhc_neural_baselines_v2 as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "mousePMHC_phase4_e9_multikernel_cnn_shared_oof"
CANDIDATE = "mousePMHC_phase4_e9_multikernel_cnn_shared_task_heads"
BASELINE_CANDIDATE = "mousePMHC_phase3_e1_shared_peptide_encoder_task_heads"
KEYS = ["sample_id", "target_tissue", "mhc_restriction"]
METRICS = ["accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "mcc"]


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def format_duration(seconds: float) -> str:
    minutes, remainder = divmod(seconds, 60.0)
    return f"{int(minutes)}m {remainder:04.1f}s" if minutes >= 1 else f"{remainder:.1f}s"


def validate_input(frame: pd.DataFrame) -> None:
    if set(frame.dataset) != {"mousePMHC"} or set(frame.split) != {"train"}:
        raise ValueError("E9 accepts mousePMHC training rows only.")
    if not frame.mhc_restriction.str.startswith("H2-").all():
        raise ValueError("E9 found a non-H2 MHC restriction.")
    peptide_lengths = frame.peptide_sequence.str.len().unique()
    if len(peptide_lengths) != 1:
        raise ValueError(f"E9 requires a fixed peptide length, found {sorted(peptide_lengths)}.")


def arrays(frame: pd.DataFrame, peptide_length: int) -> list[np.ndarray]:
    return [
        base.encode_peptides(frame.peptide_sequence, peptide_length).copy(),
        frame.task_id.to_numpy(dtype=np.int64, copy=True),
        frame.label.to_numpy(dtype=np.int64, copy=True),
    ]


def define_model(torch: Any, nn: Any, peptide_length: int, n_tasks: int, args: argparse.Namespace) -> Any:
    """Build an E1-compatible task-head model with only its encoder replaced."""

    class MultiKernelCnnSharedTaskHeads(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.peptide_length = peptide_length
            self.embedding = nn.Embedding(
                len(base.AA_TO_INDEX) + 1,
                args.embedding_dim,
                padding_idx=base.PAD_INDEX,
            )
            self.convolutions = nn.ModuleList([
                nn.Conv1d(args.embedding_dim, args.conv_channels, kernel_size=kernel, padding=kernel // 2)
                for kernel in args.kernel_sizes
            ])
            encoded_dim = peptide_length * args.conv_channels * len(args.kernel_sizes)
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(encoded_dim, args.hidden_dim),
                nn.ReLU(),
                nn.Dropout(args.dropout),
            )
            self.heads = nn.ModuleList([nn.Linear(args.hidden_dim, 1) for _ in range(n_tasks)])

        def encode(self, peptide_ids: Any) -> Any:
            channels_first = self.embedding(peptide_ids).transpose(1, 2)
            features: list[Any] = []
            for convolution in self.convolutions:
                # Symmetric padding with an even kernel creates one extra slot.
                # Crop it so every kernel retains exactly the original 9 positions.
                convolved = torch.relu(convolution(channels_first))[..., :self.peptide_length]
                features.append(convolved.transpose(1, 2))
            return self.encoder(torch.cat(features, dim=2))

        def forward(self, peptide_ids: Any, task_ids: Any) -> Any:
            encoded = self.encode(peptide_ids)
            logits = encoded.new_empty(encoded.shape[0])
            for task_id in torch.unique(task_ids):
                mask = task_ids == task_id
                logits[mask] = self.heads[int(task_id.item())](encoded[mask]).squeeze(-1)
            return logits

    return MultiKernelCnnSharedTaskHeads()


def build_loader(torch: Any, DataLoader: Any, TensorDataset: Any, frame: pd.DataFrame,
                 peptide_length: int, batch_size: int, shuffle: bool) -> Any:
    encoded = arrays(frame, peptide_length)
    dataset = TensorDataset(*[torch.as_tensor(values.copy()) for values in encoded])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(args: argparse.Namespace, torch: Any, nn: Any, DataLoader: Any, TensorDataset: Any,
                fitting: pd.DataFrame, peptide_length: int, n_tasks: int, device: str,
                seed: int, fold: int) -> Any:
    model = define_model(torch, nn, peptide_length, n_tasks, args).to(device)
    loader = build_loader(torch, DataLoader, TensorDataset, fitting, peptide_length, args.batch_size, True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_function = nn.BCEWithLogitsLoss()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        losses: list[float] = []
        for peptide_ids, task_ids, labels in loader:
            peptide_ids, task_ids, labels = peptide_ids.to(device), task_ids.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(peptide_ids, task_ids), labels.float())
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(
            f"E9 seed={seed} fold={fold + 1}/{args.oof_folds} epoch={epoch}/{args.epochs} "
            f"bce={np.mean(losses):.5f} elapsed={format_duration(time.perf_counter() - epoch_started)}",
            flush=True,
        )
    return model


def predict(torch: Any, DataLoader: Any, TensorDataset: Any, model: Any, held_out: pd.DataFrame,
            peptide_length: int, batch_size: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    loader = build_loader(torch, DataLoader, TensorDataset, held_out, peptide_length, batch_size, False)
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for peptide_ids, task_ids, batch_labels in loader:
            logits = model(peptide_ids.to(device), task_ids.to(device))
            labels.append(batch_labels.numpy())
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(labels).astype(np.int64), np.concatenate(scores).astype(float)


def metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    for (seed, tissue, h2), task in predictions.groupby(["seed", "target_tissue", "mhc_restriction"], sort=True):
        records.append({
            "experiment_name": EXPERIMENT,
            "candidate": CANDIDATE,
            "seed": int(seed),
            "target_tissue": tissue,
            "mhc_restriction": h2,
            "oof_rows": len(task),
            **base.evaluate(task.label.to_numpy(dtype=int), task.score.to_numpy(dtype=float)),
        })
    per_task = pd.DataFrame(records)
    summary_rows: list[dict[str, object]] = []
    for seed, tasks in per_task.groupby("seed", sort=True):
        row: dict[str, object] = {"experiment_name": EXPERIMENT, "candidate": CANDIDATE, "seed": int(seed)}
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
            "candidate": CANDIDATE,
            "metric": metric,
            "n_independent_seeds": len(summary),
            "seed_mean": float(summary[metric].mean()),
            "seed_sd": float(summary[metric].std(ddof=1)),
            "seed_min": float(summary[metric].min()),
            "seed_max": float(summary[metric].max()),
        })
    return per_task, summary, pd.DataFrame(stability_rows)


def load_matched_baseline(paths: list[Path], seeds: list[int], expected: pd.DataFrame) -> pd.DataFrame:
    parts = [pd.read_csv(path) for path in paths]
    baseline = pd.concat(parts, ignore_index=True)
    required = {"split", "candidate", "seed", *KEYS, "label", "score"}
    missing = required - set(baseline.columns)
    if missing:
        raise ValueError(f"E9 matched E1 source lacks columns: {sorted(missing)}")
    if set(baseline.candidate.dropna().unique()) != {BASELINE_CANDIDATE}:
        raise ValueError("E9 matched-baseline source is not the frozen E1 candidate.")
    baseline = baseline[(baseline.split == "oof") & baseline.seed.isin(seeds)].copy()
    if baseline.duplicated(["seed", "sample_id"]).any():
        raise AssertionError("E9 matched E1 sources contain duplicate seed/sample predictions.")
    for seed, candidate_rows in expected.groupby("seed", sort=True):
        actual = baseline[baseline.seed == seed]
        merged = candidate_rows[["sample_id", "target_tissue", "mhc_restriction", "label"]].merge(
            actual[["sample_id", "target_tissue", "mhc_restriction", "label", "score"]],
            on=["sample_id", "target_tissue", "mhc_restriction", "label"], how="outer", indicator=True,
        )
        if len(merged) != len(candidate_rows) or not (merged._merge == "both").all():
            raise AssertionError(f"E9 matched E1 seed {seed} does not align exactly with E9 OOF rows.")
    return baseline


def matched_comparison(e9_per_task: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (seed, tissue, h2), task in baseline.groupby(["seed", "target_tissue", "mhc_restriction"], sort=True):
        values = base.evaluate(task.label.to_numpy(dtype=int), task.score.to_numpy(dtype=float))
        e9 = e9_per_task[
            (e9_per_task.seed == seed)
            & (e9_per_task.target_tissue == tissue)
            & (e9_per_task.mhc_restriction == h2)
        ]
        if len(e9) != 1:
            raise AssertionError("E9 matched comparison cannot find exactly one CNN task result.")
        row: dict[str, object] = {"seed": int(seed), "target_tissue": tissue, "mhc_restriction": h2, "oof_rows": len(task)}
        for metric in METRICS:
            row[f"e1_{metric}"] = float(values[metric])
            row[f"e9_{metric}"] = float(e9.iloc[0][metric])
            row[f"delta_{metric}"] = row[f"e9_{metric}"] - row[f"e1_{metric}"]
        rows.append(row)
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    total_started = time.perf_counter()
    if not args.kernel_sizes or any(kernel < 1 for kernel in args.kernel_sizes):
        raise ValueError("--kernel-sizes must contain positive integers.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("E9 seeds must be unique.")
    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    raw = base.read_dataset(args.train)
    validate_input(raw)
    train, _, mappings = base.add_task_columns(raw, raw.copy())
    if args.max_tasks:
        keep = set(mappings["tasks"][:args.max_tasks])
        train = train[train.task_name.isin(keep)].copy()
        train, _, mappings = base.add_task_columns(train, train.copy())
    peptide_length = int(train.peptide_sequence.str.len().iloc[0])
    assignments = folds.make_pair_grouped_folds(train, args.oof_folds, args.oof_split_seed)
    prediction_parts: list[pd.DataFrame] = []
    parameter_count: int | None = None
    for seed in args.seeds:
        seed_started = time.perf_counter()
        for fold in range(args.oof_folds):
            fold_started = time.perf_counter()
            fitting, held_out = train[assignments != fold].copy(), train[assignments == fold].copy()
            base.set_seed(int(seed), torch)
            print(
                f"E9 seed={seed} fold={fold + 1}/{args.oof_folds} start "
                f"fit_rows={len(fitting)} holdout_rows={len(held_out)} device={device}",
                flush=True,
            )
            model = train_model(args, torch, nn, DataLoader, TensorDataset, fitting, peptide_length,
                                len(mappings["tasks"]), device, int(seed), fold)
            if parameter_count is None:
                parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
            labels, scores = predict(torch, DataLoader, TensorDataset, model, held_out, peptide_length, args.batch_size, device)
            if not np.array_equal(labels, held_out.label.to_numpy(dtype=np.int64)):
                raise AssertionError("E9 OOF labels are not aligned with held-out rows.")
            output = held_out[KEYS + ["label"]].copy()
            output.insert(0, "split", "oof")
            output.insert(1, "candidate", CANDIDATE)
            output.insert(2, "seed", int(seed))
            output["score"] = scores
            prediction_parts.append(output[["split", "candidate", "seed", *KEYS, "label", "score"]])
            print(
                f"E9 seed={seed} fold={fold + 1}/{args.oof_folds} complete "
                f"elapsed={format_duration(time.perf_counter() - fold_started)}",
                flush=True,
            )
        print(f"E9 seed={seed} complete elapsed={format_duration(time.perf_counter() - seed_started)}", flush=True)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    expected_rows = len(train) * len(args.seeds)
    if len(predictions) != expected_rows or predictions.duplicated(["seed", "sample_id"]).any():
        raise AssertionError("E9 OOF predictions must cover every training row exactly once per seed.")
    per_task, summary, stability = metric_tables(predictions)
    comparison: pd.DataFrame | None = None
    if not args.skip_baseline_comparison:
        if args.max_tasks:
            raise ValueError("--max-tasks requires --skip-baseline-comparison because it is not the frozen 24-task benchmark.")
        baseline = load_matched_baseline(args.baseline_predictions, args.seeds, predictions)
        comparison = matched_comparison(per_task, baseline)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "mousePMHC_phase4_e9_oof_predictions.csv", index=False)
    per_task.to_csv(args.output_dir / "mousePMHC_phase4_e9_oof_per_task_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "mousePMHC_phase4_e9_oof_summary_metrics.csv", index=False)
    stability.to_csv(args.output_dir / "mousePMHC_phase4_e9_oof_stability_metrics.csv", index=False)
    if comparison is not None:
        comparison.to_csv(args.output_dir / "mousePMHC_phase4_e9_matched_e1_comparison.csv", index=False)
    metadata = {
        "experiment_name": EXPERIMENT,
        "candidate": CANDIDATE,
        "human_method_source": "tissuePMHC Phase 2 E29 multi-kernel CNN peptide encoder",
        "matched_baseline": BASELINE_CANDIDATE,
        "test_data_read": False,
        "train": str(args.train),
        "n_rows": len(train),
        "n_pairs": int(train.pair_id.nunique()),
        "n_tasks": len(mappings["tasks"]),
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
        "kernel_sizes": args.kernel_sizes,
        "conv_channels": args.conv_channels,
        "parameter_count": parameter_count,
        "encoder": "Embedding -> Conv1d(k=2,3,5) -> crop-to-length -> concatenate positions -> Flatten/MLP",
        "training": "E1-compatible sample-level BCE; no task balancing, auxiliary loss, fusion, or fixed-test access",
        "baseline_predictions": [str(path) for path in args.baseline_predictions],
        "baseline_comparison_written": comparison is not None,
    }
    (args.output_dir / "mousePMHC_phase4_e9_oof_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("E9 OOF summary", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"E9 total complete elapsed={format_duration(time.perf_counter() - total_started)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=project_path("data/mousePMHC/mousePMHC_train.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=project_path("results/mousePMHC_phase4_e9_multikernel_cnn_shared_oof"))
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
    parser.add_argument("--kernel-sizes", nargs="+", type=int, default=[2, 3, 5])
    parser.add_argument("--conv-channels", type=int, default=32)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--baseline-predictions", nargs="+", type=Path,
        default=[
            project_path("results/mousePMHC_phase3_e1_oof/mousePMHC_phase3_e1_oof_predictions.csv"),
            project_path("results/mousePMHC_phase3_e1_oof_additional_seeds/mousePMHC_phase3_e1_additional_oof_predictions.csv"),
        ],
    )
    parser.add_argument("--skip-baseline-comparison", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=0, help="Smoke-test only; incompatible with matched E1 comparison.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
