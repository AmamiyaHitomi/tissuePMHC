#!/usr/bin/env python3
"""Run Phase 4 E12: H2/tissue auxiliary supervision on the matched E1 backbone.

E12 preserves E1's shared peptide encoder and task-specific binary heads.  It
adds two classifiers on the same representation, with fixed loss weights
lambda_H2=0.10 and lambda_tissue=0.02.  Development uses only pair-grouped
train-only OOF.  The script prints epoch, seed-fold, seed, and total elapsed
time to the terminal and writes no timing-specific result file.
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
EXPERIMENT = "mousePMHC_phase4_e12_auxiliary_shared_oof"
CANDIDATE = "mousePMHC_phase4_e12_h2_tissue_auxiliary_shared_task_heads"
BASELINE_CANDIDATE = "mousePMHC_phase3_e1_shared_peptide_encoder_task_heads"
KEYS = ["sample_id", "target_tissue", "mhc_restriction", "label"]
METRICS = ["accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "mcc"]


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def format_duration(seconds: float) -> str:
    minutes, remainder = divmod(seconds, 60.0)
    return f"{int(minutes)}m {remainder:04.1f}s" if minutes >= 1 else f"{remainder:.1f}s"


def validate_input(frame: pd.DataFrame) -> None:
    if set(frame.dataset) != {"mousePMHC"} or set(frame.split) != {"train"}:
        raise ValueError("E12 accepts mousePMHC training rows only.")
    if not frame.mhc_restriction.str.startswith("H2-").all():
        raise ValueError("E12 found a non-H2 MHC restriction.")


def arrays(frame: pd.DataFrame, peptide_length: int) -> list[np.ndarray]:
    return [
        base.encode_peptides(frame.peptide_sequence, peptide_length).copy(),
        frame.task_id.to_numpy(dtype=np.int64, copy=True),
        frame.tissue_id.to_numpy(dtype=np.int64, copy=True),
        frame.hla_id.to_numpy(dtype=np.int64, copy=True),
        frame.label.to_numpy(dtype=np.int64, copy=True),
    ]


def define_model(torch: Any, nn: Any, peptide_length: int, n_tasks: int, n_tissues: int, n_h2: int,
                 args: argparse.Namespace) -> Any:
    """Exact E1 encoder/head capacity plus two auxiliary classifiers."""

    class AuxiliarySharedTaskHeads(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(len(base.AA_TO_INDEX) + 1, args.embedding_dim, padding_idx=base.PAD_INDEX)
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(peptide_length * args.embedding_dim, args.hidden_dim),
                nn.ReLU(),
                nn.Dropout(args.dropout),
            )
            self.heads = nn.ModuleList([nn.Linear(args.hidden_dim, 1) for _ in range(n_tasks)])
            self.tissue_classifier = nn.Linear(args.hidden_dim, n_tissues)
            self.h2_classifier = nn.Linear(args.hidden_dim, n_h2)

        def encode(self, peptide_ids: Any) -> Any:
            return self.encoder(self.embedding(peptide_ids))

        def main_logits(self, encoded: Any, task_ids: Any) -> Any:
            logits = encoded.new_empty(encoded.shape[0])
            for task_id in torch.unique(task_ids):
                mask = task_ids == task_id
                logits[mask] = self.heads[int(task_id.item())](encoded[mask]).squeeze(-1)
            return logits

        def forward(self, peptide_ids: Any, task_ids: Any) -> Any:
            return self.main_logits(self.encode(peptide_ids), task_ids)

    return AuxiliarySharedTaskHeads()


def build_loader(torch: Any, DataLoader: Any, TensorDataset: Any, frame: pd.DataFrame,
                 peptide_length: int, batch_size: int, shuffle: bool) -> Any:
    dataset = TensorDataset(*[torch.as_tensor(value.copy()) for value in arrays(frame, peptide_length)])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def gradient_cosine(torch: Any, first_loss: Any, second_loss: Any, parameters: list[Any]) -> float:
    """Cosine of two loss gradients at the shared encoder, evaluated before update."""
    first = torch.autograd.grad(first_loss, parameters, retain_graph=True, allow_unused=True)
    second = torch.autograd.grad(second_loss, parameters, retain_graph=True, allow_unused=True)
    first_values = [value.detach().flatten() for value in first if value is not None]
    second_values = [value.detach().flatten() for value in second if value is not None]
    if not first_values or not second_values:
        return float("nan")
    first_vector = torch.cat(first_values)
    second_vector = torch.cat(second_values)
    denominator = torch.linalg.vector_norm(first_vector) * torch.linalg.vector_norm(second_vector)
    return float((torch.dot(first_vector, second_vector) / denominator.clamp_min(1e-12)).detach().cpu())


def train_model(args: argparse.Namespace, torch: Any, nn: Any, DataLoader: Any, TensorDataset: Any,
                fitting: pd.DataFrame, peptide_length: int, n_tasks: int, n_tissues: int, n_h2: int,
                device: str, seed: int, fold: int) -> tuple[Any, list[dict[str, object]]]:
    model = define_model(torch, nn, peptide_length, n_tasks, n_tissues, n_h2, args).to(device)
    loader = build_loader(torch, DataLoader, TensorDataset, fitting, peptide_length, args.batch_size, True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()
    cross_entropy = nn.CrossEntropyLoss()
    encoder_parameters = list(model.embedding.parameters()) + list(model.encoder.parameters())
    diagnostics: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        totals: list[float] = []; mains: list[float] = []; tissues: list[float] = []; h2s: list[float] = []
        tissue_accs: list[float] = []; h2_accs: list[float] = []
        gradient_values: dict[str, float] = {}
        for batch_index, batch in enumerate(loader):
            peptide_ids, task_ids, tissue_ids, h2_ids, labels = [item.to(device) for item in batch]
            optimizer.zero_grad(set_to_none=True)
            encoded = model.encode(peptide_ids)
            main_loss = bce(model.main_logits(encoded, task_ids), labels.float())
            tissue_logits = model.tissue_classifier(encoded)
            h2_logits = model.h2_classifier(encoded)
            tissue_loss = cross_entropy(tissue_logits, tissue_ids)
            h2_loss = cross_entropy(h2_logits, h2_ids)
            if batch_index == 0:
                weighted_aux = args.tissue_loss_weight * tissue_loss + args.h2_loss_weight * h2_loss
                gradient_values = {
                    "gradient_cosine_main_tissue": gradient_cosine(torch, main_loss, tissue_loss, encoder_parameters),
                    "gradient_cosine_main_h2": gradient_cosine(torch, main_loss, h2_loss, encoder_parameters),
                    "gradient_cosine_main_weighted_aux": gradient_cosine(torch, main_loss, weighted_aux, encoder_parameters),
                }
            total_loss = main_loss + args.tissue_loss_weight * tissue_loss + args.h2_loss_weight * h2_loss
            total_loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            totals.append(float(total_loss.detach().cpu())); mains.append(float(main_loss.detach().cpu()))
            tissues.append(float(tissue_loss.detach().cpu())); h2s.append(float(h2_loss.detach().cpu()))
            tissue_accs.append(float((tissue_logits.argmax(dim=1) == tissue_ids).float().mean().detach().cpu()))
            h2_accs.append(float((h2_logits.argmax(dim=1) == h2_ids).float().mean().detach().cpu()))
        diagnostics.append({
            "experiment_name": EXPERIMENT, "candidate": CANDIDATE, "seed": seed, "fold": fold,
            "epoch": epoch, "mean_total_loss": float(np.mean(totals)), "mean_main_bce_loss": float(np.mean(mains)),
            "mean_tissue_loss": float(np.mean(tissues)), "mean_h2_loss": float(np.mean(h2s)),
            "mean_tissue_accuracy": float(np.mean(tissue_accs)), "mean_h2_accuracy": float(np.mean(h2_accs)),
            "tissue_loss_weight": args.tissue_loss_weight, "h2_loss_weight": args.h2_loss_weight,
            **gradient_values,
        })
        print(
            f"E12 seed={seed} fold={fold + 1}/{args.oof_folds} epoch={epoch}/{args.epochs} "
            f"total={np.mean(totals):.5f} main_bce={np.mean(mains):.5f} "
            f"tissue_acc={np.mean(tissue_accs):.4f} h2_acc={np.mean(h2_accs):.4f} "
            f"elapsed={format_duration(time.perf_counter() - epoch_started)}",
            flush=True,
        )
    return model, diagnostics


def predict(torch: Any, DataLoader: Any, TensorDataset: Any, model: Any, held_out: pd.DataFrame,
            peptide_length: int, batch_size: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    loader = build_loader(torch, DataLoader, TensorDataset, held_out, peptide_length, batch_size, False)
    labels: list[np.ndarray] = []; scores: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for peptide_ids, task_ids, _, _, batch_labels in loader:
            logits = model(peptide_ids.to(device), task_ids.to(device))
            labels.append(batch_labels.numpy()); scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(labels).astype(np.int64), np.concatenate(scores).astype(float)


def metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    for (seed, tissue, h2), task in predictions.groupby(["seed", "target_tissue", "mhc_restriction"], sort=True):
        records.append({"experiment_name": EXPERIMENT, "candidate": CANDIDATE, "seed": int(seed),
                        "target_tissue": tissue, "mhc_restriction": h2, "oof_rows": len(task),
                        **base.evaluate(task.label.to_numpy(dtype=int), task.score.to_numpy(dtype=float))})
    per_task = pd.DataFrame(records); summary_rows: list[dict[str, object]] = []
    for seed, tasks in per_task.groupby("seed", sort=True):
        row: dict[str, object] = {"experiment_name": EXPERIMENT, "candidate": CANDIDATE, "seed": int(seed)}
        for metric in METRICS:
            row[f"mean_task_{metric}"] = float(tasks[metric].mean())
        row["worst_task_auroc"] = float(tasks.auroc.min()); row["worst6_task_auroc"] = float(tasks.nsmallest(6, "auroc").auroc.mean())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows); stability_rows: list[dict[str, object]] = []
    for metric in [column for column in summary if column.startswith("mean_task_") or column.startswith("worst")]:
        stability_rows.append({"experiment_name": EXPERIMENT, "candidate": CANDIDATE, "metric": metric,
                               "n_independent_seeds": len(summary), "seed_mean": float(summary[metric].mean()),
                               "seed_sd": float(summary[metric].std(ddof=1)), "seed_min": float(summary[metric].min()),
                               "seed_max": float(summary[metric].max())})
    return per_task, summary, pd.DataFrame(stability_rows)


def load_matched_e1(paths: list[Path], seeds: list[int], expected: pd.DataFrame) -> pd.DataFrame:
    baseline = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    required = {"split", "candidate", "seed", *KEYS, "score"}
    missing = required - set(baseline.columns)
    if missing:
        raise ValueError(f"E12 E1 source is missing columns: {sorted(missing)}")
    baseline = baseline[(baseline.split == "oof") & (baseline.candidate == BASELINE_CANDIDATE) & baseline.seed.isin(seeds)].copy()
    if baseline.duplicated(["seed", "sample_id"]).any():
        raise AssertionError("E12 E1 source has duplicate seed/sample predictions.")
    for seed, candidate_rows in expected.groupby("seed", sort=True):
        source_rows = baseline[baseline.seed == seed]
        merged = candidate_rows[KEYS].merge(source_rows[KEYS + ["score"]], on=KEYS, how="outer", indicator=True)
        if len(merged) != len(candidate_rows) or not (merged._merge == "both").all():
            raise AssertionError(f"E12 E1 seed {seed} does not align exactly with E12 OOF rows.")
    return baseline


def matched_comparison(candidate_per_task: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (seed, tissue, h2), task in baseline.groupby(["seed", "target_tissue", "mhc_restriction"], sort=True):
        e1_metrics = base.evaluate(task.label.to_numpy(dtype=int), task.score.to_numpy(dtype=float))
        e12 = candidate_per_task[(candidate_per_task.seed == seed) & (candidate_per_task.target_tissue == tissue) & (candidate_per_task.mhc_restriction == h2)]
        if len(e12) != 1:
            raise AssertionError("E12 cannot find one matched candidate task metric.")
        row: dict[str, object] = {"seed": int(seed), "target_tissue": tissue, "mhc_restriction": h2, "oof_rows": len(task)}
        for metric in METRICS:
            row[f"e1_{metric}"] = float(e1_metrics[metric]); row[f"e12_{metric}"] = float(e12.iloc[0][metric])
            row[f"delta_{metric}"] = row[f"e12_{metric}"] - row[f"e1_{metric}"]
        rows.append(row)
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    total_started = time.perf_counter()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("E12 seeds must be unique.")
    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    raw = base.read_dataset(args.train); validate_input(raw)
    train, _, mappings = base.add_task_columns(raw, raw.copy())
    peptide_length = int(train.peptide_sequence.str.len().iloc[0])
    assignments = folds.make_pair_grouped_folds(train, args.oof_folds, args.oof_split_seed)
    parts: list[pd.DataFrame] = []; diagnostic_rows: list[dict[str, object]] = []; parameter_count: int | None = None
    for seed in args.seeds:
        seed_started = time.perf_counter()
        for fold in range(args.oof_folds):
            fold_started = time.perf_counter(); fitting, held_out = train[assignments != fold].copy(), train[assignments == fold].copy()
            base.set_seed(int(seed), torch)
            print(f"E12 seed={seed} fold={fold + 1}/{args.oof_folds} start fit_rows={len(fitting)} holdout_rows={len(held_out)} device={device}", flush=True)
            model, diagnostics = train_model(args, torch, nn, DataLoader, TensorDataset, fitting, peptide_length,
                                             len(mappings["tasks"]), len(mappings["tissue_to_id"]), len(mappings["hla_to_id"]), device, int(seed), fold)
            diagnostic_rows.extend(diagnostics)
            if parameter_count is None:
                parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
            labels, scores = predict(torch, DataLoader, TensorDataset, model, held_out, peptide_length, args.batch_size, device)
            if not np.array_equal(labels, held_out.label.to_numpy(dtype=np.int64)):
                raise AssertionError("E12 OOF labels are not aligned with held-out rows.")
            output = held_out[[*KEYS[:-1], "label"]].copy(); output.insert(0, "split", "oof"); output.insert(1, "candidate", CANDIDATE); output.insert(2, "seed", int(seed)); output["score"] = scores
            parts.append(output[["split", "candidate", "seed", *KEYS, "score"]])
            print(f"E12 seed={seed} fold={fold + 1}/{args.oof_folds} complete elapsed={format_duration(time.perf_counter() - fold_started)}", flush=True)
        print(f"E12 seed={seed} complete elapsed={format_duration(time.perf_counter() - seed_started)}", flush=True)
    predictions = pd.concat(parts, ignore_index=True)
    if len(predictions) != len(train) * len(args.seeds) or predictions.duplicated(["seed", "sample_id"]).any():
        raise AssertionError("E12 OOF predictions must cover every training row exactly once per seed.")
    per_task, summary, stability = metric_tables(predictions)
    baseline = load_matched_e1(args.baseline_predictions, args.seeds, predictions)
    comparison = matched_comparison(per_task, baseline)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "mousePMHC_phase4_e12_oof_predictions.csv", index=False)
    per_task.to_csv(args.output_dir / "mousePMHC_phase4_e12_oof_per_task_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "mousePMHC_phase4_e12_oof_summary_metrics.csv", index=False)
    stability.to_csv(args.output_dir / "mousePMHC_phase4_e12_oof_stability_metrics.csv", index=False)
    comparison.to_csv(args.output_dir / "mousePMHC_phase4_e12_matched_e1_comparison.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(args.output_dir / "mousePMHC_phase4_e12_auxiliary_diagnostics.csv", index=False)
    metadata = {
        "experiment_name": EXPERIMENT, "candidate": CANDIDATE, "human_method_source": "tissuePMHC Phase 1 E13 auxiliary tissue/HLA supervision",
        "matched_baseline": BASELINE_CANDIDATE, "test_data_read": False, "train": str(args.train), "n_rows": len(train),
        "n_pairs": int(train.pair_id.nunique()), "n_tasks": len(mappings["tasks"]), "n_tissues": len(mappings["tissue_to_id"]), "n_h2": len(mappings["hla_to_id"]),
        "seeds": args.seeds, "oof_folds": args.oof_folds, "oof_split_seed": args.oof_split_seed, "device": device,
        "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "embedding_dim": args.embedding_dim, "hidden_dim": args.hidden_dim, "dropout": args.dropout, "max_grad_norm": args.max_grad_norm,
        "tissue_loss_weight": args.tissue_loss_weight, "h2_loss_weight": args.h2_loss_weight, "parameter_count": parameter_count,
        "architecture": "E1 shared Flatten-MLP encoder plus task heads, tissue classifier, and H2 classifier",
        "diagnostics": "epoch/fold auxiliary losses, auxiliary accuracies, and first-batch shared-encoder gradient cosines", "baseline_predictions": [str(path) for path in args.baseline_predictions],
    }
    (args.output_dir / "mousePMHC_phase4_e12_oof_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print("E12 OOF summary", flush=True); print(summary.to_string(index=False), flush=True)
    print(f"E12 total complete elapsed={format_duration(time.perf_counter() - total_started)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=project_path("data/mousePMHC/mousePMHC_train.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=project_path("results/mousePMHC_phase4_e12_auxiliary_shared_oof"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260704, 20260705, 20260706])
    parser.add_argument("--oof-folds", type=int, default=3); parser.add_argument("--oof-split-seed", type=int, default=20260711)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=25); parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3); parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=16); parser.add_argument("--hidden-dim", type=int, default=128); parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0); parser.add_argument("--tissue-loss-weight", type=float, default=0.02); parser.add_argument("--h2-loss-weight", type=float, default=0.10)
    parser.add_argument("--baseline-predictions", nargs="+", type=Path, default=[project_path("results/mousePMHC_phase3_e1_oof/mousePMHC_phase3_e1_oof_predictions.csv"), project_path("results/mousePMHC_phase3_e1_oof_additional_seeds/mousePMHC_phase3_e1_additional_oof_predictions.csv")])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
