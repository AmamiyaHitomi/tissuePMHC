#!/usr/bin/env python3
"""Run E10 MMoE selective sharing on the E2/E8 performance line.

Roadmap role: E10-on-E2/E8 performance line.

MMoE means Multi-gate Mixture-of-Experts. In this script, each tissue-HLA task
has its own gate over several shared peptide experts:

    peptide encoder -> shared experts -> task gate -> task-specific head

This tests whether the sharing structure discovered manually by E6/E7/E8 can be
learned automatically by a selective-sharing neural model.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_tissuepmhc_neural_baselines_v2 as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPEAT_SEEDS = [20260704, 20260705, 20260706]
MODEL_NAME = "e10_mmoe"
EXPERIMENT_NAME = "E10_MMoE"
METRICS = ["accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "mcc"]

GATE_COLUMNS = [
    "experiment_name",
    "seed",
    "model",
    "epoch",
    "task_name",
    "target_tissue",
    "mhc_restriction",
    "expert_id",
    "gate_weight",
]

COMPARISON_COLUMNS = [
    "seed",
    "target_tissue",
    "mhc_restriction",
    "baseline_model",
    "candidate_model",
    "baseline_source",
    "delta_accuracy",
    "delta_balanced_accuracy",
    "delta_auroc",
    "delta_auprc",
    "delta_f1",
    "delta_mcc",
]


def project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


def set_seed(seed: int, torch: Any) -> None:
    base.set_seed(seed, torch)
    np.random.seed(seed)


def define_mmoe_model(args: argparse.Namespace, torch: Any, nn: Any, peptide_length: int, n_tasks: int) -> Any:
    class MMoEModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(len(base.AA_TO_INDEX) + 1, args.embedding_dim, padding_idx=base.PAD_INDEX)
            input_dim = peptide_length * args.embedding_dim
            self.shared_input = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, args.hidden_dim),
                nn.ReLU(),
                nn.Dropout(args.dropout),
            )
            self.experts = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(args.hidden_dim, args.expert_dim),
                        nn.ReLU(),
                        nn.Dropout(args.dropout),
                        nn.Linear(args.expert_dim, args.expert_dim),
                        nn.ReLU(),
                    )
                    for _ in range(args.n_experts)
                ]
            )
            self.task_gates = nn.Embedding(n_tasks, args.n_experts)
            self.heads = nn.ModuleList([nn.Linear(args.expert_dim, 1) for _ in range(n_tasks)])

        def expert_features(self, peptide_ids: Any) -> Any:
            shared = self.shared_input(self.embedding(peptide_ids))
            return torch.stack([expert(shared) for expert in self.experts], dim=1)

        def gate_weights(self, task_ids: Any) -> Any:
            return torch.softmax(self.task_gates(task_ids), dim=1)

        def forward(self, peptide_ids: Any, task_ids: Any) -> Any:
            expert_outputs = self.expert_features(peptide_ids)
            gates = self.gate_weights(task_ids)
            mixed = (expert_outputs * gates.unsqueeze(-1)).sum(dim=1)
            logits = mixed.new_empty(mixed.shape[0])
            for task_id in torch.unique(task_ids):
                mask = task_ids == task_id
                logits[mask] = self.heads[int(task_id.item())](mixed[mask]).squeeze(-1)
            return logits

        def all_task_gate_weights(self) -> Any:
            task_ids = torch.arange(n_tasks, device=self.task_gates.weight.device)
            return self.gate_weights(task_ids)

    return MMoEModel()


def build_loader(
    args: argparse.Namespace,
    torch: Any,
    DataLoader: Any,
    TensorDataset: Any,
    df: pd.DataFrame,
    peptide_length: int,
    shuffle: bool,
) -> Any:
    x = base.encode_peptides(df["peptide_sequence"], peptide_length)
    task_ids = df["task_id"].to_numpy(dtype=np.int64).copy()
    labels = df["label"].to_numpy(dtype=np.int64).copy()
    return base.build_loader(torch, DataLoader, TensorDataset, [x, task_ids, labels], args.batch_size, shuffle)


def train_mmoe(
    args: argparse.Namespace,
    torch: Any,
    nn: Any,
    DataLoader: Any,
    TensorDataset: Any,
    train_df: pd.DataFrame,
    mappings: dict[str, Any],
    peptide_length: int,
    device: str,
    seed: int,
) -> tuple[Any, list[dict[str, object]]]:
    model = define_mmoe_model(args, torch, nn, peptide_length, len(mappings["tasks"])).to(device)
    loader = build_loader(args, torch, DataLoader, TensorDataset, train_df, peptide_length, True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    gate_rows: list[dict[str, object]] = []

    print(f"  train setup: epochs={args.epochs} batches_per_epoch={len(loader)}")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        losses = []
        for batch in loader:
            peptide_ids, task_ids, labels = [item.to(device) for item in batch]
            optimizer.zero_grad(set_to_none=True)
            logits = model(peptide_ids, task_ids)
            loss = loss_fn(logits, labels.float())
            if args.gate_entropy_weight != 0:
                gates = model.gate_weights(task_ids)
                gate_entropy = -(gates * torch.log(gates.clamp_min(1e-12))).sum(dim=1).mean()
                loss = loss - args.gate_entropy_weight * gate_entropy
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        with torch.no_grad():
            gates = model.all_task_gate_weights().detach().cpu().numpy()
        for task_name in mappings["tasks"]:
            tissue, hla = task_name.split("||", 1)
            task_id = int(mappings["task_to_id"][task_name])
            for expert_id, gate_weight in enumerate(gates[task_id]):
                gate_rows.append(
                    {
                        "experiment_name": args.experiment_name,
                        "seed": seed,
                        "model": args.model_name,
                        "epoch": epoch,
                        "task_name": task_name,
                        "target_tissue": tissue,
                        "mhc_restriction": hla,
                        "expert_id": expert_id,
                        "gate_weight": float(gate_weight),
                    }
                )

        print(
            f"  epoch {epoch:02d}/{args.epochs} "
            f"time={time.perf_counter() - epoch_start:.2f}s "
            f"mean_loss={float(np.mean(losses)):.4f}"
        )

    return model, gate_rows


def evaluate_mmoe(
    args: argparse.Namespace,
    torch: Any,
    DataLoader: Any,
    TensorDataset: Any,
    model: Any,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mappings: dict[str, Any],
    peptide_length: int,
    device: str,
) -> list[dict[str, object]]:
    rows = []
    model.eval()
    for task_name in mappings["tasks"]:
        train_task = train_df[train_df["task_name"] == task_name]
        test_task = test_df[test_df["task_name"] == task_name]
        loader = build_loader(args, torch, DataLoader, TensorDataset, test_task, peptide_length, False)
        scores = []
        labels = []
        with torch.no_grad():
            for batch in loader:
                peptide_ids, task_ids, y = [item.to(device) for item in batch]
                logits = model(peptide_ids, task_ids)
                scores.append(torch.sigmoid(logits).cpu().numpy())
                labels.append(y.cpu().numpy())
        y_true = np.concatenate(labels)
        y_score = np.concatenate(scores)
        rows.append(base.make_metric_row(args.model_name, train_task, test_task, base.evaluate(y_true, y_score)))
    return rows


def add_experiment_context(rows: list[dict[str, object]], seed: int, experiment_name: str) -> list[dict[str, object]]:
    for row in rows:
        row["experiment_name"] = experiment_name
        row["seed"] = seed
    return rows


def read_external_rows(path: Path, model: str) -> dict[tuple[int, str, str], dict[str, object]]:
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    if "model" in table.columns:
        table = table[table["model"] == model].copy()
    return {
        (int(row["seed"]), str(row["target_tissue"]), str(row["mhc_restriction"])): row
        for row in table.to_dict("records")
    }


def compare_external(
    rows: list[dict[str, object]],
    baseline_rows: dict[tuple[int, str, str], dict[str, object]],
    baseline_model: str,
    baseline_source: str,
) -> list[dict[str, object]]:
    comparisons = []
    for row in rows:
        key = (int(row["seed"]), str(row["target_tissue"]), str(row["mhc_restriction"]))
        baseline = baseline_rows.get(key)
        if baseline is None:
            continue
        comparison = {
            "seed": key[0],
            "target_tissue": key[1],
            "mhc_restriction": key[2],
            "baseline_model": baseline_model,
            "candidate_model": str(row["model"]),
            "baseline_source": baseline_source,
        }
        for metric in METRICS:
            comparison[f"delta_{metric}"] = float(row[metric]) - float(baseline[metric])
        comparisons.append(comparison)
    return comparisons


def run_one_seed(
    args: argparse.Namespace,
    torch: Any,
    nn: Any,
    DataLoader: Any,
    TensorDataset: Any,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mappings: dict[str, Any],
    peptide_length: int,
    device: str,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    set_seed(seed, torch)
    seed_start = time.perf_counter()
    print(f"experiment: {args.experiment_name} seed={seed} model={args.model_name}")
    model, gate_rows = train_mmoe(
        args,
        torch,
        nn,
        DataLoader,
        TensorDataset,
        train_df,
        mappings,
        peptide_length,
        device,
        seed,
    )
    metric_rows = evaluate_mmoe(
        args,
        torch,
        DataLoader,
        TensorDataset,
        model,
        train_df,
        test_df,
        mappings,
        peptide_length,
        device,
    )
    print(f"seed/model time: seed={seed} model={args.model_name} time={time.perf_counter() - seed_start:.2f}s")
    return add_experiment_context(metric_rows, seed, args.experiment_name), gate_rows


def run(args: argparse.Namespace) -> None:
    run_start = time.perf_counter()
    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_df = base.read_dataset(args.train)
    test_df = base.read_dataset(args.test)
    train_df, test_df, mappings = base.add_task_columns(train_df, test_df)
    if args.max_tasks:
        keep_tasks = set(mappings["tasks"][: args.max_tasks])
        train_df = train_df[train_df["task_name"].isin(keep_tasks)].copy()
        test_df = test_df[test_df["task_name"].isin(keep_tasks)].copy()
        train_df, test_df, mappings = base.add_task_columns(train_df, test_df)

    peptide_length = int(max(train_df["peptide_sequence"].str.len().max(), test_df["peptide_sequence"].str.len().max()))
    result_rows: list[dict[str, object]] = []
    gate_rows: list[dict[str, object]] = []

    print(f"device: {device}")
    print(f"n_tasks: {len(mappings['tasks'])}")
    print(f"model: {args.model_name}")
    print(f"n_experts: {args.n_experts}")

    for seed in args.seeds:
        metrics, gates = run_one_seed(
            args,
            torch,
            nn,
            DataLoader,
            TensorDataset,
            train_df,
            test_df,
            mappings,
            peptide_length,
            device,
            seed,
        )
        result_rows.extend(metrics)
        gate_rows.extend(gates)

    summary_rows = base.summarize_results(result_rows)
    stability_rows = base.summarize_seed_stability(summary_rows)

    e2_external = read_external_rows(args.e2_baseline_per_task, "shared_peptide_encoder_task_heads")
    famo_external = read_external_rows(args.famo_per_task, "e5_famo")
    e7_external = read_external_rows(args.e7_per_task, "e7_selective_hla_or_global")
    e8_external = read_external_rows(args.e8_per_task, "e8a_fixed_average")
    e9_external = read_external_rows(args.e9_per_task, "e9_e2_cagrad")
    external_comparisons = []
    external_comparisons.extend(
        compare_external(result_rows, e2_external, "shared_peptide_encoder_task_heads", str(args.e2_baseline_per_task))
    )
    external_comparisons.extend(compare_external(result_rows, famo_external, "e5_famo", str(args.famo_per_task)))
    external_comparisons.extend(
        compare_external(result_rows, e7_external, "e7_selective_hla_or_global", str(args.e7_per_task))
    )
    external_comparisons.extend(compare_external(result_rows, e8_external, "e8a_fixed_average", str(args.e8_per_task)))
    external_comparisons.extend(compare_external(result_rows, e9_external, "e9_e2_cagrad", str(args.e9_per_task)))
    external_comparisons.sort(
        key=lambda row: (str(row["baseline_model"]), int(row["seed"]), -float(row["delta_auroc"]))
    )

    base.write_csv(args.per_task_output, base.METRIC_COLUMNS, result_rows)
    base.write_csv(args.summary_output, base.SUMMARY_COLUMNS, summary_rows)
    base.write_csv(args.stability_output, base.STABILITY_COLUMNS, stability_rows)
    base.write_csv(args.external_comparison_output, COMPARISON_COLUMNS, external_comparisons)
    base.write_csv(args.gate_output, GATE_COLUMNS, gate_rows)

    metadata = {
        "train": str(args.train),
        "test": str(args.test),
        "e2_baseline_per_task": str(args.e2_baseline_per_task),
        "famo_per_task": str(args.famo_per_task),
        "e7_per_task": str(args.e7_per_task),
        "e8_per_task": str(args.e8_per_task),
        "e9_per_task": str(args.e9_per_task),
        "per_task_output": str(args.per_task_output),
        "summary_output": str(args.summary_output),
        "stability_output": str(args.stability_output),
        "external_comparison_output": str(args.external_comparison_output),
        "gate_output": str(args.gate_output),
        "n_tasks": len(mappings["tasks"]),
        "model": args.model_name,
        "experiment_name": args.experiment_name,
        "seeds": args.seeds,
        "device": device,
        "peptide_length": peptide_length,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "expert_dim": args.expert_dim,
        "n_experts": args.n_experts,
        "gate_entropy_weight": args.gate_entropy_weight,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "task_mapping": mappings["task_to_id"],
        "tissue_mapping": mappings["tissue_to_id"],
        "hla_mapping": mappings["hla_to_id"],
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrote: {args.per_task_output}")
    print(f"wrote: {args.summary_output}")
    print(f"wrote: {args.stability_output}")
    print(f"wrote: {args.external_comparison_output}")
    print(f"wrote: {args.gate_output}")
    print(f"wrote: {args.metadata_output}")
    print(f"run total time: {time.perf_counter() - run_start:.2f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=project_path("data/tissuePMHC/tissuePMHC_train.csv.gz"))
    parser.add_argument("--test", type=Path, default=project_path("data/tissuePMHC/tissuePMHC_test.csv.gz"))
    parser.add_argument(
        "--e2-baseline-per-task",
        type=Path,
        default=project_path("results/tissuePMHC_neural_baselines_v2/per_task_metrics.csv"),
    )
    parser.add_argument(
        "--famo-per-task",
        type=Path,
        default=project_path("results/tissuePMHC_famo/per_task_metrics.csv"),
    )
    parser.add_argument(
        "--e7-per-task",
        type=Path,
        default=project_path("results/tissuePMHC_selective_grouping/per_task_metrics.csv"),
    )
    parser.add_argument(
        "--e8-per-task",
        type=Path,
        default=project_path("results/tissuePMHC_soft_ensemble/per_task_metrics.csv"),
    )
    parser.add_argument(
        "--e9-per-task",
        type=Path,
        default=project_path("results/tissuePMHC_cagrad/per_task_metrics.csv"),
    )
    parser.add_argument(
        "--per-task-output",
        type=Path,
        default=project_path("results/tissuePMHC_mmoe/per_task_metrics.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=project_path("results/tissuePMHC_mmoe/summary_metrics.csv"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=project_path("results/tissuePMHC_mmoe/metadata.json"),
    )
    parser.add_argument(
        "--stability-output",
        type=Path,
        default=project_path("results/tissuePMHC_mmoe/stability_metrics.csv"),
    )
    parser.add_argument(
        "--external-comparison-output",
        type=Path,
        default=project_path("results/tissuePMHC_mmoe/external_comparison_metrics.csv"),
    )
    parser.add_argument(
        "--gate-output",
        type=Path,
        default=project_path("results/tissuePMHC_mmoe/gate_weight_history.csv"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_REPEAT_SEEDS)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--expert-dim", type=int, default=128)
    parser.add_argument("--n-experts", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--gate-entropy-weight",
        type=float,
        default=0.0,
        help="Positive values encourage broader gate distributions by maximizing gate entropy.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-tasks", type=int, default=0, help="Optional smoke-test limit on the number of tasks.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
