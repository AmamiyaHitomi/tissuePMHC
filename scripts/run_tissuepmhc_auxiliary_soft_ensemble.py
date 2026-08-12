#!/usr/bin/env python3
"""Run E14 auxiliary soft ensemble for tissuePMHC.

Roadmap role: personal test after E8/E13, not part of the formal report.

E14 tests whether the two strongest ideas can be stacked:

1. E8a fixed-average global/HLA soft ensemble.
2. E13 tissue/HLA auxiliary supervision for representation learning.

The main evaluated variants are:

    e14a_global_aux_hla_plain
        global branch uses auxiliary tissue/HLA supervision;
        HLA branch is the original E8-style shared-head branch.

    e14b_global_aux_hla_aux
        both global branch and HLA branch use auxiliary supervision.

Both variants use the fixed E8a-style score:

    final_score = 0.5 * global_score + 0.5 * hla_score
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_tissuepmhc_neural_baselines_v2 as base
import run_tissuepmhc_selective_grouping as e7


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPEAT_SEEDS = [20260704, 20260705, 20260706]
METRICS = ["accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "mcc"]
MODEL_NAMES = ["e14a_global_aux_hla_plain", "e14b_global_aux_hla_aux"]

ENSEMBLE_COLUMNS = [
    *base.METRIC_COLUMNS,
    "global_branch",
    "hla_branch",
    "hla_weight",
    "global_weight",
]

CANDIDATE_COLUMNS = [
    "experiment_name",
    "seed",
    "branch",
    "branch_model",
    "group_name",
    "target_tissue",
    "mhc_restriction",
    "train_rows",
    "test_rows",
    "train_positive",
    "train_negative",
    "test_positive",
    "test_negative",
    *METRICS,
]

DIAGNOSTIC_COLUMNS = [
    "experiment_name",
    "seed",
    "branch",
    "branch_model",
    "group_name",
    "epoch",
    "mean_total_loss",
    "mean_bce_loss",
    "mean_tissue_loss",
    "mean_hla_loss",
    "mean_tissue_accuracy",
    "mean_hla_accuracy",
]

# Persisted for downstream fixed-rule fusion experiments (E15 onward).  These
# rows are deliberately sample-level so branch predictions can be aligned
# without relying on their original dataframe ordering.
BRANCH_PREDICTION_COLUMNS = [
    "experiment_name",
    "seed",
    "branch",
    "branch_model",
    "sample_id",
    "target_tissue",
    "mhc_restriction",
    "label",
    "probability",
    "logit",
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


def format_duration(seconds: float) -> str:
    minutes, remaining_seconds = divmod(seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours}h {minutes:02d}m {remaining_seconds:05.2f}s"
    if minutes:
        return f"{minutes}m {remaining_seconds:05.2f}s"
    return f"{remaining_seconds:.2f}s"


def set_seed(seed: int, torch: Any) -> None:
    base.set_seed(seed, torch)
    random.seed(seed)
    np.random.seed(seed)


def count_labels(rows: pd.DataFrame) -> tuple[int, int]:
    positive = int((rows["label"] == 1).sum())
    return positive, int(len(rows) - positive)


def define_aux_shared_heads_model(
    args: argparse.Namespace,
    torch: Any,
    nn: Any,
    peptide_length: int,
    n_tasks: int,
    n_tissues: int,
    n_hlas: int,
) -> Any:
    class AuxSharedHeadsModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(len(base.AA_TO_INDEX) + 1, args.embedding_dim, padding_idx=base.PAD_INDEX)
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(peptide_length * args.embedding_dim, args.hidden_dim),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_dim, args.hidden_dim),
                nn.ReLU(),
            )
            self.heads = nn.ModuleList([nn.Linear(args.hidden_dim, 1) for _ in range(n_tasks)])
            self.tissue_classifier = nn.Linear(args.hidden_dim, n_tissues)
            self.hla_classifier = nn.Linear(args.hidden_dim, n_hlas)

        def encode(self, peptide_ids: Any) -> Any:
            return self.encoder(self.embedding(peptide_ids))

        def forward(self, peptide_ids: Any, task_ids: Any) -> Any:
            encoded = self.encode(peptide_ids)
            logits = encoded.new_empty(encoded.shape[0])
            for task_id in torch.unique(task_ids):
                mask = task_ids == task_id
                logits[mask] = self.heads[int(task_id.item())](encoded[mask]).squeeze(-1)
            return logits

        def auxiliary_logits(self, peptide_ids: Any) -> tuple[Any, Any]:
            encoded = self.encode(peptide_ids)
            return self.tissue_classifier(encoded), self.hla_classifier(encoded)

    return AuxSharedHeadsModel()


def build_aux_loader(
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
    tissue_ids = df["tissue_id"].to_numpy(dtype=np.int64).copy()
    hla_ids = df["hla_id"].to_numpy(dtype=np.int64).copy()
    labels = df["label"].to_numpy(dtype=np.int64).copy()
    return base.build_loader(
        torch,
        DataLoader,
        TensorDataset,
        [x, task_ids, tissue_ids, hla_ids, labels],
        args.batch_size,
        shuffle,
    )


def train_aux_branch(
    args: argparse.Namespace,
    torch: Any,
    nn: Any,
    DataLoader: Any,
    TensorDataset: Any,
    train_df: pd.DataFrame,
    task_to_id: dict[str, int],
    n_tissues: int,
    n_hlas: int,
    peptide_length: int,
    device: str,
    seed: int,
    branch: str,
    group_name: str,
    use_aux: bool,
) -> tuple[Any, list[dict[str, object]]]:
    train_mapped = e7.prepare_with_mapping(train_df, task_to_id)
    model = define_aux_shared_heads_model(args, torch, nn, peptide_length, len(task_to_id), n_tissues, n_hlas).to(device)
    loader = build_aux_loader(args, torch, DataLoader, TensorDataset, train_mapped, peptide_length, True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    diagnostics: list[dict[str, object]] = []
    branch_model = "aux_shared_heads" if use_aux else "plain_shared_heads"

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_losses = []
        bce_losses = []
        tissue_losses = []
        hla_losses = []
        tissue_accuracies = []
        hla_accuracies = []
        for batch in loader:
            peptide_ids, task_ids, tissue_ids, hla_ids, labels = [item.to(device) for item in batch]
            optimizer.zero_grad(set_to_none=True)
            logits = model(peptide_ids, task_ids)
            bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
            tissue_logits, hla_logits = model.auxiliary_logits(peptide_ids)
            tissue_loss = torch.nn.functional.cross_entropy(tissue_logits, tissue_ids)
            hla_loss = torch.nn.functional.cross_entropy(hla_logits, hla_ids)
            if use_aux:
                loss = bce_loss + args.tissue_loss_weight * tissue_loss + args.hla_loss_weight * hla_loss
            else:
                loss = bce_loss
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            total_losses.append(float(loss.detach().cpu()))
            bce_losses.append(float(bce_loss.detach().cpu()))
            tissue_losses.append(float(tissue_loss.detach().cpu()))
            hla_losses.append(float(hla_loss.detach().cpu()))
            tissue_accuracies.append(float((tissue_logits.argmax(dim=1) == tissue_ids).float().mean().detach().cpu()))
            hla_accuracies.append(float((hla_logits.argmax(dim=1) == hla_ids).float().mean().detach().cpu()))

        diagnostics.append(
            {
                "experiment_name": "E14_auxiliary_soft_ensemble",
                "seed": seed,
                "branch": branch,
                "branch_model": branch_model,
                "group_name": group_name,
                "epoch": epoch,
                "mean_total_loss": float(np.mean(total_losses)),
                "mean_bce_loss": float(np.mean(bce_losses)),
                "mean_tissue_loss": float(np.mean(tissue_losses)),
                "mean_hla_loss": float(np.mean(hla_losses)),
                "mean_tissue_accuracy": float(np.mean(tissue_accuracies)),
                "mean_hla_accuracy": float(np.mean(hla_accuracies)),
            }
        )
        print(
            f"    time epoch branch={branch} group={group_name} "
            f"epoch={epoch}/{args.epochs} duration={format_duration(time.perf_counter() - epoch_start)}"
        )

    return model, diagnostics


def predict_branch(
    args: argparse.Namespace,
    torch: Any,
    DataLoader: Any,
    TensorDataset: Any,
    model: Any,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    task_to_id: dict[str, int],
    peptide_length: int,
    device: str,
    seed: int,
    branch: str,
    group_name: str,
    use_aux: bool,
) -> tuple[dict[tuple[str, str], dict[str, object]], list[dict[str, object]]]:
    train_mapped = e7.prepare_with_mapping(train_df, task_to_id)
    test_mapped = e7.prepare_with_mapping(test_df, task_to_id)
    predictions: dict[tuple[str, str], dict[str, object]] = {}
    candidate_rows = []
    branch_model = "aux_shared_heads" if use_aux else "plain_shared_heads"

    for task_name in sorted(set(train_mapped["task_name"]) & set(test_mapped["task_name"])):
        train_task = train_mapped[train_mapped["task_name"] == task_name]
        test_task = test_mapped[test_mapped["task_name"] == task_name]
        loader = build_aux_loader(args, torch, DataLoader, TensorDataset, test_task, peptide_length, False)
        scores = []
        logits_list = []
        labels = []
        with torch.no_grad():
            model.eval()
            for batch in loader:
                peptide_ids, task_ids, _, _, y = [item.to(device) for item in batch]
                logits = model(peptide_ids, task_ids)
                scores.append(torch.sigmoid(logits).cpu().numpy())
                logits_list.append(logits.cpu().numpy())
                labels.append(y.cpu().numpy())
        y_true = np.concatenate(labels)
        y_score = np.concatenate(scores)
        y_logit = np.concatenate(logits_list)
        metrics = base.evaluate(y_true, y_score)
        train_positive, train_negative = count_labels(train_task)
        test_positive, test_negative = count_labels(test_task)
        first = test_task.iloc[0]
        key = (str(first["target_tissue"]), str(first["mhc_restriction"]))
        row = {
            "experiment_name": "E14_auxiliary_soft_ensemble",
            "seed": seed,
            "branch": branch,
            "branch_model": branch_model,
            "group_name": group_name,
            "target_tissue": first["target_tissue"],
            "mhc_restriction": first["mhc_restriction"],
            "train_rows": len(train_task),
            "test_rows": len(test_task),
            "train_positive": train_positive,
            "train_negative": train_negative,
            "test_positive": test_positive,
            "test_negative": test_negative,
            **metrics,
        }
        candidate_rows.append(row)
        predictions[key] = {
            "train_task": train_task,
            "test_task": test_task,
            "y_true": y_true,
            "y_score": y_score,
            "y_logit": y_logit,
            "metrics": metrics,
            "candidate_row": row,
        }

    return predictions, candidate_rows


def train_and_predict_global_branch(
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
    use_aux: bool,
) -> tuple[dict[tuple[str, str], dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    branch_name = "global_aux" if use_aux else "global_plain"
    print(f"  train {branch_name} branch")
    start = time.perf_counter()
    model, diagnostics = train_aux_branch(
        args,
        torch,
        nn,
        DataLoader,
        TensorDataset,
        train_df,
        mappings["task_to_id"],
        len(mappings["tissue_to_id"]),
        len(mappings["hla_to_id"]),
        peptide_length,
        device,
        seed,
        branch_name,
        "all_tasks",
        use_aux,
    )
    print(f"    time {branch_name} duration={format_duration(time.perf_counter() - start)}")
    predictions, candidate_rows = predict_branch(
        args,
        torch,
        DataLoader,
        TensorDataset,
        model,
        train_df,
        test_df,
        mappings["task_to_id"],
        peptide_length,
        device,
        seed,
        branch_name,
        "all_tasks",
        use_aux,
    )
    return predictions, candidate_rows, diagnostics


def train_and_predict_hla_branches(
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
    use_aux: bool,
) -> tuple[dict[tuple[str, str], dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    all_predictions: dict[tuple[str, str], dict[str, object]] = {}
    all_candidate_rows: list[dict[str, object]] = []
    all_diagnostics: list[dict[str, object]] = []
    branch_name = "hla_aux" if use_aux else "hla_plain"
    hla_groups = sorted(set(train_df["mhc_restriction"]) & set(test_df["mhc_restriction"]))

    for group_index, hla in enumerate(hla_groups, start=1):
        hla_train = train_df[train_df["mhc_restriction"] == hla].copy()
        hla_test = test_df[test_df["mhc_restriction"] == hla].copy()
        local_tasks = sorted(set(hla_train["task_name"]) & set(hla_test["task_name"]))
        if not local_tasks:
            continue
        local_task_to_id = {task: index for index, task in enumerate(local_tasks)}
        print(f"  train {branch_name} branch {group_index:02d}/{len(hla_groups)} {hla} n_tasks={len(local_tasks)}")
        start = time.perf_counter()
        model, diagnostics = train_aux_branch(
            args,
            torch,
            nn,
            DataLoader,
            TensorDataset,
            hla_train,
            local_task_to_id,
            len(mappings["tissue_to_id"]),
            len(mappings["hla_to_id"]),
            peptide_length,
            device,
            seed,
            branch_name,
            hla,
            use_aux,
        )
        print(f"    time {branch_name} {hla} duration={format_duration(time.perf_counter() - start)}")
        predictions, candidate_rows = predict_branch(
            args,
            torch,
            DataLoader,
            TensorDataset,
            model,
            hla_train,
            hla_test,
            local_task_to_id,
            peptide_length,
            device,
            seed,
            branch_name,
            hla,
            use_aux,
        )
        all_predictions.update(predictions)
        all_candidate_rows.extend(candidate_rows)
        all_diagnostics.extend(diagnostics)

    return all_predictions, all_candidate_rows, all_diagnostics


def make_ensemble_row(
    seed: int,
    model_name: str,
    global_prediction: dict[str, object],
    hla_prediction: dict[str, object],
    metrics: dict[str, float],
) -> dict[str, object]:
    train_task = global_prediction["train_task"]
    test_task = global_prediction["test_task"]
    train_positive, train_negative = count_labels(train_task)
    test_positive, test_negative = count_labels(test_task)
    first = test_task.iloc[0]
    if model_name == "e14a_global_aux_hla_plain":
        global_branch = "global_aux"
        hla_branch = "hla_plain"
    else:
        global_branch = "global_aux"
        hla_branch = "hla_aux"
    return {
        "experiment_name": "E14_auxiliary_soft_ensemble",
        "seed": seed,
        "model": model_name,
        "target_tissue": first["target_tissue"],
        "mhc_restriction": first["mhc_restriction"],
        "train_rows": len(train_task),
        "test_rows": len(test_task),
        "train_positive": train_positive,
        "train_negative": train_negative,
        "test_positive": test_positive,
        "test_negative": test_negative,
        **metrics,
        "global_branch": global_branch,
        "hla_branch": hla_branch,
        "hla_weight": 0.5,
        "global_weight": 0.5,
    }


def build_ensembles(
    seed: int,
    global_aux: dict[tuple[str, str], dict[str, object]],
    hla_plain: dict[tuple[str, str], dict[str, object]],
    hla_aux: dict[tuple[str, str], dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for key in sorted(set(global_aux) & set(hla_plain)):
        global_prediction = global_aux[key]
        hla_prediction = hla_plain[key]
        if not np.array_equal(global_prediction["y_true"], hla_prediction["y_true"]):
            raise ValueError(f"Mismatched labels for E14a task {key}")
        y_true = global_prediction["y_true"]
        score = 0.5 * global_prediction["y_score"] + 0.5 * hla_prediction["y_score"]
        rows.append(
            make_ensemble_row(
                seed,
                "e14a_global_aux_hla_plain",
                global_prediction,
                hla_prediction,
                base.evaluate(y_true, score),
            )
        )

    for key in sorted(set(global_aux) & set(hla_aux)):
        global_prediction = global_aux[key]
        hla_prediction = hla_aux[key]
        if not np.array_equal(global_prediction["y_true"], hla_prediction["y_true"]):
            raise ValueError(f"Mismatched labels for E14b task {key}")
        y_true = global_prediction["y_true"]
        score = 0.5 * global_prediction["y_score"] + 0.5 * hla_prediction["y_score"]
        rows.append(
            make_ensemble_row(
                seed,
                "e14b_global_aux_hla_aux",
                global_prediction,
                hla_prediction,
                base.evaluate(y_true, score),
            )
        )

    rows.sort(key=lambda row: (int(row["seed"]), str(row["model"]), str(row["target_tissue"]), str(row["mhc_restriction"])))
    return rows


def build_branch_prediction_rows(
    seed: int,
    predictions: dict[tuple[str, str], dict[str, object]],
) -> list[dict[str, object]]:
    """Convert a branch's task predictions to explicitly aligned CSV rows."""
    rows: list[dict[str, object]] = []
    for prediction in predictions.values():
        test_task = prediction["test_task"]
        candidate = prediction["candidate_row"]
        for sample_id, label, probability, logit in zip(
            test_task["sample_id"], prediction["y_true"], prediction["y_score"], prediction["y_logit"], strict=True
        ):
            rows.append(
                {
                    "experiment_name": "E14_auxiliary_soft_ensemble",
                    "seed": seed,
                    "branch": candidate["branch"],
                    "branch_model": candidate["branch_model"],
                    "sample_id": sample_id,
                    "target_tissue": candidate["target_tissue"],
                    "mhc_restriction": candidate["mhc_restriction"],
                    "label": int(label),
                    "probability": float(probability),
                    "logit": float(logit),
                }
            )
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
            "candidate_model": row["model"],
            "baseline_source": baseline_source,
        }
        for metric in METRICS:
            comparison[f"delta_{metric}"] = float(row[metric]) - float(baseline[metric])
        comparisons.append(comparison)
    return comparisons


def build_comparisons(args: argparse.Namespace, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    comparisons: list[dict[str, object]] = []
    baselines = [
        ("shared_peptide_encoder_task_heads", args.e2_per_task),
        ("e8a_fixed_average", args.e8_per_task),
        ("e13_aux_tissue_hla", args.e13_per_task),
    ]
    for baseline_model, path in baselines:
        baseline_rows = read_external_rows(path, baseline_model)
        comparisons.extend(compare_external(rows, baseline_rows, baseline_model, str(path)))
    comparisons.sort(key=lambda row: (str(row["candidate_model"]), str(row["baseline_model"]), int(row["seed"]), -float(row["delta_auroc"])))
    return comparisons


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
    candidate_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    branch_prediction_rows: list[dict[str, object]] = []

    print(f"device: {device}")
    print(f"n_tasks: {len(mappings['tasks'])}")
    print(f"models: {MODEL_NAMES}")
    for seed in args.seeds:
        seed_start = time.perf_counter()
        set_seed(seed, torch)
        print(f"experiment: E14_auxiliary_soft_ensemble seed={seed}")
        global_aux, rows, diagnostics = train_and_predict_global_branch(
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
            use_aux=True,
        )
        candidate_rows.extend(rows)
        diagnostic_rows.extend(diagnostics)
        branch_prediction_rows.extend(build_branch_prediction_rows(seed, global_aux))

        hla_plain, rows, diagnostics = train_and_predict_hla_branches(
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
            use_aux=False,
        )
        candidate_rows.extend(rows)
        diagnostic_rows.extend(diagnostics)
        branch_prediction_rows.extend(build_branch_prediction_rows(seed, hla_plain))

        hla_aux, rows, diagnostics = train_and_predict_hla_branches(
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
            use_aux=True,
        )
        candidate_rows.extend(rows)
        diagnostic_rows.extend(diagnostics)

        seed_rows = build_ensembles(seed, global_aux, hla_plain, hla_aux)
        result_rows.extend(seed_rows)
        for model_name in MODEL_NAMES:
            values = [float(row["auroc"]) for row in seed_rows if row["model"] == model_name]
            print(f"  {model_name} mean_auroc={float(np.mean(values)):.4f}")
        print(f"  time seed_total seed={seed} duration={format_duration(time.perf_counter() - seed_start)}")

    summary_rows = base.summarize_results(result_rows)
    stability_rows = base.summarize_seed_stability(summary_rows)
    comparison_rows = build_comparisons(args, result_rows)

    base.write_csv(args.per_task_output, ENSEMBLE_COLUMNS, result_rows)
    base.write_csv(args.summary_output, base.SUMMARY_COLUMNS, summary_rows)
    base.write_csv(args.stability_output, base.STABILITY_COLUMNS, stability_rows)
    base.write_csv(args.candidate_output, CANDIDATE_COLUMNS, candidate_rows)
    base.write_csv(args.diagnostic_output, DIAGNOSTIC_COLUMNS, diagnostic_rows)
    base.write_csv(args.branch_predictions_output, BRANCH_PREDICTION_COLUMNS, branch_prediction_rows)
    base.write_csv(args.comparison_output, COMPARISON_COLUMNS, comparison_rows)

    metadata = {
        "train": str(args.train),
        "test": str(args.test),
        "e2_per_task": str(args.e2_per_task),
        "e8_per_task": str(args.e8_per_task),
        "e13_per_task": str(args.e13_per_task),
        "per_task_output": str(args.per_task_output),
        "summary_output": str(args.summary_output),
        "stability_output": str(args.stability_output),
        "candidate_output": str(args.candidate_output),
        "diagnostic_output": str(args.diagnostic_output),
        "branch_predictions_output": str(args.branch_predictions_output),
        "comparison_output": str(args.comparison_output),
        "n_tasks": len(mappings["tasks"]),
        "models": MODEL_NAMES,
        "seeds": args.seeds,
        "device": device,
        "peptide_length": peptide_length,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "tissue_loss_weight": args.tissue_loss_weight,
        "hla_loss_weight": args.hla_loss_weight,
        "max_grad_norm": args.max_grad_norm,
        "ensemble_formula": "0.5 * global_score + 0.5 * hla_score",
        "formal_report_status": "personal_test_only_not_written_to_report",
        "task_mapping": mappings["task_to_id"],
        "tissue_mapping": mappings["tissue_to_id"],
        "hla_mapping": mappings["hla_to_id"],
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrote: {args.per_task_output}")
    print(f"wrote: {args.summary_output}")
    print(f"wrote: {args.stability_output}")
    print(f"wrote: {args.candidate_output}")
    print(f"wrote: {args.diagnostic_output}")
    print(f"wrote: {args.branch_predictions_output}")
    print(f"wrote: {args.comparison_output}")
    print(f"wrote: {args.metadata_output}")
    print(f"run total time: {format_duration(time.perf_counter() - run_start)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=project_path("data/tissuePMHC/tissuePMHC_train.csv.gz"))
    parser.add_argument("--test", type=Path, default=project_path("data/tissuePMHC/tissuePMHC_test.csv.gz"))
    parser.add_argument(
        "--e2-per-task",
        type=Path,
        default=project_path("results/tissuePMHC_neural_baselines_v2/per_task_metrics.csv"),
    )
    parser.add_argument(
        "--e8-per-task",
        type=Path,
        default=project_path("results/tissuePMHC_soft_ensemble/per_task_metrics.csv"),
    )
    parser.add_argument(
        "--e13-per-task",
        type=Path,
        default=project_path("results/tissuePMHC_auxiliary_tasks/per_task_metrics.csv"),
    )
    parser.add_argument(
        "--per-task-output",
        type=Path,
        default=project_path("results/tissuePMHC_auxiliary_soft_ensemble/per_task_metrics.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=project_path("results/tissuePMHC_auxiliary_soft_ensemble/summary_metrics.csv"),
    )
    parser.add_argument(
        "--stability-output",
        type=Path,
        default=project_path("results/tissuePMHC_auxiliary_soft_ensemble/stability_metrics.csv"),
    )
    parser.add_argument(
        "--candidate-output",
        type=Path,
        default=project_path("results/tissuePMHC_auxiliary_soft_ensemble/candidate_metrics.csv"),
    )
    parser.add_argument(
        "--diagnostic-output",
        type=Path,
        default=project_path("results/tissuePMHC_auxiliary_soft_ensemble/auxiliary_diagnostics.csv"),
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=project_path("results/tissuePMHC_auxiliary_soft_ensemble/external_comparison_metrics.csv"),
    )
    parser.add_argument(
        "--branch-predictions-output",
        type=Path,
        default=project_path("results/tissuePMHC_auxiliary_soft_ensemble/branch_predictions.csv"),
        help="Sample-level E14 branch predictions used by E15 fixed-rule fusion.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=project_path("results/tissuePMHC_auxiliary_soft_ensemble/metadata.json"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_REPEAT_SEEDS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--tissue-loss-weight", type=float, default=0.1)
    parser.add_argument("--hla-loss-weight", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-tasks", type=int, default=0, help="Optional smoke-test limit on the number of tasks.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
