#!/usr/bin/env python3
"""Tune mouse occurrence-equal TissuePMHC without selecting on the fixed test set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
PROJECT_ROOT = EXPERIMENT_ROOT.parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

import common  # noqa: E402


SEEDS = [20260704, 20260705, 20260706]
RESULT_LABEL = "TissuePMHC (tuned; mouse-training-CV-selected)"
RESULT_LABEL_ZH = "tissuePMHC（调参后版本；小鼠训练集CV选择）"
METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "auroc",
    "auprc",
    "f1",
    "mcc",
    "pair_accuracy",
]


@dataclass(frozen=True)
class Config:
    name: str
    stage: str
    sampling: str = "task_balanced"
    embedding_dim: int = 16
    conv_channels: int = 32
    hidden_dim: int = 128
    kernel_sizes: tuple[int, ...] = (2, 3, 5)
    dropout: float = 0.2
    tissue_loss_weight: float = 0.1
    hla_loss_weight: float = 0.1
    fusion_global_weight: float = 0.5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    task_batch_size: int = 32


class RunContext:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = output_dir / "run.log"
        self.timing_path = output_dir / "timing_results.csv"
        self.started = time.perf_counter()

    def emit(self, message: str) -> None:
        stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def timing(self, **row: Any) -> None:
        fields = [
            "scope",
            "stage",
            "config",
            "fold",
            "seed",
            "branch",
            "epoch",
            "elapsed_seconds",
            "status",
        ]
        record = {field: row.get(field, "") for field in fields}
        new_file = not self.timing_path.exists()
        with self.timing_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if new_file:
                writer.writeheader()
            writer.writerow(record)


def stable_seed(*parts: Any) -> int:
    payload = "||".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def model_args(config: Config, epochs: int, batch_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
        embedding_dim=int(config.embedding_dim),
        hidden_dim=int(config.hidden_dim),
        dropout=float(config.dropout),
        tissue_loss_weight=float(config.tissue_loss_weight),
        hla_loss_weight=float(config.hla_loss_weight),
        max_grad_norm=1.0,
        kernel_sizes=list(config.kernel_sizes),
        conv_channels=int(config.conv_channels),
    )


def rank_fuse(
    global_predictions: pd.DataFrame,
    hla_predictions: pd.DataFrame,
    global_weight: float,
    keys: list[str],
) -> pd.DataFrame:
    merged = global_predictions.merge(
        hla_predictions,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_global", "_hla"),
    )
    if len(merged) != len(global_predictions) or len(merged) != len(hla_predictions):
        raise ValueError("Global and H2 predictions do not cover identical samples.")
    if not np.array_equal(merged["label_global"].to_numpy(), merged["label_hla"].to_numpy()):
        raise ValueError("Global and H2 prediction labels disagree.")

    output: list[pd.DataFrame] = []
    for _, task in merged.groupby(["target_tissue", "mhc_restriction"], sort=True):
        item = task[keys + ["label_global"]].rename(columns={"label_global": "label"}).copy()
        global_rank = task["score_global"].rank(method="average", pct=True).to_numpy()
        hla_rank = task["score_hla"].rank(method="average", pct=True).to_numpy()
        item["score"] = global_weight * global_rank + (1.0 - global_weight) * hla_rank
        output.append(item)
    return pd.concat(output, ignore_index=True).sort_values(keys).reset_index(drop=True)


def make_balanced_batch(
    arrays: dict[str, np.ndarray],
    task_indices: list[np.ndarray],
    task_batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, ...]:
    sampled = [rng.choice(indices, size=task_batch_size, replace=True) for indices in task_indices]
    selected = np.concatenate(sampled)
    rng.shuffle(selected)
    return tuple(
        arrays[key][selected]
        for key in ["peptides", "task_ids", "tissue_ids", "hla_ids", "labels"]
    )


def train_branch(
    *,
    config: Config,
    args: SimpleNamespace,
    torch: Any,
    nn: Any,
    DataLoader: Any,
    TensorDataset: Any,
    e29: Any,
    train_df: pd.DataFrame,
    task_to_id: dict[str, int],
    peptide_length: int,
    device: str,
    branch: str,
    group_name: str,
    use_aux: bool,
    n_tissues: int,
    n_hlas: int,
    seed: int,
    ctx: RunContext,
    stage: str,
    fold: int | str,
) -> Any:
    torch.manual_seed(stable_seed(seed, branch, group_name, "model"))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(stable_seed(seed, branch, group_name, "cuda"))
    model = e29.define_cnn_shared_heads_model(
        args, nn, peptide_length, len(task_to_id), n_tissues, n_hlas, use_aux
    ).to(device)
    arrays = e29.mapped_arrays(train_df, task_to_id, peptide_length)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    bce = nn.BCEWithLogitsLoss()
    cross_entropy = nn.CrossEntropyLoss()

    loader = None
    task_indices: list[np.ndarray] = []
    steps_per_epoch = 0
    rng = np.random.default_rng(stable_seed(seed, branch, group_name, "sampler"))
    if config.sampling == "row":
        loader = e29.build_loader(torch, DataLoader, TensorDataset, arrays, args.batch_size, True)
        steps_per_epoch = len(loader)
    elif config.sampling == "task_balanced":
        task_indices = [np.flatnonzero(arrays["task_ids"] == task_id) for task_id in range(len(task_to_id))]
        if any(len(indices) == 0 for indices in task_indices):
            raise ValueError(f"Empty task encountered in balanced branch {group_name}.")
        steps_per_epoch = int(math.ceil(max(map(len, task_indices)) / config.task_batch_size))
    else:
        raise ValueError(f"Unknown sampling mode: {config.sampling}")

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        epoch_losses: list[float] = []
        if loader is not None:
            batches = loader
        else:
            batches = (
                make_balanced_batch(arrays, task_indices, config.task_batch_size, rng)
                for _ in range(steps_per_epoch)
            )
        for batch in batches:
            peptide_ids, task_ids, tissue_ids, hla_ids, labels = [
                torch.as_tensor(item, device=device) for item in batch
            ]
            optimizer.zero_grad(set_to_none=True)
            logits = model(peptide_ids, task_ids)
            loss = bce(logits, labels.float())
            if use_aux:
                tissue_logits, hla_logits = model.auxiliary_logits(peptide_ids)
                loss = loss + args.tissue_loss_weight * cross_entropy(tissue_logits, tissue_ids)
                loss = loss + args.hla_loss_weight * cross_entropy(hla_logits, hla_ids)
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        elapsed = time.perf_counter() - epoch_started
        mean_loss = float(np.mean(epoch_losses))
        ctx.emit(
            f"EPOCH stage={stage} config={config.name} fold={fold} seed={seed} "
            f"branch={branch}:{group_name} epoch={epoch}/{args.epochs} "
            f"steps={steps_per_epoch} loss={mean_loss:.6f} time={elapsed:.3f}s"
        )
        ctx.timing(
            scope="epoch",
            stage=stage,
            config=config.name,
            fold=fold,
            seed=seed,
            branch=f"{branch}:{group_name}",
            epoch=epoch,
            elapsed_seconds=f"{elapsed:.6f}",
            status="completed",
        )
    return model


def predict_configuration(
    *,
    config: Config,
    epochs: int,
    batch_size: int,
    torch: Any,
    nn: Any,
    DataLoader: Any,
    TensorDataset: Any,
    e29: Any,
    base: Any,
    fitting: pd.DataFrame,
    prediction: pd.DataFrame,
    mappings: dict[str, Any],
    peptide_length: int,
    device: str,
    seed: int,
    ctx: RunContext,
    stage: str,
    fold: int | str,
) -> pd.DataFrame:
    del base
    started = time.perf_counter()
    args = model_args(config, epochs, batch_size)
    global_model = train_branch(
        config=config,
        args=args,
        torch=torch,
        nn=nn,
        DataLoader=DataLoader,
        TensorDataset=TensorDataset,
        e29=e29,
        train_df=fitting,
        task_to_id=mappings["task_to_id"],
        peptide_length=peptide_length,
        device=device,
        branch="global_aux",
        group_name="all",
        use_aux=True,
        n_tissues=len(mappings["tissue_to_id"]),
        n_hlas=len(mappings["hla_to_id"]),
        seed=seed,
        ctx=ctx,
        stage=stage,
        fold=fold,
    )
    global_predictions = e29.predict_branch(
        args,
        torch,
        DataLoader,
        TensorDataset,
        global_model,
        fitting,
        prediction,
        mappings["task_to_id"],
        peptide_length,
        device,
    )
    del global_model
    if device == "cuda":
        torch.cuda.empty_cache()

    hla_parts: list[pd.DataFrame] = []
    hlas = sorted(set(fitting["mhc_restriction"]) & set(prediction["mhc_restriction"]))
    for hla in hlas:
        hla_fit = fitting[fitting["mhc_restriction"] == hla].copy()
        hla_prediction = prediction[prediction["mhc_restriction"] == hla].copy()
        tasks = sorted(set(hla_fit["task_name"]) & set(hla_prediction["task_name"]))
        task_to_id = {task: index for index, task in enumerate(tasks)}
        hla_model = train_branch(
            config=config,
            args=args,
            torch=torch,
            nn=nn,
            DataLoader=DataLoader,
            TensorDataset=TensorDataset,
            e29=e29,
            train_df=hla_fit,
            task_to_id=task_to_id,
            peptide_length=peptide_length,
            device=device,
            branch="h2_plain",
            group_name=hla,
            use_aux=False,
            n_tissues=len(mappings["tissue_to_id"]),
            n_hlas=len(mappings["hla_to_id"]),
            seed=seed,
            ctx=ctx,
            stage=stage,
            fold=fold,
        )
        hla_parts.append(
            e29.predict_branch(
                args,
                torch,
                DataLoader,
                TensorDataset,
                hla_model,
                hla_fit,
                hla_prediction,
                task_to_id,
                peptide_length,
                device,
            )
        )
        del hla_model
        if device == "cuda":
            torch.cuda.empty_cache()

    result = rank_fuse(
        global_predictions,
        pd.concat(hla_parts, ignore_index=True),
        config.fusion_global_weight,
        e29.KEYS,
    )
    elapsed = time.perf_counter() - started
    ctx.timing(
        scope="fit_predict",
        stage=stage,
        config=config.name,
        fold=fold,
        seed=seed,
        branch="all",
        epoch="",
        elapsed_seconds=f"{elapsed:.6f}",
        status="completed",
    )
    return result


def evaluate_predictions(
    model_name: str,
    seed: int,
    train_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    base: Any,
) -> tuple[dict[str, Any], pd.DataFrame]:
    train_counts = train_df.groupby(["target_tissue", "mhc_restriction"]).size()
    rows: list[dict[str, Any]] = []
    for (tissue, hla), task in prediction_df.groupby(
        ["target_tissue", "mhc_restriction"], sort=True
    ):
        metrics = base.evaluate(
            task["label"].to_numpy(dtype=np.int64), task["score"].to_numpy(dtype=np.float64)
        )
        pair_accuracy = float("nan")
        if "pair_id" in task.columns:
            wide = task.pivot(index="pair_id", columns="label", values="score")
            pair_accuracy = float((wide[1] > wide[0]).mean())
        rows.append(
            {
                "model": model_name,
                "seed": seed,
                "target_tissue": tissue,
                "mhc_restriction": hla,
                "train_rows": int(train_counts.loc[(tissue, hla)]),
                "test_rows": int(len(task)),
                **metrics,
                "pair_accuracy": pair_accuracy,
            }
        )
    per_task = pd.DataFrame(rows)
    worst_n = min(5, len(per_task))
    summary = {
        "model": model_name,
        "seed": seed,
        "n_tasks": int(len(per_task)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(prediction_df)),
        "mean_task_auroc": float(per_task["auroc"].mean()),
        "mean_task_auprc": float(per_task["auprc"].mean()),
        "mean_task_accuracy": float(per_task["accuracy"].mean()),
        "mean_task_mcc": float(per_task["mcc"].mean()),
        "mean_task_pair_accuracy": float(per_task["pair_accuracy"].mean()),
        "worst_5_mean_auroc": float(per_task.nsmallest(worst_n, "auroc")["auroc"].mean()),
    }
    summary["selection_score"] = 0.5 * (
        summary["mean_task_auroc"] + summary["mean_task_auprc"]
    )
    return summary, per_task


def attach_pair_ids(predictions: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    pair_map = source[["sample_id", "pair_id"]]
    return predictions.merge(pair_map, on="sample_id", how="left", validate="one_to_one")


def make_stage_a(smoke: bool) -> list[Config]:
    configs = [
        Config("a0_original_row", "A", sampling="row"),
        Config("a1_balanced_default", "A"),
        Config("a2_balanced_c16_h64", "A", conv_channels=16, hidden_dim=64),
        Config("a3_balanced_c16_h128", "A", conv_channels=16, hidden_dim=128),
        Config("a4_balanced_c32_h64", "A", conv_channels=32, hidden_dim=64),
        Config("a5_balanced_c8_h64", "A", conv_channels=8, hidden_dim=64),
    ]
    return configs[:2] if smoke else configs


def make_stage_b(base_config: Config, smoke: bool) -> list[Config]:
    variants = [
        replace(base_config, name="b0_aux_zero", stage="B", tissue_loss_weight=0.0, hla_loss_weight=0.0),
        replace(base_config, name="b1_aux_002", stage="B", tissue_loss_weight=0.02, hla_loss_weight=0.02),
        replace(base_config, name="b2_aux_005", stage="B", tissue_loss_weight=0.05, hla_loss_weight=0.05),
        replace(base_config, name="b3_aux_t005_h01", stage="B", tissue_loss_weight=0.05, hla_loss_weight=0.1),
        replace(base_config, name="b4_fusion_g025", stage="B", fusion_global_weight=0.25),
        replace(base_config, name="b5_fusion_g075", stage="B", fusion_global_weight=0.75),
        replace(base_config, name="b6_dropout_01", stage="B", dropout=0.1),
        replace(base_config, name="b7_dropout_035", stage="B", dropout=0.35),
        replace(base_config, name="b8_lr_0003", stage="B", learning_rate=3e-4),
        replace(base_config, name="b9_lr_003", stage="B", learning_rate=3e-3),
        replace(base_config, name="b10_wd_001", stage="B", weight_decay=1e-3),
        replace(base_config, name="b11_kernel_3", stage="B", kernel_sizes=(3,)),
        replace(base_config, name="b12_kernel_35", stage="B", kernel_sizes=(3, 5)),
    ]
    return variants[:1] if smoke else variants


def cv_rankings(cv_results: pd.DataFrame) -> pd.DataFrame:
    ranked = (
        cv_results.groupby(["config", "stage"], as_index=False)
        .agg(
            n_folds=("fold", "nunique"),
            mean_auroc=("mean_task_auroc", "mean"),
            sd_auroc=("mean_task_auroc", "std"),
            mean_auprc=("mean_task_auprc", "mean"),
            sd_auprc=("mean_task_auprc", "std"),
            mean_worst5_auroc=("worst_5_mean_auroc", "mean"),
            mean_selection_score=("selection_score", "mean"),
        )
        .sort_values(
            ["mean_selection_score", "mean_auroc", "mean_worst5_auroc"], ascending=False
        )
        .reset_index(drop=True)
    )
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def evaluate_cv_config(
    config: Config,
    folds: pd.Series,
    n_folds: int,
    train: pd.DataFrame,
    mappings: dict[str, Any],
    peptide_length: int,
    runtime: dict[str, Any],
    epochs: int,
    batch_size: int,
    ctx: RunContext,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    summaries: list[dict[str, Any]] = []
    task_frames: list[pd.DataFrame] = []
    for fold in range(n_folds):
        fitting = train[folds != fold].copy()
        held = train[folds == fold].copy()
        # Common random numbers make configuration comparisons less noisy:
        # every candidate sees the same fold-specific initialization/sampling seed.
        seed = 20260803 + fold
        ctx.emit(
            f"FOLD START stage={config.stage} config={config.name} fold={fold + 1}/{n_folds} "
            f"seed={seed} train_rows={len(fitting)} validation_rows={len(held)}"
        )
        started = time.perf_counter()
        predictions = predict_configuration(
            config=config,
            epochs=epochs,
            batch_size=batch_size,
            fitting=fitting,
            prediction=held,
            mappings=mappings,
            peptide_length=peptide_length,
            seed=seed,
            ctx=ctx,
            stage=config.stage,
            fold=fold,
            **runtime,
        )
        predictions = attach_pair_ids(predictions, held)
        summary, per_task = evaluate_predictions(config.name, seed, fitting, predictions, runtime["base"])
        summary.update({"config": config.name, "stage": config.stage, "fold": fold})
        per_task.insert(0, "fold", fold)
        per_task.insert(0, "stage", config.stage)
        per_task.insert(0, "config", config.name)
        summaries.append(summary)
        task_frames.append(per_task)
        elapsed = time.perf_counter() - started
        ctx.emit(
            f"FOLD DONE stage={config.stage} config={config.name} fold={fold + 1}/{n_folds} "
            f"AUROC={summary['mean_task_auroc']:.6f} AUPRC={summary['mean_task_auprc']:.6f} "
            f"score={summary['selection_score']:.6f} time={elapsed:.3f}s"
        )
        ctx.timing(
            scope="fold",
            stage=config.stage,
            config=config.name,
            fold=fold,
            seed=seed,
            branch="all",
            epoch="",
            elapsed_seconds=f"{elapsed:.6f}",
            status="completed",
        )
    return summaries, task_frames


def merge_best_stage_b(base_config: Config, rankings: pd.DataFrame, config_map: dict[str, Config]) -> Config:
    top_names = rankings[rankings["stage"] == "B"].head(4)["config"].tolist()
    result = base_config
    used_categories: set[str] = set()
    for name in top_names:
        candidate = config_map[name]
        if name.startswith("b0_aux") or name.startswith("b1_aux") or name.startswith("b2_aux") or name.startswith("b3_aux"):
            if "aux" in used_categories:
                continue
            result = replace(
                result,
                tissue_loss_weight=candidate.tissue_loss_weight,
                hla_loss_weight=candidate.hla_loss_weight,
            )
            used_categories.add("aux")
        elif name.startswith("b4_fusion") or name.startswith("b5_fusion"):
            if "fusion" in used_categories:
                continue
            result = replace(result, fusion_global_weight=candidate.fusion_global_weight)
            used_categories.add("fusion")
        elif name.startswith("b6_dropout") or name.startswith("b7_dropout"):
            if "dropout" in used_categories:
                continue
            result = replace(result, dropout=candidate.dropout)
            used_categories.add("dropout")
        elif name.startswith("b8_lr") or name.startswith("b9_lr"):
            if "lr" in used_categories:
                continue
            result = replace(result, learning_rate=candidate.learning_rate)
            used_categories.add("lr")
        elif name.startswith("b10_wd"):
            if "weight_decay" in used_categories:
                continue
            result = replace(result, weight_decay=candidate.weight_decay)
            used_categories.add("weight_decay")
        elif name.startswith("b11_kernel") or name.startswith("b12_kernel"):
            if "kernel" in used_categories:
                continue
            result = replace(result, kernel_sizes=candidate.kernel_sizes)
            used_categories.add("kernel")
    return replace(result, name="c0_combined_top4", stage="C")


def aggregate_final(
    predictions: pd.DataFrame,
    train: pd.DataFrame,
    base: Any,
    config_name: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    ensemble = (
        predictions.groupby(
            ["sample_id", "pair_id", "target_tissue", "mhc_restriction", "label"], as_index=False
        )["score"]
        .mean()
    )
    summary, per_task = evaluate_predictions(
        config_name + "_3seed_ensemble", 0, train, ensemble, base
    )
    return summary, per_task, ensemble


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--cv-epochs", type=int, default=15)
    parser.add_argument("--final-epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    if cli.smoke:
        cli.output_dir = HERE / "smoke_results"
        cli.cv_folds = 2
        cli.cv_epochs = 1
        cli.final_epochs = 1
    output_dir = cli.output_dir.resolve()
    ctx = RunContext(output_dir)
    ctx.emit(f"RUN START output={output_dir} smoke={cli.smoke}")

    common.enable_original_modules()
    import run_tissuepmhc_e29_multikernel_cnn_oof as e29
    import run_tissuepmhc_neural_baselines_v2 as base

    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = common.resolve_device(cli.device, torch)
    train, test, mappings, peptide_length = common.load_premium_data(base)
    runtime = {
        "torch": torch,
        "nn": nn,
        "DataLoader": DataLoader,
        "TensorDataset": TensorDataset,
        "e29": e29,
        "base": base,
        "device": device,
    }
    ctx.emit(
        f"DATA device={device} train_rows={len(train)} test_rows={len(test)} "
        f"tasks={len(mappings['tasks'])} peptide_length={peptide_length}"
    )
    folds = e29.make_pair_grouped_folds(train, cli.cv_folds, 20260803)
    fold_frame = train[["sample_id", "pair_id", "target_tissue", "mhc_restriction"]].copy()
    fold_frame["fold"] = folds.to_numpy()
    fold_frame.to_csv(output_dir / "cv_fold_assignments.csv", index=False)

    all_summaries: list[dict[str, Any]] = []
    all_tasks: list[pd.DataFrame] = []
    config_map: dict[str, Config] = {}

    stage_a = make_stage_a(cli.smoke)
    for config in stage_a:
        config_map[config.name] = config
        summaries, tasks = evaluate_cv_config(
            config,
            folds,
            cli.cv_folds,
            train,
            mappings,
            peptide_length,
            runtime,
            cli.cv_epochs,
            cli.batch_size,
            ctx,
        )
        all_summaries.extend(summaries)
        all_tasks.extend(tasks)
        pd.DataFrame(all_summaries).to_csv(output_dir / "cv_results.csv", index=False)

    rankings = cv_rankings(pd.DataFrame(all_summaries))
    best_a_name = rankings[rankings["stage"] == "A"].iloc[0]["config"]
    best_a = config_map[str(best_a_name)]
    ctx.emit(f"STAGE A WINNER config={best_a.name}")

    stage_b = make_stage_b(best_a, cli.smoke)
    for config in stage_b:
        config_map[config.name] = config
        summaries, tasks = evaluate_cv_config(
            config,
            folds,
            cli.cv_folds,
            train,
            mappings,
            peptide_length,
            runtime,
            cli.cv_epochs,
            cli.batch_size,
            ctx,
        )
        all_summaries.extend(summaries)
        all_tasks.extend(tasks)
        pd.DataFrame(all_summaries).to_csv(output_dir / "cv_results.csv", index=False)

    rankings = cv_rankings(pd.DataFrame(all_summaries))
    combined = merge_best_stage_b(best_a, rankings, config_map)
    config_map[combined.name] = combined
    summaries, tasks = evaluate_cv_config(
        combined,
        folds,
        cli.cv_folds,
        train,
        mappings,
        peptide_length,
        runtime,
        cli.cv_epochs,
        cli.batch_size,
        ctx,
    )
    all_summaries.extend(summaries)
    all_tasks.extend(tasks)

    cv_results = pd.DataFrame(all_summaries)
    cv_tasks = pd.concat(all_tasks, ignore_index=True)
    rankings = cv_rankings(cv_results)
    cv_results.to_csv(output_dir / "cv_results.csv", index=False)
    cv_tasks.to_csv(output_dir / "cv_per_task_metrics.csv", index=False)
    rankings.to_csv(output_dir / "cv_rankings.csv", index=False)

    locked_name = str(rankings.iloc[0]["config"])
    locked = config_map[locked_name]
    locked_payload = {
        "result_label": RESULT_LABEL,
        "result_label_zh": RESULT_LABEL_ZH,
        **asdict(locked),
    }
    (output_dir / "locked_config.json").write_text(
        json.dumps(locked_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    ctx.emit(
        f"LOCKED CONFIG config={locked.name} CV_AUROC={rankings.iloc[0]['mean_auroc']:.6f} "
        f"CV_AUPRC={rankings.iloc[0]['mean_auprc']:.6f}"
    )

    final_seeds = SEEDS[:1] if cli.smoke else SEEDS
    final_summaries: list[dict[str, Any]] = []
    final_tasks: list[pd.DataFrame] = []
    final_predictions: list[pd.DataFrame] = []
    for seed in final_seeds:
        seed_started = time.perf_counter()
        ctx.emit(f"SEED START seed={seed} config={locked.name}")
        prediction = predict_configuration(
            config=locked,
            epochs=cli.final_epochs,
            batch_size=cli.batch_size,
            fitting=train,
            prediction=test,
            mappings=mappings,
            peptide_length=peptide_length,
            seed=seed,
            ctx=ctx,
            stage="FINAL",
            fold="fixed_test",
            **runtime,
        )
        prediction = attach_pair_ids(prediction, test)
        prediction.insert(0, "seed", seed)
        prediction.insert(0, "model", locked.name)
        summary, per_task = evaluate_predictions(locked.name, seed, train, prediction, base)
        final_summaries.append(summary)
        final_tasks.append(per_task)
        final_predictions.append(prediction)
        elapsed = time.perf_counter() - seed_started
        ctx.emit(
            f"SEED DONE seed={seed} AUROC={summary['mean_task_auroc']:.6f} "
            f"AUPRC={summary['mean_task_auprc']:.6f} time={elapsed:.3f}s"
        )
        ctx.timing(
            scope="seed",
            stage="FINAL",
            config=locked.name,
            fold="fixed_test",
            seed=seed,
            branch="all",
            epoch="",
            elapsed_seconds=f"{elapsed:.6f}",
            status="completed",
        )

    final_summary_df = pd.DataFrame(final_summaries)
    final_task_df = pd.concat(final_tasks, ignore_index=True)
    final_prediction_df = pd.concat(final_predictions, ignore_index=True)
    final_summary_df.to_csv(output_dir / "final_seed_summary.csv", index=False)
    final_task_df.to_csv(output_dir / "final_per_task_metrics.csv", index=False)
    final_prediction_df.to_csv(output_dir / "final_predictions.csv", index=False)

    ensemble_summary, ensemble_tasks, ensemble_predictions = aggregate_final(
        final_prediction_df, train, base, locked.name
    )
    pd.DataFrame([ensemble_summary]).to_csv(output_dir / "ensemble_summary.csv", index=False)
    ensemble_tasks.to_csv(output_dir / "ensemble_per_task_metrics.csv", index=False)
    ensemble_predictions.to_csv(output_dir / "ensemble_predictions.csv", index=False)

    comparison_rows = [
        {
            "method": RESULT_LABEL,
            "aggregation": "mean_seed_metrics",
            "mean_auroc": float(final_summary_df["mean_task_auroc"].mean()),
            "sd_auroc": float(final_summary_df["mean_task_auroc"].std(ddof=1)) if len(final_summary_df) > 1 else float("nan"),
            "mean_auprc": float(final_summary_df["mean_task_auprc"].mean()),
            "sd_auprc": float(final_summary_df["mean_task_auprc"].std(ddof=1)) if len(final_summary_df) > 1 else float("nan"),
        },
        {
            "method": RESULT_LABEL,
            "aggregation": "row_prediction_ensemble",
            "mean_auroc": ensemble_summary["mean_task_auroc"],
            "sd_auroc": float("nan"),
            "mean_auprc": ensemble_summary["mean_task_auprc"],
            "sd_auprc": float("nan"),
        },
    ]
    db_path = (
        EXPERIMENT_ROOT
        / "results"
        / "v7_full_rerun"
        / "dbmtl"
        / "stability_metrics.csv"
    )
    if db_path.exists() and not cli.smoke:
        db = pd.read_csv(db_path).query("model == 'e11_dbmtl'").iloc[0]
        comparison_rows.append(
            {
                "method": "DB-MTL shared heads",
                "aggregation": "mean_seed_metrics",
                "mean_auroc": float(db["mean_auroc_mean"]),
                "sd_auroc": float(db["mean_auroc_std"]),
                "mean_auprc": float(db["mean_auprc_mean"]),
                "sd_auprc": float(db["mean_auprc_std"]),
            }
        )
    pd.DataFrame(comparison_rows).to_csv(output_dir / "comparison_to_dbmtl.csv", index=False)

    metadata = {
        "result_label": RESULT_LABEL,
        "result_label_zh": RESULT_LABEL_ZH,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "device": device,
        "cuda_device": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "train": str(common.TRAIN_PATH),
        "test": str(common.TEST_PATH),
        "cv_folds": cli.cv_folds,
        "cv_epochs": cli.cv_epochs,
        "final_epochs": cli.final_epochs,
        "selection_metric": "0.5 * (task-macro AUROC + task-macro AUPRC)",
        "test_selection_policy": "The fixed test set was not used to choose the locked configuration.",
        "locked_config": asdict(locked),
        "all_configs": [asdict(config) for config in config_map.values()],
        "seeds": final_seeds,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    total = time.perf_counter() - ctx.started
    ctx.timing(
        scope="total",
        stage="ALL",
        config=locked.name,
        fold="",
        seed="",
        branch="all",
        epoch="",
        elapsed_seconds=f"{total:.6f}",
        status="completed",
    )
    ctx.emit(
        f"TOTAL DONE locked={locked.name} mean_seed_AUROC={final_summary_df['mean_task_auroc'].mean():.6f} "
        f"mean_seed_AUPRC={final_summary_df['mean_task_auprc'].mean():.6f} "
        f"ensemble_AUROC={ensemble_summary['mean_task_auroc']:.6f} "
        f"ensemble_AUPRC={ensemble_summary['mean_task_auprc']:.6f} time={total:.3f}s"
    )


if __name__ == "__main__":
    main()
