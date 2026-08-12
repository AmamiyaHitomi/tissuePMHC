#!/usr/bin/env python3
"""Run the mouse-selected Factorized MMoE on the human 157-task benchmark.

The architecture and equal-weight task-balanced training rule are transferred
without human-test-set model selection.  Each seed first produces pair-grouped
OOF predictions and is then fitted on the complete training split for the
fixed human test split.  The final reported candidate is the equal-weight
probability average across all requested seeds.

Progress output includes elapsed time for every epoch, fold, seed, and the
complete experiment.  Completed fold/seed prediction shards are reused by
default after an interrupted run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_mousepmhc_phase3_e3_factorized_mmoe_oof as factorized  # noqa: E402
import run_mousepmhc_phase3_e5_famo_mmoe_oof as balanced  # noqa: E402
import run_tissuepmhc_e26_all_in_one as folds  # noqa: E402
import run_tissuepmhc_neural_baselines_v2 as base  # noqa: E402


EXPERIMENT = "human_157tasks_selected_factorized_mmoe"
MEMBER_CANDIDATE = "selected_factorized_mmoe_seed_member"
ENSEMBLE_CANDIDATE = "selected_factorized_mmoe_probability_mean"
MODEL_NAME = "selected_factorized_mmoe"
DEFAULT_SEEDS = [20260704, 20260705, 20260706]
KEYS = ["sample_id", "target_tissue", "mhc_restriction", "label"]
METRICS = ["accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "mcc"]

DEFAULT_TRAIN = (
    PROJECT_ROOT
    / "data"
    / "tissuePMHC_phase7_min200"
    / "tissuePMHC_phase7_min200_train.csv.gz"
)
DEFAULT_TEST = (
    PROJECT_ROOT
    / "data"
    / "tissuePMHC_phase7_min200"
    / "tissuePMHC_phase7_min200_test.csv.gz"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "tissuePMHC_phase7_min200_migrated_44tasks"
    / "e15_selected_factorized_mmoe"
)


def format_duration(seconds: float) -> str:
    hours, remainder = divmod(max(0.0, seconds), 3600.0)
    minutes, seconds = divmod(remainder, 60.0)
    if hours >= 1:
        return f"{int(hours)}h {int(minutes):02d}m {seconds:04.1f}s"
    if minutes >= 1:
        return f"{int(minutes)}m {seconds:04.1f}s"
    return f"{seconds:.1f}s"


def validate_dataset(frame: pd.DataFrame, expected_split: str) -> None:
    required = {
        "dataset",
        "split",
        "sample_id",
        "target_tissue",
        "mhc_restriction",
        "pair_id",
        "label",
        "peptide_sequence",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Human {expected_split} data are missing columns: {missing}")
    if set(frame["split"]) != {expected_split}:
        raise ValueError(
            f"Expected only split={expected_split!r}, found "
            f"{sorted(frame['split'].astype(str).unique())}."
        )
    if not frame["mhc_restriction"].astype(str).str.startswith("HLA-").all():
        raise ValueError("Human data contain a non-HLA MHC restriction.")
    if not frame["peptide_sequence"].astype(str).str.fullmatch(
        r"[ACDEFGHIKLMNPQRSTVWY]+"
    ).all():
        raise ValueError("Human data contain an empty or non-standard peptide.")
    if not frame["label"].isin([0, 1]).all():
        raise ValueError("Labels must be binary 0/1 values.")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample_id values found in {expected_split}.")


def build_mappings(
    raw_train: pd.DataFrame, raw_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_tasks = set(
        raw_train["target_tissue"].astype(str)
        + "||"
        + raw_train["mhc_restriction"].astype(str)
    )
    test_tasks = set(
        raw_test["target_tissue"].astype(str)
        + "||"
        + raw_test["mhc_restriction"].astype(str)
    )
    if train_tasks != test_tasks:
        raise ValueError(
            "Human train/test task sets differ: "
            f"train_only={len(train_tasks - test_tasks)}, "
            f"test_only={len(test_tasks - train_tasks)}."
        )
    train, test, mappings = base.add_task_columns(raw_train, raw_test)
    if len(mappings["tasks"]) != 157:
        raise ValueError(
            f"Expected the Phase-7 157-task benchmark, found "
            f"{len(mappings['tasks'])} tasks."
        )
    return train, test, mappings


def train_model(
    args: argparse.Namespace,
    torch: Any,
    nn: Any,
    fitting: pd.DataFrame,
    mappings: dict[str, Any],
    peptide_length: int,
    device: str,
    seed: int,
    stage: str,
    fold: int | None,
) -> tuple[Any, list[dict[str, object]]]:
    model = factorized.define_model(
        torch,
        nn,
        peptide_length,
        len(mappings["tasks"]),
        len(mappings["tissue_to_id"]),
        len(mappings["hla_to_id"]),
        args,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    task_arrays = balanced.task_arrays(fitting, mappings, peptide_length)
    empty = [item["task_name"] for item in task_arrays if len(item["label"]) == 0]
    if empty:
        raise ValueError(
            f"{stage} fitting data have {len(empty)} empty task(s); "
            "pair-grouped OOF cannot train the full 157-head model."
        )
    steps = args.steps_per_epoch or int(
        np.ceil(max(len(item["label"]) for item in task_arrays) / args.task_batch_size)
    )
    rng = np.random.default_rng(seed)
    location = (
        f"fold={fold + 1}/{args.oof_folds}" if fold is not None else "full-train"
    )
    history: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        losses: list[float] = []
        entropies: list[float] = []
        model.train()
        for _ in range(steps):
            batch = balanced.sample_balanced_batch(
                rng, task_arrays, args.task_batch_size
            )
            peptide, task, tissue, hla, label = [
                torch.as_tensor(value, device=device) for value in batch
            ]
            optimizer.zero_grad(set_to_none=True)
            task_losses, gates = balanced.task_loss_vector(
                torch,
                model,
                peptide,
                task,
                tissue,
                hla,
                label,
                len(task_arrays),
                args.task_batch_size,
            )
            entropy = -(
                gates * torch.log(gates.clamp_min(1e-12))
            ).sum(dim=1).mean()
            objective = task_losses.mean() - args.gate_entropy_weight * entropy
            objective.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                )
            optimizer.step()
            losses.append(float(task_losses.mean().detach().cpu()))
            entropies.append(float(entropy.detach().cpu()))

        history.append(
            {
                "experiment_name": EXPERIMENT,
                "candidate": MEMBER_CANDIDATE,
                "stage": stage,
                "seed": seed,
                "fold": fold + 1 if fold is not None else 0,
                "epoch": epoch,
                "steps": steps,
                "mean_task_balanced_bce": float(np.mean(losses)),
                "mean_gate_entropy": float(np.mean(entropies)),
                "elapsed_seconds": time.perf_counter() - epoch_started,
            }
        )
        print(
            f"Human Factorized MMoE stage={stage} seed={seed} {location} "
            f"epoch={epoch}/{args.epochs} "
            f"task_balanced_bce={np.mean(losses):.5f} "
            f"gate_entropy={np.mean(entropies):.5f} "
            f"elapsed={format_duration(time.perf_counter() - epoch_started)}",
            flush=True,
        )
    return model, history


def predict(
    torch: Any,
    DataLoader: Any,
    TensorDataset: Any,
    model: Any,
    frame: pd.DataFrame,
    peptide_length: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    return balanced.predict(
        torch,
        DataLoader,
        TensorDataset,
        model,
        frame,
        peptide_length,
        batch_size,
        device,
    )


def prediction_shard_valid(
    path: Path, expected: pd.DataFrame, seed: int, split: str
) -> bool:
    if not path.is_file():
        return False
    try:
        frame = pd.read_csv(path)
    except Exception:
        return False
    required = {"split", "candidate", "seed", *KEYS, "score"}
    return (
        required.issubset(frame.columns)
        and len(frame) == len(expected)
        and set(frame["sample_id"]) == set(expected["sample_id"])
        and not frame["sample_id"].duplicated().any()
        and set(frame["seed"]) == {seed}
        and set(frame["split"]) == {split}
    )


def make_prediction_frame(
    source: pd.DataFrame, scores: np.ndarray, seed: int, split: str
) -> pd.DataFrame:
    output = source[KEYS].copy()
    output.insert(0, "split", split)
    output.insert(1, "candidate", MEMBER_CANDIDATE)
    output.insert(2, "seed", seed)
    output["score"] = scores
    return output[["split", "candidate", "seed", *KEYS, "score"]]


def ensemble_predictions(
    members: pd.DataFrame, seeds: list[int], split: str
) -> pd.DataFrame:
    expected_rows = len(members) // len(seeds)
    if (
        len(members) != expected_rows * len(seeds)
        or members.duplicated(["seed", "sample_id"]).any()
        or set(members["seed"]) != set(seeds)
    ):
        raise AssertionError(f"{split} member predictions are incomplete.")
    grouped = members.groupby(KEYS, as_index=False).agg(
        score=("score", "mean"),
        prediction_std_across_seeds=(
            "score",
            lambda values: float(np.std(values, ddof=0)),
        ),
        n_members=("seed", "nunique"),
    )
    if len(grouped) != expected_rows or not (
        grouped["n_members"] == len(seeds)
    ).all():
        raise AssertionError(f"{split} ensemble does not contain every seed.")
    grouped.insert(0, "split", split)
    grouped.insert(1, "candidate", ENSEMBLE_CANDIDATE)
    return grouped[
        [
            "split",
            "candidate",
            *KEYS,
            "score",
            "prediction_std_across_seeds",
            "n_members",
        ]
    ]


def metric_rows(
    predictions: pd.DataFrame,
    candidate: str,
    split: str,
    seed: int | str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (tissue, hla), task in predictions.groupby(
        ["target_tissue", "mhc_restriction"], sort=True
    ):
        rows.append(
            {
                "experiment_name": EXPERIMENT,
                "seed": seed,
                "model": MODEL_NAME,
                "candidate": candidate,
                "split": split,
                "target_tissue": tissue,
                "mhc_restriction": hla,
                "test_rows": len(task),
                **base.evaluate(
                    task["label"].to_numpy(dtype=int),
                    task["score"].to_numpy(dtype=float),
                ),
            }
        )
    return rows


def summary_row(
    rows: pd.DataFrame, seed: int | str, candidate: str, split: str
) -> dict[str, object]:
    summary: dict[str, object] = {
        "experiment_name": EXPERIMENT,
        "seed": seed,
        "model": MODEL_NAME,
        "candidate": candidate,
        "split": split,
        "n_tasks": len(rows),
    }
    weights = rows["test_rows"].to_numpy(dtype=float)
    for metric in METRICS:
        values = rows[metric].to_numpy(dtype=float)
        summary[f"mean_{metric}"] = float(np.mean(values))
        summary[f"median_{metric}"] = float(np.median(values))
    for metric in ["accuracy", "auroc", "auprc"]:
        summary[f"weighted_mean_{metric}"] = float(
            np.average(rows[metric].to_numpy(dtype=float), weights=weights)
        )
    for metric in ["auroc", "auprc"]:
        values = np.sort(rows[metric].to_numpy(dtype=float))
        summary[f"worst_10_mean_{metric}"] = float(
            np.mean(values[: min(10, len(values))])
        )
    return summary


def stability_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split, split_rows in summary.groupby("split", sort=True):
        item: dict[str, object] = {
            "experiment_name": EXPERIMENT,
            "model": MODEL_NAME,
            "split": split,
            "n_seeds": len(split_rows),
        }
        for metric in [
            "mean_auroc",
            "mean_auprc",
            "mean_accuracy",
            "mean_mcc",
            "worst_10_mean_auroc",
        ]:
            values = split_rows[metric].to_numpy(dtype=float)
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
        rows.append(item)
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    total_started = time.perf_counter()
    args.train = args.train.resolve()
    args.test = args.test.resolve()
    args.output_dir = args.output_dir.resolve()
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("--seeds must not contain duplicates.")
    if not args.seeds:
        raise ValueError("At least one seed is required.")

    if args.dry_run:
        payload = {
            "experiment_name": EXPERIMENT,
            "train": str(args.train),
            "test": str(args.test),
            "output_dir": str(args.output_dir),
            "seeds": args.seeds,
            "oof_folds": args.oof_folds,
            "epochs": args.epochs,
            "device": args.device,
            "resume": args.resume,
            "fits": len(args.seeds) * (args.oof_folds + 1),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
        return

    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    raw_train = base.read_dataset(args.train)
    raw_test = base.read_dataset(args.test)
    validate_dataset(raw_train, "train")
    validate_dataset(raw_test, "test")
    if set(raw_train["sample_id"]) & set(raw_test["sample_id"]):
        raise ValueError("Train/test sample_id values overlap.")
    train, test, mappings = build_mappings(raw_train, raw_test)
    peptide_length = int(
        max(
            train["peptide_sequence"].str.len().max(),
            test["peptide_sequence"].str.len().max(),
        )
    )
    assignments = folds.make_pair_grouped_folds(
        train, args.oof_folds, args.oof_split_seed
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.output_dir / "_resume_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    resume_signature = {
        "train": str(args.train),
        "test": str(args.test),
        "seeds": args.seeds,
        "oof_folds": args.oof_folds,
        "oof_split_seed": args.oof_split_seed,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "task_batch_size": args.task_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "expert_dim": args.expert_dim,
        "condition_dim": args.condition_dim,
        "gate_hidden_dim": args.gate_hidden_dim,
        "n_experts": args.n_experts,
        "dropout": args.dropout,
        "gate_entropy_weight": args.gate_entropy_weight,
        "max_grad_norm": args.max_grad_norm,
    }
    signature_path = shard_dir / "resume_signature.json"
    if args.resume and signature_path.is_file():
        previous_signature = json.loads(signature_path.read_text(encoding="utf-8"))
        if previous_signature != resume_signature:
            raise ValueError(
                "Existing resume shards were created with different data or "
                "training arguments. Use --no-resume to retrain and replace them."
            )
    signature_path.write_text(
        json.dumps(resume_signature, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    contract = {
        "migration": (
            "mouse-selected Factorized MMoE -> human Phase-7 157-task benchmark"
        ),
        "experiment": EXPERIMENT,
        "architecture_selection_source": "mouse project only",
        "human_test_used_for_selection": False,
        "train": str(args.train),
        "test": str(args.test),
        "output_dir": str(args.output_dir),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "migration_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    oof_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    history_rows: list[dict[str, object]] = []
    parameter_count: int | None = None

    for seed_index, seed in enumerate(args.seeds, start=1):
        seed_started = time.perf_counter()
        print(
            f"\nHuman Factorized MMoE seed={seed} "
            f"({seed_index}/{len(args.seeds)}) start",
            flush=True,
        )
        for fold in range(args.oof_folds):
            fitting = train[assignments != fold].copy()
            held_out = train[assignments == fold].copy()
            shard = shard_dir / f"oof_seed{seed}_fold{fold + 1}.csv"
            history_shard = (
                shard_dir / f"oof_seed{seed}_fold{fold + 1}_history.csv"
            )
            if args.resume and prediction_shard_valid(
                shard, held_out, seed, "oof"
            ):
                print(
                    f"Human Factorized MMoE seed={seed} "
                    f"fold={fold + 1}/{args.oof_folds} [SKIP completed shard]",
                    flush=True,
                )
                oof_parts.append(pd.read_csv(shard))
                if history_shard.is_file():
                    history_rows.extend(
                        pd.read_csv(history_shard).to_dict(orient="records")
                    )
                continue

            fold_started = time.perf_counter()
            base.set_seed(seed, torch)
            print(
                f"Human Factorized MMoE seed={seed} "
                f"fold={fold + 1}/{args.oof_folds} start "
                f"fit_rows={len(fitting)} holdout_rows={len(held_out)} "
                f"device={device}",
                flush=True,
            )
            model, fold_history = train_model(
                args,
                torch,
                nn,
                fitting,
                mappings,
                peptide_length,
                device,
                seed,
                "oof",
                fold,
            )
            if parameter_count is None:
                parameter_count = int(
                    sum(parameter.numel() for parameter in model.parameters())
                )
            scores = predict(
                torch,
                DataLoader,
                TensorDataset,
                model,
                held_out,
                peptide_length,
                args.batch_size,
                device,
            )
            output = make_prediction_frame(held_out, scores, seed, "oof")
            output.to_csv(shard, index=False)
            pd.DataFrame(fold_history).to_csv(history_shard, index=False)
            oof_parts.append(output)
            history_rows.extend(fold_history)
            print(
                f"Human Factorized MMoE seed={seed} "
                f"fold={fold + 1}/{args.oof_folds} complete "
                f"elapsed={format_duration(time.perf_counter() - fold_started)}",
                flush=True,
            )

        test_shard = shard_dir / f"test_seed{seed}.csv"
        test_history_shard = shard_dir / f"test_seed{seed}_history.csv"
        if args.resume and prediction_shard_valid(
            test_shard, test, seed, "test"
        ):
            print(
                f"Human Factorized MMoE seed={seed} full-train "
                "[SKIP completed shard]",
                flush=True,
            )
            test_parts.append(pd.read_csv(test_shard))
            if test_history_shard.is_file():
                history_rows.extend(
                    pd.read_csv(test_history_shard).to_dict(orient="records")
                )
        else:
            full_started = time.perf_counter()
            base.set_seed(seed, torch)
            print(
                f"Human Factorized MMoE seed={seed} full-train start "
                f"train_rows={len(train)} test_rows={len(test)} device={device}",
                flush=True,
            )
            model, full_history = train_model(
                args,
                torch,
                nn,
                train,
                mappings,
                peptide_length,
                device,
                seed,
                "fixed-test",
                None,
            )
            if parameter_count is None:
                parameter_count = int(
                    sum(parameter.numel() for parameter in model.parameters())
                )
            scores = predict(
                torch,
                DataLoader,
                TensorDataset,
                model,
                test,
                peptide_length,
                args.batch_size,
                device,
            )
            output = make_prediction_frame(test, scores, seed, "test")
            output.to_csv(test_shard, index=False)
            pd.DataFrame(full_history).to_csv(test_history_shard, index=False)
            test_parts.append(output)
            history_rows.extend(full_history)
            print(
                f"Human Factorized MMoE seed={seed} full-train complete "
                f"elapsed={format_duration(time.perf_counter() - full_started)}",
                flush=True,
            )

        print(
            f"Human Factorized MMoE seed={seed} "
            f"({seed_index}/{len(args.seeds)}) complete "
            f"elapsed={format_duration(time.perf_counter() - seed_started)}",
            flush=True,
        )

    oof_members = pd.concat(oof_parts, ignore_index=True)
    test_members = pd.concat(test_parts, ignore_index=True)
    expected_oof = len(train) * len(args.seeds)
    expected_test = len(test) * len(args.seeds)
    if (
        len(oof_members) != expected_oof
        or oof_members.duplicated(["seed", "sample_id"]).any()
    ):
        raise AssertionError("OOF predictions do not cover each row once per seed.")
    if (
        len(test_members) != expected_test
        or test_members.duplicated(["seed", "sample_id"]).any()
    ):
        raise AssertionError("Test predictions do not cover each row once per seed.")

    oof_ensemble = ensemble_predictions(oof_members, args.seeds, "oof")
    test_ensemble = ensemble_predictions(test_members, args.seeds, "test")
    oof_members.to_csv(args.output_dir / "oof_member_predictions.csv", index=False)
    test_members.to_csv(args.output_dir / "test_member_predictions.csv", index=False)
    oof_ensemble.to_csv(
        args.output_dir / "oof_ensemble_predictions.csv", index=False
    )
    test_ensemble.to_csv(
        args.output_dir / "test_ensemble_predictions.csv", index=False
    )

    per_task_rows: list[dict[str, object]] = []
    for split, members, ensemble in [
        ("oof", oof_members, oof_ensemble),
        ("test", test_members, test_ensemble),
    ]:
        for seed in args.seeds:
            per_task_rows.extend(
                metric_rows(
                    members[members["seed"] == seed],
                    MEMBER_CANDIDATE,
                    split,
                    seed,
                )
            )
        per_task_rows.extend(
            metric_rows(
                ensemble,
                ENSEMBLE_CANDIDATE,
                split,
                f"ensemble_{len(args.seeds)}seed",
            )
        )
    per_task = pd.DataFrame(per_task_rows)
    per_task.to_csv(args.output_dir / "per_task_metrics.csv", index=False)

    summaries: list[dict[str, object]] = []
    for (split, seed, candidate), rows in per_task.groupby(
        ["split", "seed", "candidate"], sort=False
    ):
        summaries.append(summary_row(rows, seed, candidate, split))
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output_dir / "summary_metrics.csv", index=False)
    individual_summary = summary[
        summary["candidate"] == MEMBER_CANDIDATE
    ].copy()
    stability_table(individual_summary).to_csv(
        args.output_dir / "stability_metrics.csv", index=False
    )
    pd.DataFrame(history_rows).to_csv(
        args.output_dir / "training_history.csv", index=False
    )

    metadata = {
        "experiment_name": EXPERIMENT,
        "species": "Homo sapiens",
        "benchmark": "tissuePMHC Phase-7 min200 157 tasks",
        "selected_structure": "mouse E3b task-balanced Factorized MMoE",
        "selection_transfer": (
            "architecture/hyperparameters frozen from mouse; no human test "
            "result used for structure or seed selection"
        ),
        "fusion": "equal-weight probability mean across requested seeds",
        "train": str(args.train),
        "test": str(args.test),
        "n_train_rows": len(train),
        "n_test_rows": len(test),
        "n_tasks": len(mappings["tasks"]),
        "n_tissues": len(mappings["tissue_to_id"]),
        "n_hlas": len(mappings["hla_to_id"]),
        "seeds": args.seeds,
        "oof_folds": args.oof_folds,
        "oof_split_seed": args.oof_split_seed,
        "device": device,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "task_batch_size": args.task_batch_size,
        "prediction_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "expert_dim": args.expert_dim,
        "condition_dim": args.condition_dim,
        "gate_hidden_dim": args.gate_hidden_dim,
        "n_experts": args.n_experts,
        "dropout": args.dropout,
        "gate_entropy_weight": args.gate_entropy_weight,
        "max_grad_norm": args.max_grad_norm,
        "parameter_count": parameter_count,
        "total_elapsed_seconds": time.perf_counter() - total_started,
        "resume_shards_retained": True,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nHuman Factorized MMoE summary", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(
        f"Human Factorized MMoE total complete "
        f"elapsed={format_duration(time.perf_counter() - total_started)}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--oof-folds", type=int, default=3)
    parser.add_argument("--oof-split-seed", type=int, default=20260711)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--task-batch-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--expert-dim", type=int, default=64)
    parser.add_argument("--condition-dim", type=int, default=16)
    parser.add_argument("--gate-hidden-dim", type=int, default=64)
    parser.add_argument("--n-experts", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gate-entropy-weight", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse complete per-fold and per-seed prediction shards.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
