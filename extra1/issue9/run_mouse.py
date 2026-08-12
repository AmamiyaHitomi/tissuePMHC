#!/usr/bin/env python3
"""Run all mouse architecture controls on the frozen E33 peptide-disjoint folds."""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

try:
    from .common import (
        ROOT,
        SCRIPT_DIR,
        Timer,
        TrainConfig,
        config_dict,
        device_name,
        format_duration,
        load_frozen_folds,
        mappings,
        prediction_rows,
        read_training_data,
        save_outputs,
        sha256,
        validate_member_predictions,
    )
    from .models import require_torch, train_factorized_mmoe, train_shared_heads
except ImportError:
    from common import (
        ROOT,
        SCRIPT_DIR,
        Timer,
        TrainConfig,
        config_dict,
        device_name,
        format_duration,
        load_frozen_folds,
        mappings,
        prediction_rows,
        read_training_data,
        save_outputs,
        sha256,
        validate_member_predictions,
    )
    from models import require_torch, train_factorized_mmoe, train_shared_heads


if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import run_tissuepmhc_baselines as traditional  # noqa: E402


SEEDS = [20260704, 20260705, 20260706, 20260707, 20260708]
MODEL_ORDER = [
    "mouse_blosum62_random_forest",
    "mouse_shared_heads",
    "mouse_factorized_mmoe",
]


def _checkpoint_path(output_dir: Path, model: str, seed: int, fold: int) -> Path:
    return output_dir / "checkpoints" / f"{model}__seed_{seed}__fold_{fold}.csv.gz"


def _checkpoint(
    path: Path,
    held: pd.DataFrame,
    model: str,
    seed: int,
    fold: int,
) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    if (
        len(frame) != len(held)
        or set(frame["sample_id"].astype(str)) != set(held["sample_id"].astype(str))
        or set(frame["model"]) != {model}
        or set(frame["seed"].astype(int)) != {seed}
        or set(frame["fold"].astype(int)) != {fold}
    ):
        raise ValueError(f"Invalid checkpoint: {path}")
    return frame


def _run_random_forest(
    fitting: pd.DataFrame,
    held: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    encoder, template = traditional.get_models(seed)["blosum62_random_forest"]
    scores = pd.Series(index=held.index, dtype=float)
    for task_name, held_task in held.groupby("task_name", sort=True):
        fitting_task = fitting[fitting["task_name"] == task_name]
        estimator = clone(template)
        estimator.fit(
            encoder(fitting_task["peptide_sequence"].tolist()),
            fitting_task["label"].to_numpy(np.int8),
        )
        scores.loc[held_task.index] = traditional.predict_scores(
            estimator, encoder(held_task["peptide_sequence"].tolist())
        )
    if scores.isna().any():
        raise AssertionError("Mouse random forest did not score all held rows.")
    return scores.loc[held.index].to_numpy(float)


def _import_frozen_mmoe(
    train: pd.DataFrame,
    source: Path,
    seeds: list[int],
    row_folds: pd.Series,
) -> pd.DataFrame:
    frame = pd.read_csv(source)
    required = {
        "sample_id",
        "pair_id",
        "target_tissue",
        "mhc_restriction",
        "peptide_sequence",
        "label",
        "seed",
        "fold",
        "score",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Frozen Factorized MMoE predictions miss columns: {missing}")
    frame = frame[frame["seed"].astype(int).isin(seeds)].copy()
    if set(frame["sample_id"].astype(str)) != set(train["sample_id"].astype(str)):
        raise ValueError("Frozen Factorized MMoE predictions do not align with mouse training data.")
    expected_fold = pd.Series(
        row_folds.to_numpy(int), index=train["sample_id"].astype(str)
    )
    observed_expected = frame["sample_id"].astype(str).map(expected_fold)
    if observed_expected.isna().any() or not np.array_equal(
        frame["fold"].to_numpy(int), observed_expected.to_numpy(int)
    ):
        raise ValueError("Frozen Factorized MMoE predictions do not use the frozen manifest folds.")
    frame["species"] = "mouse"
    frame["model"] = "mouse_factorized_mmoe"
    return frame[
        [
            "species",
            "model",
            "seed",
            "fold",
            "sample_id",
            "pair_id",
            "target_tissue",
            "mhc_restriction",
            "peptide_sequence",
            "label",
            "score",
        ]
    ]


def run(args: argparse.Namespace) -> None:
    timer = Timer()
    models = list(dict.fromkeys(args.models))
    unknown = sorted(set(models) - set(MODEL_ORDER))
    if unknown:
        raise ValueError(f"Unknown mouse models: {unknown}")
    seeds = [int(seed) for seed in args.seeds]
    train = read_training_data(args.train, "mouse")
    row_folds, _, split_audit = load_frozen_folds(train, args.fold_manifest, args.folds)
    maps = mappings(train)
    peptide_length = int(train["peptide_sequence"].str.len().max())
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        max_grad_norm=args.max_grad_norm,
        expert_dim=args.expert_dim,
        condition_dim=args.condition_dim,
        gate_hidden_dim=args.gate_hidden_dim,
        n_experts=args.n_experts,
        gate_entropy_weight=args.gate_entropy_weight,
    )
    torch, _, _, _ = require_torch()
    device = device_name(args.device, torch)
    parts: list[pd.DataFrame] = []
    parameter_counts: dict[str, list[int]] = {model: [] for model in models}
    model_seconds: dict[str, float] = defaultdict(float)
    trained_models = [
        model
        for model in models
        if model != "mouse_factorized_mmoe" or args.retrain_factorized
    ]

    for seed in seeds:
        seed_started = time.perf_counter()
        print(f"MOUSE SEED START seed={seed}", flush=True)
        for fold in range(args.folds):
            fold_started = time.perf_counter()
            fitting = train[row_folds != fold].copy()
            held = train[row_folds == fold].copy()
            print(
                f"mouse seed={seed} fold={fold + 1}/{args.folds} "
                f"fit_rows={len(fitting)} held_rows={len(held)}",
                flush=True,
            )
            for model in trained_models:
                path = _checkpoint_path(args.output_dir, model, seed, fold)
                cached = _checkpoint(path, held, model, seed, fold)
                if cached is not None:
                    parts.append(cached)
                    print(f"  reuse {path.name}", flush=True)
                    continue
                operation_started = time.perf_counter()
                print(
                    f"  MODEL START model={model} seed={seed} "
                    f"fold={fold + 1}/{args.folds}",
                    flush=True,
                )
                if model == "mouse_blosum62_random_forest":
                    scores = _run_random_forest(fitting, held, seed)
                    count = 0
                elif model == "mouse_shared_heads":
                    fitted = train_shared_heads(
                        fitting,
                        held,
                        maps["task_to_id"],
                        peptide_length,
                        config,
                        seed,
                        device,
                        f"{model} fold={fold + 1}",
                    )
                    scores, count = fitted.scores, fitted.parameter_count
                elif model == "mouse_factorized_mmoe":
                    fitted = train_factorized_mmoe(
                        fitting,
                        held,
                        maps,
                        peptide_length,
                        config,
                        seed,
                        device,
                        f"{model} fold={fold + 1}",
                    )
                    scores, count = fitted.scores, fitted.parameter_count
                else:
                    raise AssertionError(model)
                output = prediction_rows(held, scores, "mouse", model, seed, fold)
                path.parent.mkdir(parents=True, exist_ok=True)
                output.to_csv(path, index=False)
                parts.append(output)
                parameter_counts[model].append(count)
                operation_seconds = time.perf_counter() - operation_started
                model_seconds[model] += operation_seconds
                print(
                    f"  MODEL DONE model={model} seed={seed} "
                    f"fold={fold + 1}/{args.folds} "
                    f"time={format_duration(operation_seconds)}",
                    flush=True,
                )
            print(
                f"MOUSE FOLD DONE seed={seed} fold={fold + 1}/{args.folds} "
                f"time={format_duration(time.perf_counter() - fold_started)}",
                flush=True,
            )
        print(
            f"MOUSE SEED DONE seed={seed} "
            f"time={format_duration(time.perf_counter() - seed_started)}",
            flush=True,
        )

    if "mouse_factorized_mmoe" in models and not args.retrain_factorized:
        operation_started = time.perf_counter()
        print("MODEL START model=mouse_factorized_mmoe source=frozen_predictions", flush=True)
        parts.append(
            _import_frozen_mmoe(
                train, args.frozen_main_predictions, seeds, row_folds
            )
        )
        operation_seconds = time.perf_counter() - operation_started
        model_seconds["mouse_factorized_mmoe_frozen_import"] += operation_seconds
        print(
            "MODEL DONE model=mouse_factorized_mmoe source=frozen_predictions "
            f"time={format_duration(operation_seconds)}",
            flush=True,
        )

    combined = pd.concat(parts, ignore_index=True)
    validate_member_predictions(
        combined, train, models, {model: seeds for model in models}
    )
    metadata = {
        "experiment": "issue9_mouse_strict_architecture_comparison",
        "status": "completed",
        "test_data_read": False,
        "train": str(args.train.resolve()),
        "train_sha256": sha256(args.train),
        "models": models,
        "seeds": seeds,
        "folds": args.folds,
        "split_audit": split_audit,
        "device": device,
        "train_config": config_dict(config),
        "factorized_source": (
            "retrained_by_issue9"
            if args.retrain_factorized
            else str(args.frozen_main_predictions.resolve())
        ),
        "factorized_source_sha256": (
            None
            if args.retrain_factorized
            else sha256(args.frozen_main_predictions)
        ),
        "parameter_counts_observed": {
            model: sorted(set(values)) for model, values in parameter_counts.items() if values
        },
        "single_seed_rule": f"Report seed {seeds[0]} as the pre-specified single-seed result.",
        "ensemble_rule": f"Equal-weight probability mean across {len(seeds)} pre-specified seeds.",
        "pair_acc_tie_policy": "primary PairAcc uses strict positive_score > negative_score; half-tie variant also reported",
    }
    save_outputs(args.output_dir, combined, metadata)
    print("MOUSE MODEL TIME SUMMARY", flush=True)
    for model, seconds in sorted(model_seconds.items()):
        print(f"  {model}: {format_duration(seconds)}", flush=True)
    print(
        f"MOUSE TOTAL DONE output={args.output_dir} "
        f"time={format_duration(timer.elapsed())}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        type=Path,
        default=ROOT / "data" / "mousePMHC" / "mousePMHC_train.csv.gz",
    )
    parser.add_argument(
        "--fold-manifest",
        type=Path,
        default=ROOT
        / "results"
        / "mousePMHC_phase6_e33_peptide_disjoint_oof"
        / "mousePMHC_phase6_e33_pair_fold_assignments.csv",
    )
    parser.add_argument(
        "--frozen-main-predictions",
        type=Path,
        default=ROOT
        / "results"
        / "mousePMHC_phase6_e33_peptide_disjoint_oof"
        / "mousePMHC_phase6_e33_member_oof_predictions.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "issue9_mouse_strict"
    )
    parser.add_argument("--models", nargs="+", default=MODEL_ORDER)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--retrain-factorized", action="store_true")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--expert-dim", type=int, default=64)
    parser.add_argument("--condition-dim", type=int, default=16)
    parser.add_argument("--gate-hidden-dim", type=int, default=64)
    parser.add_argument("--n-experts", type=int, default=3)
    parser.add_argument("--gate-entropy-weight", type=float, default=0.01)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
