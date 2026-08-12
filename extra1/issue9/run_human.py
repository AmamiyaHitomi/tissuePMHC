#!/usr/bin/env python3
"""Run all human architecture controls on the frozen E31 peptide-disjoint folds."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

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
    from .models import (
        percentile_rank_fusion,
        require_torch,
        train_mlp_branch,
        train_shared_heads,
    )
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
    from models import (
        percentile_rank_fusion,
        require_torch,
        train_mlp_branch,
        train_shared_heads,
    )


if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import run_tissuepmhc_baselines as traditional  # noqa: E402


SEEDS = [20260704, 20260705, 20260706]
MODEL_ORDER = [
    "human_onehot_logistic_regression",
    "human_blosum62_random_forest",
    "human_shared_heads",
    "human_mlp_dual_branch",
    "human_auxiliary_dual_branch",
    "human_tissuepmhc_net",
]
TRAINED_MODELS = MODEL_ORDER[:-1]


def _checkpoint_path(output_dir: Path, model: str, seed: int, fold: int) -> Path:
    return output_dir / "checkpoints" / f"{model}__seed_{seed}__fold_{fold}.csv.gz"


def _load_checkpoint(
    path: Path,
    held: pd.DataFrame,
    model: str,
    seed: int,
    fold: int,
) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    expected = set(held["sample_id"].astype(str))
    if (
        len(frame) != len(held)
        or set(frame["sample_id"].astype(str)) != expected
        or set(frame["model"]) != {model}
        or set(frame["seed"].astype(int)) != {seed}
        or set(frame["fold"].astype(int)) != {fold}
    ):
        raise ValueError(f"Invalid checkpoint: {path}")
    return frame


def _write_checkpoint(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _traditional_spec(model: str, seed: int) -> tuple[Callable[[list[str]], np.ndarray], Any]:
    models = traditional.get_models(seed)
    key = {
        "human_onehot_logistic_regression": "onehot_logistic_regression",
        "human_blosum62_random_forest": "blosum62_random_forest",
    }[model]
    return models[key]


def _run_traditional(
    model: str,
    fitting: pd.DataFrame,
    held: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    encoder, estimator_template = _traditional_spec(model, seed)
    scores = pd.Series(index=held.index, dtype=float)
    for task_name, held_task in held.groupby("task_name", sort=True):
        fitting_task = fitting[fitting["task_name"] == task_name]
        estimator = clone(estimator_template)
        x_fit = encoder(fitting_task["peptide_sequence"].tolist())
        x_held = encoder(held_task["peptide_sequence"].tolist())
        estimator.fit(x_fit, fitting_task["label"].to_numpy(np.int8))
        scores.loc[held_task.index] = traditional.predict_scores(estimator, x_held)
    if scores.isna().any():
        raise AssertionError(f"{model} did not score all held-out rows.")
    return scores.loc[held.index].to_numpy(float)


def _run_dual_models(
    fitting: pd.DataFrame,
    held: pd.DataFrame,
    maps: dict[str, Any],
    peptide_length: int,
    config: TrainConfig,
    seed: int,
    fold: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    global_plain = train_mlp_branch(
        fitting,
        held,
        maps["task_to_id"],
        maps["tissue_to_id"],
        maps["mhc_to_id"],
        peptide_length,
        config,
        seed,
        device,
        False,
        f"human_mlp_global_plain fold={fold + 1}",
    )
    global_aux = train_mlp_branch(
        fitting,
        held,
        maps["task_to_id"],
        maps["tissue_to_id"],
        maps["mhc_to_id"],
        peptide_length,
        config,
        seed,
        device,
        True,
        f"human_aux_global fold={fold + 1}",
    )
    mhc_scores = pd.Series(index=held.index, dtype=float)
    mhc_parameters = 0
    for mhc, held_mhc in held.groupby("mhc_restriction", sort=True):
        fitting_mhc = fitting[fitting["mhc_restriction"] == mhc]
        local_tasks = sorted(set(fitting_mhc["task_name"]) & set(held_mhc["task_name"]))
        local_mapping = {task: index for index, task in enumerate(local_tasks)}
        result = train_mlp_branch(
            fitting_mhc,
            held_mhc,
            local_mapping,
            maps["tissue_to_id"],
            maps["mhc_to_id"],
            peptide_length,
            config,
            seed,
            device,
            False,
            f"human_hla_plain[{mhc}] fold={fold + 1}",
        )
        mhc_scores.loc[held_mhc.index] = result.scores
        mhc_parameters += result.parameter_count
    if mhc_scores.isna().any():
        raise AssertionError("HLA branch did not score every held-out row.")
    hla = mhc_scores.loc[held.index].to_numpy(float)
    plain_fused = percentile_rank_fusion(held, global_plain.scores, hla)
    auxiliary_fused = percentile_rank_fusion(held, global_aux.scores, hla)
    counts = {
        "human_mlp_dual_branch": global_plain.parameter_count + mhc_parameters,
        "human_auxiliary_dual_branch": global_aux.parameter_count + mhc_parameters,
    }
    return plain_fused, auxiliary_fused, counts


def _import_frozen_main(
    train: pd.DataFrame,
    source: Path,
    selected_seeds: list[int],
    row_folds: pd.Series,
) -> pd.DataFrame:
    frozen = pd.read_csv(source)
    required = {"sample_id", "seed", "fold", "score", "label"}
    missing = sorted(required - set(frozen.columns))
    if missing:
        raise ValueError(f"Frozen TissuePMHC predictions miss columns: {missing}")
    frozen = frozen[frozen["seed"].astype(int).isin(selected_seeds)].copy()
    annotations = train[["sample_id", "pair_id", "peptide_sequence"]]
    frozen = frozen.merge(annotations, on="sample_id", how="left", validate="many_to_one")
    if frozen[["pair_id", "peptide_sequence"]].isna().any().any():
        raise ValueError("Frozen main-model predictions do not align with training rows.")
    expected_fold = pd.Series(
        row_folds.to_numpy(int), index=train["sample_id"].astype(str)
    )
    observed_expected = frozen["sample_id"].astype(str).map(expected_fold)
    if observed_expected.isna().any() or not np.array_equal(
        frozen["fold"].to_numpy(int), observed_expected.to_numpy(int)
    ):
        raise ValueError("Frozen TissuePMHC predictions do not use the frozen manifest folds.")
    frozen["species"] = "human"
    frozen["model"] = "human_tissuepmhc_net"
    return frozen[
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
    selected_models = list(dict.fromkeys(args.models))
    unknown = sorted(set(selected_models) - set(MODEL_ORDER))
    if unknown:
        raise ValueError(f"Unknown human models: {unknown}")
    seeds = [int(seed) for seed in args.seeds]
    if not seeds:
        raise ValueError("At least one seed is required.")

    train = read_training_data(args.train, "human")
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
        tissue_loss_weight=args.tissue_loss_weight,
        mhc_loss_weight=args.mhc_loss_weight,
        max_grad_norm=args.max_grad_norm,
    )
    torch, _, _, _ = require_torch()
    device = device_name(args.device, torch)
    members: list[pd.DataFrame] = []
    parameter_counts: dict[str, list[int]] = {model: [] for model in selected_models}
    model_seconds: dict[str, float] = defaultdict(float)

    requested_trained = [model for model in selected_models if model in TRAINED_MODELS]
    for seed in seeds:
        seed_started = time.perf_counter()
        print(f"HUMAN SEED START seed={seed}", flush=True)
        for fold in range(args.folds):
            fold_started = time.perf_counter()
            fitting = train[row_folds != fold].copy()
            held = train[row_folds == fold].copy()
            print(
                f"human seed={seed} fold={fold + 1}/{args.folds} "
                f"fit_rows={len(fitting)} held_rows={len(held)}",
                flush=True,
            )
            for model in requested_trained:
                checkpoint = _checkpoint_path(args.output_dir, model, seed, fold)
                cached = _load_checkpoint(checkpoint, held, model, seed, fold)
                if cached is not None:
                    print(f"  reuse {checkpoint.name}", flush=True)
                    members.append(cached)
                    continue
                if model in {
                    "human_mlp_dual_branch",
                    "human_auxiliary_dual_branch",
                }:
                    peer_models = {
                        "human_mlp_dual_branch",
                        "human_auxiliary_dual_branch",
                    } & set(requested_trained)
                    missing_peers = [
                        peer
                        for peer in peer_models
                        if _load_checkpoint(
                            _checkpoint_path(args.output_dir, peer, seed, fold),
                            held,
                            peer,
                            seed,
                            fold,
                        )
                        is None
                    ]
                    if not missing_peers:
                        continue
                    operation_started = time.perf_counter()
                    print(
                        f"  MODEL START model=human_dual_branch_bundle seed={seed} "
                        f"fold={fold + 1}/{args.folds}",
                        flush=True,
                    )
                    plain, auxiliary, counts = _run_dual_models(
                        fitting, held, maps, peptide_length, config, seed, fold, device
                    )
                    operation_seconds = time.perf_counter() - operation_started
                    model_seconds["human_dual_branch_bundle"] += operation_seconds
                    print(
                        f"  MODEL DONE model=human_dual_branch_bundle seed={seed} "
                        f"fold={fold + 1}/{args.folds} "
                        f"time={format_duration(operation_seconds)}",
                        flush=True,
                    )
                    for peer, scores in [
                        ("human_mlp_dual_branch", plain),
                        ("human_auxiliary_dual_branch", auxiliary),
                    ]:
                        if peer not in peer_models:
                            continue
                        output = prediction_rows(
                            held, scores, "human", peer, seed, fold
                        )
                        _write_checkpoint(
                            _checkpoint_path(args.output_dir, peer, seed, fold), output
                        )
                        parameter_counts[peer].append(counts[peer])
                    cached = _load_checkpoint(checkpoint, held, model, seed, fold)
                    if cached is None:
                        raise AssertionError(f"Dual-branch checkpoint was not created for {model}.")
                    members.append(cached)
                    continue
                if model in {
                    "human_onehot_logistic_regression",
                    "human_blosum62_random_forest",
                }:
                    operation_started = time.perf_counter()
                    print(
                        f"  MODEL START model={model} seed={seed} "
                        f"fold={fold + 1}/{args.folds}",
                        flush=True,
                    )
                    scores = _run_traditional(model, fitting, held, seed)
                    count = 0
                elif model == "human_shared_heads":
                    operation_started = time.perf_counter()
                    print(
                        f"  MODEL START model={model} seed={seed} "
                        f"fold={fold + 1}/{args.folds}",
                        flush=True,
                    )
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
                else:
                    raise AssertionError(model)
                output = prediction_rows(held, scores, "human", model, seed, fold)
                _write_checkpoint(checkpoint, output)
                parameter_counts[model].append(count)
                members.append(output)
                operation_seconds = time.perf_counter() - operation_started
                model_seconds[model] += operation_seconds
                print(
                    f"  MODEL DONE model={model} seed={seed} "
                    f"fold={fold + 1}/{args.folds} "
                    f"time={format_duration(operation_seconds)}",
                    flush=True,
                )
            print(
                f"HUMAN FOLD DONE seed={seed} fold={fold + 1}/{args.folds} "
                f"time={format_duration(time.perf_counter() - fold_started)}",
                flush=True,
            )
        print(
            f"HUMAN SEED DONE seed={seed} "
            f"time={format_duration(time.perf_counter() - seed_started)}",
            flush=True,
        )

    # Load any dual checkpoint created while processing its peer but not appended.
    loaded_keys = {
        (str(frame["model"].iloc[0]), int(frame["seed"].iloc[0]), int(frame["fold"].iloc[0]))
        for frame in members
        if len(frame)
    }
    for model in requested_trained:
        for seed in seeds:
            for fold in range(args.folds):
                key = (model, seed, fold)
                if key not in loaded_keys:
                    held = train[row_folds == fold]
                    cached = _load_checkpoint(
                        _checkpoint_path(args.output_dir, model, seed, fold),
                        held,
                        model,
                        seed,
                        fold,
                    )
                    if cached is None:
                        raise AssertionError(f"Missing completed checkpoint for {key}.")
                    members.append(cached)
                    loaded_keys.add(key)

    if "human_tissuepmhc_net" in selected_models:
        operation_started = time.perf_counter()
        print("MODEL START model=human_tissuepmhc_net source=frozen_predictions", flush=True)
        members.append(
            _import_frozen_main(train, args.frozen_main_predictions, seeds, row_folds)
        )
        operation_seconds = time.perf_counter() - operation_started
        model_seconds["human_tissuepmhc_net_frozen_import"] += operation_seconds
        print(
            "MODEL DONE model=human_tissuepmhc_net source=frozen_predictions "
            f"time={format_duration(operation_seconds)}",
            flush=True,
        )

    combined = pd.concat(members, ignore_index=True)
    seeds_by_model = {model: seeds for model in selected_models}
    validate_member_predictions(combined, train, selected_models, seeds_by_model)
    metadata = {
        "experiment": "issue9_human_strict_architecture_comparison",
        "status": "completed",
        "test_data_read": False,
        "train": str(args.train.resolve()),
        "train_sha256": sha256(args.train),
        "models": selected_models,
        "seeds": seeds,
        "folds": args.folds,
        "split_audit": split_audit,
        "device": device,
        "train_config": config_dict(config),
        "parameter_counts_observed": {
            model: sorted(set(values)) for model, values in parameter_counts.items() if values
        },
        "frozen_main_predictions": (
            str(args.frozen_main_predictions.resolve())
            if "human_tissuepmhc_net" in selected_models
            else None
        ),
        "frozen_main_predictions_sha256": (
            sha256(args.frozen_main_predictions)
            if "human_tissuepmhc_net" in selected_models
            else None
        ),
        "pair_acc_tie_policy": "primary PairAcc uses strict positive_score > negative_score; half-tie variant also reported",
    }
    save_outputs(args.output_dir, combined, metadata)
    print("HUMAN MODEL TIME SUMMARY", flush=True)
    for model, seconds in sorted(model_seconds.items()):
        print(f"  {model}: {format_duration(seconds)}", flush=True)
    print(
        f"HUMAN TOTAL DONE output={args.output_dir} "
        f"time={format_duration(timer.elapsed())}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        type=Path,
        default=ROOT
        / "data"
        / "tissuePMHC_phase7_min200"
        / "tissuePMHC_phase7_min200_train.csv.gz",
    )
    parser.add_argument(
        "--fold-manifest",
        type=Path,
        default=ROOT
        / "results"
        / "tissuePMHC_phase7_min200_e31_peptide_disjoint_oof"
        / "pair_fold_assignments.csv",
    )
    parser.add_argument(
        "--frozen-main-predictions",
        type=Path,
        default=ROOT
        / "results"
        / "tissuePMHC_phase7_min200_e31_peptide_disjoint_oof"
        / "member_oof_predictions.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "issue9_human_strict"
    )
    parser.add_argument("--models", nargs="+", default=MODEL_ORDER)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--tissue-loss-weight", type=float, default=0.1)
    parser.add_argument("--mhc-loss-weight", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
