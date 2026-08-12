#!/usr/bin/env python3
"""Run the three mouse controls missing from the matched strict table.

All models use the frozen E33 connected-component peptide-disjoint folds and
read only the mouse training pool. The implementations mirror the corresponding
human Issue 9 controls:

* one-hot logistic regression fitted separately in every tissue--H2 task;
* an unguided global/H2-specific MLP dual branch with task-wise rank fusion;
* the same dual branch with tissue and H2 auxiliary supervision on the global
  branch.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from extra1.issue9.common import (  # noqa: E402
    Timer,
    TrainConfig,
    atomic_json,
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
from extra1.issue9.models import (  # noqa: E402
    percentile_rank_fusion,
    require_torch,
    train_mlp_branch,
)

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
import run_tissuepmhc_baselines as traditional  # noqa: E402


SEEDS = [20260704, 20260705, 20260706, 20260707, 20260708]
MODEL_ORDER = [
    "mouse_onehot_logistic_regression",
    "mouse_mlp_dual_branch",
    "mouse_auxiliary_dual_branch",
]
DUAL_MODELS = {
    "mouse_mlp_dual_branch",
    "mouse_auxiliary_dual_branch",
}
DEFAULT_TRAIN = PROJECT_ROOT / "data" / "mousePMHC" / "mousePMHC_train.csv.gz"
DEFAULT_FOLD_MANIFEST = (
    PROJECT_ROOT
    / "results"
    / "mousePMHC_phase6_e33_peptide_disjoint_oof"
    / "mousePMHC_phase6_e33_pair_fold_assignments.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "mousePMHC_human_method_transfer"
    / "strict_missing_three"
)


def checkpoint_path(output_dir: Path, model: str, seed: int, fold: int) -> Path:
    return output_dir / "checkpoints" / f"{model}__seed_{seed}__fold_{fold}.csv.gz"


def load_checkpoint(
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
        or set(frame["sample_id"].astype(str))
        != set(held["sample_id"].astype(str))
        or set(frame["model"]) != {model}
        or set(frame["seed"].astype(int)) != {seed}
        or set(frame["fold"].astype(int)) != {fold}
    ):
        raise ValueError(f"Invalid checkpoint: {path}")
    return frame


def write_checkpoint(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def run_onehot(
    fitting: pd.DataFrame,
    held: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    encoder, template = traditional.get_models(seed)["onehot_logistic_regression"]
    scores = pd.Series(index=held.index, dtype=float)
    for task_name, held_task in held.groupby("task_name", sort=True):
        fitting_task = fitting[fitting["task_name"] == task_name]
        estimator = clone(template)
        estimator.fit(
            encoder(fitting_task["peptide_sequence"].tolist()),
            fitting_task["label"].to_numpy(np.int8),
        )
        scores.loc[held_task.index] = traditional.predict_scores(
            estimator,
            encoder(held_task["peptide_sequence"].tolist()),
        )
    if scores.isna().any():
        raise AssertionError("Mouse one-hot logistic regression missed held-out rows.")
    return scores.loc[held.index].to_numpy(float)


def run_dual_bundle(
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
        f"mouse_mlp_global_plain fold={fold + 1}",
    )
    global_auxiliary = train_mlp_branch(
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
        f"mouse_auxiliary_global fold={fold + 1}",
    )

    h2_scores = pd.Series(index=held.index, dtype=float)
    h2_parameters = 0
    for restriction, held_h2 in held.groupby("mhc_restriction", sort=True):
        fitting_h2 = fitting[fitting["mhc_restriction"] == restriction]
        local_tasks = sorted(
            set(fitting_h2["task_name"]) & set(held_h2["task_name"])
        )
        local_mapping = {task: index for index, task in enumerate(local_tasks)}
        branch = train_mlp_branch(
            fitting_h2,
            held_h2,
            local_mapping,
            maps["tissue_to_id"],
            maps["mhc_to_id"],
            peptide_length,
            config,
            seed,
            device,
            False,
            f"mouse_h2_plain[{restriction}] fold={fold + 1}",
        )
        h2_scores.loc[held_h2.index] = branch.scores
        h2_parameters += branch.parameter_count
    if h2_scores.isna().any():
        raise AssertionError("Mouse H2-specific branch missed held-out rows.")

    ordered_h2 = h2_scores.loc[held.index].to_numpy(float)
    plain_fused = percentile_rank_fusion(held, global_plain.scores, ordered_h2)
    auxiliary_fused = percentile_rank_fusion(
        held, global_auxiliary.scores, ordered_h2
    )
    counts = {
        "mouse_mlp_dual_branch": global_plain.parameter_count + h2_parameters,
        "mouse_auxiliary_dual_branch": (
            global_auxiliary.parameter_count + h2_parameters
        ),
    }
    return plain_fused, auxiliary_fused, counts


def dry_run(args: argparse.Namespace) -> None:
    missing = [
        path
        for path in (args.train.resolve(), args.fold_manifest.resolve())
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing strict input(s):\n" + "\n".join(f"  {path}" for path in missing)
        )
    payload = {
        "experiment": "mouse_strict_missing_matched_methods",
        "models": list(dict.fromkeys(args.models)),
        "train": str(args.train.resolve()),
        "fold_manifest": str(args.fold_manifest.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "seeds": [int(seed) for seed in args.seeds],
        "folds": args.folds,
        "epochs": args.epochs,
        "device": args.device,
        "fixed_test_read": False,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def run(args: argparse.Namespace) -> None:
    if args.dry_run:
        dry_run(args)
        return

    timer = Timer()
    models = list(dict.fromkeys(args.models))
    unknown = sorted(set(models) - set(MODEL_ORDER))
    if unknown:
        raise ValueError(f"Unknown strict mouse models: {unknown}")
    if not models:
        raise ValueError("At least one model is required.")
    seeds = [int(seed) for seed in args.seeds]
    if not seeds:
        raise ValueError("At least one seed is required.")

    train = read_training_data(args.train.resolve(), "mouse")
    row_folds, _, split_audit = load_frozen_folds(
        train, args.fold_manifest.resolve(), args.folds
    )
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
    parameter_counts: dict[str, list[int]] = {model: [] for model in models}
    model_seconds: dict[str, float] = defaultdict(float)

    for seed in seeds:
        print(f"MOUSE STRICT SEED START seed={seed}", flush=True)
        for fold in range(args.folds):
            fitting = train[row_folds != fold].copy()
            held = train[row_folds == fold].copy()
            print(
                f"mouse strict seed={seed} fold={fold + 1}/{args.folds} "
                f"fit_rows={len(fitting)} held_rows={len(held)}",
                flush=True,
            )

            if "mouse_onehot_logistic_regression" in models:
                model = "mouse_onehot_logistic_regression"
                path = checkpoint_path(args.output_dir, model, seed, fold)
                cached = load_checkpoint(path, held, model, seed, fold)
                if cached is not None:
                    members.append(cached)
                    print(f"  reuse {path.name}", flush=True)
                else:
                    started = time.perf_counter()
                    scores = run_onehot(fitting, held, seed)
                    output = prediction_rows(
                        held, scores, "mouse", model, seed, fold
                    )
                    write_checkpoint(path, output)
                    members.append(output)
                    model_seconds[model] += time.perf_counter() - started

            requested_dual = [model for model in models if model in DUAL_MODELS]
            cached_dual: dict[str, pd.DataFrame] = {}
            missing_dual: list[str] = []
            for model in requested_dual:
                path = checkpoint_path(args.output_dir, model, seed, fold)
                cached = load_checkpoint(path, held, model, seed, fold)
                if cached is None:
                    missing_dual.append(model)
                else:
                    cached_dual[model] = cached
                    members.append(cached)
                    print(f"  reuse {path.name}", flush=True)

            if missing_dual:
                started = time.perf_counter()
                plain, auxiliary, counts = run_dual_bundle(
                    fitting,
                    held,
                    maps,
                    peptide_length,
                    config,
                    seed,
                    fold,
                    device,
                )
                bundle_seconds = time.perf_counter() - started
                score_map = {
                    "mouse_mlp_dual_branch": plain,
                    "mouse_auxiliary_dual_branch": auxiliary,
                }
                for model in missing_dual:
                    output = prediction_rows(
                        held, score_map[model], "mouse", model, seed, fold
                    )
                    write_checkpoint(
                        checkpoint_path(args.output_dir, model, seed, fold),
                        output,
                    )
                    members.append(output)
                    parameter_counts[model].append(counts[model])
                    model_seconds[model] += bundle_seconds

    combined = pd.concat(members, ignore_index=True)
    validate_member_predictions(
        combined,
        train,
        models,
        {model: seeds for model in models},
    )
    metadata = {
        "experiment": "mouse_strict_missing_matched_methods",
        "status": "completed",
        "test_data_read": False,
        "human_method_mapping": {
            "mouse_onehot_logistic_regression": "human_onehot_logistic_regression",
            "mouse_mlp_dual_branch": "human_mlp_dual_branch",
            "mouse_auxiliary_dual_branch": "human_auxiliary_dual_branch",
        },
        "train": str(args.train.resolve()),
        "train_sha256": sha256(args.train.resolve()),
        "fold_manifest": str(args.fold_manifest.resolve()),
        "fold_manifest_sha256": sha256(args.fold_manifest.resolve()),
        "models": models,
        "seeds": seeds,
        "folds": args.folds,
        "split_audit": split_audit,
        "device": device,
        "train_config": config_dict(config),
        "fusion": (
            "Within-task percentile ranks from the global and H2-specific "
            "branches are averaged with equal weight."
        ),
        "parameter_counts_observed": {
            model: sorted(set(values))
            for model, values in parameter_counts.items()
            if values
        },
        "ensemble_rule": (
            f"Equal-weight row-level score mean across {len(seeds)} seeds."
        ),
        "comparison_scope": (
            "Identical frozen E33 peptide-disjoint folds; mouse training pool only."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.output_dir / "transfer_contract.json",
        {
            "transfer": "three human strict controls -> mouse strict benchmark",
            "source_human_runner": str(
                PROJECT_ROOT / "extra1" / "issue9" / "run_human.py"
            ),
            **metadata,
        },
    )
    save_outputs(args.output_dir, combined, metadata)
    print("MOUSE STRICT MODEL TIME SUMMARY", flush=True)
    for model, seconds in sorted(model_seconds.items()):
        print(f"  {model}: {format_duration(seconds)}", flush=True)
    print(
        f"MOUSE STRICT DONE output={args.output_dir} "
        f"time={format_duration(timer.elapsed())}",
        flush=True,
    )


def parse_args(
    default_models: list[str] | None = None,
    default_output: Path = DEFAULT_OUTPUT,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--fold-manifest", type=Path, default=DEFAULT_FOLD_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_ORDER,
        default=list(default_models or MODEL_ORDER),
    )
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


def main(
    default_models: list[str] | None = None,
    default_output: Path = DEFAULT_OUTPUT,
) -> None:
    run(parse_args(default_models=default_models, default_output=default_output))


if __name__ == "__main__":
    main()
