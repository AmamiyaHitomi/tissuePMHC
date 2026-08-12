#!/usr/bin/env python3
"""Run the one-time Phase 6 E32 fixed-test confirmation of frozen E15.

This runner deliberately does not repeat OOF training or model selection.  It
verifies the frozen Phase 4 E15 gate/configuration, trains the exact five
predeclared E3b members on all mousePMHC training rows, and reads the fixed
test exactly once.  Results are written to a new Phase 6 directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd

import run_mousepmhc_phase4_e15_five_seed_confirmation as e15
import run_tissuepmhc_neural_baselines_v2 as base


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "mousePMHC_phase6_e32_e15_fixed_test"
SEEDS = [20260704, 20260705, 20260706, 20260707, 20260708]
EXPECTED_OOF = {
    "mean_task_auroc": 0.83921974942963,
    "mean_task_auprc": 0.8315786629862018,
    "worst6_task_auroc": 0.7101160474059883,
}


def path(relative: str) -> Path:
    return ROOT / relative


def sha256(file: Path) -> str:
    digest = hashlib.sha256()
    with file.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_e15(metadata_path: Path, gate_path: Path) -> tuple[dict, dict]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if metadata.get("test_data_read") is not False or metadata.get("fixed_test_requested") is not False:
        raise RuntimeError("Frozen E15 metadata no longer records an unread fixed test.")
    if metadata.get("frozen_structure") != "E3b task-balanced Factorized MMoE":
        raise RuntimeError("Unexpected frozen E15 structure.")
    if metadata.get("fusion") != "equal-weight probability mean":
        raise RuntimeError("Unexpected frozen E15 fusion.")
    if metadata.get("original_seeds") + metadata.get("new_seeds") != SEEDS:
        raise RuntimeError("Unexpected frozen E15 seed list.")
    if metadata.get("oof_folds") != 3 or metadata.get("oof_split_seed") != 20260711:
        raise RuntimeError("Unexpected frozen E15 OOF protocol.")
    if not gate.get("passed"):
        raise RuntimeError("Frozen E15 OOF gate did not pass.")
    observed = gate.get("five_seed_metrics", {})
    for key, expected in EXPECTED_OOF.items():
        if abs(float(observed.get(key, float("nan"))) - expected) > 1e-12:
            raise RuntimeError(f"Frozen E15 metric mismatch for {key}.")
    return metadata, gate


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite one-time E32 output: {args.output_dir}")
    frozen_metadata, frozen_gate = verify_frozen_e15(args.frozen_metadata, args.frozen_gate)
    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    raw_train = base.read_dataset(args.train)
    e15.validate_train(raw_train)
    # The test is first opened inside this frozen evaluation call.
    ensemble, metrics, parameter_count = e15.full_train_test_predictions(
        args, torch, nn, DataLoader, TensorDataset, raw_train, device, SEEDS
    )
    ensemble["experiment_name"] = EXPERIMENT
    metrics["experiment_name"] = EXPERIMENT
    metrics["candidate"] = "mousePMHC_phase6_e32_frozen_e15_5seed_probability_mean"
    ensemble["candidate"] = "mousePMHC_phase6_e32_frozen_e15_5seed_probability_mean"
    args.output_dir.mkdir(parents=True, exist_ok=False)
    prediction_path = args.output_dir / "mousePMHC_phase6_e32_fixed_test_predictions.csv"
    metric_path = args.output_dir / "mousePMHC_phase6_e32_fixed_test_metrics.csv"
    ensemble.to_csv(prediction_path, index=False)
    metrics.to_csv(metric_path, index=False)
    metadata = {
        "experiment_name": EXPERIMENT,
        "status": "completed_one_time_fixed_test",
        "test_data_read": True,
        "model_selection_on_test": False,
        "frozen_structure": frozen_metadata["frozen_structure"],
        "fusion": frozen_metadata["fusion"],
        "seeds": SEEDS,
        "train": str(args.train),
        "test": str(args.test),
        "train_sha256": sha256(args.train),
        "test_sha256": sha256(args.test),
        "frozen_metadata": str(args.frozen_metadata),
        "frozen_metadata_sha256": sha256(args.frozen_metadata),
        "frozen_gate": str(args.frozen_gate),
        "frozen_gate_sha256": sha256(args.frozen_gate),
        "verified_oof_gate": frozen_gate,
        "device": device,
        "epochs": args.epochs,
        "task_batch_size": args.task_batch_size,
        "batch_size": args.batch_size,
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
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {"predictions": prediction_path.name, "metrics": metric_path.name},
        "interpretation_rule": "Report as confirmation; never change structure, seeds, weights, or thresholds from this test result.",
    }
    (args.output_dir / "mousePMHC_phase6_e32_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(metrics.tail(1).to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=path("data/mousePMHC/mousePMHC_train.csv.gz"))
    parser.add_argument("--test", type=Path, default=path("data/mousePMHC/mousePMHC_test.csv.gz"))
    parser.add_argument("--frozen-metadata", type=Path, default=path("results/mousePMHC_phase4_e15_five_seed_confirmation/mousePMHC_phase4_e15_metadata.json"))
    parser.add_argument("--frozen-gate", type=Path, default=path("results/mousePMHC_phase4_e15_five_seed_confirmation/mousePMHC_phase4_e15_oof_gate.json"))
    parser.add_argument("--output-dir", type=Path, default=path("results/mousePMHC_phase6_e32_e15_fixed_test"))
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
    parser.add_argument("--oof-folds", type=int, default=3, help="Retained only for exact E15 training log formatting.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
