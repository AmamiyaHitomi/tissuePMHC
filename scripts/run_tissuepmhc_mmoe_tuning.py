#!/usr/bin/env python3
"""Run E10b tuned MMoE configurations for tissuePMHC.

Roadmap role: E10b-on-E10 tuning.

E10 showed that MMoE selective sharing is useful but still weaker than E8 soft
ensemble. This script keeps the same E2/E8 main line and tests a small set of
tuned MMoE variants:

1. More experts.
2. Wider experts.
3. Mild gate entropy regularization.

`gate entropy regularization` means adding a small term that encourages gates
not to collapse too early onto a single expert. Positive values maximize gate
entropy, so each task can keep more than one expert active while learning.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_tissuepmhc_mmoe as e10
import run_tissuepmhc_neural_baselines_v2 as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPEAT_SEEDS = [20260704, 20260705, 20260706]

CONFIGS: dict[str, dict[str, Any]] = {
    "e10b_6experts_128": {
        "n_experts": 6,
        "expert_dim": 128,
        "gate_entropy_weight": 0.0,
    },
    "e10b_8experts_128": {
        "n_experts": 8,
        "expert_dim": 128,
        "gate_entropy_weight": 0.0,
    },
    "e10b_4experts_256": {
        "n_experts": 4,
        "expert_dim": 256,
        "gate_entropy_weight": 0.0,
    },
    "e10b_6experts_entropy": {
        "n_experts": 6,
        "expert_dim": 128,
        "gate_entropy_weight": 0.01,
    },
}


def project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


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
        for metric in e10.METRICS:
            comparison[f"delta_{metric}"] = float(row[metric]) - float(baseline[metric])
        comparisons.append(comparison)
    return comparisons


def args_for_config(args: argparse.Namespace, config_name: str) -> argparse.Namespace:
    config_args = copy.copy(args)
    for key, value in CONFIGS[config_name].items():
        setattr(config_args, key, value)
    config_args.model_name = config_name
    config_args.experiment_name = "E10b_MMoE_tuning"
    return config_args


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
    selected_configs = args.configs
    result_rows: list[dict[str, object]] = []
    gate_rows: list[dict[str, object]] = []

    print(f"device: {device}")
    print(f"n_tasks: {len(mappings['tasks'])}")
    print(f"configs: {selected_configs}")

    for config_name in selected_configs:
        config_args = args_for_config(args, config_name)
        print(
            f"config: {config_name} "
            f"n_experts={config_args.n_experts} "
            f"expert_dim={config_args.expert_dim} "
            f"gate_entropy_weight={config_args.gate_entropy_weight}"
        )
        for seed in args.seeds:
            metrics, gates = e10.run_one_seed(
                config_args,
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
    e8_external = read_external_rows(args.e8_per_task, "e8a_fixed_average")
    e10_external = read_external_rows(args.e10_per_task, "e10_mmoe")
    e9_external = read_external_rows(args.e9_per_task, "e9_e2_cagrad")
    external_comparisons = []
    external_comparisons.extend(
        compare_external(result_rows, e2_external, "shared_peptide_encoder_task_heads", str(args.e2_baseline_per_task))
    )
    external_comparisons.extend(compare_external(result_rows, e8_external, "e8a_fixed_average", str(args.e8_per_task)))
    external_comparisons.extend(compare_external(result_rows, e10_external, "e10_mmoe", str(args.e10_per_task)))
    external_comparisons.extend(compare_external(result_rows, e9_external, "e9_e2_cagrad", str(args.e9_per_task)))
    external_comparisons.sort(
        key=lambda row: (str(row["candidate_model"]), str(row["baseline_model"]), int(row["seed"]))
    )

    base.write_csv(args.per_task_output, base.METRIC_COLUMNS, result_rows)
    base.write_csv(args.summary_output, base.SUMMARY_COLUMNS, summary_rows)
    base.write_csv(args.stability_output, base.STABILITY_COLUMNS, stability_rows)
    base.write_csv(args.external_comparison_output, e10.COMPARISON_COLUMNS, external_comparisons)
    base.write_csv(args.gate_output, e10.GATE_COLUMNS, gate_rows)

    metadata = {
        "train": str(args.train),
        "test": str(args.test),
        "configs": {name: CONFIGS[name] for name in selected_configs},
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
        "--configs",
        nargs="+",
        choices=sorted(CONFIGS),
        default=["e10b_6experts_128", "e10b_4experts_256"],
        help="Tuned MMoE configs to run. Default runs two lower-cost configs.",
    )
    parser.add_argument(
        "--e2-baseline-per-task",
        type=Path,
        default=project_path("results/tissuePMHC_neural_baselines_v2/per_task_metrics.csv"),
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
        "--e10-per-task",
        type=Path,
        default=project_path("results/tissuePMHC_mmoe/per_task_metrics.csv"),
    )
    parser.add_argument(
        "--per-task-output",
        type=Path,
        default=project_path("results/tissuePMHC_mmoe_tuning/per_task_metrics.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=project_path("results/tissuePMHC_mmoe_tuning/summary_metrics.csv"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=project_path("results/tissuePMHC_mmoe_tuning/metadata.json"),
    )
    parser.add_argument(
        "--stability-output",
        type=Path,
        default=project_path("results/tissuePMHC_mmoe_tuning/stability_metrics.csv"),
    )
    parser.add_argument(
        "--external-comparison-output",
        type=Path,
        default=project_path("results/tissuePMHC_mmoe_tuning/external_comparison_metrics.csv"),
    )
    parser.add_argument(
        "--gate-output",
        type=Path,
        default=project_path("results/tissuePMHC_mmoe_tuning/gate_weight_history.csv"),
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
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-tasks", type=int, default=0, help="Optional smoke-test limit on the number of tasks.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
