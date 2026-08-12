#!/usr/bin/env python3
"""Shared, path-safe adapter for human methods not yet evaluated in mouse."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
RESULTS_ROOT = (PROJECT_ROOT / "results").resolve()
DEFAULT_TRAIN = PROJECT_ROOT / "data" / "mousePMHC" / "mousePMHC_train.csv.gz"
DEFAULT_TEST = PROJECT_ROOT / "data" / "mousePMHC" / "mousePMHC_test.csv.gz"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "mousePMHC_human_method_transfer"
DEFAULT_H2_PSEUDO = PROJECT_ROOT / "data" / "processed" / "h2_pseudo_sequences.csv"
DEFAULT_SEEDS = (20260704, 20260705, 20260706)


@dataclass(frozen=True)
class TransferExperiment:
    script: str
    description: str
    human_methods: tuple[str, ...]
    dependencies: dict[str, tuple[str, str]] = field(default_factory=dict)
    overrides: dict[str, Any] = field(default_factory=dict)
    requires_h2_pseudo: bool = False
    allowed_result_inputs: tuple[str, ...] = ()


EXPERIMENTS: dict[str, TransferExperiment] = {
    "neural_single_conditioned": TransferExperiment(
        "run_tissuepmhc_neural_baselines_v2.py",
        "Neural single-task and tissue/H2-conditioned baselines.",
        ("Neural single-task MLP", "Conditioned tissue+MHC ID model"),
        overrides={
            "experiment_plan": "custom",
            "models": [
                "neural_single_task",
                "shared_peptide_encoder_task_heads",
                "conditioned_tissue_hla",
            ],
        },
    ),
    "h2_pseudoseq": TransferExperiment(
        "run_tissuepmhc_hla_pseudoseq.py",
        "H2 pseudo-sequence conditioned model.",
        ("MHC pseudo-sequence conditioned model",),
        requires_h2_pseudo=True,
    ),
    "h2_hybrid": TransferExperiment(
        "run_tissuepmhc_hla_hybrid.py",
        "H2 ID plus pseudo-sequence hybrid.",
        ("MHC ID + pseudo-sequence hybrid",),
        requires_h2_pseudo=True,
    ),
    "tissue_grouping": TransferExperiment(
        "run_tissuepmhc_task_grouping.py",
        "Tissue-grouped hard sharing.",
        ("Tissue-grouped shared heads",),
        dependencies={
            "baseline_per_task": (
                "neural_single_conditioned",
                "per_task_metrics.csv",
            )
        },
        overrides={"models": ["tissue_grouped"]},
    ),
    "selective_grouping": TransferExperiment(
        "run_tissuepmhc_selective_grouping.py",
        "Validation-selected global/H2 hard sharing.",
        ("Selective global/MHC grouping",),
        dependencies={
            "baseline_per_task": (
                "neural_single_conditioned",
                "per_task_metrics.csv",
            )
        },
    ),
    "adaptive_soft_ensemble": TransferExperiment(
        "run_tissuepmhc_soft_ensemble.py",
        "Validation-delta and validation-softmax global/H2 ensembles.",
        (
            "Validation-delta clipped soft ensemble",
            "Validation-softmax ensemble",
        ),
        dependencies={
            "baseline_per_task": (
                "neural_single_conditioned",
                "per_task_metrics.csv",
            )
        },
    ),
    "cagrad": TransferExperiment(
        "run_tissuepmhc_cagrad.py",
        "CAGrad shared-head optimization.",
        ("CAGrad shared heads",),
    ),
    "mmoe_tuning": TransferExperiment(
        "run_tissuepmhc_mmoe_tuning.py",
        "Expanded MMoE expert configurations.",
        ("MMoE 4x256", "MMoE 6x128"),
    ),
    "dbmtl": TransferExperiment(
        "run_tissuepmhc_dbmtl.py",
        "DB-MTL dynamic task balancing.",
        ("DB-MTL shared heads",),
    ),
    "auxiliary_soft": TransferExperiment(
        "run_tissuepmhc_auxiliary_soft_ensemble.py",
        "Auxiliary global plus H2-specific soft ensembles.",
        (
            "Auxiliary-global + plain-MHC dual branch",
            "Auxiliary-global + auxiliary-MHC dual branch",
        ),
    ),
    "mlp_dual_seed_ensemble": TransferExperiment(
        "run_tissuepmhc_e17_seed_ensemble.py",
        "Multi-seed row-level MLP dual-branch ensemble.",
        ("MLP dual-branch seed ensemble",),
        dependencies={
            "branch_predictions": ("auxiliary_soft", "branch_predictions.csv")
        },
    ),
    "tissuepmhc_full": TransferExperiment(
        "run_tissuepmhc_e29_multikernel_cnn_oof.py",
        "Full multi-kernel auxiliary/global plus H2 dual-branch TissuePMHC.",
        ("Full TissuePMHC multi-kernel dual branch",),
        dependencies={
            "e17_per_task": (
                "mlp_dual_seed_ensemble",
                "per_task_metrics.csv",
            )
        },
        overrides={
            "baseline_oof_predictions": (
                PROJECT_ROOT
                / "results"
                / "mousePMHC_phase4_e15_five_seed_confirmation"
                / "mousePMHC_phase4_e15_oof_predictions.csv"
            ),
            "matching_baseline_candidate": (
                "mousePMHC_phase4_e15_e3b_3seed_probability_mean"
            ),
            "fusion_baseline_candidate": (
                "mousePMHC_phase4_e15_e3b_3seed_probability_mean"
            ),
            "experiment_name": "mouse_human_transfer_full_tissuepmhc",
            "candidate_prefix": "mouse_full_tissuepmhc",
        },
        allowed_result_inputs=("baseline_oof_predictions",),
    ),
}


def resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def import_runner(script: Path):
    sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(
        f"mouse_human_transfer_{script.stem}", script
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "parse_args") or not hasattr(module, "run"):
        raise RuntimeError(f"{script.name} must expose parse_args() and run(args)")
    return module


def default_namespace(module, script: Path):
    original = sys.argv
    try:
        sys.argv = [str(script)]
        return module.parse_args()
    finally:
        sys.argv = original


def redirect_result_paths(namespace, target_dir: Path) -> None:
    for name, value in vars(namespace).items():
        if not isinstance(value, Path):
            continue
        candidate = value if value.is_absolute() else PROJECT_ROOT / value
        if is_relative_to(resolved(candidate), RESULTS_ROOT):
            setattr(namespace, name, target_dir / value.name)


def apply_common(namespace, args: argparse.Namespace) -> None:
    if hasattr(namespace, "train"):
        namespace.train = resolved(args.train)
    if hasattr(namespace, "test"):
        namespace.test = resolved(args.test)
    if hasattr(namespace, "device"):
        namespace.device = args.device
    if hasattr(namespace, "seeds"):
        namespace.seeds = list(args.seeds)
    if hasattr(namespace, "seed"):
        namespace.seed = args.seeds[0]
    if args.epochs is not None and hasattr(namespace, "epochs"):
        namespace.epochs = args.epochs
    if hasattr(namespace, "max_tasks"):
        namespace.max_tasks = args.max_tasks


def audit_paths(
    namespace,
    output_root: Path,
    allowed_result_inputs: tuple[str, ...],
) -> None:
    safe_root = resolved(output_root)
    allowed = set(allowed_result_inputs)
    violations = []
    for name, value in vars(namespace).items():
        if not isinstance(value, Path) or name in allowed:
            continue
        candidate = value if value.is_absolute() else PROJECT_ROOT / value
        candidate = resolved(candidate)
        if is_relative_to(candidate, RESULTS_ROOT) and not is_relative_to(
            candidate, safe_root
        ):
            violations.append((name, candidate))
    if violations:
        detail = "\n".join(f"  {name}: {path}" for name, path in violations)
        raise RuntimeError(f"Result-path isolation audit failed:\n{detail}")


def serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [serialize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serialize(item) for key, item in value.items()}
    return value


def parse_entry_args(name: str) -> argparse.Namespace:
    experiment = EXPERIMENTS[name]
    parser = argparse.ArgumentParser(
        description=f"{experiment.description} Mouse human-method transfer."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--h2-pseudo-sequences", type=Path, default=DEFAULT_H2_PSEUDO)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.experiment_name = name
    return args


def run_entry(name: str) -> None:
    experiment = EXPERIMENTS[name]
    args = parse_entry_args(name)
    output_root = resolved(args.output_root)
    target_dir = output_root / name
    if (
        not args.dry_run
        and target_dir.exists()
        and any(target_dir.iterdir())
        and not args.overwrite
    ):
        raise FileExistsError(f"Refusing to overwrite nonempty {target_dir}")
    script = SCRIPTS_DIR / experiment.script
    module = import_runner(script)
    namespace = default_namespace(module, script)
    redirect_result_paths(namespace, target_dir)
    apply_common(namespace, args)

    for argument, (upstream, filename) in experiment.dependencies.items():
        dependency = output_root / upstream / filename
        if not args.dry_run and not dependency.is_file():
            raise FileNotFoundError(
                f"Missing {argument}: {dependency}. Run {upstream} first."
            )
        setattr(namespace, argument, dependency)

    if experiment.requires_h2_pseudo:
        pseudo = resolved(args.h2_pseudo_sequences)
        if not args.dry_run and not pseudo.is_file():
            raise FileNotFoundError(
                f"Missing experimentally sourced H2 pseudo-sequences: {pseudo}. "
                "Do not substitute fabricated sequences."
            )
        namespace.pseudo_sequences = pseudo

    for argument, value in experiment.overrides.items():
        setattr(namespace, argument, value)
    audit_paths(namespace, output_root, experiment.allowed_result_inputs)

    contract = {
        "transfer": "human method -> mousePMHC benchmark",
        "experiment": name,
        "human_methods": list(experiment.human_methods),
        "source_script": str(script),
        "train": str(resolved(args.train)),
        "test": str(resolved(args.test)),
        "output_root": str(output_root),
        "target_dir": str(target_dir),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {key: serialize(value) for key, value in vars(namespace).items()},
    }
    if args.dry_run:
        print(json.dumps(contract, indent=2, ensure_ascii=False))
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "transfer_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    module.run(namespace)
