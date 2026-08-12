#!/usr/bin/env python3
"""Path-safe adapters for the remaining Human occurrence-equal experiments.

This module imports the original method implementations from ``scripts`` but
forces every dataset, dependency, and output path into the Human
occurrence-equal experiment tree.  It deliberately refuses to read legacy
``results`` artifacts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
LEGACY_RESULTS_ROOT = (PROJECT_ROOT / "results").resolve()
DEFAULT_TRAIN = PROJECT_ROOT / "data" / "humanPMHC_occurence_equal_dataset" / "humanPMHC_train.csv.gz"
DEFAULT_TEST = PROJECT_ROOT / "data" / "humanPMHC_occurence_equal_dataset" / "humanPMHC_test.csv.gz"
DEFAULT_OUTPUT_ROOT = HERE / "results" / "v7_full_rerun"
DEFAULT_HLA_PSEUDO = PROJECT_ROOT / "data" / "processed" / "hla_pseudo_sequences_occurrence_equal.csv"
DEFAULT_SEEDS = (20260704, 20260705, 20260706)


@dataclass(frozen=True)
class Experiment:
    script: str
    paper_methods: tuple[str, ...]
    overrides: dict[str, Any] = field(default_factory=dict)
    dependencies: dict[str, tuple[str, str]] = field(default_factory=dict)
    requires_pseudo: bool = False


# Each experiment maps exactly to rows retained in the Human v7 architecture
# survey.  Completed occurrence-equal E0/E2/E14a/E29 runs are not listed here.
EXPERIMENTS: dict[str, Experiment] = {
    "neural_single_conditioned": Experiment(
        "run_tissuepmhc_neural_baselines_v2.py",
        ("Neural single-task MLP", "Conditioned tissue and HLA IDs"),
        overrides={"experiment_plan": "custom", "models": ["neural_single_task", "conditioned_tissue_hla"]},
    ),
    "hla_pseudoseq": Experiment(
        "run_tissuepmhc_hla_pseudoseq.py",
        ("HLA pseudo-sequence conditioned",),
        overrides={"models": ["conditioned_hla_pseudoseq"]},
        requires_pseudo=True,
    ),
    "hla_hybrid": Experiment(
        "run_tissuepmhc_hla_hybrid.py",
        ("HLA ID + pseudo-sequence hybrid",),
        overrides={"models": ["conditioned_hla_hybrid"]},
        requires_pseudo=True,
    ),
    "task_grouping": Experiment(
        "run_tissuepmhc_task_grouping.py",
        ("HLA-grouped hard sharing", "Tissue-grouped hard sharing"),
        overrides={"models": ["hla_grouped", "tissue_grouped"]},
        dependencies={"baseline_per_task": ("neural_single_conditioned", "per_task_metrics.csv")},
    ),
    "selective_grouping": Experiment(
        "run_tissuepmhc_selective_grouping.py",
        ("Selective global/HLA grouping",),
        dependencies={"baseline_per_task": ("neural_single_conditioned", "per_task_metrics.csv")},
    ),
    "adaptive_soft_ensemble": Experiment(
        "run_tissuepmhc_soft_ensemble.py",
        ("Fixed global/HLA dual-branch average", "Validation-delta clipped ensemble", "Validation-softmax ensemble"),
        dependencies={"baseline_per_task": ("neural_single_conditioned", "per_task_metrics.csv")},
    ),
    "famo": Experiment(
        "run_tissuepmhc_famo.py", ("FAMO shared heads",), overrides={"models": ["e5_famo"]}
    ),
    "mmoe": Experiment("run_tissuepmhc_mmoe.py", ("MMoE",)),
    "mmoe_tuning": Experiment(
        "run_tissuepmhc_mmoe_tuning.py", ("MMoE, 4 experts x width 256", "MMoE, 6 experts x width 128")
    ),
    "dbmtl": Experiment(
        "run_tissuepmhc_dbmtl.py", ("DB-MTL shared heads",), overrides={"models": ["e11_dbmtl"]}
    ),
    "pair_ranking": Experiment(
        "run_tissuepmhc_pair_ranking.py", ("Pair-ranking objective",), overrides={"models": ["e12_pair_ranking"]}
    ),
    "auxiliary_tasks": Experiment(
        "run_tissuepmhc_auxiliary_tasks.py",
        ("Tissue/HLA auxiliary supervision",),
        overrides={"models": ["e13_aux_tissue_hla"]},
    ),
    # E14a itself is reused from the completed new-data run.  This source run
    # is still needed to produce the previously missing E14b and branch-level
    # predictions consumed by the E17 row-level ensemble.
    "auxiliary_soft": Experiment(
        "run_tissuepmhc_auxiliary_soft_ensemble.py",
        ("Auxiliary-global + auxiliary-HLA dual branch", "E17 branch inputs"),
    ),
    "mlp_dual_seed_ensemble": Experiment(
        "run_tissuepmhc_e17_seed_ensemble.py",
        ("MLP dual-branch rank average",),
        dependencies={"branch_predictions": ("auxiliary_soft", "branch_predictions.csv")},
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


def import_runner(script: Path) -> ModuleType:
    sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(f"human_occurrence_equal_{script.stem}", script)
    if spec is None or spec.loader is None:
        raise ImportError(script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "parse_args") or not hasattr(module, "run"):
        raise RuntimeError(f"{script.name} must expose parse_args() and run(args)")
    return module


def default_namespace(module: ModuleType, script: Path) -> argparse.Namespace:
    original = sys.argv
    try:
        sys.argv = [str(script)]
        return module.parse_args()
    finally:
        sys.argv = original


def redirect_result_paths(namespace: argparse.Namespace, target_dir: Path) -> None:
    for name, value in vars(namespace).items():
        if not isinstance(value, Path):
            continue
        candidate = resolved(value if value.is_absolute() else PROJECT_ROOT / value)
        if is_relative_to(candidate, LEGACY_RESULTS_ROOT):
            setattr(namespace, name, target_dir / value.name)


def patch_literal_na_tissue(module: ModuleType) -> list[tuple[ModuleType, Any]]:
    """Preserve the literal Human tissue label ``NA`` in every imported base."""
    patched: list[tuple[ModuleType, Any]] = []
    candidates = [module]
    candidates.extend(value for value in vars(module).values() if isinstance(value, ModuleType))
    seen: set[int] = set()
    for candidate in candidates:
        if id(candidate) in seen or not hasattr(candidate, "read_dataset"):
            continue
        seen.add(id(candidate))
        original = candidate.read_dataset

        def read_dataset(path: Path, _original=original):
            frame = _original(path)
            if "target_tissue" in frame:
                frame["target_tissue"] = frame["target_tissue"].fillna("NA")
            return frame

        candidate.read_dataset = read_dataset
        patched.append((candidate, original))
    return patched


def audit_paths(namespace: argparse.Namespace, output_root: Path) -> None:
    violations = []
    for name, value in vars(namespace).items():
        if not isinstance(value, Path):
            continue
        candidate = resolved(value if value.is_absolute() else PROJECT_ROOT / value)
        if is_relative_to(candidate, LEGACY_RESULTS_ROOT):
            violations.append((name, candidate))
    if violations:
        detail = "\n".join(f"  {name}: {path}" for name, path in violations)
        raise RuntimeError(f"Legacy-result isolation audit failed:\n{detail}")
    if not is_relative_to(resolved(output_root), resolved(HERE)):
        raise RuntimeError(f"Human output root must remain under {HERE}: {output_root}")


def serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [serialize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serialize(item) for key, item in value.items()}
    return value


def configure(name: str, cli: argparse.Namespace) -> tuple[ModuleType, argparse.Namespace, Path, dict[str, Any]]:
    experiment = EXPERIMENTS[name]
    script = SCRIPTS_DIR / experiment.script
    module = import_runner(script)
    namespace = default_namespace(module, script)
    output_root = resolved(cli.output_root)
    target_dir = output_root / name
    redirect_result_paths(namespace, target_dir)
    for attribute, value in (("train", cli.train), ("test", cli.test), ("device", cli.device)):
        if hasattr(namespace, attribute):
            setattr(namespace, attribute, resolved(value) if isinstance(value, Path) else value)
    if hasattr(namespace, "seeds"):
        namespace.seeds = list(cli.seeds)
    if hasattr(namespace, "seed"):
        namespace.seed = int(cli.seeds[0])
    if cli.epochs is not None and hasattr(namespace, "epochs"):
        namespace.epochs = cli.epochs
    if hasattr(namespace, "max_tasks"):
        namespace.max_tasks = cli.max_tasks
    dependency_root = resolved(cli.dependency_root) if cli.dependency_root else output_root
    for argument, (upstream, filename) in experiment.dependencies.items():
        setattr(namespace, argument, dependency_root / upstream / filename)
    if experiment.requires_pseudo:
        namespace.pseudo_sequences = resolved(cli.hla_pseudo_sequences)
    for argument, value in experiment.overrides.items():
        setattr(namespace, argument, value)
    audit_paths(namespace, output_root)
    contract = {
        "suite": "human_v7_occurrence_equal_only",
        "experiment": name,
        "paper_methods": list(experiment.paper_methods),
        "source_script": str(script.resolve()),
        "train": str(resolved(cli.train)),
        "test": str(resolved(cli.test)),
        "target_dir": str(target_dir),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {key: serialize(value) for key, value in vars(namespace).items()},
    }
    return module, namespace, target_dir, contract


def run_entry(name: str, cli: argparse.Namespace) -> None:
    module, namespace, target_dir, contract = configure(name, cli)
    if cli.dry_run:
        print(json.dumps(contract, indent=2, ensure_ascii=False))
        return
    if target_dir.exists() and any(target_dir.iterdir()) and not cli.overwrite:
        raise FileExistsError(f"Refusing to overwrite nonempty {target_dir}")
    for argument, value in vars(namespace).items():
        if argument in EXPERIMENTS[name].dependencies and not value.is_file():
            raise FileNotFoundError(f"Missing dependency {argument}: {value}")
    if EXPERIMENTS[name].requires_pseudo and not namespace.pseudo_sequences.is_file():
        raise FileNotFoundError(namespace.pseudo_sequences)
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "human_run_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    patched = patch_literal_na_tissue(module)
    try:
        module.run(namespace)
    finally:
        for patched_module, original in patched:
            patched_module.read_dataset = original


def entry_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=tuple(EXPERIMENTS))
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dependency-root", type=Path)
    parser.add_argument("--hla-pseudo-sequences", type=Path, default=DEFAULT_HLA_PSEUDO)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser
