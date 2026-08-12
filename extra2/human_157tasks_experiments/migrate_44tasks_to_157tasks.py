#!/usr/bin/env python3
"""Run legacy 44-task human experiments on the Phase-7 157-task benchmark.

This is an adapter around the archived experiment implementations in ``scripts/``.
It changes the input dataset and redirects every result path into an isolated
migration root.  The archived 44-task results are never used as fallbacks.

Examples
--------
List the supported migrations without training:

    python extra2/human_157tasks_experiments/migrate_44tasks_to_157tasks.py --list

Preview the experiments intended for the expanded Table 5:

    python extra2/human_157tasks_experiments/migrate_44tasks_to_157tasks.py \
        --preset table5 --dry-run

Run every compatible archived experiment:

    python extra2/human_157tasks_experiments/migrate_44tasks_to_157tasks.py \
        --preset all --device cuda

Run selected experiments and store them under a custom isolated root:

    python extra2/human_157tasks_experiments/migrate_44tasks_to_157tasks.py \
        --experiments e0_traditional e6_task_grouping e10_mmoe \
        --output-root results/my_157task_migration
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
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
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "results" / "tissuePMHC_phase7_min200_migrated_44tasks"
)
RESULTS_ROOT = (PROJECT_ROOT / "results").resolve()
DEFAULT_SEEDS = (20260704, 20260705, 20260706)
HLA_PSEUDO_EXPERIMENTS = frozenset({"e4_hla_pseudoseq", "e4b_hla_hybrid"})


@dataclass(frozen=True)
class Experiment:
    script: str
    description: str
    dependencies: dict[str, tuple[str, str]] = field(default_factory=dict)
    optional_dependencies: dict[str, tuple[str, str]] = field(default_factory=dict)
    overrides: dict[str, Any] = field(default_factory=dict)


EXPERIMENTS: dict[str, Experiment] = {
    "e0_traditional": Experiment(
        "run_tissuepmhc_baselines.py",
        "E0: five per-task traditional baselines (one-hot/BLOSUM62).",
    ),
    "e1_e2_e3_neural": Experiment(
        "run_tissuepmhc_neural_baselines_v2.py",
        "E1/E2/E3: single-task neural, shared heads, and conditioned model.",
        overrides={
            "experiment_plan": "custom",
            "models": [
                "neural_single_task",
                "shared_peptide_encoder_task_heads",
                "conditioned_tissue_hla",
            ],
        },
    ),
    "e4_hla_pseudoseq": Experiment(
        "run_tissuepmhc_hla_pseudoseq.py",
        "E4: HLA pseudo-sequence conditioned model.",
    ),
    "e4b_hla_hybrid": Experiment(
        "run_tissuepmhc_hla_hybrid.py",
        "E4b: HLA ID plus pseudo-sequence hybrid.",
    ),
    "e5_famo": Experiment(
        "run_tissuepmhc_famo.py",
        "E5: FAMO-style adaptive task weighting.",
    ),
    "e6_task_grouping": Experiment(
        "run_tissuepmhc_task_grouping.py",
        "E6: HLA-grouped and tissue-grouped hard sharing.",
        dependencies={
            "baseline_per_task": ("e1_e2_e3_neural", "per_task_metrics.csv"),
        },
    ),
    "e7_selective_grouping": Experiment(
        "run_tissuepmhc_selective_grouping.py",
        "E7: validation-selected global/HLA grouping.",
        dependencies={
            "baseline_per_task": ("e1_e2_e3_neural", "per_task_metrics.csv"),
        },
    ),
    "e8_soft_ensemble": Experiment(
        "run_tissuepmhc_soft_ensemble.py",
        "E8: global/HLA soft-ensemble variants.",
        dependencies={
            "baseline_per_task": ("e1_e2_e3_neural", "per_task_metrics.csv"),
        },
    ),
    "e9_cagrad": Experiment(
        "run_tissuepmhc_cagrad.py",
        "E9: CAGrad gradient-conflict handling.",
        dependencies={
            "e2_baseline_per_task": ("e1_e2_e3_neural", "per_task_metrics.csv"),
            "famo_per_task": ("e5_famo", "per_task_metrics.csv"),
            "e7_per_task": ("e7_selective_grouping", "per_task_metrics.csv"),
            "e8_per_task": ("e8_soft_ensemble", "per_task_metrics.csv"),
        },
    ),
    "e10_mmoe": Experiment(
        "run_tissuepmhc_mmoe.py",
        "E10: multi-gate mixture-of-experts.",
        dependencies={
            "e2_baseline_per_task": ("e1_e2_e3_neural", "per_task_metrics.csv"),
            "famo_per_task": ("e5_famo", "per_task_metrics.csv"),
            "e7_per_task": ("e7_selective_grouping", "per_task_metrics.csv"),
            "e8_per_task": ("e8_soft_ensemble", "per_task_metrics.csv"),
        },
        optional_dependencies={
            "e9_per_task": ("e9_cagrad", "per_task_metrics.csv"),
        },
    ),
    "e10b_mmoe_tuning": Experiment(
        "run_tissuepmhc_mmoe_tuning.py",
        "E10b: wider/more-expert MMoE variants.",
        dependencies={
            "e2_baseline_per_task": ("e1_e2_e3_neural", "per_task_metrics.csv"),
            "e8_per_task": ("e8_soft_ensemble", "per_task_metrics.csv"),
            "e10_per_task": ("e10_mmoe", "per_task_metrics.csv"),
        },
        optional_dependencies={
            "e9_per_task": ("e9_cagrad", "per_task_metrics.csv"),
        },
    ),
    "e11_dbmtl": Experiment(
        "run_tissuepmhc_dbmtl.py",
        "E11: DB-MTL dynamic task balancing.",
        dependencies={
            "e2_baseline_per_task": ("e1_e2_e3_neural", "per_task_metrics.csv"),
            "e8_per_task": ("e8_soft_ensemble", "per_task_metrics.csv"),
            "e10_per_task": ("e10_mmoe", "per_task_metrics.csv"),
            "e10b_per_task": ("e10b_mmoe_tuning", "per_task_metrics.csv"),
        },
        optional_dependencies={
            "e9_per_task": ("e9_cagrad", "per_task_metrics.csv"),
        },
    ),
    "e12_pair_ranking": Experiment(
        "run_tissuepmhc_pair_ranking.py",
        "E12: BCE plus pair-ranking objective.",
        dependencies={
            "e2_baseline_per_task": ("e1_e2_e3_neural", "per_task_metrics.csv"),
            "e8_per_task": ("e8_soft_ensemble", "per_task_metrics.csv"),
            "e10_per_task": ("e10_mmoe", "per_task_metrics.csv"),
            "e11_per_task": ("e11_dbmtl", "per_task_metrics.csv"),
        },
    ),
    "e13_auxiliary": Experiment(
        "run_tissuepmhc_auxiliary_tasks.py",
        "E13: tissue/HLA auxiliary supervision.",
        dependencies={
            "e2_baseline_per_task": ("e1_e2_e3_neural", "per_task_metrics.csv"),
            "e8_per_task": ("e8_soft_ensemble", "per_task_metrics.csv"),
            "e10_per_task": ("e10_mmoe", "per_task_metrics.csv"),
            "e12_per_task": ("e12_pair_ranking", "per_task_metrics.csv"),
        },
    ),
    "e14_auxiliary_soft": Experiment(
        "run_tissuepmhc_auxiliary_soft_ensemble.py",
        "E14: auxiliary global plus HLA-specific soft ensemble.",
        dependencies={
            "e2_per_task": ("e1_e2_e3_neural", "per_task_metrics.csv"),
            "e8_per_task": ("e8_soft_ensemble", "per_task_metrics.csv"),
            "e13_per_task": ("e13_auxiliary", "per_task_metrics.csv"),
        },
    ),
    "e15_fusion_ablation": Experiment(
        "run_tissuepmhc_e15_fusion_ablation.py",
        "E15: probability/rank fusion ablation.",
        dependencies={
            "branch_predictions": ("e14_auxiliary_soft", "branch_predictions.csv"),
        },
    ),
    "e16_mc_dropout": Experiment(
        "run_tissuepmhc_e16_mc_dropout_ensemble.py",
        "E16: MC-dropout ensemble.",
    ),
    "e18_global_weight": Experiment(
        "run_tissuepmhc_e18_global_weight_selection.py",
        "E18: validation-only global fusion-weight selection.",
        dependencies={
            "e14_branch_predictions": (
                "e14_auxiliary_soft",
                "branch_predictions.csv",
            ),
        },
    ),
    "e19_training_ensemble": Experiment(
        "run_tissuepmhc_e19_training_ensemble.py",
        "E19: training-trajectory ensemble.",
    ),
    "e20_swa": Experiment(
        "run_tissuepmhc_e20_swa.py",
        "E20: stochastic weight averaging.",
    ),
    "e21_gradient_similarity": Experiment(
        "run_tissuepmhc_e21_gradient_similarity_auxiliary.py",
        "E21: gradient-similarity auxiliary gating.",
        dependencies={
            "e14_branch_predictions": (
                "e14_auxiliary_soft",
                "branch_predictions.csv",
            ),
        },
    ),
    "e22_nash_mtl": Experiment(
        "run_tissuepmhc_e22_periodic_nash_mtl.py",
        "E22: periodic Nash-MTL weighting.",
        dependencies={
            "e14_branch_predictions": (
                "e14_auxiliary_soft",
                "branch_predictions.csv",
            ),
        },
    ),
    "e23_forkmerge": Experiment(
        "run_tissuepmhc_e23_forkmerge.py",
        "E23: ForkMerge-style auxiliary weighting.",
        dependencies={
            "e14_branch_predictions": (
                "e14_auxiliary_soft",
                "branch_predictions.csv",
            ),
        },
    ),
    "e24_auto_lambda": Experiment(
        "run_tissuepmhc_e24_auto_lambda.py",
        "E24: Auto-Lambda-style auxiliary weighting.",
        dependencies={
            "e14_branch_predictions": (
                "e14_auxiliary_soft",
                "branch_predictions.csv",
            ),
        },
    ),
    "e25_hla_ple": Experiment(
        "run_tissuepmhc_e25_hla_structured_ple.py",
        "E25: HLA-structured PLE branch.",
        dependencies={
            "e14_global_candidate_metrics": (
                "e14_auxiliary_soft",
                "candidate_metrics.csv",
            ),
        },
    ),
}


TABLE5_EXPERIMENTS = [
    "e0_traditional",
    "e1_e2_e3_neural",
    "e4_hla_pseudoseq",
    "e4b_hla_hybrid",
    "e5_famo",
    "e6_task_grouping",
    "e7_selective_grouping",
    "e8_soft_ensemble",
    "e9_cagrad",
    "e10_mmoe",
    "e10b_mmoe_tuning",
    "e11_dbmtl",
    "e12_pair_ranking",
    "e13_auxiliary",
    "e14_auxiliary_soft",
]


def resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def import_experiment(script: Path):
    sys.path.insert(0, str(SCRIPTS_DIR))
    module_name = f"phase7_migration_{script.stem}"
    spec = importlib.util.spec_from_file_location(module_name, script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "parse_args") or not hasattr(module, "run"):
        raise RuntimeError(f"{script.name} must expose parse_args() and run(args)")
    return module


def default_namespace(module, script: Path):
    original_argv = sys.argv
    try:
        sys.argv = [str(script)]
        return module.parse_args()
    finally:
        sys.argv = original_argv


def redirect_result_paths(namespace, target_dir: Path) -> None:
    """Redirect every archived result Path to the isolated experiment folder."""
    for name, value in vars(namespace).items():
        if not isinstance(value, Path):
            continue
        candidate = value if value.is_absolute() else PROJECT_ROOT / value
        if is_relative_to(resolved(candidate), RESULTS_ROOT):
            setattr(namespace, name, target_dir / value.name)


def apply_common_overrides(
    namespace,
    *,
    train: Path,
    test: Path,
    device: str,
    seeds: tuple[int, ...],
    epochs: int | None,
    max_tasks: int,
) -> None:
    if hasattr(namespace, "train"):
        namespace.train = train
    if hasattr(namespace, "test"):
        namespace.test = test
    if hasattr(namespace, "device"):
        namespace.device = device
    if hasattr(namespace, "seeds"):
        namespace.seeds = list(seeds)
    if hasattr(namespace, "seed"):
        namespace.seed = seeds[0]
    if epochs is not None and hasattr(namespace, "epochs"):
        namespace.epochs = epochs
    if hasattr(namespace, "max_tasks"):
        namespace.max_tasks = max_tasks


def apply_dependencies(
    namespace,
    experiment: Experiment,
    output_root: Path,
    *,
    require_files: bool,
) -> None:
    for argument, (upstream, filename) in experiment.dependencies.items():
        dependency = output_root / upstream / filename
        if require_files and not dependency.is_file():
            raise FileNotFoundError(
                f"Missing dependency for {argument}: {dependency}. "
                f"Run {upstream} first."
            )
        setattr(namespace, argument, dependency)
    for argument, (upstream, filename) in experiment.optional_dependencies.items():
        setattr(namespace, argument, output_root / upstream / filename)


def audit_paths(namespace, output_root: Path) -> None:
    """Refuse any remaining path into the archived result tree."""
    safe_root = resolved(output_root)
    violations: list[tuple[str, Path]] = []
    for name, value in vars(namespace).items():
        if not isinstance(value, Path):
            continue
        candidate = value if value.is_absolute() else PROJECT_ROOT / value
        candidate = resolved(candidate)
        if is_relative_to(candidate, RESULTS_ROOT) and not is_relative_to(
            candidate, safe_root
        ):
            violations.append((name, candidate))
    if violations:
        detail = "\n".join(f"  {name}: {path}" for name, path in violations)
        raise RuntimeError(
            "Migration isolation audit failed; archived result paths remain:\n"
            f"{detail}"
        )


def serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [serialize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serialize(item) for key, item in value.items()}
    return value


def prepare_phase7_hla_pseudo_sequences(
    train: Path,
    output_root: Path,
    dry_run: bool,
) -> Path:
    """Build a migration-local table from the repository's IPD-IMGT/HLA FASTAs."""
    output = output_root / "_derived_inputs" / "hla_pseudo_sequences.csv"
    if dry_run:
        return output

    builder_script = SCRIPTS_DIR / "build_hla_pseudo_sequences.py"
    builder = import_experiment(builder_script)
    namespace = default_namespace(builder, builder_script)
    namespace.train = train
    namespace.output = output
    namespace.from_train = True
    for fasta in namespace.fasta:
        if not resolved(fasta).is_file():
            raise FileNotFoundError(
                "Missing IPD-IMGT/HLA protein FASTA required for E4/E4b: "
                f"{resolved(fasta)}"
            )
    builder.run(namespace)
    return output


def run_one(args: argparse.Namespace) -> None:
    name = args._run_one
    experiment = EXPERIMENTS[name]
    output_root = resolved(args.output_root)
    target_dir = output_root / name
    script = SCRIPTS_DIR / experiment.script

    if (
        not args.dry_run
        and target_dir.exists()
        and any(target_dir.iterdir())
        and not args.overwrite
    ):
        raise FileExistsError(
            f"{target_dir} is not empty. Refusing to overwrite it; "
            "use --overwrite only after reviewing the existing run."
        )
    if not args.dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)

    module = import_experiment(script)
    namespace = default_namespace(module, script)
    redirect_result_paths(namespace, target_dir)
    apply_common_overrides(
        namespace,
        train=resolved(args.train),
        test=resolved(args.test),
        device=args.device,
        seeds=tuple(args.seeds),
        epochs=args.epochs,
        max_tasks=args.max_tasks,
    )
    for key, value in experiment.overrides.items():
        setattr(namespace, key, value)
    if name in HLA_PSEUDO_EXPERIMENTS:
        namespace.pseudo_sequences = prepare_phase7_hla_pseudo_sequences(
            train=resolved(args.train),
            output_root=output_root,
            dry_run=args.dry_run,
        )
    apply_dependencies(
        namespace,
        experiment,
        output_root,
        require_files=not args.dry_run,
    )
    audit_paths(namespace, output_root)

    contract = {
        "migration": "44-task implementation -> Phase-7 157-task benchmark",
        "experiment": name,
        "description": experiment.description,
        "source_script": str(script),
        "train": str(resolved(args.train)),
        "test": str(resolved(args.test)),
        "output_root": str(output_root),
        "target_dir": str(target_dir),
        "archived_44task_results_read": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {key: serialize(value) for key, value in vars(namespace).items()},
    }
    if args.dry_run:
        print(json.dumps(contract, indent=2, ensure_ascii=False))
        return
    (target_dir / "migration_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    module.run(namespace)


def dependency_closure(names: list[str]) -> list[str]:
    ordered: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            raise RuntimeError(f"Cyclic migration dependency at {name}")
        visiting.add(name)
        for upstream, _ in EXPERIMENTS[name].dependencies.values():
            visit(upstream)
        visiting.remove(name)
        visited.add(name)
        ordered.append(name)

    for name in names:
        visit(name)
    return ordered


def select_experiments(args: argparse.Namespace) -> list[str]:
    if args.experiments:
        requested = args.experiments
    elif args.preset == "all":
        requested = list(EXPERIMENTS)
    else:
        requested = TABLE5_EXPERIMENTS
    unknown = sorted(set(requested) - set(EXPERIMENTS))
    if unknown:
        raise ValueError(f"Unknown experiments: {', '.join(unknown)}")
    return dependency_closure(requested)


def child_command(args: argparse.Namespace, name: str) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_run-one",
        name,
        "--train",
        str(resolved(args.train)),
        "--test",
        str(resolved(args.test)),
        "--output-root",
        str(resolved(args.output_root)),
        "--device",
        args.device,
        "--seeds",
        *[str(seed) for seed in args.seeds],
    ]
    if args.epochs is not None:
        command.extend(["--epochs", str(args.epochs)])
    if args.max_tasks:
        command.extend(["--max-tasks", str(args.max_tasks)])
    if args.dry_run:
        command.append("--dry-run")
    if args.overwrite:
        command.append("--overwrite")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["table5", "all"], default="table5")
    parser.add_argument("--experiments", nargs="+", choices=sorted(EXPERIMENTS))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional common epoch override; omitted means each archived runner's default.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=0,
        help="Smoke-test limit for runners that support it; 0 means all 157 tasks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow a nonempty target experiment directory.",
    )
    parser.add_argument("--_run-one", choices=sorted(EXPERIMENTS), help=argparse.SUPPRESS)
    return parser.parse_args()


def parse_single_experiment_args(
    experiment_name: str,
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Parse the public CLI used by one dedicated Phase-7 migration script."""
    experiment = EXPERIMENTS[experiment_name]
    parser = argparse.ArgumentParser(
        description=(
            f"{experiment.description} Run only this archived implementation "
            "on the Phase-7 157-task benchmark with isolated outputs."
        )
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epoch override; omitted means the archived runner's default.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=0,
        help="Smoke-test limit when supported; 0 means all 157 tasks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow a nonempty target experiment directory.",
    )
    args = parser.parse_args(argv)
    args._run_one = experiment_name
    args.preset = "table5"
    args.experiments = None
    args.list = False
    return args


def run_single_experiment_entry(experiment_name: str) -> None:
    """Public entry point for the 15 dedicated Table-5 migration scripts."""
    if experiment_name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    run_one(parse_single_experiment_args(experiment_name))


def main() -> None:
    args = parse_args()
    if args.list:
        for name, experiment in EXPERIMENTS.items():
            dependencies = sorted(
                {upstream for upstream, _ in experiment.dependencies.values()}
            )
            suffix = f" [depends on: {', '.join(dependencies)}]" if dependencies else ""
            print(f"{name:26s} {experiment.description}{suffix}")
        return
    if args._run_one:
        run_one(args)
        return

    for required in (resolved(args.train), resolved(args.test)):
        if not required.is_file():
            raise FileNotFoundError(required)

    selected = select_experiments(args)
    print("Migration order:")
    for index, name in enumerate(selected, start=1):
        print(f"  {index:02d}. {name}: {EXPERIMENTS[name].description}")
    print(f"Isolated output root: {resolved(args.output_root)}")

    for name in selected:
        command = child_command(args, name)
        print(f"\n=== {name} ===", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
