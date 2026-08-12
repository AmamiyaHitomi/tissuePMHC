#!/usr/bin/env python3
"""Post-selection full-data retraining and exploratory legacy-test ablations.

The program refuses to start until nested CV has written FINAL_WEIGHT_LOCK.json.
It then trains the locked auxiliary setting, 0.10, 0.20, no auxiliary, and the
formal architecture ablations with the five frozen seeds.  The previously
viewed occurrence-equal test split is labelled exploratory in every contract.
Interpretation-ready checkpoints are saved for the locked and no-aux models.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_human_auxiliary_nested_cv as nested  # noqa: E402
import run_occurrence_equal_ablation_mhc_only as ablation  # noqa: E402
import run_tissuepmhc_neural_baselines_v2 as base  # noqa: E402


NESTED_ROOT = ROOT / "results" / "human_aux_nested_cv_v1"
DEFAULT_OUTPUT = ROOT / "results" / "human_postselection_formal_v1"


def atomic_csv(frame: pd.DataFrame, path: Path, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, compression=compression)
    os.replace(temporary, path)


def condition_complete(path: Path, expected_models: set[str]) -> bool:
    summary = path / "summary_metrics.csv"
    predictions = path / "test_predictions.csv.gz"
    if not summary.is_file() or not predictions.is_file():
        return False
    try:
        observed = set(pd.read_csv(summary, usecols=["model"])["model"])
        pd.read_csv(predictions, nrows=1)
        return observed == expected_models
    except Exception:
        return False


def evaluate_named(
    named: dict[str, pd.DataFrame], train: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_parts: list[pd.DataFrame] = []
    per_task_parts: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for model, prediction in named.items():
        item = prediction.copy()
        item.insert(0, "seed", seed)
        item.insert(0, "model", model)
        prediction_parts.append(item)
        per_task, summary = ablation.evaluate_prediction(
            prediction, train, "human", model, seed, ablation.SPECIES_CONFIGS["human"].worst_k
        )
        summary["mean_task_f1"] = float(per_task.f1.mean())
        per_task_parts.append(per_task)
        summaries.append(summary)
        print(
            f"[FORMAL METRIC] seed={seed} model={model} "
            f"AUPRC={summary['mean_task_auprc']:.6f} AUROC={summary['mean_task_auroc']:.6f} "
            f"F1={summary['mean_task_f1']:.6f} MCC={summary['mean_task_mcc']:.6f}",
            flush=True,
        )
    return (
        pd.concat(prediction_parts, ignore_index=True),
        pd.concat(per_task_parts, ignore_index=True),
        pd.DataFrame(summaries),
    )


def save_condition(
    target: Path,
    named: dict[str, pd.DataFrame],
    train: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions, per_task, summary = evaluate_named(named, train, seed)
    atomic_csv(predictions, target / "test_predictions.csv.gz", "gzip")
    atomic_csv(per_task, target / "per_task_metrics.csv")
    # Completion marker is written last.
    atomic_csv(summary, target / "summary_metrics.csv")
    return predictions, per_task, summary


def load_condition(target: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(target / "test_predictions.csv.gz", keep_default_na=False),
        pd.read_csv(target / "per_task_metrics.csv", keep_default_na=False),
        pd.read_csv(target / "summary_metrics.csv", keep_default_na=False),
    )


def paired_seed_table(summaries: pd.DataFrame, output: Path) -> None:
    subset = summaries[summaries.model.isin({"no_auxiliary", "aux_weight_0.1", "aux_weight_0.2"})]
    rows: list[pd.DataFrame] = []
    stats: list[dict[str, Any]] = []
    for metric in ["mean_task_auprc", "mean_task_auroc", "mean_task_f1", "mean_task_mcc"]:
        wide = subset.pivot(index="seed", columns="model", values=metric).reset_index()
        wide["delta_0.1_minus_noaux"] = wide["aux_weight_0.1"] - wide.no_auxiliary
        wide["delta_0.2_minus_noaux"] = wide["aux_weight_0.2"] - wide.no_auxiliary
        item = wide.copy()
        item.insert(0, "metric", metric)
        rows.append(item)
        for candidate in ("aux_weight_0.1", "aux_weight_0.2"):
            values = (wide[candidate] - wide.no_auxiliary).to_numpy(dtype=float)
            low, high = ablation.bootstrap_mean_ci(
                values, ablation.stable_seed("postselection", metric, candidate), 10000
            )
            nonzero = values[values != 0]
            stats.append({
                "metric": metric,
                "candidate": candidate,
                "reference": "no_auxiliary",
                "mean_difference": float(values.mean()),
                "sd_difference": float(values.std(ddof=1)),
                "ci95_low": low,
                "ci95_high": high,
                "wins": int((values > 0).sum()),
                "ties": int((values == 0).sum()),
                "losses": int((values < 0).sum()),
                "wilcoxon_p": float(wilcoxon(nonzero).pvalue) if len(nonzero) else 1.0,
                "interpretation": "exploratory legacy fixed test; not used for model selection",
            })
    atomic_csv(pd.concat(rows, ignore_index=True), output / "paired_seed_raw_scores_and_differences.csv")
    atomic_csv(pd.DataFrame(stats), output / "paired_seed_statistics.csv")


def gradient_summary(path: Path, output: Path) -> None:
    data = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for (config, seed, epoch), group in data.groupby(["config", "seed", "epoch"], sort=True):
        for auxiliary, column in (
            ("tissue", "primary_tissue_cosine"), ("mhc", "primary_mhc_cosine")
        ):
            values = group[column].to_numpy(dtype=float)
            rows.append({
                "config": config,
                "seed": seed,
                "epoch": epoch,
                "auxiliary_task": auxiliary,
                "batches": len(values),
                "mean_cosine": float(values.mean()),
                "median_cosine": float(np.median(values)),
                "negative_batch_fraction": float((values < 0).mean()),
                "mean_primary_gradient_norm": float(group.primary_gradient_norm.mean()),
                "mean_auxiliary_gradient_norm": float(
                    group[f"{auxiliary}_gradient_norm"].mean()
                ),
            })
    per_epoch = pd.DataFrame(rows)
    atomic_csv(per_epoch, output / "gradient_per_epoch_summary.csv")
    aggregate = per_epoch.groupby(["config", "auxiliary_task"], as_index=False).agg(
        mean_cosine=("mean_cosine", "mean"),
        median_epoch_cosine=("median_cosine", "median"),
        mean_negative_batch_fraction=("negative_batch_fraction", "mean"),
        epochs=("epoch", "size"),
    )
    atomic_csv(aggregate, output / "gradient_weight_summary.csv")


def hardlink_publish(source: Path, target: Path) -> None:
    """Publish a compatibility checkpoint name without duplicating model bytes."""
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        return
    temporary = target.with_name(target.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    os.link(source, temporary)
    os.replace(temporary, target)


def build_interpretability_views(output: Path, selected_weight: float) -> None:
    """Create old-script-compatible checkpoint names and branch prediction tables."""
    conditions = {
        "locked_auxiliary": selected_weight,
        "no_auxiliary": 0.0,
    }
    for label, weight in conditions.items():
        prediction_parts: list[pd.DataFrame] = []
        source_name = "no_auxiliary" if weight == 0 else f"aux_weight_{weight:g}"
        run_name = "aux_weight_0" if weight == 0 else f"aux_weight_{weight:g}"
        for seed in nested.SEEDS:
            source_global = output / "checkpoints" / source_name / f"seed_{seed}" / "global.pt"
            target_seed = output / "interpretability" / label / "checkpoints" / f"seed_{seed}"
            hardlink_publish(source_global, target_seed / "global_aux.pt")
            hla_source = output / "checkpoints" / "no_auxiliary" / f"seed_{seed}"
            for checkpoint in hla_source.glob("mhc_branch__*.pt"):
                suffix = checkpoint.name.removeprefix("mhc_branch__")
                hardlink_publish(checkpoint, target_seed / f"hla_plain__{suffix}")
            run_root = output / "seed_runs" / f"seed_{seed}" / run_name
            global_prediction = pd.read_csv(run_root / "branch_global.csv.gz", keep_default_na=False)
            mhc_prediction = pd.read_csv(run_root / "branch_mhc.csv.gz", keep_default_na=False)
            fused_prediction = pd.read_csv(run_root / "branch_rank.csv.gz", keep_default_na=False)
            keys = ablation.PREDICTION_KEYS
            merged = global_prediction[keys + ["score"]].rename(columns={"score": "global_score"}).merge(
                mhc_prediction[keys + ["score"]].rename(columns={"score": "hla_score"}),
                on=keys, validate="one_to_one",
            ).merge(
                fused_prediction[keys + ["score"]].rename(columns={"score": "fused_score"}),
                on=keys, validate="one_to_one",
            )
            merged.insert(0, "seed", seed)
            prediction_parts.append(merged)
        target = output / "interpretability" / label
        atomic_csv(pd.concat(prediction_parts, ignore_index=True), target / "branch_predictions.csv.gz", "gzip")
        (target / "manifest.json").write_text(json.dumps({
            "condition": label,
            "auxiliary_weight": weight,
            "seeds": nested.SEEDS,
            "hla_checkpoints": "paired no-auxiliary HLA branches; weight-independent",
            "selection_use": False,
        }, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nested-root", type=Path, default=NESTED_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lock_path = args.nested_root / "FINAL_WEIGHT_LOCK.json"
    contract_path = args.nested_root / "run_contract.json"
    if not lock_path.is_file() or not contract_path.is_file():
        raise RuntimeError("Nested CV is incomplete; FINAL_WEIGHT_LOCK.json is required")
    nested_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if nested_contract.get("status") != "completed":
        raise RuntimeError("Nested CV contract is not completed")
    if nested.sha256(nested.PROTOCOL) != nested.PROTOCOL_SHA256:
        raise RuntimeError("Frozen protocol hash mismatch")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    selected_weight = float(lock["final_weight"])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch_parts = base.require_torch()
    torch = torch_parts[0]
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    train, test, mappings, peptide_length = ablation.read_data(
        ablation.SPECIES_CONFIGS["human"], None
    )
    timing = ablation.TimingLogger(output / "timing_results.csv")
    gradients = ablation.GradientLogger(output / "gradient_batch_diagnostics.csv")
    contract = {
        "status": "running",
        "nested_lock": str(lock_path.resolve()),
        "selected_weight": selected_weight,
        "seeds": nested.SEEDS,
        "fixed_test_role": "supplementary exploratory; viewed during earlier development",
        "fixed_test_used_for_selection": False,
        "checkpoint_contract": "selected and no-aux global models; selected reuses paired no-aux HLA checkpoints",
        "device": device,
    }
    (output / "run_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    all_predictions: list[pd.DataFrame] = []
    all_per_tasks: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []
    started = time.perf_counter()
    try:
        unique_weights = tuple(sorted({0.0, 0.1, 0.2, selected_weight}))
        for seed in nested.SEEDS:
            seed_root = output / "seed_runs" / f"seed_{seed}"
            zero_target = seed_root / "aux_weight_0"
            if condition_complete(zero_target, {"no_auxiliary"}):
                zero_outputs = load_condition(zero_target)
                print(f"[FORMAL SKIP] seed={seed} condition=no_auxiliary", flush=True)
            else:
                config = replace(ablation.SPECIES_CONFIGS["human"], tissue_loss_weight=0, mhc_loss_weight=0)
                branches, _ = ablation.fit_two_branch(
                    train=train, test=test, mappings=mappings, peptide_length=peptide_length,
                    config=config, config_name="formal_no_auxiliary", encoder_kind="cnn",
                    use_auxiliary=False, seed=seed, epochs=config.epochs, device=device,
                    torch_parts=torch_parts, timing=timing,
                    checkpoint_dir=output / "checkpoints" / "no_auxiliary" / f"seed_{seed}",
                )
                for name, frame in branches.items():
                    atomic_csv(frame, zero_target / f"branch_{name}.csv.gz", "gzip")
                zero_outputs = save_condition(zero_target, {"no_auxiliary": branches["rank"]}, train, seed)
            all_predictions.append(zero_outputs[0]); all_per_tasks.append(zero_outputs[1]); all_summaries.append(zero_outputs[2])
            mhc_prediction = pd.read_csv(zero_target / "branch_mhc.csv.gz", keep_default_na=False)

            weighted_branches: dict[float, dict[str, pd.DataFrame]] = {}
            for weight in unique_weights:
                if weight == 0:
                    continue
                target = seed_root / f"aux_weight_{weight:g}"
                model_name = f"aux_weight_{weight:g}"
                if condition_complete(target, {model_name}):
                    outputs = load_condition(target)
                    branches = {
                        name: pd.read_csv(target / f"branch_{name}.csv.gz", keep_default_na=False)
                        for name in ("rank", "probability", "global", "mhc")
                    }
                    print(f"[FORMAL SKIP] seed={seed} condition={model_name}", flush=True)
                else:
                    config = replace(
                        ablation.SPECIES_CONFIGS["human"],
                        tissue_loss_weight=weight, mhc_loss_weight=weight,
                    )
                    branches, _ = ablation.fit_two_branch(
                        train=train, test=test, mappings=mappings, peptide_length=peptide_length,
                        config=config, config_name=f"formal_aux_weight_{weight:g}", encoder_kind="cnn",
                        use_auxiliary=True, seed=seed, epochs=config.epochs, device=device,
                        torch_parts=torch_parts, timing=timing, gradient_logger=gradients,
                        gradient_batches_per_epoch=5, precomputed_mhc_prediction=mhc_prediction,
                        checkpoint_dir=output / "checkpoints" / f"aux_weight_{weight:g}" / f"seed_{seed}",
                    )
                    for name, frame in branches.items():
                        atomic_csv(frame, target / f"branch_{name}.csv.gz", "gzip")
                    outputs = save_condition(target, {model_name: branches["rank"]}, train, seed)
                weighted_branches[weight] = branches
                all_predictions.append(outputs[0]); all_per_tasks.append(outputs[1]); all_summaries.append(outputs[2])

            selected = weighted_branches[selected_weight] if selected_weight > 0 else {
                name: pd.read_csv(zero_target / f"branch_{name}.csv.gz", keep_default_na=False)
                for name in ("rank", "probability", "global", "mhc")
            }
            derived_target = seed_root / "formal_selected_derived"
            if condition_complete(
                derived_target, {"selected_full_rank_fusion", "no_mhc_branch", "no_rank_fusion"}
            ):
                outputs = load_condition(derived_target)
            else:
                outputs = save_condition(derived_target, {
                    "selected_full_rank_fusion": selected["rank"],
                    "no_mhc_branch": selected["global"],
                    "no_rank_fusion": selected["probability"],
                }, train, seed)
            all_predictions.append(outputs[0]); all_per_tasks.append(outputs[1]); all_summaries.append(outputs[2])

            mlp_target = seed_root / "no_multikernel"
            if condition_complete(mlp_target, {"no_multikernel"}):
                outputs = load_condition(mlp_target)
            else:
                config = replace(
                    ablation.SPECIES_CONFIGS["human"],
                    tissue_loss_weight=selected_weight, mhc_loss_weight=selected_weight,
                )
                branches, _ = ablation.fit_two_branch(
                    train=train, test=test, mappings=mappings, peptide_length=peptide_length,
                    config=config, config_name="formal_no_multikernel", encoder_kind="mlp",
                    use_auxiliary=selected_weight > 0, seed=seed, epochs=config.epochs,
                    device=device, torch_parts=torch_parts, timing=timing,
                )
                outputs = save_condition(mlp_target, {"no_multikernel": branches["rank"]}, train, seed)
            all_predictions.append(outputs[0]); all_per_tasks.append(outputs[1]); all_summaries.append(outputs[2])

            mhc_only_target = seed_root / "mhc_only"
            if condition_complete(mhc_only_target, {"mhc_only_cnn"}):
                outputs = load_condition(mhc_only_target)
            else:
                prediction, _ = ablation.fit_mhc_only(
                    train=train, test=test, mappings=mappings, peptide_length=peptide_length,
                    config=ablation.SPECIES_CONFIGS["human"], seed=seed,
                    epochs=ablation.SPECIES_CONFIGS["human"].epochs, device=device,
                    torch_parts=torch_parts, timing=timing,
                )
                outputs = save_condition(mhc_only_target, {"mhc_only_cnn": prediction}, train, seed)
            all_predictions.append(outputs[0]); all_per_tasks.append(outputs[1]); all_summaries.append(outputs[2])
            print(f"[FORMAL SEED COMPLETE] seed={seed}", flush=True)

        predictions = pd.concat(all_predictions, ignore_index=True)
        per_tasks = pd.concat(all_per_tasks, ignore_index=True)
        summaries = pd.concat(all_summaries, ignore_index=True)
        atomic_csv(predictions, output / "all_seed_exploratory_test_predictions.csv.gz", "gzip")
        atomic_csv(per_tasks, output / "all_seed_per_task_metrics.csv")
        atomic_csv(summaries, output / "per_seed_summary.csv")
        paired_seed_table(summaries, output)
        gradient_summary(output / "gradient_batch_diagnostics.csv", output)
        build_interpretability_views(output, selected_weight)
        contract["status"] = "completed"
    finally:
        elapsed = time.perf_counter() - started
        contract["elapsed_seconds"] = elapsed
        (output / "run_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
        timing.write(scope="formal_total", species="human", config="all_formal_conditions",
                     elapsed_seconds=f"{elapsed:.6f}", status=contract["status"])
        print(f"[FORMAL TOTAL TIME] elapsed_seconds={elapsed:.3f} status={contract['status']}", flush=True)


if __name__ == "__main__":
    main()
