#!/usr/bin/env python3
"""Frozen nested-CV selection of Human auxiliary-loss weights.

This runner reads only the occurrence-equal Human training split.  It uses
five outer folds, three inner folds, seven tied auxiliary-weight candidates,
and five paired training seeds.  Every atomic result is written immediately,
so an interrupted multi-hour run can resume without recomputing completed
seed/weight cells.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_occurrence_equal_ablation_mhc_only as ablation  # noqa: E402
import run_tissuepmhc_neural_baselines_v2 as base  # noqa: E402


PROTOCOL = ROOT / "protocols" / "human_aux_nested_cv_v1.json"
PROTOCOL_SHA256 = "1FFD059ABD8F58498AB377292AD3BF1702BD6DC9C6D672897301B0FFDDB98CD6"
TRAIN_SHA256 = "E381B3A71FD1023C20995431A288B7107B917BE87899C62E1CCCCDF53DBBA1AC"
WEIGHTS = (0.0, 0.01, 0.03, 0.05, 0.10, 0.20, 0.30)
SEEDS = (20260721, 20260722, 20260723, 20260724, 20260725)
OUTER_FOLDS = 5
INNER_FOLDS = 3
OUTER_SPLIT_SEED = 2026081101
INNER_SPLIT_SEEDS = (2026081201, 2026081202, 2026081203, 2026081204, 2026081205)
TIE_THRESHOLD = 0.002
MISSING_GROUP_VALUES = {"", "NA", "N/A", "NONE", "NULL", "NAN"}
OUTPUT = ROOT / "results" / "human_aux_nested_cv_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


class DSU:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def add_leakage_components(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join pairs, exact/near peptides, and available source-protein IDs."""
    dsu = DSU()
    peptide_values = sorted(frame["peptide_sequence"].astype(str).unique())
    wildcard_owner: dict[str, str] = {}
    for peptide in peptide_values:
        peptide_node = f"peptide:{peptide}"
        dsu.find(peptide_node)
        for position in range(len(peptide)):
            signature = f"{position}:{peptide[:position]}*{peptide[position + 1:]}"
            if signature in wildcard_owner:
                dsu.union(peptide_node, wildcard_owner[signature])
            else:
                wildcard_owner[signature] = peptide_node
    protein_columns = ("source_molecule_uniprot_id", "molecule_parent_uniprot_id")
    for row in frame.itertuples(index=False):
        pair_node = f"pair:{row.pair_id}"
        dsu.union(pair_node, f"peptide:{row.peptide_sequence}")
        for column in protein_columns:
            value = str(getattr(row, column)).strip()
            if value.upper() not in MISSING_GROUP_VALUES:
                dsu.union(pair_node, f"protein:{value}")
    output = frame.copy()
    output["component_id"] = output["pair_id"].astype(str).map(
        lambda value: dsu.find(f"pair:{value}")
    )
    component_pairs = output[["component_id", "pair_id"]].drop_duplicates().groupby("component_id").size()
    audit = {
        "rows": int(len(output)),
        "pairs": int(output["pair_id"].nunique()),
        "peptides": int(output["peptide_sequence"].nunique()),
        "components": int(output["component_id"].nunique()),
        "largest_component_pairs": int(component_pairs.max()),
        "multi_pair_components": int((component_pairs > 1).sum()),
        "group_links": [
            "pair_id", "exact peptide", "peptide Hamming distance <= 1",
            "source_molecule_uniprot_id", "molecule_parent_uniprot_id",
        ],
    }
    return output, audit


def stable_uint64(value: str, seed: int) -> int:
    payload = f"{seed}|{value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def assign_component_folds(frame: pd.DataFrame, n_folds: int, seed: int) -> pd.Series:
    """Greedily balance task-by-label cells without splitting components."""
    cells = sorted({(str(task), int(label)) for task, label in zip(frame.task_name, frame.label)})
    cell_to_index = {cell: index for index, cell in enumerate(cells)}
    vectors: dict[str, np.ndarray] = {}
    for component, group in frame.groupby("component_id", sort=True):
        vector = np.zeros(len(cells), dtype=np.float64)
        for (task, label), count in group.groupby(["task_name", "label"]).size().items():
            vector[cell_to_index[(str(task), int(label))]] = float(count)
        vectors[str(component)] = vector
    totals = np.sum(list(vectors.values()), axis=0)
    if (totals < n_folds).any():
        unsupported = [cells[index] for index, count in enumerate(totals) if count < n_folds]
        raise ValueError(f"Task/label cells cannot support {n_folds} folds: {unsupported[:10]}")
    target = totals / n_folds
    denominator = np.maximum(target, 1.0)
    ordered = sorted(
        vectors,
        key=lambda component: (
            -float(vectors[component].sum()),
            -int((vectors[component] > 0).sum()),
            stable_uint64(component, seed),
        ),
    )
    fold_counts = np.zeros((n_folds, len(cells)), dtype=np.float64)
    assignment: dict[str, int] = {}
    for component in ordered:
        contribution = vectors[component]
        costs: list[tuple[float, float, int]] = []
        for fold in range(n_folds):
            proposed = fold_counts.copy()
            proposed[fold] += contribution
            imbalance = float(np.square((proposed - target) / denominator).sum())
            costs.append((imbalance, float(fold_counts[fold].sum()), fold))
        chosen = min(costs)[2]
        assignment[component] = chosen
        fold_counts[chosen] += contribution
    folds = frame["component_id"].astype(str).map(assignment)
    if folds.isna().any():
        raise AssertionError("Unassigned leakage component")
    return folds.astype(int)


def split_audit(frame: pd.DataFrame, folds: pd.Series, n_folds: int, scope: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    protein_columns = ("source_molecule_uniprot_id", "molecule_parent_uniprot_id")
    for fold in range(n_folds):
        held = frame.loc[folds == fold]
        fit = frame.loc[folds != fold]
        row: dict[str, Any] = {
            "scope": scope,
            "fold": fold,
            "fit_rows": len(fit),
            "held_rows": len(held),
            "fit_pairs": fit.pair_id.nunique(),
            "held_pairs": held.pair_id.nunique(),
            "held_tasks": held.task_name.nunique(),
            "component_overlap": len(set(fit.component_id) & set(held.component_id)),
            "pair_overlap": len(set(fit.pair_id) & set(held.pair_id)),
            "peptide_overlap": len(set(fit.peptide_sequence) & set(held.peptide_sequence)),
        }
        for column in protein_columns:
            left = {str(value) for value in fit[column] if str(value).strip().upper() not in MISSING_GROUP_VALUES}
            right = {str(value) for value in held[column] if str(value).strip().upper() not in MISSING_GROUP_VALUES}
            row[f"{column}_overlap"] = len(left & right)
        rows.append(row)
        overlap_fields = [value for key, value in row.items() if key.endswith("overlap")]
        if any(overlap_fields):
            raise AssertionError(f"{scope} fold {fold} leakage: {row}")
        if held.task_name.nunique() != frame.task_name.nunique():
            raise AssertionError(f"{scope} fold {fold} omits tasks")
    return pd.DataFrame(rows)


def metric_outputs(
    prediction: pd.DataFrame,
    fitting: pd.DataFrame,
    seed: int,
    weight: float,
    outer_fold: int,
    inner_fold: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = f"aux_weight_{weight:g}"
    per_task, summary = ablation.evaluate_prediction(
        prediction, fitting, "human", model, seed, ablation.SPECIES_CONFIGS["human"].worst_k
    )
    summary["mean_task_f1"] = float(per_task["f1"].mean())
    per_task.insert(0, "outer_fold", outer_fold)
    per_task.insert(1, "inner_fold", -1 if inner_fold is None else inner_fold)
    per_task.insert(4, "auxiliary_weight", weight)
    summary.update({
        "outer_fold": outer_fold,
        "inner_fold": -1 if inner_fold is None else inner_fold,
        "auxiliary_weight": weight,
    })
    return per_task, pd.DataFrame([summary])


def atomic_to_csv(frame: pd.DataFrame, path: Path, *, compression: str | None = None) -> None:
    """Write a CSV completely before atomically publishing its final name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, compression=compression)
    os.replace(temporary, path)


def valid_atomic_result(path: Path, weight: float, seed: int) -> bool:
    if not path.is_file():
        return False
    try:
        row = pd.read_csv(path).iloc[0]
        valid = math.isclose(float(row.auxiliary_weight), weight) and int(row.seed) == seed
        required = [path.parent / "validation_predictions.csv.gz", path.parent / "per_task_metrics.csv"]
        if weight == 0:
            required.append(path.parent / "mhc_predictions.csv.gz")
        if not valid or not all(item.is_file() for item in required):
            return False
        # A gzip footer is not validated by existence alone.  Reading one row
        # makes a corrupt no-auxiliary cache fail closed and rerun that cell.
        if weight == 0:
            pd.read_csv(path.parent / "mhc_predictions.csv.gz", nrows=1)
        return True
    except Exception:
        return False


def run_weight_cell(
    *,
    target: Path,
    fitting: pd.DataFrame,
    validation: pd.DataFrame,
    mappings: dict[str, Any],
    peptide_length: int,
    weight: float,
    seed: int,
    outer_fold: int,
    inner_fold: int | None,
    device: str,
    torch_parts: tuple[Any, Any, Any, Any],
    timing: ablation.TimingLogger,
    mhc_prediction: pd.DataFrame | None,
) -> pd.DataFrame:
    summary_path = target / "summary_metrics.csv"
    if valid_atomic_result(summary_path, weight, seed):
        print(
            f"[CELL SKIP] outer={outer_fold} inner={inner_fold} seed={seed} weight={weight:g}",
            flush=True,
        )
        return pd.read_csv(summary_path)
    target.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config = replace(
        ablation.SPECIES_CONFIGS["human"],
        tissue_loss_weight=weight,
        mhc_loss_weight=weight,
    )
    predictions, _ = ablation.fit_two_branch(
        train=fitting,
        test=validation,
        mappings=mappings,
        peptide_length=peptide_length,
        config=config,
        config_name=f"nested_aux_weight_{weight:g}",
        encoder_kind="cnn",
        use_auxiliary=weight > 0,
        seed=seed,
        epochs=config.epochs,
        device=device,
        torch_parts=torch_parts,
        timing=timing,
        precomputed_mhc_prediction=mhc_prediction,
    )
    prediction = predictions["rank"]
    per_task, summary = metric_outputs(
        prediction, fitting, seed, weight, outer_fold, inner_fold
    )
    item = prediction.copy()
    item.insert(0, "seed", seed)
    item.insert(0, "auxiliary_weight", weight)
    atomic_to_csv(item, target / "validation_predictions.csv.gz", compression="gzip")
    atomic_to_csv(per_task, target / "per_task_metrics.csv")
    if mhc_prediction is None:
        atomic_to_csv(predictions["mhc"], target / "mhc_predictions.csv.gz", compression="gzip")
    # The summary is the completion marker and is deliberately written last.
    atomic_to_csv(summary, summary_path)
    elapsed = time.perf_counter() - started
    timing.write(
        scope="nested_cell", species="human", config=f"aux_weight_{weight:g}", seed=seed,
        epochs=config.epochs, elapsed_seconds=f"{elapsed:.6f}", status="completed",
    )
    print(
        f"[CELL METRIC] outer={outer_fold} inner={inner_fold} seed={seed} weight={weight:g} "
        f"AUPRC={summary.iloc[0]['mean_task_auprc']:.6f} "
        f"AUROC={summary.iloc[0]['mean_task_auroc']:.6f} elapsed_seconds={elapsed:.3f}",
        flush=True,
    )
    return summary


def run_partition_grid(
    *,
    root: Path,
    fitting: pd.DataFrame,
    validation: pd.DataFrame,
    outer_fold: int,
    inner_fold: int | None,
    weights: tuple[float, ...],
    device: str,
    torch_parts: tuple[Any, Any, Any, Any],
    timing: ablation.TimingLogger,
) -> pd.DataFrame:
    fitting, validation, mappings = base.add_task_columns(fitting.copy(), validation.copy())
    peptide_length = int(fitting.peptide_sequence.str.len().iloc[0])
    summaries: list[pd.DataFrame] = []
    for seed in SEEDS:
        seed_root = root / f"seed_{seed}"
        zero_root = seed_root / "weight_0"
        mhc_path = zero_root / "mhc_predictions.csv.gz"
        zero_summary = run_weight_cell(
            target=zero_root, fitting=fitting, validation=validation, mappings=mappings,
            peptide_length=peptide_length, weight=0.0, seed=seed, outer_fold=outer_fold,
            inner_fold=inner_fold, device=device, torch_parts=torch_parts, timing=timing,
            mhc_prediction=None,
        )
        summaries.append(zero_summary)
        if not mhc_path.is_file():
            raise FileNotFoundError(f"Missing resumable MHC prediction: {mhc_path}")
        mhc_prediction = pd.read_csv(mhc_path, keep_default_na=False)
        for weight in weights:
            if weight == 0:
                continue
            summaries.append(run_weight_cell(
                target=seed_root / f"weight_{weight:g}", fitting=fitting, validation=validation,
                mappings=mappings, peptide_length=peptide_length, weight=weight, seed=seed,
                outer_fold=outer_fold, inner_fold=inner_fold, device=device,
                torch_parts=torch_parts, timing=timing, mhc_prediction=mhc_prediction,
            ))
    return pd.concat(summaries, ignore_index=True)


def select_weight(inner_summaries: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    scores = (
        inner_summaries.groupby("auxiliary_weight", as_index=False)
        .agg(
            mean_inner_auprc=("mean_task_auprc", "mean"),
            sd_inner_auprc=("mean_task_auprc", "std"),
            mean_inner_auroc=("mean_task_auroc", "mean"),
            observations=("mean_task_auprc", "size"),
        )
        .sort_values("auxiliary_weight")
    )
    best = float(scores.mean_inner_auprc.max())
    eligible = scores[scores.mean_inner_auprc >= best - TIE_THRESHOLD]
    selected = float(eligible.auxiliary_weight.min())
    scores["within_tie_threshold"] = scores.mean_inner_auprc >= best - TIE_THRESHOLD
    scores["selected"] = scores.auxiliary_weight == selected
    return selected, scores


def final_weight_rule(outer_scores: pd.DataFrame) -> tuple[float, pd.DataFrame, dict[str, Any]]:
    working = outer_scores.copy()
    working["rank_within_outer"] = working.groupby("outer_fold")["mean_inner_auprc"].transform(
        lambda values: rankdata(-values.to_numpy(), method="average")
    )
    aggregate = working.groupby("auxiliary_weight", as_index=False).agg(
        mean_inner_auprc=("mean_inner_auprc", "mean"),
        sd_across_outer=("mean_inner_auprc", "std"),
        mean_rank=("rank_within_outer", "mean"),
        outer_selected_count=("selected", "sum"),
    )
    aggregate = aggregate.sort_values("auxiliary_weight").reset_index(drop=True)
    best = float(aggregate.mean_inner_auprc.max())
    eligible = aggregate[aggregate.mean_inner_auprc >= best - TIE_THRESHOLD]
    selected = float(eligible.auxiliary_weight.min())
    top_two = aggregate.nlargest(2, "mean_inner_auprc").auxiliary_weight.tolist()
    special: dict[str, Any] = {"applied": False}
    if set(top_two) == {0.1, 0.2}:
        by_outer = working.pivot(index="outer_fold", columns="auxiliary_weight", values="mean_inner_auprc")
        gain = float(aggregate.loc[aggregate.auxiliary_weight == 0.2, "mean_inner_auprc"].iloc[0] -
                     aggregate.loc[aggregate.auxiliary_weight == 0.1, "mean_inner_auprc"].iloc[0])
        not_worse = int((by_outer[0.2] >= by_outer[0.1]).sum())
        selected = 0.2 if gain > TIE_THRESHOLD and not_worse >= 4 else 0.1
        special = {"applied": True, "gain_0p20_minus_0p10": gain,
                   "outer_folds_0p20_not_worse": not_worse, "selected": selected}
    aggregate["within_tie_threshold"] = aggregate.mean_inner_auprc >= best - TIE_THRESHOLD
    aggregate["selected_final"] = aggregate.auxiliary_weight == selected
    return selected, aggregate, special


def permutation_pvalue(values: np.ndarray, seed: int, replicates: int = 200000) -> float:
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    completed = 0
    for start in range(0, replicates, 10000):
        count = min(10000, replicates - start)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(count, len(values)))
        exceed += int((np.abs((signs * values).mean(axis=1)) >= observed).sum())
        completed += count
    return float((exceed + 1) / (completed + 1))


def paired_outer_statistics(outer: pd.DataFrame, output: Path) -> None:
    metrics = ["mean_task_auprc", "mean_task_auroc", "mean_task_f1", "mean_task_mcc"]
    rows: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []
    for metric in metrics:
        wide = outer.pivot(index=["outer_fold", "seed"], columns="role", values=metric)
        if not {"selected_auxiliary", "no_auxiliary"}.issubset(wide.columns):
            raise AssertionError("Incomplete outer paired comparison")
        wide = wide.reset_index()
        wide["difference_selected_minus_noaux"] = wide.selected_auxiliary - wide.no_auxiliary
        item = wide.copy()
        item.insert(0, "metric", metric)
        details.append(item)
        values = wide.difference_selected_minus_noaux.to_numpy(dtype=float)
        rng = np.random.default_rng(ablation.stable_seed("nested_cv", metric, "outer_bootstrap"))
        folds = sorted(wide.outer_fold.unique())
        draws = []
        for _ in range(10000):
            sampled = rng.choice(folds, size=len(folds), replace=True)
            draws.append(float(np.mean([wide.loc[wide.outer_fold == fold, "difference_selected_minus_noaux"].mean()
                                        for fold in sampled])))
        nonzero = values[values != 0]
        rows.append({
            "metric": metric,
            "paired_observations": len(values),
            "mean_difference": float(values.mean()),
            "sd_difference": float(values.std(ddof=1)),
            "ci95_low": float(np.quantile(draws, 0.025)),
            "ci95_high": float(np.quantile(draws, 0.975)),
            "wins": int((values > 0).sum()),
            "ties": int((values == 0).sum()),
            "losses": int((values < 0).sum()),
            "paired_permutation_p": permutation_pvalue(
                values, ablation.stable_seed("nested_cv", metric, "permutation")
            ),
            "wilcoxon_p": float(wilcoxon(nonzero).pvalue) if len(nonzero) else 1.0,
        })
    pd.concat(details, ignore_index=True).to_csv(output / "outer_paired_raw_scores_and_differences.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "outer_paired_statistics.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--max-outer-folds", type=int, default=OUTER_FOLDS, help=argparse.SUPPRESS)
    parser.add_argument("--max-inner-folds", type=int, default=INNER_FOLDS, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("Frozen protocol hash mismatch; refusing to run")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    train_path = ROOT / protocol["data"]["development_train"]
    if sha256(train_path) != TRAIN_SHA256:
        raise RuntimeError("Frozen training-data hash mismatch; refusing to run")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "frozen_protocol.json").write_bytes(PROTOCOL.read_bytes())
    source = base.read_dataset(train_path)
    source["target_tissue"] = source["target_tissue"].fillna("NA")
    source, _, _ = base.add_task_columns(source, source.copy())
    source, component_audit = add_leakage_components(source)
    outer_assignments = assign_component_folds(source, OUTER_FOLDS, OUTER_SPLIT_SEED)
    source[["sample_id", "pair_id", "task_name", "component_id"]].assign(
        outer_fold=outer_assignments.to_numpy()
    ).to_csv(output / "outer_fold_assignments.csv.gz", index=False, compression="gzip")
    outer_audit = split_audit(source, outer_assignments, OUTER_FOLDS, "outer")
    outer_audit.to_csv(output / "outer_split_audit.csv", index=False)
    (output / "component_audit.json").write_text(json.dumps(component_audit, indent=2), encoding="utf-8")
    print(f"[SPLIT AUDIT] {json.dumps(component_audit)}", flush=True)
    if args.audit_only:
        print(f"[AUDIT ONLY COMPLETE] output={output}", flush=True)
        return

    torch_parts = base.require_torch()
    torch = torch_parts[0]
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    timing = ablation.TimingLogger(output / "timing_results.csv")
    contract_path = output / "run_contract.json"
    contract = {
        "status": "running",
        "protocol_sha256": PROTOCOL_SHA256,
        "train_sha256": TRAIN_SHA256,
        "device": device,
        "cuda_device": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "weights": WEIGHTS,
        "seeds": SEEDS,
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "fixed_config": asdict(ablation.SPECIES_CONFIGS["human"]),
        "legacy_fixed_test_accessed": False,
    }
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    all_started = time.perf_counter()
    outer_score_parts: list[pd.DataFrame] = []
    outer_result_parts: list[pd.DataFrame] = []
    try:
        for outer_fold in range(min(args.max_outer_folds, OUTER_FOLDS)):
            outer_train = source.loc[outer_assignments != outer_fold].copy()
            outer_holdout = source.loc[outer_assignments == outer_fold].copy()
            inner_assignments = assign_component_folds(
                outer_train, INNER_FOLDS, INNER_SPLIT_SEEDS[outer_fold]
            )
            (output / f"outer_{outer_fold}").mkdir(parents=True, exist_ok=True)
            split_audit(
                outer_train, inner_assignments, INNER_FOLDS, f"outer_{outer_fold}_inner"
            ).to_csv(output / f"outer_{outer_fold}" / "inner_split_audit.csv", index=False)
            inner_assignment_path = output / f"outer_{outer_fold}" / "inner_fold_assignments.csv.gz"
            inner_assignment_path.parent.mkdir(parents=True, exist_ok=True)
            outer_train[["sample_id", "pair_id", "task_name", "component_id"]].assign(
                inner_fold=inner_assignments.to_numpy()
            ).to_csv(inner_assignment_path, index=False, compression="gzip")
            inner_summaries: list[pd.DataFrame] = []
            for inner_fold in range(min(args.max_inner_folds, INNER_FOLDS)):
                print(f"[INNER START] outer={outer_fold} inner={inner_fold}", flush=True)
                inner_summaries.append(run_partition_grid(
                    root=output / f"outer_{outer_fold}" / f"inner_{inner_fold}",
                    fitting=outer_train.loc[inner_assignments != inner_fold].copy(),
                    validation=outer_train.loc[inner_assignments == inner_fold].copy(),
                    outer_fold=outer_fold,
                    inner_fold=inner_fold,
                    weights=WEIGHTS,
                    device=device,
                    torch_parts=torch_parts,
                    timing=timing,
                ))
            if args.max_inner_folds != INNER_FOLDS:
                continue
            inner_frame = pd.concat(inner_summaries, ignore_index=True)
            inner_frame.to_csv(output / f"outer_{outer_fold}" / "inner_all_seed_metrics.csv", index=False)
            selected, scores = select_weight(inner_frame)
            scores.insert(0, "outer_fold", outer_fold)
            scores.to_csv(output / f"outer_{outer_fold}" / "inner_weight_scores.csv", index=False)
            outer_score_parts.append(scores)
            (output / f"outer_{outer_fold}" / "selected_weight.json").write_text(
                json.dumps({"selected_weight": selected, "primary_metric": "mean_task_auprc",
                            "tie_threshold": TIE_THRESHOLD}, indent=2), encoding="utf-8"
            )
            print(f"[OUTER WEIGHT LOCK] outer={outer_fold} selected={selected:g}", flush=True)
            refit_weights = (0.0,) if selected == 0 else (0.0, selected)
            refit = run_partition_grid(
                root=output / f"outer_{outer_fold}" / "outer_refit",
                fitting=outer_train,
                validation=outer_holdout,
                outer_fold=outer_fold,
                inner_fold=None,
                weights=refit_weights,
                device=device,
                torch_parts=torch_parts,
                timing=timing,
            )
            noaux = refit[refit.auxiliary_weight == 0].copy()
            noaux["role"] = "no_auxiliary"
            selected_frame = refit[refit.auxiliary_weight == selected].copy()
            selected_frame["role"] = "selected_auxiliary"
            outer_result_parts.extend([noaux, selected_frame])
        if args.max_outer_folds == OUTER_FOLDS and args.max_inner_folds == INNER_FOLDS:
            outer_scores = pd.concat(outer_score_parts, ignore_index=True)
            outer_scores.to_csv(output / "all_outer_inner_weight_scores.csv", index=False)
            final_weight, aggregate, special = final_weight_rule(outer_scores)
            aggregate.to_csv(output / "final_weight_training_only_summary.csv", index=False)
            lock = {
                "final_weight": final_weight,
                "selection_data": "nested inner validation metrics only",
                "tie_threshold": TIE_THRESHOLD,
                "special_0p10_vs_0p20_gate": special,
                "legacy_fixed_test_accessed": False,
            }
            (output / "FINAL_WEIGHT_LOCK.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
            outer_results = pd.concat(outer_result_parts, ignore_index=True)
            outer_results.to_csv(output / "outer_unbiased_per_seed_metrics.csv", index=False)
            paired_outer_statistics(outer_results, output)
            contract["status"] = "completed"
            contract["final_weight"] = final_weight
        else:
            contract["status"] = "partial_smoke"
    finally:
        elapsed = time.perf_counter() - all_started
        contract["elapsed_seconds"] = elapsed
        contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
        timing.write(scope="nested_cv_total", species="human", config="all_weights",
                     elapsed_seconds=f"{elapsed:.6f}", status=contract["status"])
        print(f"[TOTAL TIME] elapsed_seconds={elapsed:.3f} status={contract['status']}", flush=True)


if __name__ == "__main__":
    main()
