#!/usr/bin/env python3
"""D1-D3 train-only OOF diagnostics for tissue-condition dependence.

The workflow trains matched C0/C2/C3 models, then performs tissue swaps,
held-out tissue-input shuffling, inference-time component ablations, and a
matched no-auxiliary retraining control. It never opens the premium test set.
Elapsed time is printed only; no timing value is persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPERIMENTS_DIR = Path(__file__).resolve().parent
EXTRA_PREMIUM_DIR = EXPERIMENTS_DIR.parent
if str(EXTRA_PREMIUM_DIR) not in sys.path:
    sys.path.insert(0, str(EXTRA_PREMIUM_DIR))

import common
import conditional_model_experiments as cexp
import tissue_aux_diagnostics as bdiag


CANDIDATES: tuple[cexp.Candidate, ...] = ("c0", "c2", "c3")
DEFAULT_SHUFFLE_SEED = 20260721


def parser() -> argparse.ArgumentParser:
    result = cexp.parser("Run premium D1-D3 condition-dependence diagnostics.")
    result.add_argument("--tissue-shuffle-seed", type=int, default=DEFAULT_SHUFFLE_SEED)
    result.add_argument(
        "--skip-auxiliary-retrain",
        action="store_true",
        help="Skip the D3 matched no-auxiliary retraining control.",
    )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _observed_tissues(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {item.strip() for item in str(value).split(";") if item.strip()}


def _predict_variant(
    candidate: cexp.Candidate,
    args: Any,
    torch: Any,
    DataLoader: Any,
    TensorDataset: Any,
    base: Any,
    model: Any,
    frame: pd.DataFrame,
    peptide_length: int,
    device: str,
    *,
    tissue_ids: np.ndarray | None = None,
    task_ids: np.ndarray | None = None,
    use_tissue: bool = True,
    use_hla: bool = True,
    use_task_residual: bool = True,
) -> np.ndarray:
    altered = frame.copy()
    if tissue_ids is not None:
        altered["tissue_id"] = np.asarray(tissue_ids, dtype=np.int64)
    if task_ids is not None:
        altered["task_id"] = np.asarray(task_ids, dtype=np.int64)
    loader = cexp.build_loader(
        args,
        torch,
        DataLoader,
        TensorDataset,
        base,
        altered,
        peptide_length,
        shuffle=False,
    )
    scores: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for peptide, task, tissue, hla, _ in loader:
            peptide = peptide.to(device)
            task = task.to(device)
            tissue = tissue.to(device)
            hla = hla.to(device)
            if candidate == "c0":
                logits = model(peptide, task)
            else:
                logits = model.ablation_logits(
                    peptide,
                    task,
                    tissue,
                    hla,
                    use_tissue=use_tissue,
                    use_hla=use_hla,
                    use_task_residual=use_task_residual,
                )
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores).astype(float)


def _fused_rows(
    candidate: cexp.Candidate,
    seed: int,
    fold: int,
    fitting: pd.DataFrame,
    held_out: pd.DataFrame,
    global_scores: np.ndarray,
    hla_scores: pd.DataFrame,
    diagnostic: str,
) -> pd.DataFrame:
    result = cexp.fold_prediction_rows(
        candidate, seed, fold, fitting, held_out, global_scores, hla_scores
    )
    result.insert(1, "diagnostic", diagnostic)
    return result


def _shuffle_tissues(
    held_out: pd.DataFrame, seed: int, fold: int
) -> np.ndarray:
    """Permute tissues within HLA so every shuffled tissue/HLA pair is legal."""
    values = held_out["tissue_id"].to_numpy(dtype=np.int64).copy()
    rng = np.random.default_rng(seed + 1009 * fold)
    for _, indices in held_out.groupby("hla_id", sort=True).groups.items():
        positions = held_out.index.get_indexer(np.asarray(list(indices)))
        original = values[positions].copy()
        if len(np.unique(original)) < 2:
            continue
        candidate = rng.permutation(original)
        for _ in range(20):
            if np.any(candidate != original):
                break
            candidate = rng.permutation(original)
        values[positions] = candidate
    return values


def _swap_rows(
    candidate: cexp.Candidate,
    args: Any,
    torch: Any,
    DataLoader: Any,
    TensorDataset: Any,
    base: Any,
    model: Any,
    held_out: pd.DataFrame,
    mappings: dict[str, Any],
    peptide_length: int,
    device: str,
    seed: int,
    fold: int,
) -> pd.DataFrame:
    task_lookup = (
        held_out[["target_tissue", "mhc_restriction", "task_id"]]
        .drop_duplicates()
        .set_index(["target_tissue", "mhc_restriction"])["task_id"]
        .to_dict()
    )
    # Some valid tasks may be absent from one held-out fold; recover their IDs
    # from the stable mapping created from the complete premium train split.
    for task_name, task_id in mappings["task_to_id"].items():
        tissue, hla = task_name.split("||", 1)
        task_lookup.setdefault((tissue, hla), int(task_id))

    parts: list[pd.DataFrame] = []
    original_tissues = held_out["target_tissue"].astype(str).to_numpy()
    original_observed = [
        tissue in _observed_tissues(value)
        for tissue, value in zip(
            original_tissues,
            held_out["reported_tissues_same_hla_uniprot"],
            strict=True,
        )
    ]
    for swapped_tissue, swapped_tissue_id in sorted(
        mappings["tissue_to_id"].items(), key=lambda item: item[1]
    ):
        valid = held_out["mhc_restriction"].map(
            lambda hla: (str(swapped_tissue), str(hla)) in task_lookup
        )
        if not valid.any():
            continue
        subset = held_out.loc[valid].copy()
        swapped_task_ids = np.asarray(
            [
                task_lookup[(str(swapped_tissue), str(hla))]
                for hla in subset["mhc_restriction"]
            ],
            dtype=np.int64,
        )
        scores = _predict_variant(
            candidate,
            args,
            torch,
            DataLoader,
            TensorDataset,
            base,
            model,
            subset,
            peptide_length,
            device,
            tissue_ids=np.full(len(subset), swapped_tissue_id, dtype=np.int64),
            task_ids=swapped_task_ids,
        )
        output = subset[
            [
                "sample_id",
                "pair_id",
                "label",
                "peptide_sequence",
                "molecule_parent_uniprot_id",
                "target_tissue",
                "mhc_restriction",
                "reported_tissues_same_hla_uniprot",
            ]
        ].copy()
        output.rename(columns={"target_tissue": "original_tissue"}, inplace=True)
        output["swapped_tissue"] = str(swapped_tissue)
        output["swapped_task_id"] = swapped_task_ids
        output["global_score"] = scores
        output["original_tissue_observed"] = np.asarray(original_observed)[
            np.flatnonzero(valid.to_numpy())
        ]
        output["swapped_tissue_observed"] = output[
            "reported_tissues_same_hla_uniprot"
        ].map(lambda value: str(swapped_tissue) in _observed_tissues(value))
        output.insert(0, "fold", fold)
        output.insert(0, "seed", seed)
        output.insert(0, "candidate", cexp.CANDIDATES[candidate]["id"])
        parts.append(output)
    return pd.concat(parts, ignore_index=True)


def _swap_summary(swap: pd.DataFrame) -> pd.DataFrame:
    keyed = swap.merge(
        swap.loc[swap["original_tissue"] == swap["swapped_tissue"], [
            "candidate", "seed", "fold", "sample_id", "global_score"
        ]].rename(columns={"global_score": "original_global_score"}),
        on=["candidate", "seed", "fold", "sample_id"],
        how="left",
        validate="many_to_one",
    )
    if keyed["original_global_score"].isna().any():
        raise AssertionError("D1 swap matrix lacks original-tissue diagonal scores.")
    keyed["score_delta"] = keyed["global_score"] - keyed["original_global_score"]
    informative = keyed[
        keyed["original_tissue_observed"] != keyed["swapped_tissue_observed"]
    ].copy()
    informative["expected_sign"] = np.where(
        informative["swapped_tissue_observed"], 1.0, -1.0
    )
    informative["direction_concordant"] = (
        informative["score_delta"] * informative["expected_sign"] > 0
    )
    rows: list[dict[str, object]] = []
    for keys, group in informative.groupby(["candidate", "seed"], sort=True):
        rows.append(
            {
                "candidate": keys[0],
                "seed": int(keys[1]),
                "informative_swaps": int(len(group)),
                "direction_concordance": float(group["direction_concordant"].mean()),
                "mean_signed_delta": float(
                    (group["score_delta"] * group["expected_sign"]).mean()
                ),
                "median_signed_delta": float(
                    (group["score_delta"] * group["expected_sign"]).median()
                ),
            }
        )
    return pd.DataFrame(rows), keyed


def diagnostic_comparison(per_task: pd.DataFrame) -> pd.DataFrame:
    """Match every D2/D3 task row to its own candidate baseline."""
    parts: list[pd.DataFrame] = []
    parsed = per_task["candidate"].astype(str).str.split("::", n=1, expand=True)
    work = per_task.copy()
    work["model_candidate"] = parsed[0]
    work["diagnostic"] = parsed[1]
    for model_candidate, model_rows in work.groupby(
        "model_candidate", sort=True
    ):
        baseline = model_rows[model_rows["diagnostic"] == "baseline"]
        for diagnostic, current in model_rows[
            model_rows["diagnostic"] != "baseline"
        ].groupby("diagnostic", sort=True):
            matched = baseline.merge(
                current,
                on=["seed", "target_tissue", "mhc_restriction"],
                suffixes=("_baseline", "_diagnostic"),
                validate="one_to_one",
            )
            output = matched[
                ["seed", "target_tissue", "mhc_restriction"]
            ].copy()
            output.insert(0, "diagnostic", diagnostic)
            output.insert(0, "candidate", model_candidate)
            for metric in cexp.METRICS:
                output[f"baseline_{metric}"] = matched[
                    f"{metric}_baseline"
                ]
                output[f"diagnostic_{metric}"] = matched[
                    f"{metric}_diagnostic"
                ]
                output[f"delta_diagnostic_minus_baseline_{metric}"] = (
                    matched[f"{metric}_diagnostic"]
                    - matched[f"{metric}_baseline"]
                )
            parts.append(output)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def run_all_d(cli_args: argparse.Namespace | None = None) -> None:
    args = cli_args or parser().parse_args()
    seeds = cexp.validate_cli(args)
    if args.tissue_shuffle_seed < 0:
        raise ValueError("--tissue-shuffle-seed must be non-negative.")
    common.enable_original_modules()
    import run_tissuepmhc_auxiliary_soft_ensemble as e14
    import run_tissuepmhc_neural_baselines_v2 as base

    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = common.resolve_device(args.device, torch)
    train, mappings, peptide_length = bdiag.load_premium_train(base)
    assignments = bdiag.make_pair_grouped_folds(
        train, args.oof_folds, args.oof_split_seed
    )
    model_args = common.model_args(
        args,
        condition_dim=args.condition_dim,
        residual_l2_weight=args.residual_l2_weight,
    )
    output_dir = common.RESULTS_ROOT / "experiments" / "D_all_condition_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    bdiag.fold_assignment_table(train, assignments).to_csv(
        output_dir / "fold_assignments.csv", index=False
    )

    prediction_parts: list[pd.DataFrame] = []
    swap_parts: list[pd.DataFrame] = []
    training_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for seed in seeds:
        for fold in range(args.oof_folds):
            fitting = train.loc[assignments != fold].copy()
            held_out = train.loc[assignments == fold].copy()
            fold_started = time.perf_counter()
            print(
                f"\n=== D seed={seed} fold={fold + 1}/{args.oof_folds} ===",
                flush=True,
            )
            e14.set_seed(seed, torch)
            hla_predictions, _, hla_diagnostics = e14.train_and_predict_hla_branches(
                model_args,
                torch,
                nn,
                DataLoader,
                TensorDataset,
                fitting,
                held_out,
                mappings,
                peptide_length,
                device,
                seed,
                False,
            )
            hla_scores = bdiag.hla_prediction_frame(hla_predictions)
            training_rows.extend(
                {
                    "candidate": "shared_hla_plain",
                    "seed": seed,
                    "fold": fold,
                    **row,
                }
                for row in hla_diagnostics
            )

            shuffled_tissues = _shuffle_tissues(
                held_out, args.tissue_shuffle_seed + seed, fold
            )
            for candidate in CANDIDATES:
                model, diagnostics, _ = cexp.train_global_model(
                    candidate,
                    model_args,
                    torch,
                    nn,
                    DataLoader,
                    TensorDataset,
                    base,
                    e14,
                    fitting,
                    mappings,
                    peptide_length,
                    device,
                    seed,
                    fold,
                )
                training_rows.extend(diagnostics)
                baseline = _predict_variant(
                    candidate, model_args, torch, DataLoader, TensorDataset,
                    base, model, held_out, peptide_length, device
                )
                prediction_parts.append(
                    _fused_rows(
                        candidate, seed, fold, fitting, held_out, baseline,
                        hla_scores, "baseline"
                    )
                )
                shuffled = _predict_variant(
                    candidate,
                    model_args,
                    torch,
                    DataLoader,
                    TensorDataset,
                    base,
                    model,
                    held_out,
                    peptide_length,
                    device,
                    tissue_ids=shuffled_tissues,
                )
                prediction_parts.append(
                    _fused_rows(
                        candidate, seed, fold, fitting, held_out, shuffled,
                        hla_scores, "D2_shuffled_tissue_input"
                    )
                )
                if candidate != "c0":
                    for diagnostic, switches in (
                        ("D3_tissue_off", {"use_tissue": False}),
                        ("D3_hla_off", {"use_hla": False}),
                        ("D3_task_residual_off", {"use_task_residual": False}),
                    ):
                        scores = _predict_variant(
                            candidate,
                            model_args,
                            torch,
                            DataLoader,
                            TensorDataset,
                            base,
                            model,
                            held_out,
                            peptide_length,
                            device,
                            **switches,
                        )
                        prediction_parts.append(
                            _fused_rows(
                                candidate, seed, fold, fitting, held_out, scores,
                                hla_scores, diagnostic
                            )
                        )
                swap_parts.append(
                    _swap_rows(
                        candidate,
                        model_args,
                        torch,
                        DataLoader,
                        TensorDataset,
                        base,
                        model,
                        held_out,
                        mappings,
                        peptide_length,
                        device,
                        seed,
                        fold,
                    )
                )
                del model

                if not args.skip_auxiliary_retrain:
                    no_aux_args = common.model_args(
                        args,
                        condition_dim=args.condition_dim,
                        residual_l2_weight=args.residual_l2_weight,
                        tissue_loss_weight=0.0,
                        hla_loss_weight=0.0,
                    )
                    no_aux_model, no_aux_diag, _ = cexp.train_global_model(
                        candidate,
                        no_aux_args,
                        torch,
                        nn,
                        DataLoader,
                        TensorDataset,
                        base,
                        e14,
                        fitting,
                        mappings,
                        peptide_length,
                        device,
                        seed,
                        fold,
                    )
                    training_rows.extend(no_aux_diag)
                    no_aux_scores = _predict_variant(
                        candidate,
                        no_aux_args,
                        torch,
                        DataLoader,
                        TensorDataset,
                        base,
                        no_aux_model,
                        held_out,
                        peptide_length,
                        device,
                    )
                    prediction_parts.append(
                        _fused_rows(
                            candidate, seed, fold, fitting, held_out,
                            no_aux_scores, hla_scores, "D3_auxiliary_off_retrained"
                        )
                    )
                    del no_aux_model
                if device == "cuda":
                    torch.cuda.empty_cache()
            print(
                f"D fold elapsed: {time.perf_counter() - fold_started:.2f}s",
                flush=True,
            )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    if predictions.duplicated(
        ["candidate", "diagnostic", "seed", "sample_id"]
    ).any():
        raise AssertionError("D diagnostic predictions contain duplicate rows.")
    predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
    metric_input = predictions.rename(columns={"candidate": "model"}).copy()
    metric_input["candidate"] = (
        metric_input["model"].astype(str) + "::" + metric_input["diagnostic"].astype(str)
    )
    tables = cexp.metric_tables(base, metric_input.drop(columns=["model"]))
    names = (
        "per_task_metrics.csv",
        "summary_metrics.csv",
        "matched_baseline_comparison.csv",
        "other_count_metrics.csv",
        "seen_unseen_metrics.csv",
        "per_hla_metrics.csv",
        "per_tissue_metrics.csv",
    )
    for table, name in zip(tables, names, strict=True):
        table.to_csv(output_dir / name, index=False)
    diagnostic_comparison(tables[0]).to_csv(
        output_dir / "matched_baseline_comparison.csv", index=False
    )
    cexp.seed_aggregate(tables[1]).to_csv(
        output_dir / "seed_aggregate.csv", index=False
    )

    swap = pd.concat(swap_parts, ignore_index=True)
    swap_summary, swap_detail = _swap_summary(swap)
    swap_detail.to_csv(output_dir / "d1_tissue_swap_scores.csv", index=False)
    swap_summary.to_csv(output_dir / "d1_tissue_swap_summary.csv", index=False)
    pd.DataFrame(training_rows).to_csv(
        output_dir / "training_diagnostics.csv", index=False
    )
    source_files = [
        Path(__file__).resolve(),
        Path(cexp.__file__).resolve(),
        Path(bdiag.__file__).resolve(),
        common.TRAIN_PATH,
    ]
    settings = {
        "experiments": ["D1", "D2", "D3"],
        "candidates": [cexp.CANDIDATES[value]["id"] for value in CANDIDATES],
        "test_data_read": False,
        "train": str(common.TRAIN_PATH),
        "device": device,
        "seeds": list(seeds),
        "oof_folds": args.oof_folds,
        "oof_split_seed": args.oof_split_seed,
        "tissue_shuffle_seed": args.tissue_shuffle_seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "global_hla_fusion": "fixed 0.5/0.5 probability mean",
        "d1_policy": "fixed peptide/HLA; enumerate only train-observed tissue/HLA tasks",
        "d2_policy": "permute held-out tissue IDs within HLA; keep original task head",
        "d3_auxiliary_policy": (
            "matched retraining with both tissue and HLA auxiliary weights zero"
            if not args.skip_auxiliary_retrain
            else "skipped by CLI"
        ),
        "file_sha256": {
            str(path): _sha256(path) for path in source_files if path.exists()
        },
    }
    (output_dir / "run_settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"\nD experiments complete: {output_dir}\n"
        f"D total elapsed: {time.perf_counter() - started:.2f}s",
        flush=True,
    )
