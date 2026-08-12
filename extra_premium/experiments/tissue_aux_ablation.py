"""Shared implementation for the premium E14 tissue-auxiliary ablations.

The public runners in this directory intentionally preserve the original
E14a layout: one global branch, one per-HLA plain branch, and a fixed 0.5/0.5
probability fusion.  Only the global branch's tissue auxiliary objective
changes.  No original script or result directory is modified.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd


EXPERIMENTS_DIR = Path(__file__).resolve().parent
EXTRA_PREMIUM_DIR = EXPERIMENTS_DIR.parent
if str(EXTRA_PREMIUM_DIR) not in sys.path:
    sys.path.insert(0, str(EXTRA_PREMIUM_DIR))

import common


AuxiliaryMode = Literal[
    "current",
    "hla_only",
    "positive_only_tissue",
    "observed_tissue_multilabel",
    "observed_tissue_masked",
    "other_tissue_count",
]

EXPERIMENTS: dict[AuxiliaryMode, dict[str, str]] = {
    "current": {
        "id": "A0_current_auxiliary",
        "description": "A0: original E14a HLA + target-tissue auxiliary supervision.",
        "tissue_policy": "all rows predict query target_tissue (original E14a semantics)",
    },
    "hla_only": {
        "id": "A1_hla_only_auxiliary",
        "description": "A1: E14a with HLA auxiliary supervision only.",
        "tissue_policy": "disabled",
    },
    "positive_only_tissue": {
        "id": "A2_positive_only_tissue_auxiliary",
        "description": "A2: E14a where target-tissue auxiliary loss is computed for positive rows only.",
        "tissue_policy": "positive rows only; negative rows contribute no tissue auxiliary loss",
    },
    "observed_tissue_multilabel": {
        "id": "A3_observed_tissue_multilabel_auxiliary",
        "description": "A3: E14a with train-only observed-tissue multi-label auxiliary supervision.",
        "tissue_policy": "train-positive observed-tissue multi-label BCE; unobserved tissues are zero",
    },
    "observed_tissue_masked": {
        "id": "A4_observed_tissue_masked_auxiliary",
        "description": "A4: E14a with train-only observed-tissue multi-label supervision and unknown-tissue masking.",
        "tissue_policy": "multi-label BCE on observed-positive tissues and the row query tissue; other tissues are unknown/masked",
    },
    "other_tissue_count": {
        "id": "A5_other_tissue_count_auxiliary",
        "description": "A5: E14a with train-only cross-tissue-count auxiliary supervision.",
        "tissue_policy": "no tissue-class auxiliary; predict train-only other-tissue count bins 0/1/2+",
    },
}

EXPERIMENT_MODES: tuple[AuxiliaryMode, ...] = tuple(EXPERIMENTS)
DEFAULT_SEEDS = (20260704, 20260705, 20260706)
SUMMARY_METRICS = (
    "mean_task_auroc",
    "mean_task_auprc",
    "mean_task_accuracy",
    "mean_task_mcc",
    "mean_task_pair_accuracy",
    "worst_10_mean_auroc",
    "global_auroc",
    "global_auprc",
    "global_accuracy",
    "global_mcc",
)

OBSERVATION_KEY_COLUMNS = [
    "mhc_restriction",
    "molecule_parent_uniprot_id",
    "peptide_sequence",
]


def prediction_frame(
    global_branch: dict[tuple[str, str], dict[str, object]],
    hla_plain: dict[tuple[str, str], dict[str, object]],
) -> pd.DataFrame:
    """Fuse matching task predictions while checking row alignment."""
    rows: list[pd.DataFrame] = []
    for key in sorted(set(global_branch) & set(hla_plain)):
        global_item = global_branch[key]
        hla_item = hla_plain[key]
        if not np.array_equal(global_item["y_true"], hla_item["y_true"]):
            raise ValueError(f"Branches disagree on labels for task {key}.")
        global_test = global_item["test_task"]
        hla_test = hla_item["test_task"]
        if not global_test["sample_id"].reset_index(drop=True).equals(
            hla_test["sample_id"].reset_index(drop=True)
        ):
            raise ValueError(f"Branches disagree on test row order for task {key}.")
        rows.append(
            pd.DataFrame(
                {
                    "sample_id": global_test["sample_id"].to_numpy(),
                    "score": 0.5 * global_item["y_score"] + 0.5 * hla_item["y_score"],
                }
            )
        )
    if not rows:
        raise RuntimeError("No matching task predictions were produced.")
    return pd.concat(rows, ignore_index=True)


def build_train_only_observed_tissue_targets(
    train_mapped: pd.DataFrame,
    n_tissues: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Build fold-safe tissue targets from positive rows in the supplied train only.

    ``reported_tissues_same_hla_uniprot`` is deliberately not used because it
    was aggregated before the premium train/test split.  A tissue is considered
    observed only when the same HLA-parent-peptide key occurs as a positive row
    in ``train_mapped``.  For the masked A4 objective, observed-positive entries
    and the current query tissue are known; all other entries are unknown.
    """
    missing = [column for column in OBSERVATION_KEY_COLUMNS if column not in train_mapped]
    if missing:
        raise ValueError(f"Train data cannot build observed-tissue targets; missing {missing}.")

    observed_by_key: dict[tuple[str, str, str], set[int]] = {}
    positives = train_mapped[train_mapped["label"] == 1]
    for row in positives.itertuples(index=False):
        key = tuple(str(getattr(row, column)) for column in OBSERVATION_KEY_COLUMNS)
        observed_by_key.setdefault(key, set()).add(int(row.tissue_id))

    targets = np.zeros((len(train_mapped), n_tissues), dtype=np.float32)
    known_mask = np.zeros((len(train_mapped), n_tissues), dtype=np.float32)
    count_bins = np.zeros(len(train_mapped), dtype=np.int64)
    negative_query_conflicts = 0

    for row_index, row in enumerate(train_mapped.itertuples(index=False)):
        key = tuple(str(getattr(row, column)) for column in OBSERVATION_KEY_COLUMNS)
        observed = observed_by_key.get(key, set())
        if observed:
            targets[row_index, list(observed)] = 1.0
            known_mask[row_index, list(observed)] = 1.0

        query_tissue_id = int(row.tissue_id)
        label = int(row.label)
        if label == 0 and query_tissue_id in observed:
            negative_query_conflicts += 1
        targets[row_index, query_tissue_id] = float(label)
        known_mask[row_index, query_tissue_id] = 1.0

        other_count = len(observed - {query_tissue_id})
        count_bins[row_index] = min(other_count, 2)

    if negative_query_conflicts:
        raise ValueError(
            "Train-only tissue reconstruction found a negative row whose query "
            f"tissue is observed positive for the same key: {negative_query_conflicts}."
        )

    stats = {
        "unique_observation_keys": len(observed_by_key),
        "rows_with_any_observed_tissue": int((targets.sum(axis=1) > 0).sum()),
        "known_multilabel_entries": int(known_mask.sum()),
        "count_bin_0_rows": int((count_bins == 0).sum()),
        "count_bin_1_rows": int((count_bins == 1).sum()),
        "count_bin_2plus_rows": int((count_bins == 2).sum()),
    }
    return targets, known_mask, count_bins, stats


def build_semantic_aux_loader(
    args: Any,
    torch: Any,
    DataLoader: Any,
    TensorDataset: Any,
    e14: Any,
    train_mapped: pd.DataFrame,
    peptide_length: int,
    tissue_targets: np.ndarray,
    tissue_known_mask: np.ndarray,
    count_bins: np.ndarray,
) -> Any:
    peptide_ids = e14.base.encode_peptides(
        train_mapped["peptide_sequence"], peptide_length
    )
    arrays = [
        peptide_ids,
        train_mapped["task_id"].to_numpy(dtype=np.int64),
        train_mapped["tissue_id"].to_numpy(dtype=np.int64),
        train_mapped["hla_id"].to_numpy(dtype=np.int64),
        train_mapped["label"].to_numpy(dtype=np.int64),
        tissue_targets,
        tissue_known_mask,
        count_bins,
    ]
    return e14.base.build_loader(
        torch,
        DataLoader,
        TensorDataset,
        [np.asarray(array).copy() for array in arrays],
        args.batch_size,
        True,
    )


def train_semantic_global_branch(
    mode: AuxiliaryMode,
    args: Any,
    torch: Any,
    nn: Any,
    DataLoader: Any,
    TensorDataset: Any,
    train_df: pd.DataFrame,
    mappings: dict[str, Any],
    peptide_length: int,
    device: str,
    seed: int,
    e14: Any,
    e7: Any,
) -> tuple[Any, list[dict[str, object]], dict[str, int]]:
    """Train A3/A4/A5 without using globally aggregated tissue metadata."""
    if mode not in {
        "observed_tissue_multilabel",
        "observed_tissue_masked",
        "other_tissue_count",
    }:
        raise ValueError(f"Semantic auxiliary trainer does not support mode={mode!r}.")

    train_mapped = e7.prepare_with_mapping(train_df, mappings["task_to_id"])
    n_tissues = len(mappings["tissue_to_id"])
    tissue_targets, tissue_known_mask, count_bins, target_stats = (
        build_train_only_observed_tissue_targets(train_mapped, n_tissues)
    )
    model = e14.define_aux_shared_heads_model(
        args,
        torch,
        nn,
        peptide_length,
        len(mappings["task_to_id"]),
        n_tissues,
        len(mappings["hla_to_id"]),
    )
    if mode == "other_tissue_count":
        model.other_tissue_count_classifier = nn.Linear(args.hidden_dim, 3)
    model = model.to(device)

    loader = build_semantic_aux_loader(
        args,
        torch,
        DataLoader,
        TensorDataset,
        e14,
        train_mapped,
        peptide_length,
        tissue_targets,
        tissue_known_mask,
        count_bins,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    diagnostics: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        total_losses: list[float] = []
        bce_losses: list[float] = []
        hla_losses: list[float] = []
        semantic_losses: list[float] = []
        hla_accuracies: list[float] = []
        query_tissue_accuracies: list[float] = []
        count_accuracies: list[float] = []
        supervised_tissue_entries = 0

        for batch in loader:
            (
                peptide_ids,
                task_ids,
                query_tissue_ids,
                hla_ids,
                labels,
                batch_tissue_targets,
                batch_tissue_mask,
                batch_count_bins,
            ) = [item.to(device) for item in batch]
            optimizer.zero_grad(set_to_none=True)
            logits = model(peptide_ids, task_ids)
            bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels.float()
            )
            tissue_logits, hla_logits = model.auxiliary_logits(peptide_ids)
            hla_loss = torch.nn.functional.cross_entropy(hla_logits, hla_ids)

            if mode == "observed_tissue_multilabel":
                semantic_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    tissue_logits, batch_tissue_targets
                )
                supervised_tissue_entries += int(batch_tissue_targets.numel())
            elif mode == "observed_tissue_masked":
                element_losses = torch.nn.functional.binary_cross_entropy_with_logits(
                    tissue_logits, batch_tissue_targets, reduction="none"
                )
                denominator = batch_tissue_mask.sum().clamp_min(1.0)
                semantic_loss = (element_losses * batch_tissue_mask).sum() / denominator
                supervised_tissue_entries += int(batch_tissue_mask.sum().item())
            else:
                encoded = model.encode(peptide_ids)
                count_logits = model.other_tissue_count_classifier(encoded)
                semantic_loss = torch.nn.functional.cross_entropy(
                    count_logits, batch_count_bins
                )
                count_accuracies.append(
                    float(
                        (count_logits.argmax(dim=1) == batch_count_bins)
                        .float()
                        .mean()
                        .detach()
                        .cpu()
                    )
                )

            loss = (
                bce_loss
                + args.hla_loss_weight * hla_loss
                + args.tissue_loss_weight * semantic_loss
            )
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            total_losses.append(float(loss.detach().cpu()))
            bce_losses.append(float(bce_loss.detach().cpu()))
            hla_losses.append(float(hla_loss.detach().cpu()))
            semantic_losses.append(float(semantic_loss.detach().cpu()))
            hla_accuracies.append(
                float(
                    (hla_logits.argmax(dim=1) == hla_ids)
                    .float()
                    .mean()
                    .detach()
                    .cpu()
                )
            )
            if mode != "other_tissue_count":
                query_predictions = (
                    tissue_logits.gather(1, query_tissue_ids.unsqueeze(1)).squeeze(1)
                    >= 0
                )
                query_tissue_accuracies.append(
                    float(
                        (query_predictions == labels.bool())
                        .float()
                        .mean()
                        .detach()
                        .cpu()
                    )
                )

        diagnostics.append(
            {
                "experiment_id": EXPERIMENTS[mode]["id"],
                "seed": seed,
                "branch": "global_semantic_aux",
                "epoch": epoch,
                "auxiliary_mode": mode,
                "mean_total_loss": float(np.mean(total_losses)),
                "mean_bce_loss": float(np.mean(bce_losses)),
                "mean_hla_loss": float(np.mean(hla_losses)),
                "mean_semantic_auxiliary_loss": float(np.mean(semantic_losses)),
                "mean_hla_accuracy": float(np.mean(hla_accuracies)),
                "mean_query_tissue_auxiliary_accuracy": (
                    float(np.mean(query_tissue_accuracies))
                    if query_tissue_accuracies
                    else np.nan
                ),
                "mean_other_tissue_count_accuracy": (
                    float(np.mean(count_accuracies)) if count_accuracies else np.nan
                ),
                "supervised_tissue_entries": supervised_tissue_entries,
                **target_stats,
            }
        )
        print(
            f"    epoch global_semantic_aux={mode} {epoch}/{args.epochs} "
            f"duration={time.perf_counter() - started:.1f}s",
            flush=True,
        )

    return model, diagnostics, target_stats


def train_custom_global_branch(
    mode: AuxiliaryMode,
    args: Any,
    torch: Any,
    nn: Any,
    DataLoader: Any,
    TensorDataset: Any,
    train_df: pd.DataFrame,
    mappings: dict[str, Any],
    peptide_length: int,
    device: str,
    seed: int,
    e14: Any,
    e7: Any,
) -> tuple[Any, list[dict[str, object]]]:
    """Train the E14 global branch with one isolated auxiliary-loss change."""
    if mode not in {"hla_only", "positive_only_tissue"}:
        raise ValueError(f"Custom trainer does not support mode={mode!r}.")

    train_mapped = e7.prepare_with_mapping(train_df, mappings["task_to_id"])
    model = e14.define_aux_shared_heads_model(
        args,
        torch,
        nn,
        peptide_length,
        len(mappings["task_to_id"]),
        len(mappings["tissue_to_id"]),
        len(mappings["hla_to_id"]),
    ).to(device)
    loader = e14.build_aux_loader(
        args, torch, DataLoader, TensorDataset, train_mapped, peptide_length, True
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    diagnostics: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        total_losses: list[float] = []
        bce_losses: list[float] = []
        hla_losses: list[float] = []
        tissue_losses: list[float] = []
        hla_accuracies: list[float] = []
        tissue_positive_accuracies: list[float] = []
        tissue_supervised_rows = 0

        for batch in loader:
            peptide_ids, task_ids, tissue_ids, hla_ids, labels = [
                item.to(device) for item in batch
            ]
            optimizer.zero_grad(set_to_none=True)
            logits = model(peptide_ids, task_ids)
            bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels.float()
            )
            tissue_logits, hla_logits = model.auxiliary_logits(peptide_ids)
            hla_loss = torch.nn.functional.cross_entropy(hla_logits, hla_ids)

            if mode == "hla_only":
                tissue_loss = logits.new_zeros(())
                supervised_mask = torch.zeros_like(labels, dtype=torch.bool)
            else:
                supervised_mask = labels == 1
                if bool(supervised_mask.any()):
                    tissue_loss = torch.nn.functional.cross_entropy(
                        tissue_logits[supervised_mask], tissue_ids[supervised_mask]
                    )
                    tissue_positive_accuracies.append(
                        float(
                            (
                                tissue_logits[supervised_mask].argmax(dim=1)
                                == tissue_ids[supervised_mask]
                            )
                            .float()
                            .mean()
                            .detach()
                            .cpu()
                        )
                    )
                    tissue_supervised_rows += int(supervised_mask.sum().item())
                else:
                    tissue_loss = logits.new_zeros(())

            loss = bce_loss + args.hla_loss_weight * hla_loss
            if mode == "positive_only_tissue":
                loss = loss + args.tissue_loss_weight * tissue_loss
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            total_losses.append(float(loss.detach().cpu()))
            bce_losses.append(float(bce_loss.detach().cpu()))
            hla_losses.append(float(hla_loss.detach().cpu()))
            tissue_losses.append(float(tissue_loss.detach().cpu()))
            hla_accuracies.append(
                float(
                    (hla_logits.argmax(dim=1) == hla_ids)
                    .float()
                    .mean()
                    .detach()
                    .cpu()
                )
            )

        diagnostics.append(
            {
                "experiment_id": EXPERIMENTS[mode]["id"],
                "seed": seed,
                "branch": "global_custom_aux",
                "epoch": epoch,
                "auxiliary_mode": mode,
                "mean_total_loss": float(np.mean(total_losses)),
                "mean_bce_loss": float(np.mean(bce_losses)),
                "mean_hla_loss": float(np.mean(hla_losses)),
                "mean_tissue_loss": float(np.mean(tissue_losses)),
                "mean_hla_accuracy": float(np.mean(hla_accuracies)),
                "mean_tissue_positive_accuracy": (
                    float(np.mean(tissue_positive_accuracies))
                    if tissue_positive_accuracies
                    else np.nan
                ),
                "tissue_supervised_rows": tissue_supervised_rows,
            }
        )
        print(
            f"    epoch global_custom_aux={mode} {epoch}/{args.epochs} "
            f"duration={time.perf_counter() - started:.1f}s",
            flush=True,
        )

    return model, diagnostics


def custom_global_predictions(
    mode: AuxiliaryMode,
    args: Any,
    torch: Any,
    nn: Any,
    DataLoader: Any,
    TensorDataset: Any,
    train: pd.DataFrame,
    test: pd.DataFrame,
    mappings: dict[str, Any],
    peptide_length: int,
    device: str,
    seed: int,
    e14: Any,
    e7: Any,
) -> tuple[dict[tuple[str, str], dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    print(f"  train global branch with auxiliary mode={mode}", flush=True)
    model, diagnostics = train_custom_global_branch(
        mode,
        args,
        torch,
        nn,
        DataLoader,
        TensorDataset,
        train,
        mappings,
        peptide_length,
        device,
        seed,
        e14,
        e7,
    )
    predictions, candidate_rows = e14.predict_branch(
        args,
        torch,
        DataLoader,
        TensorDataset,
        model,
        train,
        test,
        mappings["task_to_id"],
        peptide_length,
        device,
        seed,
        f"global_{mode}",
        "all_tasks",
        True,
    )
    return predictions, candidate_rows, diagnostics


def semantic_global_predictions(
    mode: AuxiliaryMode,
    args: Any,
    torch: Any,
    nn: Any,
    DataLoader: Any,
    TensorDataset: Any,
    train: pd.DataFrame,
    test: pd.DataFrame,
    mappings: dict[str, Any],
    peptide_length: int,
    device: str,
    seed: int,
    e14: Any,
    e7: Any,
) -> tuple[
    dict[tuple[str, str], dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, int],
]:
    print(f"  train global branch with semantic auxiliary mode={mode}", flush=True)
    model, diagnostics, target_stats = train_semantic_global_branch(
        mode,
        args,
        torch,
        nn,
        DataLoader,
        TensorDataset,
        train,
        mappings,
        peptide_length,
        device,
        seed,
        e14,
        e7,
    )
    predictions, candidate_rows = e14.predict_branch(
        args,
        torch,
        DataLoader,
        TensorDataset,
        model,
        train,
        test,
        mappings["task_to_id"],
        peptide_length,
        device,
        seed,
        f"global_{mode}",
        "all_tasks",
        True,
    )
    return predictions, candidate_rows, diagnostics, target_stats


def experiment_parser(description: str) -> Any:
    parser = common.basic_parser(description)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help=(
            "Training seeds. The default runs the common three-seed panel: "
            + " ".join(str(seed) for seed in DEFAULT_SEEDS)
        ),
    )
    return parser


def validate_seeds(raw_seeds: list[int]) -> tuple[int, ...]:
    seeds = tuple(int(seed) for seed in raw_seeds)
    if not seeds:
        raise ValueError("--seeds requires at least one integer.")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"--seeds contains duplicates: {seeds}")
    if any(seed < 0 for seed in seeds):
        raise ValueError(f"Seeds must be non-negative: {seeds}")
    return seeds


def aggregate_seed_summaries(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Return one mean/std row per experiment without pooling test rows."""
    rows: list[dict[str, object]] = []
    for experiment_id, group in per_seed.groupby("experiment_id", sort=False):
        row: dict[str, object] = {
            "experiment_id": experiment_id,
            "n_seeds": int(group["seed"].nunique()),
            "seeds": ",".join(str(value) for value in sorted(group["seed"].unique())),
        }
        for metric in SUMMARY_METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def run_single_seed(
    mode: AuxiliaryMode,
    seed: int,
    cli: Any,
    torch: Any,
    nn: Any,
    DataLoader: Any,
    TensorDataset: Any,
    device: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    mappings: dict[str, Any],
    peptide_length: int,
    base: Any,
    e14: Any,
    e7: Any,
) -> dict[str, Any]:
    spec = EXPERIMENTS[mode]
    args = common.model_args(cli)
    # Keep repeated seeds and experiments independent even if a frozen helper
    # adds temporary columns in place.
    train_run = train.copy(deep=True)
    test_run = test.copy(deep=True)

    # Every branch starts from the same explicit experiment seed. Resetting
    # before the HLA branch prevents A5's extra classifier initialization (or
    # any other global-branch implementation detail) from changing its RNG.
    e14.set_seed(seed, torch)
    started = time.perf_counter()
    print(
        f"{spec['id']}: device={device}, seed={seed}, "
        f"train_rows={len(train_run)}, test_rows={len(test_run)}, "
        f"tasks={len(mappings['tasks'])}",
        flush=True,
    )

    target_stats: dict[str, int] = {}
    if mode == "current":
        global_predictions, _, global_diagnostics = e14.train_and_predict_global_branch(
            args,
            torch,
            nn,
            DataLoader,
            TensorDataset,
            train_run,
            test_run,
            mappings,
            peptide_length,
            device,
            seed,
            True,
        )
    elif mode in {"hla_only", "positive_only_tissue"}:
        global_predictions, _, global_diagnostics = custom_global_predictions(
            mode,
            args,
            torch,
            nn,
            DataLoader,
            TensorDataset,
            train_run,
            test_run,
            mappings,
            peptide_length,
            device,
            seed,
            e14,
            e7,
        )
    else:
        (
            global_predictions,
            _,
            global_diagnostics,
            target_stats,
        ) = semantic_global_predictions(
            mode,
            args,
            torch,
            nn,
            DataLoader,
            TensorDataset,
            train_run,
            test_run,
            mappings,
            peptide_length,
            device,
            seed,
            e14,
            e7,
        )

    e14.set_seed(seed, torch)
    hla_predictions, _, hla_diagnostics = e14.train_and_predict_hla_branches(
        args,
        torch,
        nn,
        DataLoader,
        TensorDataset,
        train_run,
        test_run,
        mappings,
        peptide_length,
        device,
        seed,
        False,
    )

    predicted = prediction_frame(global_predictions, hla_predictions)
    aligned = test_run[["sample_id"]].merge(
        predicted, on="sample_id", how="left", validate="one_to_one"
    )
    if aligned["score"].isna().any():
        raise ValueError("The fused model did not produce every premium test score.")

    model_name = f"experiments/{spec['id']}/seed_{seed}"
    summary = common.save_basic_test_results(
        model_name,
        train_run,
        test_run,
        aligned["score"].to_numpy(),
        base,
        {
            "experiment_id": spec["id"],
            "source_model": "scripts/run_tissuepmhc_auxiliary_soft_ensemble.py",
            "architecture": "E14 global branch plus per-HLA plain branch, fixed 0.5 probability fusion",
            "global_tissue_auxiliary_policy": spec["tissue_policy"],
            "global_hla_auxiliary_policy": "all rows",
            "auxiliary_target_source": {
                "current": "original query target_tissue metadata",
                "positive_only_tissue": "original query target_tissue metadata on positive rows only",
                "hla_only": "not applicable; tissue auxiliary supervision is disabled",
                "observed_tissue_multilabel": "premium train positive rows only; globally aggregated reported-tissues column is not used",
                "observed_tissue_masked": "premium train positive rows only; globally aggregated reported-tissues column is not used",
                "other_tissue_count": "premium train positive rows only; globally aggregated other-tissue count is not used",
            }[mode],
            "train_only_target_statistics": target_stats,
            "device": device,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "experiment_seed": seed,
            "branch_seed_policy": (
                "reset the complete RNG state to experiment_seed immediately "
                "before both the global and HLA branch"
            ),
            "tissue_loss_weight": args.tissue_loss_weight,
            "hla_loss_weight": args.hla_loss_weight,
            "elapsed_seconds": time.perf_counter() - started,
        },
        seed=seed,
    )

    diagnostics = pd.DataFrame([*global_diagnostics, *hla_diagnostics])
    output_dir = common.RESULTS_ROOT / model_name
    diagnostics.to_csv(output_dir / "training_diagnostics.csv", index=False)
    print(f"wrote: {output_dir / 'training_diagnostics.csv'}", flush=True)
    return {"experiment_id": spec["id"], **summary}


def run_experiment_modes(modes: tuple[AuxiliaryMode, ...], cli: Any) -> None:
    if cli.epochs < 1:
        raise ValueError("--epochs must be positive.")
    if cli.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    seeds = validate_seeds(cli.seeds)

    common.enable_original_modules()
    import run_tissuepmhc_auxiliary_soft_ensemble as e14
    import run_tissuepmhc_neural_baselines_v2 as base
    import run_tissuepmhc_selective_grouping as e7

    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = common.resolve_device(cli.device, torch)
    train, test, mappings, peptide_length = common.load_premium_data(base)

    all_summaries: list[dict[str, Any]] = []
    total_runs = len(modes) * len(seeds)
    run_number = 0
    for mode in modes:
        experiment_summaries: list[dict[str, Any]] = []
        for seed in seeds:
            run_number += 1
            print(
                f"\n=== A experiment run {run_number}/{total_runs}: "
                f"{EXPERIMENTS[mode]['id']} seed={seed} ===",
                flush=True,
            )
            summary = run_single_seed(
                mode,
                seed,
                cli,
                torch,
                nn,
                DataLoader,
                TensorDataset,
                device,
                train,
                test,
                mappings,
                peptide_length,
                base,
                e14,
                e7,
            )
            experiment_summaries.append(summary)
            all_summaries.append(summary)

        experiment_dir = (
            common.RESULTS_ROOT / "experiments" / EXPERIMENTS[mode]["id"]
        )
        per_seed = pd.DataFrame(experiment_summaries)
        per_seed.to_csv(experiment_dir / "seed_summary.csv", index=False)
        aggregate_seed_summaries(per_seed).to_csv(
            experiment_dir / "seed_aggregate.csv", index=False
        )
        print(f"wrote: {experiment_dir / 'seed_summary.csv'}", flush=True)
        print(f"wrote: {experiment_dir / 'seed_aggregate.csv'}", flush=True)

    if len(modes) > 1:
        results_dir = common.RESULTS_ROOT / "experiments"
        per_seed = pd.DataFrame(all_summaries)
        per_seed.to_csv(results_dir / "A_experiments_seed_summary.csv", index=False)
        aggregate_seed_summaries(per_seed).to_csv(
            results_dir / "A_experiments_seed_aggregate.csv", index=False
        )
        print(
            f"wrote: {results_dir / 'A_experiments_seed_summary.csv'}",
            flush=True,
        )
        print(
            f"wrote: {results_dir / 'A_experiments_seed_aggregate.csv'}",
            flush=True,
        )


def run_experiment(mode: AuxiliaryMode) -> None:
    cli = experiment_parser(EXPERIMENTS[mode]["description"]).parse_args()
    run_experiment_modes((mode,), cli)


def run_all_experiments() -> None:
    cli = experiment_parser(
        "Run A0-A5 tissue-auxiliary ablations with a shared seed panel."
    ).parse_args()
    run_experiment_modes(EXPERIMENT_MODES, cli)
