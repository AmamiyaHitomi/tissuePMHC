"""Premium train-only OOF diagnostics for tissue auxiliary supervision.

B1 reports tissue/HLA auxiliary behavior separately for positive and negative
rows. B2 measures main/auxiliary gradient cosines at the shared peptide
encoder. B3 compares the matched current auxiliary against a fitting-fold-only
permutation of tissue auxiliary labels.

The premium fixed test file is deliberately never opened by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


BPart = Literal["b1", "b2", "b3"]
DEFAULT_SEEDS = (20260704, 20260705, 20260706)
DEFAULT_OOF_FOLDS = 3
DEFAULT_OOF_SPLIT_SEED = 20260711
DEFAULT_TISSUE_SHUFFLE_SEED = 20260712

CANDIDATE_CURRENT = "B3_current_tissue_auxiliary"
CANDIDATE_SHUFFLED = "B3_shuffled_tissue_auxiliary"
METRICS = (
    "accuracy",
    "balanced_accuracy",
    "auroc",
    "auprc",
    "f1",
    "mcc",
)


def parser(description: str) -> argparse.ArgumentParser:
    result = common.basic_parser(description)
    result.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Independent training seeds. B defaults to the shared three-seed protocol.",
    )
    result.add_argument("--oof-folds", type=int, default=DEFAULT_OOF_FOLDS)
    result.add_argument(
        "--oof-split-seed", type=int, default=DEFAULT_OOF_SPLIT_SEED
    )
    result.add_argument(
        "--tissue-shuffle-seed",
        type=int,
        default=DEFAULT_TISSUE_SHUFFLE_SEED,
        help="Defines one fitting-fold tissue-label permutation shared by model seeds.",
    )
    result.add_argument(
        "--gradient-max-batches",
        type=int,
        default=0,
        help="0 audits every held-out batch; a positive value limits B2 for smoke runs.",
    )
    return result


def validate_cli(args: argparse.Namespace) -> tuple[int, ...]:
    seeds = tuple(int(seed) for seed in args.seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain one or more unique integers.")
    if any(seed < 0 for seed in seeds):
        raise ValueError("--seeds must be non-negative.")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive.")
    if args.oof_folds < 2:
        raise ValueError("--oof-folds must be at least two.")
    if args.gradient_max_batches < 0:
        raise ValueError("--gradient-max-batches must be non-negative.")
    return seeds


def load_premium_train(base: Any) -> tuple[pd.DataFrame, dict[str, Any], int]:
    """Load and validate premium train without opening the fixed test file."""
    raw = base.read_dataset(common.TRAIN_PATH)
    missing = common.REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"premium train is missing columns: {sorted(missing)}")
    if "other_tissue_presentation_count" not in raw:
        raise ValueError("premium train lacks other_tissue_presentation_count.")
    if set(raw["dataset"].astype(str)) != {"humanPMHC"}:
        raise ValueError("B experiments require humanPMHC rows only.")
    if set(raw["split"].astype(str)) != {"train"}:
        raise ValueError("B experiments may read premium train rows only.")
    if not raw["peptide_sequence"].astype(str).str.fullmatch(
        r"[ACDEFGHIKLMNPQRSTVWY]{9}"
    ).all():
        raise ValueError("B experiments require canonical 9-mer peptides.")
    pair_labels = raw.groupby("pair_id", sort=False)["label"].agg(list)
    if not pair_labels.map(
        lambda values: sorted(int(value) for value in values) == [0, 1]
    ).all():
        raise ValueError("Every train pair must contain exactly one 0 and one 1.")

    train, duplicate, mappings = base.add_task_columns(raw, raw.copy())
    if len(train) != len(raw) or len(duplicate) != len(raw):
        raise ValueError("Train-only task mapping unexpectedly removed rows.")
    peptide_length = int(train["peptide_sequence"].str.len().max())
    return train, mappings, peptide_length


def make_pair_grouped_folds(
    train: pd.DataFrame, n_folds: int, split_seed: int
) -> pd.Series:
    """Assign pairs inside each task, keeping every pair in one OOF fold."""
    rng = np.random.default_rng(split_seed)
    assignments = pd.Series(index=train.index, dtype="int64")
    for task_name, task in train.groupby("task_name", sort=True):
        pairs = np.asarray(sorted(task["pair_id"].astype(str).unique()))
        if len(pairs) < n_folds:
            raise ValueError(
                f"Task {task_name!r} has {len(pairs)} pairs, fewer than {n_folds} folds."
            )
        shuffled = rng.permutation(pairs)
        pair_to_fold = {
            pair_id: position % n_folds
            for position, pair_id in enumerate(shuffled)
        }
        assignments.loc[task.index] = (
            task["pair_id"].astype(str).map(pair_to_fold).astype(int)
        )
    if assignments.isna().any():
        raise AssertionError("Some train rows did not receive an OOF fold.")
    for fold in range(n_folds):
        fitting_pairs = set(train.loc[assignments != fold, "pair_id"].astype(str))
        held_out_pairs = set(train.loc[assignments == fold, "pair_id"].astype(str))
        if fitting_pairs & held_out_pairs:
            raise AssertionError(f"pair_id leakage in fold {fold}.")
    return assignments.astype(int)


def fold_assignment_table(
    train: pd.DataFrame, assignments: pd.Series
) -> pd.DataFrame:
    result = train[
        [
            "sample_id",
            "pair_id",
            "task_name",
            "target_tissue",
            "mhc_restriction",
            "label",
        ]
    ].copy()
    result["fold"] = assignments.to_numpy(dtype=np.int64)
    return result.sort_values(["fold", "task_name", "pair_id", "label"])


def other_count_group(values: pd.Series) -> pd.Series:
    numeric = values.astype(int)
    return numeric.map(lambda value: "1" if value == 1 else ("2" if value == 2 else "3+"))


def build_loader(
    args: argparse.Namespace,
    torch: Any,
    DataLoader: Any,
    TensorDataset: Any,
    base: Any,
    frame: pd.DataFrame,
    peptide_length: int,
    *,
    shuffle: bool,
    auxiliary_tissue_ids: np.ndarray | None = None,
) -> Any:
    if auxiliary_tissue_ids is None:
        tissue_ids = frame["tissue_id"].to_numpy(dtype=np.int64)
    else:
        tissue_ids = np.asarray(auxiliary_tissue_ids, dtype=np.int64)
        if tissue_ids.shape != (len(frame),):
            raise ValueError("Auxiliary tissue labels do not align with fitting rows.")
    arrays = [
        base.encode_peptides(frame["peptide_sequence"], peptide_length),
        frame["task_id"].to_numpy(dtype=np.int64),
        tissue_ids,
        frame["tissue_id"].to_numpy(dtype=np.int64),
        frame["hla_id"].to_numpy(dtype=np.int64),
        frame["label"].to_numpy(dtype=np.int64),
    ]
    return base.build_loader(
        torch,
        DataLoader,
        TensorDataset,
        [np.asarray(value).copy() for value in arrays],
        args.batch_size,
        shuffle,
    )


def train_global_model(
    args: argparse.Namespace,
    torch: Any,
    nn: Any,
    DataLoader: Any,
    TensorDataset: Any,
    base: Any,
    e14: Any,
    fitting: pd.DataFrame,
    mappings: dict[str, Any],
    peptide_length: int,
    device: str,
    seed: int,
    fold: int,
    candidate: str,
    auxiliary_tissue_ids: np.ndarray,
) -> tuple[Any, list[dict[str, object]]]:
    model = e14.define_aux_shared_heads_model(
        args,
        torch,
        nn,
        peptide_length,
        len(mappings["task_to_id"]),
        len(mappings["tissue_to_id"]),
        len(mappings["hla_to_id"]),
    ).to(device)
    loader = build_loader(
        args,
        torch,
        DataLoader,
        TensorDataset,
        base,
        fitting,
        peptide_length,
        shuffle=True,
        auxiliary_tissue_ids=auxiliary_tissue_ids,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    rows: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        total_values: list[float] = []
        main_values: list[float] = []
        tissue_values: list[float] = []
        hla_values: list[float] = []
        auxiliary_tissue_accuracies: list[float] = []
        query_tissue_accuracies: list[float] = []
        hla_accuracies: list[float] = []
        model.train()
        for batch in loader:
            (
                peptide_ids,
                task_ids,
                auxiliary_tissue_targets,
                query_tissue_ids,
                hla_ids,
                labels,
            ) = [value.to(device) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            main_logits = model(peptide_ids, task_ids)
            tissue_logits, hla_logits = model.auxiliary_logits(peptide_ids)
            main_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                main_logits, labels.float()
            )
            tissue_loss = torch.nn.functional.cross_entropy(
                tissue_logits, auxiliary_tissue_targets
            )
            hla_loss = torch.nn.functional.cross_entropy(hla_logits, hla_ids)
            total_loss = (
                main_loss
                + args.tissue_loss_weight * tissue_loss
                + args.hla_loss_weight * hla_loss
            )
            total_loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                )
            optimizer.step()

            tissue_predictions = tissue_logits.argmax(dim=1)
            total_values.append(float(total_loss.detach().cpu()))
            main_values.append(float(main_loss.detach().cpu()))
            tissue_values.append(float(tissue_loss.detach().cpu()))
            hla_values.append(float(hla_loss.detach().cpu()))
            auxiliary_tissue_accuracies.append(
                float(
                    (tissue_predictions == auxiliary_tissue_targets)
                    .float()
                    .mean()
                    .detach()
                    .cpu()
                )
            )
            query_tissue_accuracies.append(
                float(
                    (tissue_predictions == query_tissue_ids)
                    .float()
                    .mean()
                    .detach()
                    .cpu()
                )
            )
            hla_accuracies.append(
                float(
                    (hla_logits.argmax(dim=1) == hla_ids)
                    .float()
                    .mean()
                    .detach()
                    .cpu()
                )
            )
        rows.append(
            {
                "candidate": candidate,
                "seed": seed,
                "fold": fold,
                "branch": "global_aux",
                "epoch": epoch,
                "mean_total_loss": float(np.mean(total_values)),
                "mean_main_bce_loss": float(np.mean(main_values)),
                "mean_tissue_loss": float(np.mean(tissue_values)),
                "mean_hla_loss": float(np.mean(hla_values)),
                "mean_auxiliary_tissue_accuracy": float(
                    np.mean(auxiliary_tissue_accuracies)
                ),
                "mean_query_tissue_accuracy": float(
                    np.mean(query_tissue_accuracies)
                ),
                "mean_hla_accuracy": float(np.mean(hla_accuracies)),
            }
        )
        print(
            f"    {candidate} epoch={epoch}/{args.epochs} "
            f"loss={np.mean(total_values):.5f} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    return model, rows


def predict_main_scores(
    args: argparse.Namespace,
    torch: Any,
    DataLoader: Any,
    TensorDataset: Any,
    base: Any,
    model: Any,
    frame: pd.DataFrame,
    peptide_length: int,
    device: str,
) -> np.ndarray:
    loader = build_loader(
        args,
        torch,
        DataLoader,
        TensorDataset,
        base,
        frame,
        peptide_length,
        shuffle=False,
    )
    values: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for peptide_ids, task_ids, _, _, _, _ in loader:
            logits = model(peptide_ids.to(device), task_ids.to(device))
            values.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(values).astype(float)


def auxiliary_prediction_rows(
    args: argparse.Namespace,
    torch: Any,
    DataLoader: Any,
    TensorDataset: Any,
    base: Any,
    model: Any,
    held_out: pd.DataFrame,
    peptide_length: int,
    device: str,
    seed: int,
    fold: int,
    id_to_tissue: dict[int, str],
) -> pd.DataFrame:
    loader = build_loader(
        args,
        torch,
        DataLoader,
        TensorDataset,
        base,
        held_out,
        peptide_length,
        shuffle=False,
    )
    tissue_predictions: list[np.ndarray] = []
    tissue_probabilities: list[np.ndarray] = []
    hla_predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for peptide_ids, _, _, query_tissue_ids, _, _ in loader:
            tissue_logits, hla_logits = model.auxiliary_logits(
                peptide_ids.to(device)
            )
            probabilities = torch.softmax(tissue_logits, dim=1)
            tissue_predictions.append(tissue_logits.argmax(dim=1).cpu().numpy())
            tissue_probabilities.append(
                probabilities.gather(
                    1, query_tissue_ids.to(device).unsqueeze(1)
                )
                .squeeze(1)
                .cpu()
                .numpy()
            )
            hla_predictions.append(hla_logits.argmax(dim=1).cpu().numpy())

    result = held_out[
        [
            "sample_id",
            "pair_id",
            "target_tissue",
            "mhc_restriction",
            "label",
            "tissue_id",
            "hla_id",
            "other_tissue_presentation_count",
        ]
    ].copy()
    result.insert(0, "fold", fold)
    result.insert(0, "seed", seed)
    result["other_count_group"] = other_count_group(
        result["other_tissue_presentation_count"]
    )
    result["predicted_tissue_id"] = np.concatenate(tissue_predictions).astype(int)
    result["predicted_tissue"] = result["predicted_tissue_id"].map(id_to_tissue)
    result["query_tissue_probability"] = np.concatenate(
        tissue_probabilities
    ).astype(float)
    result["tissue_correct"] = (
        result["predicted_tissue_id"] == result["tissue_id"]
    )
    result["tissue_nll"] = -np.log(
        result["query_tissue_probability"].clip(lower=1e-12)
    )
    result["hla_correct"] = (
        np.concatenate(hla_predictions).astype(int) == result["hla_id"].to_numpy()
    )
    return result


def b1_metric_table(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    def add(dimension: str, group_columns: list[str]) -> None:
        for keys, group in rows.groupby(
            ["seed", "fold", *group_columns], sort=True, dropna=False
        ):
            if not isinstance(keys, tuple):
                keys = (keys,)
            record: dict[str, object] = {
                "dimension": dimension,
                "seed": int(keys[0]),
                "fold": int(keys[1]),
                "n_rows": int(len(group)),
                "tissue_accuracy": float(group["tissue_correct"].mean()),
                "mean_query_tissue_probability": float(
                    group["query_tissue_probability"].mean()
                ),
                "mean_tissue_nll": float(group["tissue_nll"].mean()),
                "hla_accuracy": float(group["hla_correct"].mean()),
            }
            for column, value in zip(group_columns, keys[2:], strict=True):
                record[column] = value
            records.append(record)

    add("label", ["label"])
    add("label_x_other_count", ["label", "other_count_group"])
    add("label_x_tissue", ["label", "target_tissue"])
    return pd.DataFrame(records)


def gradient_stats(
    torch: Any,
    first_loss: Any,
    second_loss: Any,
    parameters: list[Any],
) -> dict[str, float]:
    first = torch.autograd.grad(
        first_loss, parameters, retain_graph=False, allow_unused=True
    )
    second = torch.autograd.grad(
        second_loss, parameters, retain_graph=False, allow_unused=True
    )
    first_values = [
        value.detach().flatten() for value in first if value is not None
    ]
    second_values = [
        value.detach().flatten() for value in second if value is not None
    ]
    if not first_values or not second_values:
        return {
            "cosine": float("nan"),
            "main_gradient_norm": float("nan"),
            "auxiliary_gradient_norm": float("nan"),
            "gradient_dot_product": float("nan"),
        }
    first_vector = torch.cat(first_values)
    second_vector = torch.cat(second_values)
    first_norm = torch.linalg.vector_norm(first_vector)
    second_norm = torch.linalg.vector_norm(second_vector)
    dot = torch.dot(first_vector, second_vector)
    cosine = dot / (first_norm * second_norm).clamp_min(1e-12)
    return {
        "cosine": float(cosine.detach().cpu()),
        "main_gradient_norm": float(first_norm.detach().cpu()),
        "auxiliary_gradient_norm": float(second_norm.detach().cpu()),
        "gradient_dot_product": float(dot.detach().cpu()),
    }


def gradient_audit_rows(
    args: argparse.Namespace,
    torch: Any,
    DataLoader: Any,
    TensorDataset: Any,
    base: Any,
    model: Any,
    held_out: pd.DataFrame,
    peptide_length: int,
    device: str,
    seed: int,
    fold: int,
) -> pd.DataFrame:
    """Audit held-out subgroup gradients with dropout disabled."""
    audit = held_out.reset_index(drop=True).copy()
    audit["other_count_group"] = other_count_group(
        audit["other_tissue_presentation_count"]
    )
    loader = build_loader(
        args,
        torch,
        DataLoader,
        TensorDataset,
        base,
        audit,
        peptide_length,
        shuffle=False,
    )
    encoder_parameters = list(model.embedding.parameters()) + list(
        model.encoder.parameters()
    )
    records: list[dict[str, object]] = []
    model.eval()
    row_start = 0
    for batch_index, batch in enumerate(loader):
        if args.gradient_max_batches and batch_index >= args.gradient_max_batches:
            break
        (
            peptide_ids,
            task_ids,
            _,
            tissue_ids,
            hla_ids,
            labels,
        ) = [value.to(device) for value in batch]
        batch_meta = audit.iloc[row_start : row_start + len(labels)].reset_index(
            drop=True
        )
        row_start += len(labels)
        batch_other_groups = batch_meta["other_count_group"].to_numpy()

        for label_value, label_name in ((1, "positive"), (0, "negative")):
            label_mask_cpu = (
                batch_meta["label"].to_numpy(dtype=np.int64) == label_value
            )
            subgroup_names = ["all", "1", "2", "3+"]
            for subgroup_name in subgroup_names:
                subgroup_mask_cpu = label_mask_cpu.copy()
                if subgroup_name != "all":
                    subgroup_mask_cpu &= batch_other_groups == subgroup_name
                positions = np.flatnonzero(subgroup_mask_cpu)
                if len(positions) < 2:
                    continue
                positions_tensor = torch.as_tensor(
                    positions, dtype=torch.long, device=device
                )
                subgroup_peptides = peptide_ids.index_select(0, positions_tensor)
                subgroup_tasks = task_ids.index_select(0, positions_tensor)
                subgroup_tissues = tissue_ids.index_select(0, positions_tensor)
                subgroup_hlas = hla_ids.index_select(0, positions_tensor)
                subgroup_labels = labels.index_select(0, positions_tensor)

                main_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    model(subgroup_peptides, subgroup_tasks),
                    subgroup_labels.float(),
                )
                tissue_logits, _ = model.auxiliary_logits(subgroup_peptides)
                tissue_loss = torch.nn.functional.cross_entropy(
                    tissue_logits, subgroup_tissues
                )
                tissue_stats = gradient_stats(
                    torch, main_loss, tissue_loss, encoder_parameters
                )

                main_loss_hla = (
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        model(subgroup_peptides, subgroup_tasks),
                        subgroup_labels.float(),
                    )
                )
                _, hla_logits = model.auxiliary_logits(subgroup_peptides)
                hla_loss = torch.nn.functional.cross_entropy(
                    hla_logits, subgroup_hlas
                )
                hla_stats = gradient_stats(
                    torch, main_loss_hla, hla_loss, encoder_parameters
                )
                records.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "batch_index": batch_index,
                        "label": label_value,
                        "label_group": label_name,
                        "other_count_group": subgroup_name,
                        "n_rows": int(len(positions)),
                        "main_loss": float(main_loss.detach().cpu()),
                        "tissue_loss": float(tissue_loss.detach().cpu()),
                        "hla_loss": float(hla_loss.detach().cpu()),
                        "gradient_cosine_main_tissue": tissue_stats["cosine"],
                        "main_gradient_norm_for_tissue": tissue_stats[
                            "main_gradient_norm"
                        ],
                        "tissue_gradient_norm": tissue_stats[
                            "auxiliary_gradient_norm"
                        ],
                        "gradient_dot_main_tissue": tissue_stats[
                            "gradient_dot_product"
                        ],
                        "gradient_cosine_main_hla": hla_stats["cosine"],
                        "main_gradient_norm_for_hla": hla_stats[
                            "main_gradient_norm"
                        ],
                        "hla_gradient_norm": hla_stats[
                            "auxiliary_gradient_norm"
                        ],
                        "gradient_dot_main_hla": hla_stats[
                            "gradient_dot_product"
                        ],
                    }
                )
    return pd.DataFrame(records)


def b2_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for keys, group in rows.groupby(
        ["seed", "label_group", "other_count_group"], sort=True
    ):
        weights = group["n_rows"].to_numpy(dtype=float)
        record: dict[str, object] = {
            "seed": int(keys[0]),
            "label_group": keys[1],
            "other_count_group": keys[2],
            "n_batches": int(len(group)),
            "n_row_observations": int(group["n_rows"].sum()),
        }
        for auxiliary in ("tissue", "hla"):
            values = group[
                f"gradient_cosine_main_{auxiliary}"
            ].to_numpy(dtype=float)
            finite = np.isfinite(values)
            record[f"mean_cosine_main_{auxiliary}"] = float(
                np.mean(values[finite])
            )
            record[f"weighted_mean_cosine_main_{auxiliary}"] = float(
                np.average(values[finite], weights=weights[finite])
            )
            record[f"median_cosine_main_{auxiliary}"] = float(
                np.median(values[finite])
            )
            record[f"conflict_fraction_main_{auxiliary}"] = float(
                np.mean(values[finite] < 0)
            )
        records.append(record)
    return pd.DataFrame(records)


def hla_prediction_frame(
    predictions: dict[tuple[str, str], dict[str, object]]
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for item in predictions.values():
        test_task = item["test_task"].reset_index(drop=True)
        rows.append(
            pd.DataFrame(
                {
                    "sample_id": test_task["sample_id"].to_numpy(),
                    "hla_score": np.asarray(item["y_score"], dtype=float),
                }
            )
        )
    result = pd.concat(rows, ignore_index=True)
    if result["sample_id"].duplicated().any():
        raise AssertionError("HLA prediction frame contains duplicate sample IDs.")
    return result


def add_seen_metadata(
    held_out: pd.DataFrame, fitting: pd.DataFrame
) -> pd.DataFrame:
    result = held_out.copy()
    seen_peptides = set(fitting["peptide_sequence"].astype(str))
    seen_hla_peptides = set(
        zip(
            fitting["mhc_restriction"].astype(str),
            fitting["peptide_sequence"].astype(str),
            strict=True,
        )
    )
    result["peptide_seen_in_fitting"] = (
        result["peptide_sequence"].astype(str).isin(seen_peptides)
    )
    result["hla_peptide_seen_in_fitting"] = [
        value in seen_hla_peptides
        for value in zip(
            result["mhc_restriction"].astype(str),
            result["peptide_sequence"].astype(str),
            strict=True,
        )
    ]
    result["other_count_group"] = other_count_group(
        result["other_tissue_presentation_count"]
    )
    pair_seen = result.groupby("pair_id", sort=False)[
        "peptide_seen_in_fitting"
    ].transform("sum")
    result["peptide_seen_count_in_pair"] = pair_seen.astype(int)
    return result


def fold_prediction_rows(
    held_out: pd.DataFrame,
    fitting: pd.DataFrame,
    hla_scores: pd.DataFrame,
    global_scores: np.ndarray,
    candidate: str,
    seed: int,
    fold: int,
) -> pd.DataFrame:
    result = add_seen_metadata(held_out, fitting)
    result["global_score"] = np.asarray(global_scores, dtype=float)
    result = result.merge(
        hla_scores, on="sample_id", how="left", validate="one_to_one"
    )
    if result["hla_score"].isna().any():
        raise AssertionError("Missing HLA scores for held-out rows.")
    result["score"] = 0.5 * result["global_score"] + 0.5 * result["hla_score"]
    result.insert(0, "fold", fold)
    result.insert(0, "seed", seed)
    result.insert(0, "candidate", candidate)
    columns = [
        "candidate",
        "seed",
        "fold",
        "sample_id",
        "pair_id",
        "target_tissue",
        "mhc_restriction",
        "peptide_sequence",
        "label",
        "other_tissue_presentation_count",
        "other_count_group",
        "peptide_seen_in_fitting",
        "hla_peptide_seen_in_fitting",
        "peptide_seen_count_in_pair",
        "global_score",
        "hla_score",
        "score",
    ]
    return result[columns]


def pair_accuracy(frame: pd.DataFrame) -> float:
    scores = frame.pivot(index="pair_id", columns="label", values="score")
    if scores.isna().any().any() or set(scores.columns) != {0, 1}:
        raise ValueError("Pair accuracy requires one positive and one negative.")
    return float((scores[1] > scores[0]).mean())


def metric_record(base: Any, frame: pd.DataFrame) -> dict[str, float]:
    return {
        **base.evaluate(
            frame["label"].to_numpy(dtype=np.int64),
            frame["score"].to_numpy(dtype=float),
        ),
        "pair_accuracy": pair_accuracy(frame),
    }


def b3_metric_tables(
    base: Any, predictions: pd.DataFrame
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    task_rows: list[dict[str, object]] = []
    for keys, task in predictions.groupby(
        ["candidate", "seed", "target_tissue", "mhc_restriction"], sort=True
    ):
        task_rows.append(
            {
                "candidate": keys[0],
                "seed": int(keys[1]),
                "target_tissue": keys[2],
                "mhc_restriction": keys[3],
                "oof_rows": int(len(task)),
                **metric_record(base, task),
            }
        )
    per_task = pd.DataFrame(task_rows)

    summary_rows: list[dict[str, object]] = []
    for keys, tasks in per_task.groupby(["candidate", "seed"], sort=True):
        row: dict[str, object] = {
            "candidate": keys[0],
            "seed": int(keys[1]),
            "n_tasks": int(len(tasks)),
        }
        for metric in (*METRICS, "pair_accuracy"):
            row[f"mean_task_{metric}"] = float(tasks[metric].mean())
        row["worst_10_mean_auroc"] = float(
            tasks.nsmallest(min(10, len(tasks)), "auroc")["auroc"].mean()
        )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    comparison = per_task[per_task["candidate"] == CANDIDATE_CURRENT].merge(
        per_task[per_task["candidate"] == CANDIDATE_SHUFFLED],
        on=["seed", "target_tissue", "mhc_restriction"],
        suffixes=("_current", "_shuffled"),
        validate="one_to_one",
    )
    comparison_rows = comparison[
        ["seed", "target_tissue", "mhc_restriction"]
    ].copy()
    for metric in (*METRICS, "pair_accuracy"):
        comparison_rows[f"current_{metric}"] = comparison[
            f"{metric}_current"
        ]
        comparison_rows[f"shuffled_{metric}"] = comparison[
            f"{metric}_shuffled"
        ]
        comparison_rows[f"delta_shuffled_minus_current_{metric}"] = (
            comparison[f"{metric}_shuffled"] - comparison[f"{metric}_current"]
        )

    other_rows: list[dict[str, object]] = []
    seen_rows: list[dict[str, object]] = []
    for keys, group in predictions.groupby(
        ["candidate", "seed", "other_count_group"], sort=True
    ):
        other_rows.append(
            {
                "candidate": keys[0],
                "seed": int(keys[1]),
                "other_count_group": keys[2],
                "n_rows": int(len(group)),
                "n_pairs": int(group["pair_id"].nunique()),
                **metric_record(base, group),
            }
        )
    for keys, group in predictions.groupby(
        ["candidate", "seed", "peptide_seen_count_in_pair"], sort=True
    ):
        seen_rows.append(
            {
                "candidate": keys[0],
                "seed": int(keys[1]),
                "peptide_seen_count_in_pair": int(keys[2]),
                "n_rows": int(len(group)),
                "n_pairs": int(group["pair_id"].nunique()),
                **metric_record(base, group),
            }
        )

    aggregation_metrics = [
        "accuracy",
        "balanced_accuracy",
        "auroc",
        "auprc",
        "f1",
        "mcc",
        "pair_accuracy",
    ]
    per_hla = (
        per_task.groupby(["candidate", "seed", "mhc_restriction"], sort=True)[
            aggregation_metrics
        ]
        .mean()
        .reset_index()
    )
    per_tissue = (
        per_task.groupby(["candidate", "seed", "target_tissue"], sort=True)[
            aggregation_metrics
        ]
        .mean()
        .reset_index()
    )
    return (
        per_task,
        summary,
        comparison_rows,
        pd.DataFrame(other_rows),
        pd.DataFrame(seen_rows),
        per_hla,
        per_tissue,
    )


def aggregate_seed_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metric_columns = [
        column
        for column in summary
        if column.startswith("mean_task_") or column == "worst_10_mean_auroc"
    ]
    for candidate, group in summary.groupby("candidate", sort=True):
        row: dict[str, object] = {
            "candidate": candidate,
            "n_seeds": int(group["seed"].nunique()),
            "seeds": ",".join(
                str(seed) for seed in sorted(group["seed"].unique())
            ),
        }
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = (
                float(group[metric].std(ddof=1)) if len(group) > 1 else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_parts(
    parts: tuple[BPart, ...],
    output_name: str,
    cli_args: argparse.Namespace | None = None,
) -> None:
    args = cli_args or parser(
        f"Run premium B diagnostics: {', '.join(parts)}"
    ).parse_args()
    seeds = validate_cli(args)
    requested = set(parts)

    common.enable_original_modules()
    import run_tissuepmhc_auxiliary_soft_ensemble as e14
    import run_tissuepmhc_neural_baselines_v2 as base

    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = common.resolve_device(args.device, torch)
    train, mappings, peptide_length = load_premium_train(base)
    assignments = make_pair_grouped_folds(
        train, args.oof_folds, args.oof_split_seed
    )
    model_args = common.model_args(
        args, gradient_max_batches=args.gradient_max_batches
    )
    output_dir = common.RESULTS_ROOT / "experiments" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_assignment_table(train, assignments).to_csv(
        output_dir / "fold_assignments.csv", index=False
    )

    id_to_tissue = {
        int(identifier): tissue
        for tissue, identifier in mappings["tissue_to_id"].items()
    }
    b1_rows: list[pd.DataFrame] = []
    b2_rows: list[pd.DataFrame] = []
    b3_rows: list[pd.DataFrame] = []
    training_rows: list[dict[str, object]] = []
    started = time.perf_counter()

    for seed in seeds:
        for fold in range(args.oof_folds):
            fitting = train.loc[assignments != fold].copy()
            held_out = train.loc[assignments == fold].copy()
            print(
                f"\n=== B diagnostics seed={seed} fold={fold + 1}/"
                f"{args.oof_folds} fit={len(fitting)} holdout={len(held_out)} ===",
                flush=True,
            )
            correct_targets = fitting["tissue_id"].to_numpy(dtype=np.int64)
            e14.set_seed(seed, torch)
            current_model, diagnostics = train_global_model(
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
                CANDIDATE_CURRENT,
                correct_targets,
            )
            training_rows.extend(diagnostics)

            if "b1" in requested:
                b1_rows.append(
                    auxiliary_prediction_rows(
                        model_args,
                        torch,
                        DataLoader,
                        TensorDataset,
                        base,
                        current_model,
                        held_out,
                        peptide_length,
                        device,
                        seed,
                        fold,
                        id_to_tissue,
                    )
                )
            if "b2" in requested:
                b2_rows.append(
                    gradient_audit_rows(
                        model_args,
                        torch,
                        DataLoader,
                        TensorDataset,
                        base,
                        current_model,
                        held_out,
                        peptide_length,
                        device,
                        seed,
                        fold,
                    )
                )
            if "b3" in requested:
                current_global_scores = predict_main_scores(
                    model_args,
                    torch,
                    DataLoader,
                    TensorDataset,
                    base,
                    current_model,
                    held_out,
                    peptide_length,
                    device,
                )

                permutation_rng = np.random.default_rng(
                    args.tissue_shuffle_seed + fold
                )
                shuffled_targets = permutation_rng.permutation(correct_targets)
                if np.array_equal(shuffled_targets, correct_targets):
                    raise AssertionError("Tissue-label permutation changed no labels.")
                e14.set_seed(seed, torch)
                shuffled_model, shuffled_diagnostics = train_global_model(
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
                    CANDIDATE_SHUFFLED,
                    shuffled_targets,
                )
                training_rows.extend(shuffled_diagnostics)
                shuffled_global_scores = predict_main_scores(
                    model_args,
                    torch,
                    DataLoader,
                    TensorDataset,
                    base,
                    shuffled_model,
                    held_out,
                    peptide_length,
                    device,
                )

                e14.set_seed(seed, torch)
                (
                    hla_predictions,
                    _,
                    hla_diagnostics,
                ) = e14.train_and_predict_hla_branches(
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
                for row in hla_diagnostics:
                    training_rows.append(
                        {
                            "candidate": "shared_hla_plain",
                            "seed": seed,
                            "fold": fold,
                            **row,
                        }
                    )
                hla_scores = hla_prediction_frame(hla_predictions)
                b3_rows.extend(
                    [
                        fold_prediction_rows(
                            held_out,
                            fitting,
                            hla_scores,
                            current_global_scores,
                            CANDIDATE_CURRENT,
                            seed,
                            fold,
                        ),
                        fold_prediction_rows(
                            held_out,
                            fitting,
                            hla_scores,
                            shuffled_global_scores,
                            CANDIDATE_SHUFFLED,
                            seed,
                            fold,
                        ),
                    ]
                )

            del current_model
            if "b3" in requested:
                del shuffled_model
            if device == "cuda":
                torch.cuda.empty_cache()

    pd.DataFrame(training_rows).to_csv(
        output_dir / "training_diagnostics.csv", index=False
    )
    if b1_rows:
        b1_predictions = pd.concat(b1_rows, ignore_index=True)
        b1_predictions.to_csv(
            output_dir / "b1_auxiliary_predictions.csv", index=False
        )
        b1_metric_table(b1_predictions).to_csv(
            output_dir / "b1_auxiliary_metrics.csv", index=False
        )
    if b2_rows:
        b2_gradients = pd.concat(b2_rows, ignore_index=True)
        b2_gradients.to_csv(
            output_dir / "b2_gradient_cosines.csv", index=False
        )
        b2_summary(b2_gradients).to_csv(
            output_dir / "b2_gradient_summary.csv", index=False
        )
    if b3_rows:
        predictions = pd.concat(b3_rows, ignore_index=True)
        expected = len(train) * len(seeds) * 2
        if len(predictions) != expected:
            raise AssertionError(
                f"B3 expected {expected} OOF rows, found {len(predictions)}."
            )
        if predictions.duplicated(["candidate", "seed", "sample_id"]).any():
            raise AssertionError("B3 OOF predictions contain duplicate rows.")
        predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
        (
            per_task,
            summary,
            comparison,
            other_metrics,
            seen_metrics,
            per_hla,
            per_tissue,
        ) = b3_metric_tables(base, predictions)
        per_task.to_csv(output_dir / "per_task_metrics.csv", index=False)
        summary.to_csv(output_dir / "summary_metrics.csv", index=False)
        aggregate_seed_summary(summary).to_csv(
            output_dir / "seed_aggregate.csv", index=False
        )
        comparison.to_csv(
            output_dir / "b3_matched_task_comparison.csv", index=False
        )
        other_metrics.to_csv(
            output_dir / "other_count_metrics.csv", index=False
        )
        seen_metrics.to_csv(
            output_dir / "seen_unseen_metrics.csv", index=False
        )
        per_hla.to_csv(output_dir / "per_hla_metrics.csv", index=False)
        per_tissue.to_csv(output_dir / "per_tissue_metrics.csv", index=False)

    source_files = [
        Path(__file__).resolve(),
        Path(common.__file__).resolve(),
        common.ORIGINAL_SCRIPTS_DIR
        / "run_tissuepmhc_auxiliary_soft_ensemble.py",
        common.ORIGINAL_SCRIPTS_DIR
        / "run_tissuepmhc_neural_baselines_v2.py",
        common.TRAIN_PATH,
    ]
    write_json(
        output_dir / "run_settings.json",
        {
            "experiment_parts": list(parts),
            "test_data_read": False,
            "train": str(common.TRAIN_PATH),
            "train_rows": len(train),
            "train_pairs": int(train["pair_id"].nunique()),
            "n_tasks": len(mappings["tasks"]),
            "n_tissues": len(mappings["tissue_to_id"]),
            "n_hlas": len(mappings["hla_to_id"]),
            "device": device,
            "seeds": list(seeds),
            "oof_folds": args.oof_folds,
            "oof_split_seed": args.oof_split_seed,
            "tissue_shuffle_seed": args.tissue_shuffle_seed,
            "tissue_shuffle_policy": (
                "permute query tissue IDs only inside each fitting fold; "
                "use the same fold permutation for every model seed"
            ),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_max_batches": args.gradient_max_batches,
            "gradient_parameter_scope": (
                "shared peptide embedding and encoder parameters only; "
                "dropout disabled during held-out audit"
            ),
            "architecture": (
                "E14 global auxiliary branch plus shared matched per-HLA plain "
                "branch with fixed 0.5/0.5 probability fusion for B3"
            ),
            "file_sha256": {
                str(path): sha256(path)
                for path in source_files
                if path.exists()
            },
        },
    )
    print(
        f"\nB diagnostics complete: {output_dir}\n"
        f"B total elapsed: {time.perf_counter() - started:.2f}s",
        flush=True,
    )


def run_b1() -> None:
    run_parts(("b1",), "B1_auxiliary_diagnostics")


def run_b2() -> None:
    run_parts(("b2",), "B2_gradient_conflict")


def run_b3() -> None:
    run_parts(("b3",), "B3_tissue_label_shuffle")


def run_all_b() -> None:
    run_parts(("b1", "b2", "b3"), "B_all_auxiliary_diagnostics")
