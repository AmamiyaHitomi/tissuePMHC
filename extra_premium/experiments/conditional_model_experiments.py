"""Premium train-only OOF experiments for tissue/HLA-conditioned models.

C0 reproduces the current peptide encoder plus task-specific head. C1 adds
late tissue/HLA concatenation. C2 uses tissue/HLA FiLM modulation. C3 adds
shared, HLA, tissue, and task residual scores. C4 removes the task residual.

All candidates retain the A0 query-tissue and HLA auxiliary objectives. Their
global predictions are fused with one shared matched per-HLA plain branch.
This module deliberately never opens the premium fixed test file.
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
import tissue_aux_diagnostics as bdiag


Candidate = Literal["c0", "c1", "c2", "c3", "c4"]
CANDIDATES: dict[Candidate, dict[str, str]] = {
    "c0": {
        "id": "C0_current_task_head",
        "description": "Current peptide encoder plus task-specific linear head.",
    },
    "c1": {
        "id": "C1_late_tissue_hla_concatenation",
        "description": "Late concatenation of peptide, query-tissue, and HLA embeddings.",
    },
    "c2": {
        "id": "C2_tissue_hla_film",
        "description": "Query-tissue/HLA FiLM modulation with a shared classifier.",
    },
    "c3": {
        "id": "C3_conditional_shared_task_residual",
        "description": "FiLM plus shared, HLA, tissue, and task residual scores.",
    },
    "c4": {
        "id": "C4_conditional_without_task_residual",
        "description": "C3 ablation retaining HLA/tissue but removing task residual.",
    },
}
CANDIDATE_ORDER: tuple[Candidate, ...] = tuple(CANDIDATES)
DEFAULT_SEEDS = (20260704, 20260705, 20260706)
METRICS = (
    "accuracy",
    "balanced_accuracy",
    "auroc",
    "auprc",
    "f1",
    "mcc",
    "pair_accuracy",
)


def parser(description: str) -> argparse.ArgumentParser:
    result = common.basic_parser(description)
    result.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Independent training seeds. C defaults to the shared three-seed protocol.",
    )
    result.add_argument(
        "--oof-folds", type=int, default=bdiag.DEFAULT_OOF_FOLDS
    )
    result.add_argument(
        "--oof-split-seed",
        type=int,
        default=bdiag.DEFAULT_OOF_SPLIT_SEED,
    )
    result.add_argument("--condition-dim", type=int, default=16)
    result.add_argument(
        "--residual-l2-weight",
        type=float,
        default=1e-4,
        help="Additional mean-square penalty for C3/C4 residual heads.",
    )
    return result


def validate_cli(args: argparse.Namespace) -> tuple[int, ...]:
    seeds = tuple(int(seed) for seed in args.seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain unique integers.")
    if any(seed < 0 for seed in seeds):
        raise ValueError("--seeds must be non-negative.")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive.")
    if args.oof_folds < 2:
        raise ValueError("--oof-folds must be at least two.")
    if args.condition_dim < 1:
        raise ValueError("--condition-dim must be positive.")
    if args.residual_l2_weight < 0:
        raise ValueError("--residual-l2-weight must be non-negative.")
    return seeds


def define_conditioned_model(
    candidate: Candidate,
    args: Any,
    torch: Any,
    nn: Any,
    base: Any,
    e14: Any,
    peptide_length: int,
    n_tasks: int,
    n_tissues: int,
    n_hlas: int,
) -> Any:
    if candidate == "c0":
        return e14.define_aux_shared_heads_model(
            args,
            torch,
            nn,
            peptide_length,
            n_tasks,
            n_tissues,
            n_hlas,
        )

    class PeptideAuxiliaryBase(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(
                len(base.AA_TO_INDEX) + 1,
                args.embedding_dim,
                padding_idx=base.PAD_INDEX,
            )
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    peptide_length * args.embedding_dim, args.hidden_dim
                ),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_dim, args.hidden_dim),
                nn.ReLU(),
            )
            self.tissue_classifier = nn.Linear(args.hidden_dim, n_tissues)
            self.hla_classifier = nn.Linear(args.hidden_dim, n_hlas)

        def encode(self, peptide_ids: Any) -> Any:
            return self.encoder(self.embedding(peptide_ids))

        def auxiliary_logits(self, peptide_ids: Any) -> tuple[Any, Any]:
            encoded = self.encode(peptide_ids)
            return (
                self.tissue_classifier(encoded),
                self.hla_classifier(encoded),
            )

        def residual_parameters(self) -> list[Any]:
            return []

    class LateConcatenationModel(PeptideAuxiliaryBase):
        def __init__(self) -> None:
            super().__init__()
            self.tissue_embedding = nn.Embedding(
                n_tissues, args.condition_dim
            )
            self.hla_embedding = nn.Embedding(n_hlas, args.condition_dim)
            self.classifier = nn.Sequential(
                nn.Linear(
                    args.hidden_dim + 2 * args.condition_dim,
                    args.hidden_dim,
                ),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_dim, 1),
            )

        def forward(
            self,
            peptide_ids: Any,
            task_ids: Any,
            tissue_ids: Any,
            hla_ids: Any,
        ) -> Any:
            del task_ids
            combined = torch.cat(
                [
                    self.encode(peptide_ids),
                    self.tissue_embedding(tissue_ids),
                    self.hla_embedding(hla_ids),
                ],
                dim=1,
            )
            return self.classifier(combined).squeeze(1)

    class FilmModel(PeptideAuxiliaryBase):
        def __init__(
            self,
            *,
            use_hla_tissue_residuals: bool,
            use_task_residual: bool,
        ) -> None:
            super().__init__()
            self.use_hla_tissue_residuals = use_hla_tissue_residuals
            self.use_task_residual = use_task_residual
            self.tissue_embedding = nn.Embedding(
                n_tissues, args.condition_dim
            )
            self.hla_embedding = nn.Embedding(n_hlas, args.condition_dim)
            self.condition_network = nn.Sequential(
                nn.Linear(2 * args.condition_dim, args.hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hidden_dim, 2 * args.hidden_dim),
            )
            self.shared_head = nn.Linear(args.hidden_dim, 1)
            if use_hla_tissue_residuals:
                self.hla_residuals = nn.ModuleList(
                    [nn.Linear(args.hidden_dim, 1) for _ in range(n_hlas)]
                )
                self.tissue_residuals = nn.ModuleList(
                    [nn.Linear(args.hidden_dim, 1) for _ in range(n_tissues)]
                )
                self._zero_residuals(self.hla_residuals)
                self._zero_residuals(self.tissue_residuals)
            if use_task_residual:
                self.task_residuals = nn.ModuleList(
                    [nn.Linear(args.hidden_dim, 1) for _ in range(n_tasks)]
                )
                self._zero_residuals(self.task_residuals)

        @staticmethod
        def _zero_residuals(modules: Any) -> None:
            for module in modules:
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)

        def conditioned_features(
            self, peptide_ids: Any, tissue_ids: Any, hla_ids: Any
        ) -> Any:
            return self.conditioned_features_with_ablation(
                peptide_ids,
                tissue_ids,
                hla_ids,
                use_tissue_condition=True,
                use_hla_condition=True,
            )

        def conditioned_features_with_ablation(
            self,
            peptide_ids: Any,
            tissue_ids: Any,
            hla_ids: Any,
            *,
            use_tissue_condition: bool,
            use_hla_condition: bool,
        ) -> Any:
            peptide = self.encode(peptide_ids)
            tissue_condition = self.tissue_embedding(tissue_ids)
            hla_condition = self.hla_embedding(hla_ids)
            if not use_tissue_condition:
                tissue_condition = torch.zeros_like(tissue_condition)
            if not use_hla_condition:
                hla_condition = torch.zeros_like(hla_condition)
            condition = torch.cat(
                [
                    tissue_condition,
                    hla_condition,
                ],
                dim=1,
            )
            gamma, beta = self.condition_network(condition).chunk(2, dim=1)
            return peptide * (1.0 + torch.tanh(gamma)) + beta

        @staticmethod
        def add_selected_residuals(
            features: Any, identifiers: Any, heads: Any, logits: Any
        ) -> Any:
            result = logits
            for identifier in torch.unique(identifiers):
                mask = identifiers == identifier
                result[mask] = (
                    result[mask]
                    + heads[int(identifier.item())](features[mask]).squeeze(1)
                )
            return result

        def forward(
            self,
            peptide_ids: Any,
            task_ids: Any,
            tissue_ids: Any,
            hla_ids: Any,
        ) -> Any:
            features = self.conditioned_features(
                peptide_ids, tissue_ids, hla_ids
            )
            logits = self.shared_head(features).squeeze(1)
            if self.use_hla_tissue_residuals:
                logits = self.add_selected_residuals(
                    features, hla_ids, self.hla_residuals, logits
                )
                logits = self.add_selected_residuals(
                    features, tissue_ids, self.tissue_residuals, logits
                )
            if self.use_task_residual:
                logits = self.add_selected_residuals(
                    features, task_ids, self.task_residuals, logits
                )
            return logits

        def ablation_logits(
            self,
            peptide_ids: Any,
            task_ids: Any,
            tissue_ids: Any,
            hla_ids: Any,
            *,
            use_tissue: bool = True,
            use_hla: bool = True,
            use_task_residual: bool = True,
        ) -> Any:
            """Inference-only component switches used by experiment D3."""
            features = self.conditioned_features_with_ablation(
                peptide_ids,
                tissue_ids,
                hla_ids,
                use_tissue_condition=use_tissue,
                use_hla_condition=use_hla,
            )
            logits = self.shared_head(features).squeeze(1)
            if self.use_hla_tissue_residuals:
                if use_hla:
                    logits = self.add_selected_residuals(
                        features, hla_ids, self.hla_residuals, logits
                    )
                if use_tissue:
                    logits = self.add_selected_residuals(
                        features, tissue_ids, self.tissue_residuals, logits
                    )
            if self.use_task_residual and use_task_residual:
                logits = self.add_selected_residuals(
                    features, task_ids, self.task_residuals, logits
                )
            return logits

        def residual_parameters(self) -> list[Any]:
            values: list[Any] = []
            if self.use_hla_tissue_residuals:
                values.extend(self.hla_residuals.parameters())
                values.extend(self.tissue_residuals.parameters())
            if self.use_task_residual:
                values.extend(self.task_residuals.parameters())
            return values

    if candidate == "c1":
        return LateConcatenationModel()
    if candidate == "c2":
        return FilmModel(
            use_hla_tissue_residuals=False, use_task_residual=False
        )
    if candidate == "c3":
        return FilmModel(
            use_hla_tissue_residuals=True, use_task_residual=True
        )
    if candidate == "c4":
        return FilmModel(
            use_hla_tissue_residuals=True, use_task_residual=False
        )
    raise ValueError(f"Unknown candidate: {candidate}")


def main_logits(
    candidate: Candidate,
    model: Any,
    peptide_ids: Any,
    task_ids: Any,
    tissue_ids: Any,
    hla_ids: Any,
) -> Any:
    if candidate == "c0":
        return model(peptide_ids, task_ids)
    return model(peptide_ids, task_ids, tissue_ids, hla_ids)


def residual_penalty(model: Any, torch: Any) -> Any:
    if not hasattr(model, "residual_parameters"):
        return next(model.parameters()).new_zeros(())
    parameters = list(model.residual_parameters())
    if not parameters:
        return next(model.parameters()).new_zeros(())
    return torch.cat([value.reshape(-1) for value in parameters]).square().mean()


def build_loader(
    args: Any,
    torch: Any,
    DataLoader: Any,
    TensorDataset: Any,
    base: Any,
    frame: pd.DataFrame,
    peptide_length: int,
    *,
    shuffle: bool,
) -> Any:
    arrays = [
        base.encode_peptides(frame["peptide_sequence"], peptide_length),
        frame["task_id"].to_numpy(dtype=np.int64),
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
    candidate: Candidate,
    args: Any,
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
) -> tuple[Any, list[dict[str, object]], dict[str, int]]:
    # Initialization and training RNG streams are reset separately. This keeps
    # batch order and dropout streams matched even when architectures consume a
    # different number of random values during construction.
    e14.set_seed(seed, torch)
    model = define_conditioned_model(
        candidate,
        args,
        torch,
        nn,
        base,
        e14,
        peptide_length,
        len(mappings["task_to_id"]),
        len(mappings["tissue_to_id"]),
        len(mappings["hla_to_id"]),
    ).to(device)
    parameter_counts = {
        "total_parameters": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "trainable_parameters": int(
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        ),
        "residual_parameters": int(
            sum(
                parameter.numel()
                for parameter in (
                    list(model.residual_parameters())
                    if hasattr(model, "residual_parameters")
                    else []
                )
            )
        ),
    }
    e14.set_seed(seed, torch)
    loader = build_loader(
        args,
        torch,
        DataLoader,
        TensorDataset,
        base,
        fitting,
        peptide_length,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    rows: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        totals: list[float] = []
        mains: list[float] = []
        tissues: list[float] = []
        hlas: list[float] = []
        penalties: list[float] = []
        tissue_accuracies: list[float] = []
        hla_accuracies: list[float] = []
        model.train()
        for batch in loader:
            peptide_ids, task_ids, tissue_ids, hla_ids, labels = [
                value.to(device) for value in batch
            ]
            optimizer.zero_grad(set_to_none=True)
            logits = main_logits(
                candidate,
                model,
                peptide_ids,
                task_ids,
                tissue_ids,
                hla_ids,
            )
            tissue_logits, hla_logits = model.auxiliary_logits(peptide_ids)
            main_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels.float()
            )
            tissue_loss = torch.nn.functional.cross_entropy(
                tissue_logits, tissue_ids
            )
            hla_loss = torch.nn.functional.cross_entropy(hla_logits, hla_ids)
            penalty = residual_penalty(model, torch)
            total_loss = (
                main_loss
                + args.tissue_loss_weight * tissue_loss
                + args.hla_loss_weight * hla_loss
                + args.residual_l2_weight * penalty
            )
            total_loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                )
            optimizer.step()
            totals.append(float(total_loss.detach().cpu()))
            mains.append(float(main_loss.detach().cpu()))
            tissues.append(float(tissue_loss.detach().cpu()))
            hlas.append(float(hla_loss.detach().cpu()))
            penalties.append(float(penalty.detach().cpu()))
            tissue_accuracies.append(
                float(
                    (tissue_logits.argmax(dim=1) == tissue_ids)
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
                "candidate": CANDIDATES[candidate]["id"],
                "seed": seed,
                "fold": fold,
                "branch": "global",
                "epoch": epoch,
                "mean_total_loss": float(np.mean(totals)),
                "mean_main_bce_loss": float(np.mean(mains)),
                "mean_tissue_loss": float(np.mean(tissues)),
                "mean_hla_loss": float(np.mean(hlas)),
                "mean_residual_penalty": float(np.mean(penalties)),
                "mean_tissue_accuracy": float(
                    np.mean(tissue_accuracies)
                ),
                "mean_hla_accuracy": float(np.mean(hla_accuracies)),
            }
        )
        print(
            f"    {candidate.upper()} epoch={epoch}/{args.epochs} "
            f"main={np.mean(mains):.5f} elapsed="
            f"{time.perf_counter() - started:.1f}s",
            flush=True,
        )
    return model, rows, parameter_counts


def predict_global(
    candidate: Candidate,
    args: Any,
    torch: Any,
    DataLoader: Any,
    TensorDataset: Any,
    base: Any,
    model: Any,
    held_out: pd.DataFrame,
    peptide_length: int,
    device: str,
) -> np.ndarray:
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
    scores: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for peptide_ids, task_ids, tissue_ids, hla_ids, _ in loader:
            logits = main_logits(
                candidate,
                model,
                peptide_ids.to(device),
                task_ids.to(device),
                tissue_ids.to(device),
                hla_ids.to(device),
            )
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores).astype(float)


def fold_prediction_rows(
    candidate: Candidate,
    seed: int,
    fold: int,
    fitting: pd.DataFrame,
    held_out: pd.DataFrame,
    global_scores: np.ndarray,
    hla_scores: pd.DataFrame,
) -> pd.DataFrame:
    result = bdiag.add_seen_metadata(held_out, fitting)
    result["global_score"] = np.asarray(global_scores, dtype=float)
    result = result.merge(
        hla_scores, on="sample_id", how="left", validate="one_to_one"
    )
    if result["hla_score"].isna().any():
        raise AssertionError("Missing shared HLA score.")
    result["score"] = 0.5 * result["global_score"] + 0.5 * result["hla_score"]
    result.insert(0, "fold", fold)
    result.insert(0, "seed", seed)
    result.insert(0, "candidate", CANDIDATES[candidate]["id"])
    return result[
        [
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
    ]


def metric_tables(
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
        ["candidate", "seed", "target_tissue", "mhc_restriction"],
        sort=True,
    ):
        task_rows.append(
            {
                "candidate": keys[0],
                "seed": int(keys[1]),
                "target_tissue": keys[2],
                "mhc_restriction": keys[3],
                "oof_rows": int(len(task)),
                **bdiag.metric_record(base, task),
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
        for metric in METRICS:
            row[f"mean_task_{metric}"] = float(tasks[metric].mean())
        row["worst_10_mean_auroc"] = float(
            tasks.nsmallest(min(10, len(tasks)), "auroc")["auroc"].mean()
        )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    baseline = per_task[
        per_task["candidate"] == CANDIDATES["c0"]["id"]
    ]
    comparison_rows: list[pd.DataFrame] = []
    for candidate in sorted(set(per_task["candidate"]) - {CANDIDATES["c0"]["id"]}):
        candidate_rows = per_task[per_task["candidate"] == candidate]
        matched = baseline.merge(
            candidate_rows,
            on=["seed", "target_tissue", "mhc_restriction"],
            suffixes=("_c0", "_candidate"),
            validate="one_to_one",
        )
        output = matched[
            ["seed", "target_tissue", "mhc_restriction"]
        ].copy()
        output.insert(0, "candidate", candidate)
        for metric in METRICS:
            output[f"c0_{metric}"] = matched[f"{metric}_c0"]
            output[f"candidate_{metric}"] = matched[
                f"{metric}_candidate"
            ]
            output[f"delta_candidate_minus_c0_{metric}"] = (
                matched[f"{metric}_candidate"] - matched[f"{metric}_c0"]
            )
        comparison_rows.append(output)
    comparison = (
        pd.concat(comparison_rows, ignore_index=True)
        if comparison_rows
        else pd.DataFrame()
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
                **bdiag.metric_record(base, group),
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
                **bdiag.metric_record(base, group),
            }
        )
    per_hla = (
        per_task.groupby(
            ["candidate", "seed", "mhc_restriction"], sort=True
        )[list(METRICS)]
        .mean()
        .reset_index()
    )
    per_tissue = (
        per_task.groupby(
            ["candidate", "seed", "target_tissue"], sort=True
        )[list(METRICS)]
        .mean()
        .reset_index()
    )
    return (
        per_task,
        summary,
        comparison,
        pd.DataFrame(other_rows),
        pd.DataFrame(seen_rows),
        per_hla,
        per_tissue,
    )


def seed_aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in summary
        if column.startswith("mean_task_") or column == "worst_10_mean_auroc"
    ]
    rows: list[dict[str, object]] = []
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


def run_candidates(
    candidates: tuple[Candidate, ...],
    output_name: str,
    cli_args: argparse.Namespace | None = None,
) -> None:
    args = cli_args or parser(
        "Run premium C conditioned-model OOF experiments."
    ).parse_args()
    seeds = validate_cli(args)
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
    output_dir = common.RESULTS_ROOT / "experiments" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    bdiag.fold_assignment_table(train, assignments).to_csv(
        output_dir / "fold_assignments.csv", index=False
    )

    prediction_parts: list[pd.DataFrame] = []
    training_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for seed in seeds:
        for fold in range(args.oof_folds):
            fitting = train.loc[assignments != fold].copy()
            held_out = train.loc[assignments == fold].copy()
            print(
                f"\n=== C experiments seed={seed} fold={fold + 1}/"
                f"{args.oof_folds} fit={len(fitting)} holdout={len(held_out)} ===",
                flush=True,
            )

            # Train one HLA branch per seed/fold and reuse it for every global
            # candidate. This makes all C comparisons exactly HLA-matched.
            e14.set_seed(seed, torch)
            hla_predictions, _, hla_diagnostics = (
                e14.train_and_predict_hla_branches(
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
            )
            hla_scores = bdiag.hla_prediction_frame(hla_predictions)
            for row in hla_diagnostics:
                training_rows.append(
                    {
                        "candidate": "shared_hla_plain",
                        "seed": seed,
                        "fold": fold,
                        **row,
                    }
                )

            for candidate in candidates:
                model, diagnostics, counts = train_global_model(
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
                parameter_rows.append(
                    {
                        "candidate": CANDIDATES[candidate]["id"],
                        "seed": seed,
                        "fold": fold,
                        **counts,
                    }
                )
                global_scores = predict_global(
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
                )
                prediction_parts.append(
                    fold_prediction_rows(
                        candidate,
                        seed,
                        fold,
                        fitting,
                        held_out,
                        global_scores,
                        hla_scores,
                    )
                )
                del model
                if device == "cuda":
                    torch.cuda.empty_cache()

    predictions = pd.concat(prediction_parts, ignore_index=True)
    expected = len(train) * len(seeds) * len(candidates)
    if len(predictions) != expected:
        raise AssertionError(
            f"Expected {expected} C OOF rows, found {len(predictions)}."
        )
    if predictions.duplicated(["candidate", "seed", "sample_id"]).any():
        raise AssertionError("C OOF predictions contain duplicate rows.")
    predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
    (
        per_task,
        summary,
        comparison,
        other_metrics,
        seen_metrics,
        per_hla,
        per_tissue,
    ) = metric_tables(base, predictions)
    per_task.to_csv(output_dir / "per_task_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)
    seed_aggregate(summary).to_csv(
        output_dir / "seed_aggregate.csv", index=False
    )
    comparison.to_csv(
        output_dir / "matched_c0_comparison.csv", index=False
    )
    other_metrics.to_csv(
        output_dir / "other_count_metrics.csv", index=False
    )
    seen_metrics.to_csv(
        output_dir / "seen_unseen_metrics.csv", index=False
    )
    per_hla.to_csv(output_dir / "per_hla_metrics.csv", index=False)
    per_tissue.to_csv(output_dir / "per_tissue_metrics.csv", index=False)
    pd.DataFrame(training_rows).to_csv(
        output_dir / "training_diagnostics.csv", index=False
    )
    pd.DataFrame(parameter_rows).to_csv(
        output_dir / "parameter_counts.csv", index=False
    )

    source_files = [
        Path(__file__).resolve(),
        Path(bdiag.__file__).resolve(),
        Path(common.__file__).resolve(),
        common.ORIGINAL_SCRIPTS_DIR
        / "run_tissuepmhc_auxiliary_soft_ensemble.py",
        common.ORIGINAL_SCRIPTS_DIR
        / "run_tissuepmhc_neural_baselines_v2.py",
        common.TRAIN_PATH,
    ]
    settings = {
        "candidates": [CANDIDATES[value]["id"] for value in candidates],
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
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "condition_dim": args.condition_dim,
        "residual_l2_weight": args.residual_l2_weight,
        "tissue_loss_weight": model_args.tissue_loss_weight,
        "hla_loss_weight": model_args.hla_loss_weight,
        "global_hla_fusion": "fixed 0.5/0.5 probability mean",
        "auxiliary_policy": (
            "A0 policy: every fitting row predicts query target_tissue and HLA "
            "from the unconditioned peptide representation"
        ),
        "rng_policy": (
            "reset to model seed before initialization and again before each "
            "global training loop; HLA branch is trained once per seed/fold"
        ),
        "file_sha256": {
            str(path): sha256(path)
            for path in source_files
            if path.exists()
        },
    }
    (output_dir / "run_settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"\nC experiments complete: {output_dir}\n"
        f"C total elapsed: {time.perf_counter() - started:.2f}s",
        flush=True,
    )


def run_candidate(candidate: Candidate) -> None:
    run_candidates((candidate,), CANDIDATES[candidate]["id"])


def run_all_c() -> None:
    run_candidates(CANDIDATE_ORDER, "C_all_conditioning_models")
