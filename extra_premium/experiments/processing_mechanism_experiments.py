#!/usr/bin/env python3
"""Processing-first E0-E4 experiments for humanPMHC premium.

E0 is a matched C4-style tissue/HLA FiLM baseline. E1 adds peptide flanks and
an external processing score. E2 is the parent-expression-only control. E3
uses an explicit interaction between tissue antigen-processing machinery and
the peptide/flank representation. E4 combines processing, machinery, and
parent expression. All candidates use the same pair-aware training protocol,
the same shared HLA branch, and train-only pair-grouped OOF folds.

The fixed premium test set is never opened.
"""

from __future__ import annotations

import argparse
import copy
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
import conditional_model_experiments as cexp
import tissue_aux_diagnostics as bdiag


Candidate = Literal["e0", "e1", "e2", "e3", "e4"]
CANDIDATES: dict[Candidate, dict[str, object]] = {
    "e0": {
        "id": "E0_matched_c4_pairwise_baseline",
        "processing": False,
        "expression": False,
        "machinery": False,
        "description": "Matched C4 architecture under the E pair-aware protocol.",
    },
    "e1": {
        "id": "E1_flank_processing",
        "processing": True,
        "expression": False,
        "machinery": False,
        "description": "C4 plus real protein flanks and processing features.",
    },
    "e2": {
        "id": "E2_parent_expression_control",
        "processing": False,
        "expression": True,
        "machinery": False,
        "description": "Parent-protein query-tissue expression only.",
    },
    "e3": {
        "id": "E3_tissue_processing_interaction",
        "processing": True,
        "expression": False,
        "machinery": True,
        "description": "Tissue machinery FiLM modulation of flank processing.",
    },
    "e4": {
        "id": "E4_full_processing_expression_mechanism",
        "processing": True,
        "expression": True,
        "machinery": True,
        "description": "Processing, tissue machinery, and parent expression.",
    },
}
CANDIDATE_ORDER: tuple[Candidate, ...] = tuple(CANDIDATES)
DEFAULT_FEATURES = (
    EXTRA_PREMIUM_DIR / "external" / "mechanism" / "e_mechanism_features.csv.gz"
)
DEFAULT_AUDIT = (
    EXTRA_PREMIUM_DIR / "external" / "mechanism" / "e_mechanism_feature_audit.json"
)
DEFAULT_SHUFFLE_SEED = 20260801
PROCESSING_NUMERIC = (
    "relative_position",
    "peptide_occurrence_count",
    "peptide_exact_match",
    "peptide_unique_match",
    "flank_missing",
    "mhcflurry_processing_score",
    "processing_score_missing",
)
EXPRESSION_NUMERIC = (
    "query_expression",
    "cross_tissue_mean_expression",
    "relative_expression",
    "is_expressed",
    "expression_missing",
)
MACHINERY_MISSING = "machinery_missing_fraction"
MAPPING_COLUMNS = (
    "tissue_mapping_quality",
    "tissue_mapping_tier",
    "low_proxy",
)
BINARY_COLUMNS = {
    "peptide_exact_match",
    "peptide_unique_match",
    "flank_missing",
    "processing_score_missing",
    "is_expressed",
    "expression_missing",
}


def parser(description: str | None = None) -> argparse.ArgumentParser:
    result = cexp.parser(description or __doc__)
    result.set_defaults(seeds=[20260704])
    result.add_argument("--feature-csv", type=Path, default=DEFAULT_FEATURES)
    result.add_argument("--feature-audit-json", type=Path, default=DEFAULT_AUDIT)
    result.add_argument("--flank-length", type=int, default=15)
    result.add_argument("--feature-dim", type=int, default=32)
    result.add_argument(
        "--pairwise-loss-weight",
        type=float,
        default=0.25,
        help="Weight of softplus(-(positive-negative)); applied equally to E0-E4.",
    )
    result.add_argument("--feature-shuffle-seed", type=int, default=DEFAULT_SHUFFLE_SEED)
    result.add_argument(
        "--processing-score-policy",
        choices=("with_mhcflurry", "without_mhcflurry"),
        default="with_mhcflurry",
        help=(
            "Strict paired ablation: either use the prepared MHCFlurry processing "
            "score or replace it with zero and set its missing mask to one."
        ),
    )
    result.add_argument(
        "--skip-shuffle-controls",
        action="store_true",
        help="Skip inference-time processing/expression/machinery negative controls.",
    )
    result.add_argument(
        "--validate-inputs-only",
        action="store_true",
        help="Audit prepared features and stop before model training.",
    )
    return result


def validate_args(args: argparse.Namespace) -> tuple[int, ...]:
    seeds = cexp.validate_cli(args)
    if args.flank_length < 1 or args.feature_dim < 1:
        raise ValueError("--flank-length and --feature-dim must be positive.")
    if args.pairwise_loss_weight < 0:
        raise ValueError("--pairwise-loss-weight must be non-negative.")
    if args.feature_shuffle_seed < 0:
        raise ValueError("--feature-shuffle-seed must be non-negative.")
    return seeds


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_feature_table(
    train: pd.DataFrame,
    feature_path: Path,
    audit_path: Path,
) -> tuple[pd.DataFrame, tuple[str, ...], dict[str, object], pd.DataFrame]:
    if not feature_path.is_file():
        raise FileNotFoundError(
            f"Prepared E features not found: {feature_path}\n"
            "Run prepare_e_mechanism_features.py first."
        )
    if not audit_path.is_file():
        raise FileNotFoundError(f"Prepared feature audit not found: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("test_data_read") is not False:
        raise ValueError("Feature audit must explicitly state test_data_read=false.")
    if audit.get("tissue_mapping_reviewed") is not True:
        raise ValueError(
            "Formal E training requires every tissue mapping row to be reviewed/approved."
        )
    features = pd.read_csv(feature_path)
    required = {
        "sample_id", "n_flank_sequence", "c_flank_sequence",
        *PROCESSING_NUMERIC, *EXPRESSION_NUMERIC, MACHINERY_MISSING,
        *MAPPING_COLUMNS,
        *(f"proxy_enabled__{column}" for column in EXPRESSION_NUMERIC),
    }
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Prepared E features missing columns: {sorted(missing)}")
    machinery_gene_columns = tuple(sorted(
        column for column in features if column.startswith("machinery__")
    ))
    if not machinery_gene_columns:
        raise ValueError("Prepared E features contain no machinery__ columns.")
    machinery_columns = (*machinery_gene_columns, MACHINERY_MISSING)
    proxy_columns = tuple(
        f"proxy_enabled__{column}"
        for column in (*EXPRESSION_NUMERIC, *machinery_columns)
    )
    missing_proxy = set(proxy_columns) - set(features.columns)
    if missing_proxy:
        raise ValueError(
            f"Prepared E features missing proxy sensitivity columns: {sorted(missing_proxy)}"
        )
    if features["sample_id"].duplicated().any():
        raise ValueError("Prepared E feature sample_id values must be unique.")
    expected = set(train["sample_id"])
    actual = set(features["sample_id"])
    if expected != actual:
        raise ValueError(
            "Prepared feature sample IDs must exactly equal premium train; "
            f"missing={len(expected - actual)}, extra={len(actual - expected)}"
        )
    features = features.set_index("sample_id").reindex(train["sample_id"])
    for column in (
        *PROCESSING_NUMERIC, *EXPRESSION_NUMERIC, *machinery_columns,
        *proxy_columns, "low_proxy",
    ):
        features[column] = pd.to_numeric(features[column], errors="raise")
    allowed_tiers = {"exact_or_synonym", "aggregate_proxy", "low_proxy", "other_proxy"}
    unexpected_tiers = set(features["tissue_mapping_tier"].astype(str)) - allowed_tiers
    if unexpected_tiers:
        raise ValueError(f"Unexpected tissue mapping tiers: {sorted(unexpected_tiers)}")

    aligned = train[["sample_id", "pair_id", "label"]].merge(
        features.reset_index(), on="sample_id", validate="one_to_one"
    )
    pair_sizes = aligned.groupby("pair_id").size()
    pair_sums = aligned.groupby("pair_id")["label"].sum()
    if not pair_sizes.eq(2).all() or not pair_sums.eq(1).all():
        raise AssertionError("Premium E protocol requires one positive and one negative per pair.")
    expression_violations = 0
    for _, pair in aligned.groupby("pair_id", sort=False):
        left = pair.iloc[0][list(EXPRESSION_NUMERIC)]
        right = pair.iloc[1][list(EXPRESSION_NUMERIC)]
        expression_violations += int(
            not (left.eq(right) | (left.isna() & right.isna())).all()
        )
    if expression_violations:
        raise AssertionError(
            f"Expression varies inside {expression_violations} same-parent pairs."
        )
    audit_rows = pd.DataFrame([
        {"check": "train_rows", "value": len(train)},
        {"check": "train_pairs", "value": train["pair_id"].nunique()},
        {"check": "feature_rows", "value": len(features)},
        {"check": "machinery_feature_count", "value": len(machinery_columns)},
        {"check": "low_proxy_row_fraction", "value": float(features["low_proxy"].mean())},
        {"check": "machinery_fully_available_fraction", "value": float(
            features[MACHINERY_MISSING].eq(0).mean()
        )},
        {"check": "unique_flank_coverage", "value": float(
            features["peptide_unique_match"].mean()
        )},
        {"check": "processing_score_coverage", "value": float(
            features["mhcflurry_processing_score"].notna().mean()
        )},
        {"check": "expression_coverage", "value": float(
            features["expression_missing"].eq(0).mean()
        )},
        {"check": "expression_pair_invariance_violations", "value": 0},
    ])
    return features, machinery_columns, audit, audit_rows


def encode_flanks(
    values: pd.Series,
    flank_length: int,
    base: Any,
    *,
    n_terminal: bool,
) -> np.ndarray:
    encoded = np.full((len(values), flank_length), base.PAD_INDEX, dtype=np.int64)
    for row_index, raw in enumerate(values.fillna("").astype(str)):
        sequence = raw.strip().upper()
        if n_terminal:
            sequence = sequence[-flank_length:]
            offset = flank_length - len(sequence)
        else:
            sequence = sequence[:flank_length]
            offset = 0
        for column_index, aa in enumerate(sequence, start=offset):
            encoded[row_index, column_index] = base.AA_TO_INDEX.get(aa, base.PAD_INDEX)
    return encoded


def transformed_numeric(
    fitting: pd.DataFrame,
    target: pd.DataFrame,
    features: pd.DataFrame,
    columns: tuple[str, ...],
    target_columns: tuple[str, ...] | None = None,
) -> np.ndarray:
    if target_columns is None:
        target_columns = columns
    if len(target_columns) != len(columns):
        raise ValueError("target_columns must align one-to-one with fitting columns.")
    fit_values = features.loc[fitting["sample_id"], list(columns)].to_numpy(float)
    target_values = features.loc[
        target["sample_id"], list(target_columns)
    ].to_numpy(float)
    for index, column in enumerate(columns):
        if column.startswith("machinery__") or column in {
            "query_expression", "cross_tissue_mean_expression"
        }:
            fit_values[:, index] = np.log1p(fit_values[:, index])
            target_values[:, index] = np.log1p(target_values[:, index])
        if column in BINARY_COLUMNS:
            fit_values[:, index] = np.nan_to_num(fit_values[:, index], nan=0.0)
            target_values[:, index] = np.nan_to_num(target_values[:, index], nan=0.0)
            continue
        observed = fit_values[:, index][np.isfinite(fit_values[:, index])]
        mean = float(observed.mean()) if len(observed) else 0.0
        std = float(observed.std(ddof=0)) if len(observed) else 1.0
        if std <= 1e-12:
            std = 1.0
        fit_values[:, index] = (fit_values[:, index] - mean) / std
        target_values[:, index] = (target_values[:, index] - mean) / std
    return np.nan_to_num(target_values, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32
    )


def feature_arrays(
    fitting: pd.DataFrame,
    target: pd.DataFrame,
    features: pd.DataFrame,
    machinery_columns: tuple[str, ...],
    flank_length: int,
    base: Any,
    *,
    proxy_enabled: bool = False,
    use_mhcflurry_score: bool = True,
) -> dict[str, np.ndarray]:
    selected = features.loc[target["sample_id"]]
    processing = transformed_numeric(
        fitting, target, features, PROCESSING_NUMERIC
    )
    if not use_mhcflurry_score:
        processing[:, PROCESSING_NUMERIC.index("mhcflurry_processing_score")] = 0.0
        processing[:, PROCESSING_NUMERIC.index("processing_score_missing")] = 1.0
    return {
        "n_flank": encode_flanks(
            selected["n_flank_sequence"], flank_length, base, n_terminal=True
        ),
        "c_flank": encode_flanks(
            selected["c_flank_sequence"], flank_length, base, n_terminal=False
        ),
        "processing": processing,
        "expression": transformed_numeric(
            fitting,
            target,
            features,
            EXPRESSION_NUMERIC,
            tuple(f"proxy_enabled__{column}" for column in EXPRESSION_NUMERIC)
            if proxy_enabled else None,
        ),
        "machinery": transformed_numeric(
            fitting,
            target,
            features,
            machinery_columns,
            tuple(f"proxy_enabled__{column}" for column in machinery_columns)
            if proxy_enabled else None,
        ),
    }


def define_model(
    candidate: Candidate,
    args: Any,
    torch: Any,
    nn: Any,
    base: Any,
    peptide_length: int,
    n_tissues: int,
    n_hlas: int,
    n_processing: int,
    n_expression: int,
    n_machinery: int,
) -> Any:
    config = CANDIDATES[candidate]
    use_processing = bool(config["processing"])
    use_expression = bool(config["expression"])
    use_machinery = bool(config["machinery"])

    class MechanismModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(
                len(base.AA_TO_INDEX) + 1,
                args.embedding_dim,
                padding_idx=base.PAD_INDEX,
            )
            self.peptide_encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(peptide_length * args.embedding_dim, args.hidden_dim),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_dim, args.hidden_dim),
                nn.ReLU(),
            )
            self.tissue_classifier = nn.Linear(args.hidden_dim, n_tissues)
            self.hla_classifier = nn.Linear(args.hidden_dim, n_hlas)
            self.tissue_embedding = nn.Embedding(n_tissues, args.condition_dim)
            self.hla_embedding = nn.Embedding(n_hlas, args.condition_dim)

            if use_processing:
                self.flank_embedding = nn.Embedding(
                    len(base.AA_TO_INDEX) + 1,
                    args.embedding_dim,
                    padding_idx=base.PAD_INDEX,
                )
                self.flank_encoder = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(2 * args.flank_length * args.embedding_dim, args.feature_dim),
                    nn.ReLU(),
                    nn.Dropout(args.dropout),
                )
                self.processing_numeric = nn.Sequential(
                    nn.Linear(n_processing, args.feature_dim), nn.ReLU()
                )
                self.processing_encoder = nn.Sequential(
                    nn.Linear(2 * args.feature_dim, args.hidden_dim), nn.ReLU()
                )
                self.peptide_processing_mixer = nn.Sequential(
                    nn.Linear(2 * args.hidden_dim, args.hidden_dim), nn.ReLU()
                )
            if use_expression:
                self.expression_projector = nn.Sequential(
                    nn.Linear(n_expression, args.feature_dim),
                    nn.ReLU(),
                    nn.Linear(args.feature_dim, args.condition_dim),
                )
            if use_machinery:
                self.machinery_projector = nn.Sequential(
                    nn.Linear(n_machinery, args.feature_dim),
                    nn.ReLU(),
                    nn.Linear(args.feature_dim, args.condition_dim),
                )
                self.processing_condition = nn.Sequential(
                    nn.Linear(args.condition_dim, args.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(args.hidden_dim, 2 * args.hidden_dim),
                )

            condition_width = 2 * args.condition_dim
            if use_expression:
                condition_width += args.condition_dim
            self.condition_network = nn.Sequential(
                nn.Linear(condition_width, args.hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hidden_dim, 2 * args.hidden_dim),
            )
            self.shared_head = nn.Linear(args.hidden_dim, 1)
            self.hla_residuals = nn.ModuleList(
                [nn.Linear(args.hidden_dim, 1) for _ in range(n_hlas)]
            )
            self.tissue_residuals = nn.ModuleList(
                [nn.Linear(args.hidden_dim, 1) for _ in range(n_tissues)]
            )
            for module in list(self.hla_residuals) + list(self.tissue_residuals):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)

        def encode_peptide(self, peptide: Any) -> Any:
            return self.peptide_encoder(self.embedding(peptide))

        def auxiliary_logits(self, peptide: Any) -> tuple[Any, Any]:
            encoded = self.encode_peptide(peptide)
            return self.tissue_classifier(encoded), self.hla_classifier(encoded)

        @staticmethod
        def add_residual(features: Any, identifiers: Any, heads: Any, logits: Any) -> Any:
            result = logits
            for identifier in torch.unique(identifiers):
                mask = identifiers == identifier
                result[mask] = result[mask] + heads[int(identifier.item())](
                    features[mask]
                ).squeeze(1)
            return result

        def forward(
            self,
            peptide: Any,
            tissue: Any,
            hla: Any,
            n_flank: Any,
            c_flank: Any,
            processing: Any,
            expression: Any,
            machinery: Any,
        ) -> Any:
            peptide_features = self.encode_peptide(peptide)
            if use_processing:
                flank = torch.cat(
                    [self.flank_embedding(n_flank), self.flank_embedding(c_flank)],
                    dim=1,
                )
                processing_features = self.processing_encoder(torch.cat([
                    self.flank_encoder(flank),
                    self.processing_numeric(processing),
                ], dim=1))
                if use_machinery:
                    gamma_p, beta_p = self.processing_condition(
                        self.machinery_projector(machinery)
                    ).chunk(2, dim=1)
                    processing_features = (
                        processing_features * (1.0 + torch.tanh(gamma_p)) + beta_p
                    )
                peptide_features = self.peptide_processing_mixer(torch.cat([
                    peptide_features, processing_features
                ], dim=1))

            conditions = [self.tissue_embedding(tissue), self.hla_embedding(hla)]
            if use_expression:
                conditions.append(self.expression_projector(expression))
            gamma, beta = self.condition_network(torch.cat(conditions, dim=1)).chunk(
                2, dim=1
            )
            features = peptide_features * (1.0 + torch.tanh(gamma)) + beta
            logits = self.shared_head(features).squeeze(1)
            logits = self.add_residual(features, hla, self.hla_residuals, logits)
            logits = self.add_residual(features, tissue, self.tissue_residuals, logits)
            return logits

        def residual_parameters(self) -> list[Any]:
            return list(self.hla_residuals.parameters()) + list(
                self.tissue_residuals.parameters()
            )

    return MechanismModel()


def row_arrays(
    frame: pd.DataFrame,
    feature_values: dict[str, np.ndarray],
    peptide_length: int,
    base: Any,
) -> list[np.ndarray]:
    return [
        base.encode_peptides(frame["peptide_sequence"], peptide_length),
        frame["tissue_id"].to_numpy(np.int64),
        frame["hla_id"].to_numpy(np.int64),
        feature_values["n_flank"],
        feature_values["c_flank"],
        feature_values["processing"],
        feature_values["expression"],
        feature_values["machinery"],
        frame["label"].to_numpy(np.int64),
    ]


def pair_loader(
    args: Any,
    torch: Any,
    DataLoader: Any,
    TensorDataset: Any,
    frame: pd.DataFrame,
    arrays: list[np.ndarray],
) -> Any:
    order = np.lexsort((-frame["label"].to_numpy(), frame["pair_id"].to_numpy()))
    ordered_pairs = frame["pair_id"].to_numpy()[order].reshape(-1, 2)
    ordered_labels = frame["label"].to_numpy()[order].reshape(-1, 2)
    if not np.all(ordered_pairs[:, 0] == ordered_pairs[:, 1]):
        raise AssertionError("Pair-aware loader failed to place pair rows together.")
    if not np.all((ordered_labels[:, 0] == 1) & (ordered_labels[:, 1] == 0)):
        raise AssertionError("Pair-aware loader requires positive then negative.")
    pair_arrays = [np.asarray(value)[order].reshape((-1, 2, *value.shape[1:])) for value in arrays]
    return DataLoader(
        TensorDataset(*[torch.as_tensor(value.copy()) for value in pair_arrays]),
        batch_size=max(1, args.batch_size // 2),
        shuffle=True,
    )


def prediction_loader(
    args: Any,
    torch: Any,
    DataLoader: Any,
    TensorDataset: Any,
    arrays: list[np.ndarray],
) -> Any:
    return DataLoader(
        TensorDataset(*[torch.as_tensor(np.asarray(value).copy()) for value in arrays]),
        batch_size=args.batch_size,
        shuffle=False,
    )


def flatten_pair_batch(batch: list[Any]) -> list[Any]:
    return [value.reshape((-1, *value.shape[2:])) for value in batch]


def train_model(
    candidate: Candidate,
    args: Any,
    torch: Any,
    nn: Any,
    DataLoader: Any,
    TensorDataset: Any,
    base: Any,
    e14: Any,
    fitting: pd.DataFrame,
    feature_values: dict[str, np.ndarray],
    mappings: dict[str, Any],
    peptide_length: int,
    n_machinery: int,
    device: str,
    seed: int,
    fold: int,
) -> tuple[Any, list[dict[str, object]], dict[str, int]]:
    e14.set_seed(seed, torch)
    model = define_model(
        candidate, args, torch, nn, base, peptide_length,
        len(mappings["tissue_to_id"]), len(mappings["hla_to_id"]),
        len(PROCESSING_NUMERIC), len(EXPRESSION_NUMERIC), n_machinery,
    ).to(device)
    counts = {
        "total_parameters": int(sum(value.numel() for value in model.parameters())),
        "trainable_parameters": int(sum(
            value.numel() for value in model.parameters() if value.requires_grad
        )),
        "residual_parameters": int(sum(
            value.numel() for value in model.residual_parameters()
        )),
    }
    arrays = row_arrays(fitting, feature_values, peptide_length, base)
    e14.set_seed(seed, torch)
    loader = pair_loader(args, torch, DataLoader, TensorDataset, fitting, arrays)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rows: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        totals: list[float] = []
        mains: list[float] = []
        pairs: list[float] = []
        tissues: list[float] = []
        hlas: list[float] = []
        penalties: list[float] = []
        model.train()
        for pair_batch in loader:
            batch = [value.to(device) for value in flatten_pair_batch(list(pair_batch))]
            peptide, tissue, hla, n_flank, c_flank, processing, expression, machinery, labels = batch
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                peptide, tissue, hla, n_flank, c_flank, processing, expression, machinery
            )
            tissue_logits, hla_logits = model.auxiliary_logits(peptide)
            main_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels.float()
            )
            pair_logits = logits.reshape(-1, 2)
            pair_loss = torch.nn.functional.softplus(
                -(pair_logits[:, 0] - pair_logits[:, 1])
            ).mean()
            tissue_loss = torch.nn.functional.cross_entropy(tissue_logits, tissue)
            hla_loss = torch.nn.functional.cross_entropy(hla_logits, hla)
            penalty = cexp.residual_penalty(model, torch)
            total_loss = (
                main_loss
                + args.pairwise_loss_weight * pair_loss
                + args.tissue_loss_weight * tissue_loss
                + args.hla_loss_weight * hla_loss
                + args.residual_l2_weight * penalty
            )
            total_loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            totals.append(float(total_loss.detach().cpu()))
            mains.append(float(main_loss.detach().cpu()))
            pairs.append(float(pair_loss.detach().cpu()))
            tissues.append(float(tissue_loss.detach().cpu()))
            hlas.append(float(hla_loss.detach().cpu()))
            penalties.append(float(penalty.detach().cpu()))
        rows.append({
            "candidate": CANDIDATES[candidate]["id"],
            "seed": seed,
            "fold": fold,
            "branch": "global_mechanism",
            "epoch": epoch,
            "mean_total_loss": float(np.mean(totals)),
            "mean_main_bce_loss": float(np.mean(mains)),
            "mean_pairwise_loss": float(np.mean(pairs)),
            "mean_tissue_loss": float(np.mean(tissues)),
            "mean_hla_loss": float(np.mean(hlas)),
            "mean_residual_penalty": float(np.mean(penalties)),
        })
        print(
            f"    {candidate.upper()} epoch={epoch}/{args.epochs} "
            f"bce={np.mean(mains):.5f} pair={np.mean(pairs):.5f} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    return model, rows, counts


def predict_model(
    args: Any,
    torch: Any,
    DataLoader: Any,
    TensorDataset: Any,
    base: Any,
    model: Any,
    frame: pd.DataFrame,
    feature_values: dict[str, np.ndarray],
    peptide_length: int,
    device: str,
) -> np.ndarray:
    arrays = row_arrays(frame, feature_values, peptide_length, base)
    loader = prediction_loader(args, torch, DataLoader, TensorDataset, arrays)
    values: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            peptide, tissue, hla, n_flank, c_flank, processing, expression, machinery, _ = [
                value.to(device) for value in batch
            ]
            logits = model(
                peptide, tissue, hla, n_flank, c_flank, processing, expression, machinery
            )
            values.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(values).astype(float)


def grouped_permutation(frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    permutation = np.arange(len(frame))
    for _, index in frame.groupby(["target_tissue", "mhc_restriction"], sort=True).groups.items():
        positions = frame.index.get_indexer(index)
        permutation[positions] = positions[rng.permutation(len(positions))]
    return permutation


def pair_grouped_permutation(frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    permutation = np.arange(len(frame))
    working = frame.reset_index(drop=True)
    for _, group in working.groupby(["target_tissue", "mhc_restriction"], sort=True):
        pairs = list(group.groupby("pair_id", sort=False).groups.values())
        if len(pairs) < 2:
            continue
        shuffled = rng.permutation(len(pairs))
        for target_positions, source_index in zip(pairs, shuffled):
            source_positions = pairs[int(source_index)]
            target_positions = np.asarray(list(target_positions), dtype=int)
            source_positions = np.asarray(list(source_positions), dtype=int)
            if len(target_positions) != len(source_positions):
                raise AssertionError("Pair shuffle encountered unequal pair sizes.")
            permutation[target_positions] = source_positions
    return permutation


def tissue_swap_machinery(
    frame: pd.DataFrame,
    machinery: np.ndarray,
) -> np.ndarray:
    result = machinery.copy()
    tissues = sorted(frame["target_tissue"].unique())
    if len(tissues) < 2:
        return result
    source = {tissue: tissues[(index + 1) % len(tissues)] for index, tissue in enumerate(tissues)}
    representatives = {
        tissue: np.nanmean(machinery[frame["target_tissue"].eq(tissue).to_numpy()], axis=0)
        for tissue in tissues
    }
    for tissue in tissues:
        result[frame["target_tissue"].eq(tissue).to_numpy()] = representatives[source[tissue]]
    return result


def shuffled_variants(
    candidate: Candidate,
    frame: pd.DataFrame,
    values: dict[str, np.ndarray],
    seed: int,
) -> list[tuple[str, dict[str, np.ndarray]]]:
    config = CANDIDATES[candidate]
    output: list[tuple[str, dict[str, np.ndarray]]] = []
    if bool(config["processing"]):
        rng = np.random.default_rng(seed + 11)
        permutation = grouped_permutation(frame.reset_index(drop=True), rng)
        altered = {key: value.copy() for key, value in values.items()}
        for key in ("n_flank", "c_flank", "processing"):
            altered[key] = altered[key][permutation]
        output.append(("processing_shuffled", altered))
    if bool(config["expression"]):
        rng = np.random.default_rng(seed + 23)
        permutation = pair_grouped_permutation(frame.reset_index(drop=True), rng)
        altered = {key: value.copy() for key, value in values.items()}
        altered["expression"] = altered["expression"][permutation]
        output.append(("expression_shuffled_by_pair", altered))
    if bool(config["machinery"]):
        altered = {key: value.copy() for key, value in values.items()}
        altered["machinery"] = tissue_swap_machinery(
            frame.reset_index(drop=True), altered["machinery"]
        )
        output.append(("machinery_tissue_swapped", altered))
    return output


def prediction_rows(
    candidate_id: str,
    variant: str,
    seed: int,
    fold: int,
    fitting: pd.DataFrame,
    held_out: pd.DataFrame,
    global_scores: np.ndarray,
    hla_scores: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    result = bdiag.add_seen_metadata(held_out, fitting)
    result["global_score"] = global_scores
    result = result.merge(hla_scores, on="sample_id", how="left", validate="one_to_one")
    if result["hla_score"].isna().any():
        raise AssertionError("Missing shared HLA score.")
    result["score"] = 0.5 * result["global_score"] + 0.5 * result["hla_score"]
    aligned = features.loc[result["sample_id"]]
    result["flank_missing"] = aligned["flank_missing"].to_numpy(float)
    result["processing_score_missing"] = aligned[
        "processing_score_missing"
    ].to_numpy(float)
    result["expression_missing"] = aligned["expression_missing"].to_numpy(float)
    result["machinery_missing_fraction"] = aligned[
        MACHINERY_MISSING
    ].to_numpy(float)
    for column in MAPPING_COLUMNS:
        result[column] = aligned[column].to_numpy()
    result["model_input_policy"] = (
        "proxy_enabled_inference_sensitivity"
        if variant == "low_proxy_values_enabled"
        else "primary_low_proxy_as_missing"
    )
    result.insert(0, "prediction_variant", variant)
    result.insert(0, "fold", fold)
    result.insert(0, "seed", seed)
    result.insert(0, "candidate", candidate_id if variant == "observed" else f"{candidate_id}__{variant}")
    return result


def matched_comparison(per_task: pd.DataFrame, baseline: str) -> pd.DataFrame:
    reference = per_task[per_task["candidate"].eq(baseline)]
    parts: list[pd.DataFrame] = []
    for candidate in sorted(set(per_task["candidate"]) - {baseline}):
        current = per_task[per_task["candidate"].eq(candidate)]
        matched = reference.merge(
            current,
            on=["seed", "target_tissue", "mhc_restriction"],
            suffixes=("_e0", "_candidate"),
            validate="one_to_one",
        )
        result = matched[["seed", "target_tissue", "mhc_restriction"]].copy()
        result.insert(0, "candidate", candidate)
        for metric in cexp.METRICS:
            result[f"e0_{metric}"] = matched[f"{metric}_e0"]
            result[f"candidate_{metric}"] = matched[f"{metric}_candidate"]
            result[f"delta_candidate_minus_e0_{metric}"] = (
                matched[f"{metric}_candidate"] - matched[f"{metric}_e0"]
            )
        parts.append(result)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def feature_subset_metrics(base: Any, predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    masks = {
        "all": np.ones(len(predictions), dtype=bool),
        "unique_flank": predictions["flank_missing"].eq(0).to_numpy(),
        "flank_missing": predictions["flank_missing"].eq(1).to_numpy(),
        "expression_mapped": predictions["expression_missing"].eq(0).to_numpy(),
        "expression_missing": predictions["expression_missing"].eq(1).to_numpy(),
        "mapping_exact_or_synonym": predictions["tissue_mapping_tier"].eq(
            "exact_or_synonym"
        ).to_numpy(),
        "mapping_aggregate_proxy": predictions["tissue_mapping_tier"].eq(
            "aggregate_proxy"
        ).to_numpy(),
        "mapping_low_proxy": predictions["tissue_mapping_tier"].eq(
            "low_proxy"
        ).to_numpy(),
        "mapping_non_low_proxy": predictions["tissue_mapping_tier"].ne(
            "low_proxy"
        ).to_numpy(),
    }
    for name, mask in masks.items():
        for keys, raw_group in predictions.loc[mask].groupby(
            ["candidate", "seed"], sort=True
        ):
            pair_check = raw_group.groupby("pair_id")["label"].agg(
                pair_size="size", positive_count="sum"
            )
            complete_pairs = pair_check.index[
                pair_check["pair_size"].eq(2)
                & pair_check["positive_count"].eq(1)
            ]
            group = raw_group[raw_group["pair_id"].isin(complete_pairs)]
            if group.empty:
                continue
            rows.append({
                "candidate": keys[0], "seed": int(keys[1]), "subset": name,
                "n_rows": len(group), "n_pairs": group["pair_id"].nunique(),
                **bdiag.metric_record(base, group),
            })
    return pd.DataFrame(rows)


def run_candidates(
    candidates: tuple[Candidate, ...],
    output_name: str,
    cli_args: argparse.Namespace | None = None,
) -> None:
    args = cli_args or parser().parse_args()
    seeds = validate_args(args)
    feature_path = args.feature_csv.resolve()
    audit_path = args.feature_audit_json.resolve()
    common.enable_original_modules()
    import run_tissuepmhc_auxiliary_soft_ensemble as e14
    import run_tissuepmhc_neural_baselines_v2 as base

    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = common.resolve_device(args.device, torch)
    train, mappings, peptide_length = bdiag.load_premium_train(base)
    features, machinery_columns, feature_audit, audit_rows = load_feature_table(
        train, feature_path, audit_path
    )
    mhcflurry_coverage = float(
        features["mhcflurry_processing_score"].notna().mean()
    )
    if args.processing_score_policy == "with_mhcflurry" and mhcflurry_coverage == 0.0:
        raise ValueError(
            "with_mhcflurry was requested but the prepared feature table contains no "
            "MHCFlurry processing scores. Re-run prepare_e_mechanism_features.py "
            "without --skip-mhcflurry."
        )
    output_dir = common.RESULTS_ROOT / "experiments" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_rows.to_csv(output_dir / "mechanism_feature_audit.csv", index=False)
    if args.validate_inputs_only:
        print(audit_rows.to_string(index=False))
        print("E feature validation complete; training skipped.")
        return

    assignments = bdiag.make_pair_grouped_folds(
        train, args.oof_folds, args.oof_split_seed
    )
    bdiag.fold_assignment_table(train, assignments).to_csv(
        output_dir / "fold_assignments.csv", index=False
    )
    model_args = common.model_args(
        args,
        condition_dim=args.condition_dim,
        feature_dim=args.feature_dim,
        flank_length=args.flank_length,
        pairwise_loss_weight=args.pairwise_loss_weight,
        residual_l2_weight=args.residual_l2_weight,
    )
    predictions: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for seed in seeds:
        for fold in range(args.oof_folds):
            fitting = train.loc[assignments != fold].copy()
            held_out = train.loc[assignments == fold].copy()
            print(
                f"\n=== E seed={seed} fold={fold + 1}/{args.oof_folds} "
                f"fit={len(fitting)} holdout={len(held_out)} ===",
                flush=True,
            )
            fitting_values = feature_arrays(
                fitting,
                fitting,
                features,
                machinery_columns,
                args.flank_length,
                base,
                use_mhcflurry_score=args.processing_score_policy == "with_mhcflurry",
            )
            held_values = feature_arrays(
                fitting,
                held_out,
                features,
                machinery_columns,
                args.flank_length,
                base,
                use_mhcflurry_score=args.processing_score_policy == "with_mhcflurry",
            )
            held_proxy_values = feature_arrays(
                fitting,
                held_out,
                features,
                machinery_columns,
                args.flank_length,
                base,
                proxy_enabled=True,
                use_mhcflurry_score=args.processing_score_policy == "with_mhcflurry",
            )
            e14.set_seed(seed, torch)
            hla_predictions, _, hla_diagnostics = e14.train_and_predict_hla_branches(
                model_args, torch, nn, DataLoader, TensorDataset,
                fitting, held_out, mappings, peptide_length, device, seed, False,
            )
            hla_scores = bdiag.hla_prediction_frame(hla_predictions)
            diagnostics.extend({
                "candidate": "shared_hla_plain", "seed": seed, "fold": fold, **row
            } for row in hla_diagnostics)

            for candidate in candidates:
                model, rows, counts = train_model(
                    candidate, model_args, torch, nn, DataLoader, TensorDataset,
                    base, e14, fitting, fitting_values, mappings, peptide_length,
                    len(machinery_columns), device, seed, fold,
                )
                diagnostics.extend(rows)
                parameter_rows.append({
                    "candidate": CANDIDATES[candidate]["id"],
                    "seed": seed, "fold": fold, **counts,
                })
                scores = predict_model(
                    model_args, torch, DataLoader, TensorDataset, base, model,
                    held_out, held_values, peptide_length, device,
                )
                predictions.append(prediction_rows(
                    str(CANDIDATES[candidate]["id"]), "observed", seed, fold,
                    fitting, held_out, scores, hla_scores, features,
                ))
                if bool(CANDIDATES[candidate]["expression"]) or bool(
                    CANDIDATES[candidate]["machinery"]
                ):
                    proxy_scores = predict_model(
                        model_args,
                        torch,
                        DataLoader,
                        TensorDataset,
                        base,
                        model,
                        held_out,
                        held_proxy_values,
                        peptide_length,
                        device,
                    )
                    predictions.append(prediction_rows(
                        str(CANDIDATES[candidate]["id"]),
                        "low_proxy_values_enabled",
                        seed,
                        fold,
                        fitting,
                        held_out,
                        proxy_scores,
                        hla_scores,
                        features,
                    ))
                if not args.skip_shuffle_controls:
                    shuffle_seed = args.feature_shuffle_seed + seed + 1009 * fold
                    for variant, altered in shuffled_variants(
                        candidate, held_out, held_values, shuffle_seed
                    ):
                        variant_scores = predict_model(
                            model_args, torch, DataLoader, TensorDataset, base, model,
                            held_out, altered, peptide_length, device,
                        )
                        predictions.append(prediction_rows(
                            str(CANDIDATES[candidate]["id"]), variant, seed, fold,
                            fitting, held_out, variant_scores, hla_scores, features,
                        ))
                del model
                if device == "cuda":
                    torch.cuda.empty_cache()

    prediction_table = pd.concat(predictions, ignore_index=True)
    if prediction_table.duplicated(["candidate", "seed", "sample_id"]).any():
        raise AssertionError("E OOF predictions contain duplicate candidate/seed/sample rows.")
    observed_ids = {str(CANDIDATES[candidate]["id"]) for candidate in candidates}
    observed = prediction_table[prediction_table["candidate"].isin(observed_ids)]
    expected = len(train) * len(seeds) * len(candidates)
    if len(observed) != expected:
        raise AssertionError(f"Expected {expected} observed E rows, found {len(observed)}")
    hla_ranges = prediction_table.pivot_table(
        index=["seed", "sample_id"], columns="candidate", values="hla_score"
    )
    if float((hla_ranges.max(axis=1) - hla_ranges.min(axis=1)).max()) != 0.0:
        raise AssertionError("Shared HLA predictions differ across E candidates.")

    prediction_table.to_csv(output_dir / "oof_predictions.csv", index=False)
    per_task, summary, _, other, seen, per_hla, per_tissue = cexp.metric_tables(
        base, prediction_table
    )
    per_task.to_csv(output_dir / "per_task_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)
    cexp.seed_aggregate(summary).to_csv(output_dir / "seed_aggregate.csv", index=False)
    matched_comparison(per_task, str(CANDIDATES["e0"]["id"])).to_csv(
        output_dir / "matched_e0_comparison.csv", index=False
    )
    other.to_csv(output_dir / "other_count_metrics.csv", index=False)
    seen.to_csv(output_dir / "seen_unseen_metrics.csv", index=False)
    per_hla.to_csv(output_dir / "per_hla_metrics.csv", index=False)
    per_tissue.to_csv(output_dir / "per_tissue_metrics.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output_dir / "training_diagnostics.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(output_dir / "parameter_counts.csv", index=False)
    subset_metrics = feature_subset_metrics(base, prediction_table)
    subset_metrics.to_csv(output_dir / "feature_coverage_metrics.csv", index=False)
    subset_metrics[
        subset_metrics["subset"].str.startswith("mapping_")
    ].to_csv(output_dir / "mapping_quality_metrics.csv", index=False)

    source_paths = [
        Path(__file__).resolve(), Path(cexp.__file__).resolve(),
        Path(bdiag.__file__).resolve(), Path(common.__file__).resolve(),
        feature_path, audit_path, common.TRAIN_PATH,
    ]
    settings = {
        "schema_version": 1,
        "candidates": [CANDIDATES[value] for value in candidates],
        "test_data_read": False,
        "train": str(common.TRAIN_PATH),
        "feature_csv": str(feature_path),
        "feature_audit": feature_audit,
        "seeds": list(seeds),
        "device": device,
        "oof_folds": args.oof_folds,
        "oof_split_seed": args.oof_split_seed,
        "epochs": args.epochs,
        "batch_size_rows": args.batch_size,
        "pairwise_loss_weight": args.pairwise_loss_weight,
        "processing_score_policy": args.processing_score_policy,
        "mhcflurry_processing_score_coverage": mhcflurry_coverage,
        "flank_length": args.flank_length,
        "feature_dim": args.feature_dim,
        "condition_dim": args.condition_dim,
        "residual_policy": "HLA and tissue residuals only; no task residual",
        "global_hla_fusion": "fixed 0.5/0.5 probability mean",
        "shuffle_controls": not args.skip_shuffle_controls,
        "normalization": (
            "continuous external features standardized using fitting-fold rows only; "
            "expression and machinery abundance use log1p first; missing values become "
            "zero after scaling and retain explicit missing masks"
        ),
        "mapping_uncertainty_policy": (
            "primary training treats low_proxy query/relative expression and tissue "
            "machinery as missing; cross-tissue mean expression remains available; "
            "proxy-enabled values are evaluated only as an inference sensitivity variant"
        ),
        "mapping_quality_subsets": [
            "exact_or_synonym", "aggregate_proxy", "low_proxy", "non_low_proxy"
        ],
        "rng_policy": (
            "reset before model initialization and again before each training loader; "
            "one HLA branch shared exactly across all candidates per seed/fold"
        ),
        "file_sha256": {str(path): sha256(path) for path in source_paths if path.exists()},
    }
    (output_dir / "run_settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"\nE experiments complete: {output_dir}\n"
        f"E total elapsed: {time.perf_counter() - started:.2f}s",
        flush=True,
    )


def run_candidate(candidate: Candidate) -> None:
    run_candidates((candidate,), str(CANDIDATES[candidate]["id"]))


def run_all_e() -> None:
    run_candidates(CANDIDATE_ORDER, "E_all_processing_mechanisms")


def run_all_e_with_mhcflurry() -> None:
    args = parser("Run single-seed E0-E4 with MHCFlurry processing score.").parse_args()
    args.processing_score_policy = "with_mhcflurry"
    run_candidates(
        CANDIDATE_ORDER,
        "E_all_processing_mechanisms_with_mhcflurry",
        args,
    )


def run_all_e_without_mhcflurry() -> None:
    args = parser("Run single-seed E0-E4 without MHCFlurry processing score.").parse_args()
    args.processing_score_policy = "without_mhcflurry"
    run_candidates(
        CANDIDATE_ORDER,
        "E_all_processing_mechanisms_without_mhcflurry",
        args,
    )


def run_all_e_mhcflurry_ablation() -> None:
    base_args = parser(
        "Run paired single-seed E0-E4 experiments with and without MHCFlurry."
    ).parse_args()
    with_mhcflurry = copy.deepcopy(base_args)
    with_mhcflurry.processing_score_policy = "with_mhcflurry"
    run_candidates(
        CANDIDATE_ORDER,
        "E_all_processing_mechanisms_with_mhcflurry",
        with_mhcflurry,
    )
    without_mhcflurry = copy.deepcopy(base_args)
    without_mhcflurry.processing_score_policy = "without_mhcflurry"
    run_candidates(
        CANDIDATE_ORDER,
        "E_all_processing_mechanisms_without_mhcflurry",
        without_mhcflurry,
    )
