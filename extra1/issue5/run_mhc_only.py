from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .common import (
        DEFAULT_RESULTS,
        ROW_KEYS,
        SPECS,
        atomic_json,
        load_strict_folds,
        make_standard_folds,
        read_benchmark,
        seed_everything,
        sha256,
    )
except ImportError:
    from common import (
        DEFAULT_RESULTS,
        ROW_KEYS,
        SPECS,
        atomic_json,
        load_strict_folds,
        make_standard_folds,
        read_benchmark,
        seed_everything,
        sha256,
    )


AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {amino: index + 1 for index, amino in enumerate(AA)}


@dataclass(frozen=True)
class Config:
    epochs: int = 25
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    embedding_dim: int = 16
    hidden_dim: int = 128
    dropout: float = 0.2
    conv_channels: int = 32
    kernel_sizes: tuple[int, ...] = (2, 3, 5)
    expert_dim: int = 64
    condition_dim: int = 16
    gate_hidden_dim: int = 64
    n_experts: int = 3
    max_grad_norm: float = 1.0


def require_torch() -> tuple[Any, Any, Any, Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    return torch, nn, DataLoader, TensorDataset


def encode_peptides(values: pd.Series) -> np.ndarray:
    return np.asarray(
        [[AA_TO_INDEX[amino] for amino in peptide] for peptide in values.astype(str)],
        dtype=np.int64,
    )


def define_model(
    species: str, n_mhcs: int, config: Config, torch: Any, nn: Any
) -> Any:
    class HumanMhcOnly(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(len(AA_TO_INDEX) + 1, config.embedding_dim)
            self.position_embedding = nn.Parameter(
                torch.zeros(1, 9, config.embedding_dim)
            )
            self.convolutions = nn.ModuleList(
                [
                    nn.Conv1d(
                        config.embedding_dim,
                        config.conv_channels,
                        kernel_size=kernel,
                        padding=kernel // 2,
                    )
                    for kernel in config.kernel_sizes
                ]
            )
            encoded_dim = 9 * config.conv_channels * len(config.kernel_sizes)
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(encoded_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
            )
            self.mhc_heads = nn.ModuleList(
                [nn.Linear(config.hidden_dim, 1) for _ in range(n_mhcs)]
            )

        def forward(self, peptide: Any, mhc: Any) -> Any:
            embedded = self.embedding(peptide) + self.position_embedding
            channel_first = embedded.transpose(1, 2)
            features = [
                torch.relu(convolution(channel_first)[..., :9]).transpose(1, 2)
                for convolution in self.convolutions
            ]
            encoded = self.encoder(torch.cat(features, dim=2))
            logits = encoded.new_empty(encoded.shape[0])
            for mhc_id in torch.unique(mhc):
                mask = mhc == mhc_id
                logits[mask] = self.mhc_heads[int(mhc_id.item())](
                    encoded[mask]
                ).squeeze(-1)
            return logits

    class MouseMhcOnly(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(len(AA_TO_INDEX) + 1, config.embedding_dim)
            self.peptide_encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(9 * config.embedding_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            )
            self.mhc_embedding = nn.Embedding(n_mhcs, config.condition_dim)
            self.experts = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(config.hidden_dim, config.expert_dim),
                        nn.ReLU(),
                        nn.Dropout(config.dropout),
                        nn.Linear(config.expert_dim, config.expert_dim),
                        nn.ReLU(),
                    )
                    for _ in range(config.n_experts)
                ]
            )
            self.gate = nn.Sequential(
                nn.Linear(
                    config.hidden_dim + config.condition_dim, config.gate_hidden_dim
                ),
                nn.ReLU(),
                nn.Linear(config.gate_hidden_dim, config.n_experts),
            )
            self.mhc_heads = nn.ModuleList(
                [nn.Linear(config.expert_dim, 1) for _ in range(n_mhcs)]
            )

        def forward(self, peptide: Any, mhc: Any) -> Any:
            encoded = self.peptide_encoder(self.embedding(peptide))
            experts = torch.stack([expert(encoded) for expert in self.experts], dim=1)
            gate_input = torch.cat([encoded, self.mhc_embedding(mhc)], dim=1)
            gates = torch.softmax(self.gate(gate_input), dim=1)
            mixed = (experts * gates.unsqueeze(-1)).sum(dim=1)
            logits = mixed.new_empty(mixed.shape[0])
            for mhc_id in torch.unique(mhc):
                mask = mhc == mhc_id
                logits[mask] = self.mhc_heads[int(mhc_id.item())](
                    mixed[mask]
                ).squeeze(-1)
            return logits

    return HumanMhcOnly() if species == "human" else MouseMhcOnly()


def arrays(frame: pd.DataFrame, mhc_to_id: dict[str, int]) -> list[np.ndarray]:
    mhc = frame["mhc_restriction"].map(mhc_to_id)
    if mhc.isna().any():
        raise ValueError("MHC mapping does not cover all rows.")
    return [
        encode_peptides(frame["peptide_sequence"]),
        mhc.to_numpy(np.int64),
        frame["label"].to_numpy(np.float32),
    ]


def loader(
    frame: pd.DataFrame,
    mhc_to_id: dict[str, int],
    batch_size: int,
    shuffle: bool,
    torch: Any,
    DataLoader: Any,
    TensorDataset: Any,
) -> Any:
    tensors = [torch.as_tensor(value.copy()) for value in arrays(frame, mhc_to_id)]
    return DataLoader(
        TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle
    )


def fit_predict(
    species: str,
    fitting: pd.DataFrame,
    held: pd.DataFrame,
    mhc_to_id: dict[str, int],
    config: Config,
    seed: int,
    device: str,
    label: str,
) -> tuple[np.ndarray, int]:
    torch, nn, DataLoader, TensorDataset = require_torch()
    seed_everything(seed, torch)
    model = define_model(species, len(mhc_to_id), config, torch, nn).to(device)
    train_loader = loader(
        fitting,
        mhc_to_id,
        config.batch_size,
        True,
        torch,
        DataLoader,
        TensorDataset,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses: list[float] = []
        for peptide, mhc, target in train_loader:
            peptide, mhc, target = [
                value.to(device) for value in (peptide, mhc, target)
            ]
            optimizer.zero_grad(set_to_none=True)
            logits = model(peptide, mhc)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(
            f"{label} epoch={epoch}/{config.epochs} loss={np.mean(losses):.6f} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    held_loader = loader(
        held,
        mhc_to_id,
        config.batch_size,
        False,
        torch,
        DataLoader,
        TensorDataset,
    )
    scores: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for peptide, mhc, _ in held_loader:
            logits = model(peptide.to(device), mhc.to(device))
            scores.append(torch.sigmoid(logits).cpu().numpy())
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    return np.concatenate(scores), parameter_count


def prediction_frame(
    held: pd.DataFrame,
    scores: np.ndarray,
    species: str,
    protocol: str,
    seed: int,
    fold: int,
) -> pd.DataFrame:
    result = held[ROW_KEYS].copy()
    result.insert(0, "species", species)
    result.insert(1, "protocol", protocol)
    result.insert(2, "model", f"{species}_mhc_only")
    result.insert(3, "seed", seed)
    result.insert(4, "fold", fold)
    result["score"] = scores
    return result


def subset_pairs(frame: pd.DataFrame, max_pairs: int) -> pd.DataFrame:
    if max_pairs <= 0 or frame["pair_id"].nunique() <= max_pairs:
        return frame
    pair_ids = (
        frame[["pair_id", "task_name"]]
        .drop_duplicates()
        .sort_values(["task_name", "pair_id"], kind="stable")
        .groupby("task_name", group_keys=False)
        .head(max(1, max_pairs // frame["task_name"].nunique()))["pair_id"]
    )
    return frame[frame["pair_id"].isin(set(pair_ids))].copy()


def run_species(
    species: str,
    output_dir: Path,
    config: Config,
    device: str,
    protocols: list[str],
    max_pairs: int,
) -> None:
    spec = SPECS[species]
    train = subset_pairs(read_benchmark(spec.train, species, "train"), max_pairs)
    test = read_benchmark(spec.test, species, "test")
    mhcs = sorted(train["mhc_restriction"].unique())
    mhc_to_id = {mhc: index for index, mhc in enumerate(mhcs)}
    if not set(test["mhc_restriction"]).issubset(mhc_to_id):
        raise ValueError("Fixed test contains MHC absent from training data.")
    torch, _, _, _ = require_torch()
    resolved_device = (
        "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        if device == "auto"
        else device
    )
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    run_contract = {
        "species": species,
        "model": f"{species}_mhc_only",
        "config": asdict(config),
        "seeds": list(spec.seeds),
        "protocols": protocols,
        "standard_split_seed": 20260711,
        "strict_manifest": str(spec.strict_manifest.resolve()),
        "strict_manifest_sha256": sha256(spec.strict_manifest),
        "train": str(spec.train.resolve()),
        "train_sha256": sha256(spec.train),
        "test": str(spec.test.resolve()),
        "test_sha256": sha256(spec.test),
        "smoke_max_pairs": max_pairs,
    }
    contract_path = output_dir / "run_contract.json"
    if contract_path.exists():
        import json

        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing != run_contract:
            raise RuntimeError(
                f"Existing checkpoints have a different run contract: {contract_path}. "
                "Use a different output directory."
            )
    else:
        atomic_json(contract_path, run_contract)

    members: list[pd.DataFrame] = []
    _, nn, _, _ = require_torch()
    template_model = define_model(species, len(mhc_to_id), config, torch, nn)
    deterministic_parameter_count = int(
        sum(parameter.numel() for parameter in template_model.parameters())
    )
    parameter_counts: dict[str, int] = {
        protocol: deterministic_parameter_count for protocol in protocols
    }
    for protocol in protocols:
        if protocol == "standard_oof":
            fold_assignment = make_standard_folds(train)
            fold_values = range(3)
        elif protocol == "strict_oof":
            if max_pairs:
                raise ValueError("Strict smoke subsetting is disabled; use full frozen manifest.")
            fold_assignment = load_strict_folds(train, spec.strict_manifest)
            fold_values = range(3)
        elif protocol == "fixed_test":
            fold_assignment = None
            fold_values = [-1]
        else:
            raise ValueError(f"Unknown protocol: {protocol}")

        for seed in spec.seeds:
            for fold in fold_values:
                checkpoint = (
                    output_dir
                    / "checkpoints"
                    / f"{species}_mhc_only__{protocol}__seed_{seed}__fold_{fold}.csv.gz"
                )
                if checkpoint.exists():
                    cached = pd.read_csv(checkpoint)
                    expected_frame = (
                        test if protocol == "fixed_test" else train[fold_assignment == fold]
                    )
                    expected_ids = set(expected_frame["sample_id"].astype(str))
                    cached_ids = set(cached["sample_id"].astype(str))
                    identity_ok = (
                        set(cached["species"]) == {species}
                        and set(cached["protocol"]) == {protocol}
                        and set(cached["seed"].astype(int)) == {seed}
                        and set(cached["fold"].astype(int)) == {fold}
                    )
                    if (
                        len(cached) != len(expected_frame)
                        or cached["sample_id"].duplicated().any()
                        or cached_ids != expected_ids
                        or not identity_ok
                    ):
                        raise ValueError(f"Incomplete checkpoint: {checkpoint}")
                    members.append(cached)
                    continue
                if protocol == "fixed_test":
                    fitting, held = train, test
                else:
                    fitting = train[fold_assignment != fold]
                    held = train[fold_assignment == fold]
                scores, parameter_count = fit_predict(
                    species,
                    fitting,
                    held,
                    mhc_to_id,
                    config,
                    seed,
                    resolved_device,
                    f"{species}/{protocol}/seed={seed}/fold={fold}",
                )
                prediction = prediction_frame(
                    held, scores, species, protocol, seed, fold
                )
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                prediction.to_csv(checkpoint, index=False)
                members.append(prediction)
                parameter_counts[protocol] = parameter_count

    member = pd.concat(members, ignore_index=True)
    ensemble_keys = ["species", "protocol", "model", *ROW_KEYS]
    ensemble = (
        member.groupby(ensemble_keys, as_index=False)
        .agg(
            score=("score", "mean"),
            score_std=("score", lambda x: float(np.std(x, ddof=0))),
            n_members=("seed", "nunique"),
            fold=("fold", "first"),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    member.to_csv(output_dir / "member_predictions.csv.gz", index=False)
    ensemble.to_csv(output_dir / "ensemble_predictions.csv.gz", index=False)
    atomic_json(
        output_dir / "metadata.json",
        {
            "species": species,
            "model": f"{species}_mhc_only",
            "model_definition": (
                "s=f(peptide,MHC); no tissue input, tissue embedding, tissue auxiliary "
                "loss, or tissue-MHC task head"
            ),
            "human_encoder": "E29-compatible position-preserving multi-kernel CNN",
            "mouse_encoder": "E15-compatible embedding/MLP plus three experts; gate uses peptide+MHC only",
            "loss": "row-level BCE; tissue/task identity is not used for weighting",
            **run_contract,
            "device": resolved_device,
            "parameter_counts": parameter_counts,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", choices=["human", "mouse"], required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=["standard_oof", "strict_oof", "fixed_test"],
        default=["standard_oof", "strict_oof", "fixed_test"],
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-pairs", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    destination = args.output_dir or DEFAULT_RESULTS / f"{args.species}_mhc_only"
    run_species(
        args.species,
        destination,
        Config(epochs=args.epochs, batch_size=args.batch_size),
        args.device,
        args.protocols,
        args.max_pairs,
    )
