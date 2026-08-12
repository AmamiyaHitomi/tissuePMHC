from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .common import SCRIPT_DIR, TrainConfig, format_duration, seed_everything
except ImportError:
    from common import SCRIPT_DIR, TrainConfig, format_duration, seed_everything


if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_tissuepmhc_neural_baselines_v2 as base  # noqa: E402


@dataclass
class FitResult:
    scores: np.ndarray
    parameter_count: int


def require_torch() -> tuple[Any, Any, Any, Any]:
    return base.require_torch()


def _loader(
    torch: Any,
    DataLoader: Any,
    TensorDataset: Any,
    arrays: list[np.ndarray],
    batch_size: int,
    shuffle: bool,
) -> Any:
    tensors = [torch.as_tensor(value.copy()) for value in arrays]
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


def _peptides(frame: pd.DataFrame, peptide_length: int) -> np.ndarray:
    return base.encode_peptides(frame["peptide_sequence"], peptide_length).copy()


def _parameter_count(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def train_shared_heads(
    fitting: pd.DataFrame,
    held: pd.DataFrame,
    task_to_id: dict[str, int],
    peptide_length: int,
    config: TrainConfig,
    seed: int,
    device: str,
    run_label: str = "shared_heads",
) -> FitResult:
    torch, nn, DataLoader, TensorDataset = require_torch()
    seed_everything(seed, torch)
    _, SharedTaskHeadsModel, _ = base.define_models(nn)
    model = SharedTaskHeadsModel(
        peptide_length,
        len(task_to_id),
        config.embedding_dim,
        config.hidden_dim,
        config.dropout,
    ).to(device)
    train_task = fitting["task_name"].map(task_to_id)
    held_task = held["task_name"].map(task_to_id)
    if train_task.isna().any() or held_task.isna().any():
        raise ValueError("Shared-head task mapping does not cover fitting and held-out rows.")
    loader = _loader(
        torch,
        DataLoader,
        TensorDataset,
        [
            _peptides(fitting, peptide_length),
            train_task.to_numpy(np.int64),
            fitting["label"].to_numpy(np.int64),
        ],
        config.batch_size,
        True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_function = torch.nn.BCEWithLogitsLoss()
    training_started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        epoch_started = time.perf_counter()
        losses: list[float] = []
        model.train()
        for peptide, task, label in loader:
            peptide, task, label = [
                value.to(device) for value in (peptide, task, label)
            ]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(peptide, task), label.float())
            loss.backward()
            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(
            f"    model={run_label} seed={seed} epoch={epoch}/{config.epochs} "
            f"loss={np.mean(losses):.6f} "
            f"epoch_time={format_duration(time.perf_counter() - epoch_started)} "
            f"train_time={format_duration(time.perf_counter() - training_started)}",
            flush=True,
        )
    held_loader = _loader(
        torch,
        DataLoader,
        TensorDataset,
        [
            _peptides(held, peptide_length),
            held_task.to_numpy(np.int64),
            held["label"].to_numpy(np.int64),
        ],
        config.batch_size,
        False,
    )
    _, scores = base.predict_scores(torch, model, held_loader, device, "task_heads")
    return FitResult(scores=scores, parameter_count=_parameter_count(model))


def _define_branch_model(
    torch: Any,
    nn: Any,
    peptide_length: int,
    n_tasks: int,
    n_tissues: int,
    n_mhcs: int,
    config: TrainConfig,
) -> Any:
    class BranchModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(
                len(base.AA_TO_INDEX) + 1,
                config.embedding_dim,
                padding_idx=base.PAD_INDEX,
            )
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(peptide_length * config.embedding_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
            )
            self.heads = nn.ModuleList([nn.Linear(config.hidden_dim, 1) for _ in range(n_tasks)])
            self.tissue_classifier = nn.Linear(config.hidden_dim, n_tissues)
            self.mhc_classifier = nn.Linear(config.hidden_dim, n_mhcs)

        def encode(self, peptides: Any) -> Any:
            return self.encoder(self.embedding(peptides))

        def forward(self, peptides: Any, task_ids: Any) -> Any:
            encoded = self.encode(peptides)
            logits = encoded.new_empty(encoded.shape[0])
            for task_id in torch.unique(task_ids):
                mask = task_ids == task_id
                logits[mask] = self.heads[int(task_id.item())](encoded[mask]).squeeze(-1)
            return logits

    return BranchModel()


def train_mlp_branch(
    fitting: pd.DataFrame,
    held: pd.DataFrame,
    task_to_id: dict[str, int],
    tissue_to_id: dict[str, int],
    mhc_to_id: dict[str, int],
    peptide_length: int,
    config: TrainConfig,
    seed: int,
    device: str,
    use_auxiliary: bool,
    run_label: str = "mlp_branch",
) -> FitResult:
    torch, nn, DataLoader, TensorDataset = require_torch()
    seed_everything(seed, torch)
    model = _define_branch_model(
        torch,
        nn,
        peptide_length,
        len(task_to_id),
        len(tissue_to_id),
        len(mhc_to_id),
        config,
    ).to(device)

    def arrays(frame: pd.DataFrame) -> list[np.ndarray]:
        task_ids = frame["task_name"].map(task_to_id)
        tissue_ids = frame["target_tissue"].map(tissue_to_id)
        mhc_ids = frame["mhc_restriction"].map(mhc_to_id)
        if task_ids.isna().any() or tissue_ids.isna().any() or mhc_ids.isna().any():
            raise ValueError("Branch categorical mapping failed.")
        return [
            _peptides(frame, peptide_length),
            task_ids.to_numpy(np.int64),
            tissue_ids.to_numpy(np.int64),
            mhc_ids.to_numpy(np.int64),
            frame["label"].to_numpy(np.int64),
        ]

    loader = _loader(
        torch, DataLoader, TensorDataset, arrays(fitting), config.batch_size, True
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    training_started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        epoch_started = time.perf_counter()
        losses: list[float] = []
        model.train()
        for peptide, task, tissue, mhc, label in loader:
            peptide, task, tissue, mhc, label = [
                value.to(device) for value in (peptide, task, tissue, mhc, label)
            ]
            optimizer.zero_grad(set_to_none=True)
            encoded = model.encode(peptide)
            logits = encoded.new_empty(encoded.shape[0])
            for task_id in torch.unique(task):
                mask = task == task_id
                logits[mask] = model.heads[int(task_id.item())](encoded[mask]).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, label.float())
            if use_auxiliary:
                loss = loss + config.tissue_loss_weight * torch.nn.functional.cross_entropy(
                    model.tissue_classifier(encoded), tissue
                )
                loss = loss + config.mhc_loss_weight * torch.nn.functional.cross_entropy(
                    model.mhc_classifier(encoded), mhc
                )
            loss.backward()
            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(
            f"    model={run_label} seed={seed} epoch={epoch}/{config.epochs} "
            f"loss={np.mean(losses):.6f} "
            f"epoch_time={format_duration(time.perf_counter() - epoch_started)} "
            f"train_time={format_duration(time.perf_counter() - training_started)}",
            flush=True,
        )

    held_loader = _loader(
        torch, DataLoader, TensorDataset, arrays(held), config.batch_size, False
    )
    scores: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for peptide, task, _, _, _ in held_loader:
            logits = model(peptide.to(device), task.to(device))
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return FitResult(scores=np.concatenate(scores), parameter_count=_parameter_count(model))


def percentile_rank_fusion(
    held: pd.DataFrame,
    global_scores: np.ndarray,
    mhc_scores: np.ndarray,
) -> np.ndarray:
    work = held[["sample_id", "task_name"]].copy()
    work["global"] = global_scores
    work["mhc"] = mhc_scores
    work["global_rank"] = work.groupby("task_name", sort=False)["global"].rank(
        method="average", pct=True
    )
    work["mhc_rank"] = work.groupby("task_name", sort=False)["mhc"].rank(
        method="average", pct=True
    )
    return (0.5 * (work["global_rank"] + work["mhc_rank"])).to_numpy(float)


def train_factorized_mmoe(
    fitting: pd.DataFrame,
    held: pd.DataFrame,
    maps: dict[str, Any],
    peptide_length: int,
    config: TrainConfig,
    seed: int,
    device: str,
    run_label: str = "factorized_mmoe",
) -> FitResult:
    torch, nn, DataLoader, TensorDataset = require_torch()
    seed_everything(seed, torch)

    class FactorizedMMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(
                len(base.AA_TO_INDEX) + 1,
                config.embedding_dim,
                padding_idx=base.PAD_INDEX,
            )
            self.peptide_encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(peptide_length * config.embedding_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            )
            self.tissue_embedding = nn.Embedding(
                len(maps["tissue_to_id"]), config.condition_dim
            )
            self.mhc_embedding = nn.Embedding(len(maps["mhc_to_id"]), config.condition_dim)
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
                    config.hidden_dim + 2 * config.condition_dim,
                    config.gate_hidden_dim,
                ),
                nn.ReLU(),
                nn.Linear(config.gate_hidden_dim, config.n_experts),
            )
            self.heads = nn.ModuleList(
                [nn.Linear(config.expert_dim, 1) for _ in maps["tasks"]]
            )

        def forward(
            self,
            peptide: Any,
            task: Any,
            tissue: Any,
            mhc: Any,
        ) -> tuple[Any, Any]:
            encoded = self.peptide_encoder(self.embedding(peptide))
            experts = torch.stack([expert(encoded) for expert in self.experts], dim=1)
            gate_input = torch.cat(
                [encoded, self.tissue_embedding(tissue), self.mhc_embedding(mhc)], dim=1
            )
            gates = torch.softmax(self.gate(gate_input), dim=1)
            mixed = (experts * gates.unsqueeze(-1)).sum(dim=1)
            logits = mixed.new_empty(mixed.shape[0])
            for task_id in torch.unique(task):
                mask = task == task_id
                logits[mask] = self.heads[int(task_id.item())](mixed[mask]).squeeze(-1)
            return logits, gates

    def arrays(frame: pd.DataFrame) -> list[np.ndarray]:
        return [
            _peptides(frame, peptide_length),
            frame["task_name"].map(maps["task_to_id"]).to_numpy(np.int64),
            frame["target_tissue"].map(maps["tissue_to_id"]).to_numpy(np.int64),
            frame["mhc_restriction"].map(maps["mhc_to_id"]).to_numpy(np.int64),
            frame["label"].to_numpy(np.int64),
        ]

    model = FactorizedMMoE().to(device)
    loader = _loader(
        torch, DataLoader, TensorDataset, arrays(fitting), config.batch_size, True
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    training_started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        epoch_started = time.perf_counter()
        losses: list[float] = []
        model.train()
        for peptide, task, tissue, mhc, label in loader:
            peptide, task, tissue, mhc, label = [
                value.to(device) for value in (peptide, task, tissue, mhc, label)
            ]
            optimizer.zero_grad(set_to_none=True)
            logits, gates = model(peptide, task, tissue, mhc)
            per_row = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, label.float(), reduction="none"
            )
            bce = torch.stack(
                [per_row[task == task_id].mean() for task_id in torch.unique(task)]
            ).mean()
            entropy = -(gates * torch.log(gates.clamp_min(1e-12))).sum(dim=1).mean()
            loss = bce - config.gate_entropy_weight * entropy
            loss.backward()
            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(
            f"    model={run_label} seed={seed} epoch={epoch}/{config.epochs} "
            f"loss={np.mean(losses):.6f} "
            f"epoch_time={format_duration(time.perf_counter() - epoch_started)} "
            f"train_time={format_duration(time.perf_counter() - training_started)}",
            flush=True,
        )

    held_loader = _loader(
        torch, DataLoader, TensorDataset, arrays(held), config.batch_size, False
    )
    scores: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for peptide, task, tissue, mhc, _ in held_loader:
            logits, _ = model(
                peptide.to(device), task.to(device), tissue.to(device), mhc.to(device)
            )
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return FitResult(scores=np.concatenate(scores), parameter_count=_parameter_count(model))
