#!/usr/bin/env python3
"""Run fixed-test component ablations and a trainable MHC-only control.

The experiment uses the occurrence-matched Human and Mouse splits, the same
three final-training seeds, and the species-specific locked TissuePMHC
training budgets.  The formal component family is:

* full_rank_fusion: multi-kernel encoder + auxiliary global branch +
  allele-restricted branch + within-task percentile-rank fusion;
* no_mhc_branch: the global auxiliary branch from the same fitted full model;
* no_rank_fusion: the two fitted full branches averaged as raw probabilities;
* no_auxiliary: both branches retained, auxiliary losses removed;
* no_multikernel: the convolutional encoder replaced by a position-preserving
  flattened MLP while all other components are retained;
* mhc_only_cnn: one multi-kernel encoder with MHC-specific heads and no tissue
  or tissue--MHC task identity supplied to the model.

Every epoch, seed, and total run time is printed and appended to
``timing_results.csv``.  Completed seed/config outputs are resumable.  The
Human runner uses branch-local parameter, dropout, and DataLoader generators
so that the full and no-auxiliary conditions are paired stochastic runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_tissuepmhc_neural_baselines_v2 as base  # noqa: E402


DEFAULT_SEEDS = (20260704, 20260705, 20260706)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "occurrence_equal_ablation_mhc_only"
CONFIGS = ("full_components", "no_auxiliary", "no_multikernel", "mhc_only")
INVARIANCE_ATOL = 1e-6
PREDICTION_MODELS = (
    "full_rank_fusion",
    "no_mhc_branch",
    "no_rank_fusion",
    "no_auxiliary",
    "no_multikernel",
    "mhc_only_cnn",
)
KEYS = ["sample_id", "pair_id", "target_tissue", "mhc_restriction", "label"]
PREDICTION_KEYS = [*KEYS, "peptide_sequence"]


@dataclass(frozen=True)
class SpeciesConfig:
    species: str
    train: str
    test: str
    n_tasks: int
    epochs: int
    batch_size: int
    sampling: str
    task_batch_size: int
    embedding_dim: int
    kernel_sizes: tuple[int, ...]
    conv_channels: int
    hidden_dim: int
    dropout: float
    tissue_loss_weight: float
    mhc_loss_weight: float
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    worst_k: int


SPECIES_CONFIGS = {
    "human": SpeciesConfig(
        species="human",
        train="data/humanPMHC_occurence_equal_dataset/humanPMHC_train.csv.gz",
        test="data/humanPMHC_occurence_equal_dataset/humanPMHC_test.csv.gz",
        n_tasks=77,
        epochs=20,
        batch_size=512,
        sampling="row",
        task_batch_size=32,
        embedding_dim=16,
        kernel_sizes=(2, 3, 5),
        conv_channels=32,
        hidden_dim=128,
        dropout=0.35,
        tissue_loss_weight=0.30,
        mhc_loss_weight=0.30,
        learning_rate=2e-3,
        weight_decay=1e-4,
        max_grad_norm=1.0,
        worst_k=10,
    ),
    "mouse": SpeciesConfig(
        species="mouse",
        train="data/mousePMHC_occurence_equal_dataset/mousePMHC_train.csv.gz",
        test="data/mousePMHC_occurence_equal_dataset/mousePMHC_test.csv.gz",
        n_tasks=11,
        epochs=25,
        batch_size=512,
        sampling="task_balanced",
        task_batch_size=32,
        embedding_dim=16,
        kernel_sizes=(3, 5),
        conv_channels=32,
        hidden_dim=128,
        dropout=0.35,
        tissue_loss_weight=0.05,
        mhc_loss_weight=0.10,
        learning_rate=3e-3,
        weight_decay=1e-4,
        max_grad_norm=1.0,
        worst_k=5,
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "||".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def set_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TimingLogger:
    fields = [
        "timestamp_utc", "scope", "species", "config", "model", "seed",
        "branch", "epoch", "epochs", "elapsed_seconds", "status",
    ]

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **row: object) -> None:
        new_file = not self.path.exists()
        payload = {field: row.get(field, "") for field in self.fields}
        payload["timestamp_utc"] = utc_now()
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields)
            if new_file:
                writer.writeheader()
            writer.writerow(payload)


class GradientLogger:
    """Append one shared-encoder gradient-conflict diagnostic per epoch."""

    fields = [
        "timestamp_utc", "species", "config", "seed", "branch", "epoch", "batch_index",
        "primary_tissue_cosine", "primary_mhc_cosine", "tissue_mhc_cosine",
        "primary_gradient_norm", "tissue_gradient_norm", "mhc_gradient_norm",
    ]

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **row: object) -> None:
        new_file = not self.path.exists()
        payload = {field: row.get(field, "") for field in self.fields}
        payload["timestamp_utc"] = utc_now()
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields)
            if new_file:
                writer.writeheader()
            writer.writerow(payload)


def read_data(config: SpeciesConfig, max_tasks: int | None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], int]:
    train_path = PROJECT_ROOT / config.train
    test_path = PROJECT_ROOT / config.test
    train = base.read_dataset(train_path)
    test = base.read_dataset(test_path)
    for frame in (train, test):
        frame["target_tissue"] = frame["target_tissue"].fillna("NA")
    train, test, mappings = base.add_task_columns(train, test)
    if max_tasks:
        keep = set(mappings["tasks"][:max_tasks])
        train = train[train["task_name"].isin(keep)].copy()
        test = test[test["task_name"].isin(keep)].copy()
        train, test, mappings = base.add_task_columns(train, test)
    expected_tasks = max_tasks if max_tasks else config.n_tasks
    if len(mappings["tasks"]) != expected_tasks:
        raise ValueError(f"{config.species}: expected {expected_tasks} tasks, observed {len(mappings['tasks'])}")
    if set(train["pair_id"].astype(str)) & set(test["pair_id"].astype(str)):
        raise ValueError(f"{config.species}: train/test pair overlap detected")
    for split_name, frame in (("train", train), ("test", test)):
        labels = frame.groupby("pair_id", sort=False)["label"].agg(lambda x: set(map(int, x)))
        if not labels.map(lambda x: x == {0, 1}).all():
            raise ValueError(f"{config.species}: invalid pairs in {split_name}")
    lengths = pd.concat([train["peptide_sequence"].str.len(), test["peptide_sequence"].str.len()])
    if lengths.nunique() != 1:
        raise ValueError(f"{config.species}: mixed peptide lengths {sorted(lengths.unique())}")
    return train, test, mappings, int(lengths.iloc[0])


def define_model(
    *,
    torch: Any,
    nn: Any,
    config: SpeciesConfig,
    peptide_length: int,
    n_heads: int,
    n_tissues: int,
    n_mhcs: int,
    encoder_kind: str,
    use_auxiliary: bool,
) -> Any:
    class PeptideHeadsModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.peptide_length = peptide_length
            self.embedding = nn.Embedding(
                len(base.AA_TO_INDEX) + 1,
                config.embedding_dim,
                padding_idx=base.PAD_INDEX,
            )
            self.position_embedding = nn.Parameter(
                torch.zeros(1, peptide_length, config.embedding_dim)
            )
            if encoder_kind == "cnn":
                self.convolutions = nn.ModuleList([
                    nn.Conv1d(
                        config.embedding_dim,
                        config.conv_channels,
                        kernel_size=kernel,
                        padding=kernel // 2,
                    )
                    for kernel in config.kernel_sizes
                ])
                input_dim = peptide_length * config.conv_channels * len(config.kernel_sizes)
            elif encoder_kind == "mlp":
                self.convolutions = None
                input_dim = peptide_length * config.embedding_dim
            else:
                raise ValueError(f"Unknown encoder_kind={encoder_kind}")
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
            )
            self.heads = nn.ModuleList([nn.Linear(config.hidden_dim, 1) for _ in range(n_heads)])
            self.use_auxiliary = use_auxiliary
            if use_auxiliary:
                self.tissue_classifier = nn.Linear(config.hidden_dim, n_tissues)
                self.mhc_classifier = nn.Linear(config.hidden_dim, n_mhcs)

        def encode(self, peptide_ids: Any) -> Any:
            embedded = self.embedding(peptide_ids) + self.position_embedding
            if self.convolutions is None:
                features = embedded
            else:
                channels_first = embedded.transpose(1, 2)
                parts = []
                for convolution in self.convolutions:
                    convolved = convolution(channels_first)[..., : self.peptide_length]
                    parts.append(torch.relu(convolved).transpose(1, 2))
                features = torch.cat(parts, dim=2)
            return self.encoder(features)

        def forward(
            self, peptide_ids: Any, head_ids: Any, return_encoded: bool = False
        ) -> Any:
            encoded = self.encode(peptide_ids)
            logits = encoded.new_empty(encoded.shape[0])
            for head_id in torch.unique(head_ids):
                mask = head_ids == head_id
                logits[mask] = self.heads[int(head_id.item())](encoded[mask]).squeeze(-1)
            return (logits, encoded) if return_encoded else logits

        def auxiliary_logits(self, peptide_ids: Any) -> tuple[Any, Any]:
            if not self.use_auxiliary:
                raise RuntimeError("Auxiliary classifiers are disabled")
            encoded = self.encode(peptide_ids)
            return self.auxiliary_logits_from_encoded(encoded)

        def auxiliary_logits_from_encoded(self, encoded: Any) -> tuple[Any, Any]:
            if not self.use_auxiliary:
                raise RuntimeError("Auxiliary classifiers are disabled")
            return self.tissue_classifier(encoded), self.mhc_classifier(encoded)

    return PeptideHeadsModel()


def arrays_for(
    frame: pd.DataFrame,
    head_column: str,
    head_to_id: dict[str, int],
    peptide_length: int,
) -> dict[str, np.ndarray]:
    heads = frame[head_column].map(head_to_id)
    if heads.isna().any():
        missing = sorted(frame.loc[heads.isna(), head_column].unique())
        raise ValueError(f"Missing head mapping: {missing}")
    return {
        "peptides": base.encode_peptides(frame["peptide_sequence"], peptide_length),
        "head_ids": heads.to_numpy(dtype=np.int64),
        "tissue_ids": frame["tissue_id"].to_numpy(dtype=np.int64),
        "mhc_ids": frame["hla_id"].to_numpy(dtype=np.int64),
        "labels": frame["label"].to_numpy(dtype=np.int64),
        "sampling_groups": frame["task_id"].to_numpy(dtype=np.int64),
    }


def row_batches(
    arrays: dict[str, np.ndarray],
    torch: Any,
    DataLoader: Any,
    TensorDataset: Any,
    batch_size: int,
    shuffle: bool,
    generator: Any | None = None,
) -> Iterable[tuple[Any, ...]]:
    columns = ["peptides", "head_ids", "tissue_ids", "mhc_ids", "labels"]
    tensors = [torch.as_tensor(arrays[column].copy()) for column in columns]
    return DataLoader(
        TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle, generator=generator
    )


def balanced_batches(
    arrays: dict[str, np.ndarray],
    task_batch_size: int,
    rng: np.random.Generator,
) -> tuple[int, Iterable[tuple[np.ndarray, ...]]]:
    groups = [
        np.flatnonzero(arrays["sampling_groups"] == group)
        for group in sorted(np.unique(arrays["sampling_groups"]))
    ]
    if any(len(indices) == 0 for indices in groups):
        raise ValueError("Empty task in balanced sampler")
    steps = int(math.ceil(max(map(len, groups)) / task_batch_size))

    def iterator() -> Iterable[tuple[np.ndarray, ...]]:
        for _ in range(steps):
            selected = np.concatenate([
                rng.choice(indices, size=task_batch_size, replace=True) for indices in groups
            ])
            rng.shuffle(selected)
            yield tuple(
                arrays[column][selected]
                for column in ["peptides", "head_ids", "tissue_ids", "mhc_ids", "labels"]
            )

    return steps, iterator()


def train_model(
    *,
    torch: Any,
    nn: Any,
    DataLoader: Any,
    TensorDataset: Any,
    config: SpeciesConfig,
    frame: pd.DataFrame,
    head_column: str,
    head_to_id: dict[str, int],
    peptide_length: int,
    encoder_kind: str,
    use_auxiliary: bool,
    seed: int,
    config_name: str,
    branch: str,
    seed_namespace: tuple[str, str],
    epochs: int,
    device: str,
    timing: TimingLogger,
    force_row_sampling: bool = False,
    gradient_logger: GradientLogger | None = None,
    gradient_batches_per_epoch: int = 1,
) -> tuple[Any, int]:
    # Branch-local streams make corresponding full/no-auxiliary runs exactly
    # paired: shared layers see the same initialization, minibatch order, and
    # dropout stream.  Auxiliary heads are created after shared layers and do
    # not alter the reset training stream below.
    parameter_seed = stable_seed(seed, *seed_namespace, "parameters")
    set_seed(parameter_seed, torch)
    sampler_seed = stable_seed(seed, *seed_namespace, "sampler")
    model = define_model(
        torch=torch,
        nn=nn,
        config=config,
        peptide_length=peptide_length,
        n_heads=len(head_to_id),
        n_tissues=int(frame["tissue_id"].max()) + 1,
        n_mhcs=int(frame["hla_id"].max()) + 1,
        encoder_kind=encoder_kind,
        use_auxiliary=use_auxiliary,
    ).to(device)
    training_seed = stable_seed(seed, *seed_namespace, "training_stream")
    set_seed(training_seed, torch)
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(sampler_seed)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    arrays = arrays_for(frame, head_column, head_to_id, peptide_length)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    bce = nn.BCEWithLogitsLoss()
    cross_entropy = nn.CrossEntropyLoss()
    rng = np.random.default_rng(sampler_seed)

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        losses: list[float] = []
        if config.sampling == "task_balanced" and not force_row_sampling:
            steps, batches = balanced_batches(arrays, config.task_batch_size, rng)
        else:
            batches = row_batches(
                arrays, torch, DataLoader, TensorDataset, config.batch_size, True, loader_generator
            )
            steps = len(batches)  # type: ignore[arg-type]
        gradient_rows: list[dict[str, float]] = []
        for batch_index, batch in enumerate(batches):
            peptide_ids, head_ids, tissue_ids, mhc_ids, labels = [
                torch.as_tensor(item, device=device) for item in batch
            ]
            optimizer.zero_grad(set_to_none=True)
            if use_auxiliary:
                logits, encoded = model(peptide_ids, head_ids, return_encoded=True)
            else:
                logits = model(peptide_ids, head_ids)
                encoded = None
            primary_loss = bce(logits, labels.float())
            loss = primary_loss
            if use_auxiliary:
                tissue_logits, mhc_logits = model.auxiliary_logits_from_encoded(encoded)
                tissue_loss = cross_entropy(tissue_logits, tissue_ids)
                mhc_loss = cross_entropy(mhc_logits, mhc_ids)
                loss = loss + config.tissue_loss_weight * tissue_loss
                loss = loss + config.mhc_loss_weight * mhc_loss
                if gradient_logger is not None and batch_index < gradient_batches_per_epoch:
                    shared_parameters = [
                        parameter for name, parameter in model.named_parameters()
                        if not name.startswith(("heads.", "tissue_classifier.", "mhc_classifier."))
                    ]

                    def flat_gradient(component: Any) -> Any:
                        gradients = torch.autograd.grad(
                            component, shared_parameters, retain_graph=True, allow_unused=True
                        )
                        return torch.cat([
                            gradient.detach().reshape(-1) if gradient is not None
                            else torch.zeros_like(parameter).reshape(-1)
                            for gradient, parameter in zip(gradients, shared_parameters)
                        ])

                    primary_gradient = flat_gradient(primary_loss)
                    tissue_gradient = flat_gradient(tissue_loss)
                    mhc_gradient = flat_gradient(mhc_loss)
                    cosine = torch.nn.functional.cosine_similarity
                    gradient_rows.append({
                        "batch_index": batch_index,
                        "primary_tissue_cosine": float(cosine(primary_gradient, tissue_gradient, dim=0).cpu()),
                        "primary_mhc_cosine": float(cosine(primary_gradient, mhc_gradient, dim=0).cpu()),
                        "tissue_mhc_cosine": float(cosine(tissue_gradient, mhc_gradient, dim=0).cpu()),
                        "primary_gradient_norm": float(torch.linalg.vector_norm(primary_gradient).cpu()),
                        "tissue_gradient_norm": float(torch.linalg.vector_norm(tissue_gradient).cpu()),
                        "mhc_gradient_norm": float(torch.linalg.vector_norm(mhc_gradient).cpu()),
                    })
            loss.backward()
            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        elapsed = time.perf_counter() - started
        mean_loss = float(np.mean(losses))
        print(
            f"[EPOCH] species={config.species} config={config_name} seed={seed} "
            f"branch={branch} epoch={epoch}/{epochs} steps={steps} "
            f"loss={mean_loss:.6f} elapsed_seconds={elapsed:.3f}",
            flush=True,
        )
        timing.write(
            scope="epoch",
            species=config.species,
            config=config_name,
            seed=seed,
            branch=branch,
            epoch=epoch,
            epochs=epochs,
            elapsed_seconds=f"{elapsed:.6f}",
            status="completed",
        )
        if gradient_logger is not None:
            for gradient_row in gradient_rows:
                gradient_logger.write(
                    species=config.species,
                    config=config_name,
                    seed=seed,
                    branch=branch,
                    epoch=epoch,
                    **gradient_row,
                )
    return model, parameter_count


def predict_model(
    *,
    model: Any,
    frame: pd.DataFrame,
    head_column: str,
    head_to_id: dict[str, int],
    peptide_length: int,
    torch: Any,
    DataLoader: Any,
    TensorDataset: Any,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    arrays = arrays_for(frame, head_column, head_to_id, peptide_length)
    loader = row_batches(arrays, torch, DataLoader, TensorDataset, batch_size, False)
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for peptide_ids, head_ids, _, _, _ in loader:
            logits = model(peptide_ids.to(device), head_ids.to(device))
            probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
    output = frame[PREDICTION_KEYS].copy().reset_index(drop=True)
    output["score"] = np.concatenate(probabilities)
    return output


def save_model_checkpoint(
    *,
    torch: Any,
    path: Path,
    model: Any,
    config: SpeciesConfig,
    peptide_length: int,
    head_to_id: dict[str, int],
    n_tissues: int,
    n_mhcs: int,
    encoder_kind: str,
    use_auxiliary: bool,
    seed: int,
    branch: str,
) -> None:
    """Save an interpretation-ready checkpoint with its full model contract."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save({
        "state_dict": {
            name: parameter.detach().cpu() for name, parameter in model.state_dict().items()
        },
        "species_config": asdict(config),
        "peptide_length": peptide_length,
        "head_to_id": head_to_id,
        "n_tissues": n_tissues,
        "n_mhcs": n_mhcs,
        "encoder_kind": encoder_kind,
        "use_auxiliary": use_auxiliary,
        "seed": seed,
        "branch": branch,
    }, temporary)
    temporary.replace(path)


def merge_branches(global_prediction: pd.DataFrame, mhc_prediction: pd.DataFrame, mode: str) -> pd.DataFrame:
    merged = global_prediction.merge(
        mhc_prediction,
        on=PREDICTION_KEYS,
        how="inner",
        validate="one_to_one",
        suffixes=("_global", "_mhc"),
    )
    if len(merged) != len(global_prediction) or len(merged) != len(mhc_prediction):
        raise ValueError("Branch prediction coverage mismatch")
    parts: list[pd.DataFrame] = []
    for _, task in merged.groupby(["target_tissue", "mhc_restriction"], sort=True):
        item = task[PREDICTION_KEYS].copy()
        if mode == "rank":
            global_score = task["score_global"].rank(method="average", pct=True).to_numpy()
            mhc_score = task["score_mhc"].rank(method="average", pct=True).to_numpy()
            item["score"] = 0.5 * (global_score + mhc_score)
        elif mode == "probability":
            item["score"] = 0.5 * (
                task["score_global"].to_numpy() + task["score_mhc"].to_numpy()
            )
        else:
            raise ValueError(f"Unknown fusion mode={mode}")
        parts.append(item)
    return pd.concat(parts, ignore_index=True).sort_values(KEYS).reset_index(drop=True)


def fit_two_branch(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    mappings: dict[str, Any],
    peptide_length: int,
    config: SpeciesConfig,
    config_name: str,
    encoder_kind: str,
    use_auxiliary: bool,
    seed: int,
    epochs: int,
    device: str,
    torch_parts: tuple[Any, Any, Any, Any],
    timing: TimingLogger,
    gradient_logger: GradientLogger | None = None,
    gradient_batches_per_epoch: int = 1,
    precomputed_mhc_prediction: pd.DataFrame | None = None,
    checkpoint_dir: Path | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    torch, nn, DataLoader, TensorDataset = torch_parts
    global_model, global_parameters = train_model(
        torch=torch,
        nn=nn,
        DataLoader=DataLoader,
        TensorDataset=TensorDataset,
        config=config,
        frame=train,
        head_column="task_name",
        head_to_id=mappings["task_to_id"],
        peptide_length=peptide_length,
        encoder_kind=encoder_kind,
        use_auxiliary=use_auxiliary,
        seed=seed,
        config_name=config_name,
        branch="global",
        seed_namespace=("global_aux", "all"),
        epochs=epochs,
        device=device,
        timing=timing,
        gradient_logger=gradient_logger,
        gradient_batches_per_epoch=gradient_batches_per_epoch,
    )
    global_prediction = predict_model(
        model=global_model,
        frame=test,
        head_column="task_name",
        head_to_id=mappings["task_to_id"],
        peptide_length=peptide_length,
        torch=torch,
        DataLoader=DataLoader,
        TensorDataset=TensorDataset,
        batch_size=config.batch_size,
        device=device,
    )
    if checkpoint_dir is not None:
        save_model_checkpoint(
            torch=torch,
            path=checkpoint_dir / "global.pt",
            model=global_model,
            config=config,
            peptide_length=peptide_length,
            head_to_id=mappings["task_to_id"],
            n_tissues=len(mappings["tissue_to_id"]),
            n_mhcs=len(mappings["hla_to_id"]),
            encoder_kind=encoder_kind,
            use_auxiliary=use_auxiliary,
            seed=seed,
            branch="global",
        )
    del global_model
    if device == "cuda":
        torch.cuda.empty_cache()

    if precomputed_mhc_prediction is not None:
        mhc_prediction = precomputed_mhc_prediction.copy()
        mhc_parameters = 0
    else:
        mhc_parts: list[pd.DataFrame] = []
        mhc_parameters = 0
        restrictions = sorted(test["mhc_restriction"].unique())
        for index, restriction in enumerate(restrictions, start=1):
            fitting = train[train["mhc_restriction"] == restriction].copy()
            prediction = test[test["mhc_restriction"] == restriction].copy()
            tasks = sorted(set(fitting["task_name"]) & set(prediction["task_name"]))
            task_to_id = {task: task_id for task_id, task in enumerate(tasks)}
            branch_name = f"mhc_{index:02d}_of_{len(restrictions):02d}_{restriction}"
            model, parameters = train_model(
                torch=torch,
                nn=nn,
                DataLoader=DataLoader,
                TensorDataset=TensorDataset,
                config=config,
                frame=fitting,
                head_column="task_name",
                head_to_id=task_to_id,
                peptide_length=peptide_length,
                encoder_kind=encoder_kind,
                use_auxiliary=False,
                seed=seed,
                config_name=config_name,
                branch=branch_name,
                seed_namespace=("h2_plain" if config.species == "mouse" else "hla_plain", restriction),
                epochs=epochs,
                device=device,
                timing=timing,
            )
            mhc_parameters += parameters
            mhc_parts.append(predict_model(
                model=model,
                frame=prediction,
                head_column="task_name",
                head_to_id=task_to_id,
                peptide_length=peptide_length,
                torch=torch,
                DataLoader=DataLoader,
                TensorDataset=TensorDataset,
                batch_size=config.batch_size,
                device=device,
            ))
            if checkpoint_dir is not None:
                safe_restriction = restriction.replace("*", "_").replace(":", "_").replace("/", "_")
                save_model_checkpoint(
                    torch=torch,
                    path=checkpoint_dir / f"mhc_branch__{safe_restriction}.pt",
                    model=model,
                    config=config,
                    peptide_length=peptide_length,
                    head_to_id=task_to_id,
                    n_tissues=len(mappings["tissue_to_id"]),
                    n_mhcs=len(mappings["hla_to_id"]),
                    encoder_kind=encoder_kind,
                    use_auxiliary=False,
                    seed=seed,
                    branch=f"mhc:{restriction}",
                )
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        mhc_prediction = pd.concat(mhc_parts, ignore_index=True)
    predictions = {
        "rank": merge_branches(global_prediction, mhc_prediction, "rank"),
        "probability": merge_branches(global_prediction, mhc_prediction, "probability"),
        "global": global_prediction.sort_values(KEYS).reset_index(drop=True),
        "mhc": mhc_prediction.sort_values(KEYS).reset_index(drop=True),
    }
    return predictions, {
        "global_parameters": global_parameters,
        "mhc_branch_parameters": mhc_parameters,
        "total_parameters": global_parameters + mhc_parameters,
    }


def fit_mhc_only(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    mappings: dict[str, Any],
    peptide_length: int,
    config: SpeciesConfig,
    seed: int,
    epochs: int,
    device: str,
    torch_parts: tuple[Any, Any, Any, Any],
    timing: TimingLogger,
) -> tuple[pd.DataFrame, dict[str, int]]:
    torch, nn, DataLoader, TensorDataset = torch_parts
    mhc_to_id = mappings["hla_to_id"]
    model, parameters = train_model(
        torch=torch,
        nn=nn,
        DataLoader=DataLoader,
        TensorDataset=TensorDataset,
        config=config,
        frame=train,
        head_column="mhc_restriction",
        head_to_id=mhc_to_id,
        peptide_length=peptide_length,
        encoder_kind="cnn",
        use_auxiliary=False,
        seed=seed,
        config_name="mhc_only",
        branch="joint_mhc_heads_no_tissue",
        seed_namespace=("mhc_only", "all"),
        epochs=epochs,
        device=device,
        timing=timing,
        force_row_sampling=True,
    )
    prediction = predict_model(
        model=model,
        frame=test,
        head_column="mhc_restriction",
        head_to_id=mhc_to_id,
        peptide_length=peptide_length,
        torch=torch,
        DataLoader=DataLoader,
        TensorDataset=TensorDataset,
        batch_size=config.batch_size,
        device=device,
    )
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return prediction, {"total_parameters": parameters}


def pair_accuracy(task: pd.DataFrame) -> float:
    wide = task.pivot(index="pair_id", columns="label", values="score")
    if set(wide.columns) != {0, 1} or wide.isna().any().any():
        raise ValueError("Pair accuracy requires exactly one row per label and pair")
    return float((wide[1] > wide[0]).mean())


def evaluate_prediction(
    prediction: pd.DataFrame,
    train: pd.DataFrame,
    species: str,
    model: str,
    seed: int,
    worst_k: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_counts = train.groupby(["target_tissue", "mhc_restriction"], sort=True).size()
    rows: list[dict[str, Any]] = []
    for (tissue, mhc), task in prediction.groupby(["target_tissue", "mhc_restriction"], sort=True):
        metrics = base.evaluate(
            task["label"].to_numpy(dtype=np.int64),
            task["score"].to_numpy(dtype=np.float64),
        )
        rows.append({
            "species": species,
            "model": model,
            "seed": seed,
            "target_tissue": tissue,
            "mhc_restriction": mhc,
            "train_rows": int(train_counts.loc[(tissue, mhc)]),
            "test_rows": int(len(task)),
            **metrics,
            "pair_accuracy": pair_accuracy(task),
        })
    per_task = pd.DataFrame(rows)
    summary = {
        "species": species,
        "model": model,
        "seed": seed,
        "n_tasks": int(len(per_task)),
        "train_rows": int(len(train)),
        "test_rows": int(len(prediction)),
        "mean_task_auroc": float(per_task["auroc"].mean()),
        "mean_task_auprc": float(per_task["auprc"].mean()),
        "mean_task_accuracy": float(per_task["accuracy"].mean()),
        "mean_task_mcc": float(per_task["mcc"].mean()),
        "mean_task_pair_accuracy": float(per_task["pair_accuracy"].mean()),
        "worst_k": int(min(worst_k, len(per_task))),
        "worst_k_mean_auroc": float(per_task.nsmallest(min(worst_k, len(per_task)), "auroc")["auroc"].mean()),
    }
    return per_task, summary


def completed_seed(path: Path, expected_models: set[str]) -> bool:
    if not path.is_file():
        return False
    try:
        observed = set(pd.read_csv(path, usecols=["model"])["model"].unique())
    except Exception:
        return False
    return observed == expected_models


def run_seed_config(
    *,
    cli: argparse.Namespace,
    species_config: SpeciesConfig,
    train: pd.DataFrame,
    test: pd.DataFrame,
    mappings: dict[str, Any],
    peptide_length: int,
    config_name: str,
    seed: int,
    epochs: int,
    device: str,
    torch_parts: tuple[Any, Any, Any, Any],
    timing: TimingLogger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target = cli.output / species_config.species / "seed_runs" / config_name / f"seed_{seed}"
    prediction_path = target / "test_predictions.csv.gz"
    per_task_path = target / "per_task_metrics.csv"
    summary_path = target / "summary_metrics.csv"
    parameters_path = target / "parameter_counts.csv"
    expected = {
        "full_components": {"full_rank_fusion", "no_mhc_branch", "no_rank_fusion"},
        "no_auxiliary": {"no_auxiliary"},
        "no_multikernel": {"no_multikernel"},
        "mhc_only": {"mhc_only_cnn"},
    }[config_name]
    if (
        not cli.overwrite
        and completed_seed(prediction_path, expected)
        and per_task_path.is_file()
        and summary_path.is_file()
        and parameters_path.is_file()
    ):
        print(
            f"[SEED SKIP] species={species_config.species} config={config_name} seed={seed}",
            flush=True,
        )
        return (
            pd.read_csv(prediction_path, keep_default_na=False),
            pd.read_csv(per_task_path, keep_default_na=False),
            pd.read_csv(summary_path, keep_default_na=False),
            pd.read_csv(parameters_path, keep_default_na=False),
        )
    target.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(
        f"[SEED START] species={species_config.species} config={config_name} "
        f"seed={seed} epochs={epochs} device={device}",
        flush=True,
    )
    if config_name == "full_components":
        branch_predictions, parameter_counts = fit_two_branch(
            train=train,
            test=test,
            mappings=mappings,
            peptide_length=peptide_length,
            config=species_config,
            config_name=config_name,
            encoder_kind="cnn",
            use_auxiliary=True,
            seed=seed,
            epochs=epochs,
            device=device,
            torch_parts=torch_parts,
            timing=timing,
        )
        named = {
            "full_rank_fusion": branch_predictions["rank"],
            "no_mhc_branch": branch_predictions["global"],
            "no_rank_fusion": branch_predictions["probability"],
        }
    elif config_name in {"no_auxiliary", "no_multikernel"}:
        branch_predictions, parameter_counts = fit_two_branch(
            train=train,
            test=test,
            mappings=mappings,
            peptide_length=peptide_length,
            config=species_config,
            config_name=config_name,
            encoder_kind="mlp" if config_name == "no_multikernel" else "cnn",
            use_auxiliary=config_name != "no_auxiliary",
            seed=seed,
            epochs=epochs,
            device=device,
            torch_parts=torch_parts,
            timing=timing,
        )
        named = {config_name: branch_predictions["rank"]}
    elif config_name == "mhc_only":
        prediction, parameter_counts = fit_mhc_only(
            train=train,
            test=test,
            mappings=mappings,
            peptide_length=peptide_length,
            config=species_config,
            seed=seed,
            epochs=epochs,
            device=device,
            torch_parts=torch_parts,
            timing=timing,
        )
        named = {"mhc_only_cnn": prediction}
    else:
        raise ValueError(f"Unknown config={config_name}")

    prediction_parts: list[pd.DataFrame] = []
    per_task_parts: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for model_name, prediction in named.items():
        item = prediction.copy()
        item.insert(0, "seed", seed)
        item.insert(0, "model", model_name)
        item.insert(0, "species", species_config.species)
        prediction_parts.append(item)
        per_task, summary = evaluate_prediction(
            prediction,
            train,
            species_config.species,
            model_name,
            seed,
            species_config.worst_k,
        )
        per_task_parts.append(per_task)
        summaries.append(summary)
        print(
            f"[SEED METRIC] species={species_config.species} config={config_name} "
            f"model={model_name} seed={seed} AUROC={summary['mean_task_auroc']:.6f} "
            f"AUPRC={summary['mean_task_auprc']:.6f} PairAcc={summary['mean_task_pair_accuracy']:.6f}",
            flush=True,
        )
    predictions = pd.concat(prediction_parts, ignore_index=True)
    per_tasks = pd.concat(per_task_parts, ignore_index=True)
    summary_frame = pd.DataFrame(summaries)
    parameters = pd.DataFrame([{
        "species": species_config.species,
        "config": config_name,
        "seed": seed,
        **parameter_counts,
    }])
    predictions.to_csv(prediction_path, index=False, compression="gzip")
    per_tasks.to_csv(per_task_path, index=False)
    summary_frame.to_csv(summary_path, index=False)
    parameters.to_csv(parameters_path, index=False)
    elapsed = time.perf_counter() - started
    timing.write(
        scope="seed_config",
        species=species_config.species,
        config=config_name,
        seed=seed,
        epochs=epochs,
        elapsed_seconds=f"{elapsed:.6f}",
        status="completed",
    )
    print(
        f"[SEED TIME] species={species_config.species} config={config_name} "
        f"seed={seed} elapsed_seconds={elapsed:.3f}",
        flush=True,
    )
    return predictions, per_tasks, summary_frame, parameters


def bootstrap_mean_ci(values: np.ndarray, seed: int, replicates: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(replicates, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def aggregate_outputs(output: Path, species: str, per_tasks: pd.DataFrame, summaries: pd.DataFrame) -> None:
    target = output / species
    target.mkdir(parents=True, exist_ok=True)
    per_tasks.to_csv(target / "all_seed_per_task_metrics.csv", index=False)
    summaries.to_csv(target / "per_seed_summary.csv", index=False)
    metric_columns = [
        "mean_task_auroc", "mean_task_auprc", "mean_task_accuracy", "mean_task_mcc",
        "mean_task_pair_accuracy", "worst_k_mean_auroc",
    ]
    seed_rows: list[dict[str, Any]] = []
    for model, group in summaries.groupby("model", sort=False):
        row: dict[str, Any] = {"species": species, "model": model, "n_seeds": int(group["seed"].nunique())}
        for metric in metric_columns:
            values = pd.to_numeric(group[metric])
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        seed_rows.append(row)
    pd.DataFrame(seed_rows).to_csv(target / "seed_metric_summary.csv", index=False)

    task_mean = (
        per_tasks.groupby(["species", "model", "target_tissue", "mhc_restriction"], as_index=False)
        [["auroc", "auprc", "accuracy", "mcc", "pair_accuracy"]]
        .mean()
    )
    baseline = task_mean[task_mean["model"] == "full_rank_fusion"]
    comparison_rows: list[dict[str, Any]] = []
    delta_rows: list[pd.DataFrame] = []
    for model in sorted(set(task_mean["model"]) - {"full_rank_fusion"}):
        candidate = task_mean[task_mean["model"] == model]
        merged = baseline.merge(
            candidate,
            on=["species", "target_tissue", "mhc_restriction"],
            validate="one_to_one",
            suffixes=("_full", "_candidate"),
        )
        for metric in ["auroc", "auprc", "pair_accuracy"]:
            delta = merged[f"{metric}_candidate"].to_numpy() - merged[f"{metric}_full"].to_numpy()
            low, high = bootstrap_mean_ci(delta, stable_seed(species, model, metric, "bootstrap"))
            nonzero = delta[delta != 0]
            p_value = float(wilcoxon(nonzero, alternative="two-sided").pvalue) if len(nonzero) else 1.0
            comparison_rows.append({
                "species": species,
                "candidate": model,
                "reference": "full_rank_fusion",
                "metric": metric,
                "n_tasks": len(delta),
                "mean_delta_candidate_minus_full": float(delta.mean()),
                "median_delta_candidate_minus_full": float(np.median(delta)),
                "ci95_low": low,
                "ci95_high": high,
                "wins": int((delta > 0).sum()),
                "ties": int((delta == 0).sum()),
                "losses": int((delta < 0).sum()),
                "wilcoxon_p": p_value,
            })
        detail = merged[["species", "target_tissue", "mhc_restriction"]].copy()
        detail.insert(1, "candidate", model)
        for metric in ["auroc", "auprc", "accuracy", "mcc", "pair_accuracy"]:
            detail[f"{metric}_delta_candidate_minus_full"] = (
                merged[f"{metric}_candidate"] - merged[f"{metric}_full"]
            )
        delta_rows.append(detail)
    pd.DataFrame(comparison_rows).to_csv(target / "paired_comparisons.csv", index=False)
    pd.concat(delta_rows, ignore_index=True).to_csv(target / "task_deltas.csv", index=False)


def write_mhc_only_invariance_audit(output: Path, species: str, predictions: pd.DataFrame) -> None:
    subset = predictions[predictions["model"] == "mhc_only_cnn"].copy()
    if subset.empty:
        return
    query = (
        subset.groupby(["seed", "peptide_sequence", "mhc_restriction"], as_index=False)
        .agg(
            rows=("score", "size"),
            tissues=("target_tissue", "nunique"),
            labels=("label", "nunique"),
            score_min=("score", "min"),
            score_max=("score", "max"),
        )
    )
    query["score_range"] = query["score_max"] - query["score_min"]
    # Identical rows can land in different CUDA batches. Float32 convolution
    # may then differ by a few ULPs even though no tissue value reaches the
    # model; use an explicit numerical tolerance and retain the raw range.
    query["invariance_violation"] = query["score_range"] > INVARIANCE_ATOL
    query.to_csv(output / species / "mhc_only_query_invariance_detail.csv", index=False)
    summary = (
        query.groupby("seed", as_index=False)
        .agg(
            unique_queries=("peptide_sequence", "size"),
            multirow_queries=("rows", lambda x: int((x > 1).sum())),
            multitissue_queries=("tissues", lambda x: int((x > 1).sum())),
            conflicting_queries=("labels", lambda x: int((x > 1).sum())),
            max_score_range=("score_range", "max"),
            invariance_violations=("invariance_violation", "sum"),
        )
    )
    summary.insert(0, "species", species)
    summary.insert(2, "invariance_atol", INVARIANCE_ATOL)
    summary.to_csv(output / species / "mhc_only_invariance_audit.csv", index=False)
    if int(summary["invariance_violations"].sum()) != 0:
        raise AssertionError(f"{species}: MHC-only scores vary across tissue rows for identical peptide--MHC queries")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", nargs="+", choices=sorted(SPECIES_CONFIGS), default=["human", "mouse"])
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs-override", type=int)
    parser.add_argument(
        "--human-aux-weight",
        type=float,
        help="Override both Human tissue and MHC auxiliary-loss weights.",
    )
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    if not cli.seeds:
        raise ValueError("At least one seed is required")
    if cli.epochs_override is not None and cli.epochs_override < 1:
        raise ValueError("--epochs-override must be positive")
    if cli.human_aux_weight is not None and cli.human_aux_weight < 0:
        raise ValueError("--human-aux-weight must be non-negative")
    cli.output = cli.output.resolve()
    cli.output.mkdir(parents=True, exist_ok=True)
    torch_parts = base.require_torch()
    torch = torch_parts[0]
    if cli.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cli.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    timing = TimingLogger(cli.output / "timing_results.csv")
    run_started = time.perf_counter()
    run_started_utc = utc_now()
    contract_path = cli.output / "run_contract.json"
    previous_contract: dict[str, Any] = {}
    if contract_path.is_file() and not cli.overwrite:
        previous_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    elif contract_path.is_file() and set(cli.species) != set(SPECIES_CONFIGS):
        # A partial overwrite replaces only the requested species. Preserve
        # provenance for completed species that are intentionally untouched.
        previous_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    combined_species = sorted(set(previous_contract.get("species", [])) | set(cli.species))
    combined_configs = [
        name for name in CONFIGS
        if name in (set(previous_contract.get("configs", [])) | set(cli.configs))
    ]
    combined_seeds = sorted(set(previous_contract.get("seeds", [])) | set(cli.seeds))
    contract: dict[str, Any] = {
        "status": "running",
        "started_utc": previous_contract.get("started_utc", run_started_utc),
        "device": device,
        "cuda_device": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "species": combined_species,
        "configs": combined_configs,
        "seeds": combined_seeds,
        "epochs_override": cli.epochs_override,
        "human_aux_weight": cli.human_aux_weight,
        "max_tasks": cli.max_tasks,
        "model_input_contract": {
            "full_and_ablations": "peptide sequence plus tissue--MHC task-specific output heads",
            "mhc_only_cnn": "peptide sequence plus MHC-specific output head; no tissue or tissue--MHC task input",
        },
        "species_configs": previous_contract.get("species_configs", {}),
        "last_invocation": {
            "started_utc": run_started_utc,
            "species": cli.species,
            "configs": cli.configs,
            "seeds": cli.seeds,
            "overwrite": cli.overwrite,
        },
    }
    all_parameter_frames: list[pd.DataFrame] = []
    try:
        for species in cli.species:
            config = SPECIES_CONFIGS[species]
            if species == "human" and cli.human_aux_weight is not None:
                config = replace(
                    config,
                    tissue_loss_weight=cli.human_aux_weight,
                    mhc_loss_weight=cli.human_aux_weight,
                )
            train_path = PROJECT_ROOT / config.train
            test_path = PROJECT_ROOT / config.test
            contract["species_configs"][species] = {
                **asdict(config),
                "train_sha256": sha256(train_path),
                "test_sha256": sha256(test_path),
            }
            train, test, mappings, peptide_length = read_data(config, cli.max_tasks)
            print(
                f"[SPECIES START] species={species} train_rows={len(train)} test_rows={len(test)} "
                f"tasks={len(mappings['tasks'])} mhcs={len(mappings['hla_to_id'])} device={device}",
                flush=True,
            )
            prediction_frames: list[pd.DataFrame] = []
            per_task_frames: list[pd.DataFrame] = []
            summary_frames: list[pd.DataFrame] = []
            parameter_frames: list[pd.DataFrame] = []
            species_started = time.perf_counter()
            epochs = cli.epochs_override or config.epochs
            for config_name in cli.configs:
                for seed in cli.seeds:
                    prediction, per_task, summary, parameters = run_seed_config(
                        cli=cli,
                        species_config=config,
                        train=train,
                        test=test,
                        mappings=mappings,
                        peptide_length=peptide_length,
                        config_name=config_name,
                        seed=seed,
                        epochs=epochs,
                        device=device,
                        torch_parts=torch_parts,
                        timing=timing,
                    )
                    prediction_frames.append(prediction)
                    per_task_frames.append(per_task)
                    summary_frames.append(summary)
                    parameter_frames.append(parameters)
            species_predictions = pd.concat(prediction_frames, ignore_index=True)
            species_per_tasks = pd.concat(per_task_frames, ignore_index=True)
            species_summaries = pd.concat(summary_frames, ignore_index=True)
            species_parameters = pd.concat(parameter_frames, ignore_index=True)
            species_predictions.to_csv(
                cli.output / species / "all_seed_test_predictions.csv.gz",
                index=False,
                compression="gzip",
            )
            aggregate_outputs(cli.output, species, species_per_tasks, species_summaries)
            write_mhc_only_invariance_audit(cli.output, species, species_predictions)
            species_parameters.to_csv(cli.output / species / "parameter_counts.csv", index=False)
            all_parameter_frames.append(species_parameters)
            elapsed = time.perf_counter() - species_started
            timing.write(
                scope="species_total",
                species=species,
                epochs=epochs,
                elapsed_seconds=f"{elapsed:.6f}",
                status="completed",
            )
            print(f"[SPECIES TOTAL TIME] species={species} elapsed_seconds={elapsed:.3f}", flush=True)
        root_parameter_frames = []
        for species in sorted(SPECIES_CONFIGS):
            species_parameters_path = cli.output / species / "parameter_counts.csv"
            if species_parameters_path.is_file():
                root_parameter_frames.append(pd.read_csv(species_parameters_path))
        pd.concat(root_parameter_frames, ignore_index=True).to_csv(
            cli.output / "parameter_counts.csv", index=False
        )
        contract["status"] = "completed"
    finally:
        total_elapsed = time.perf_counter() - run_started
        contract["finished_utc"] = utc_now()
        contract["elapsed_seconds"] = float(previous_contract.get("elapsed_seconds", 0.0)) + total_elapsed
        contract["last_invocation"]["finished_utc"] = contract["finished_utc"]
        contract["last_invocation"]["elapsed_seconds"] = total_elapsed
        contract_path.write_text(
            json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        timing.write(
            scope="run_total",
            model="all_requested_experiments",
            elapsed_seconds=f"{total_elapsed:.6f}",
            status=contract["status"],
        )
        print(
            f"[TOTAL TIME] elapsed_seconds={total_elapsed:.3f} status={contract['status']} "
            f"output={cli.output}",
            flush=True,
        )


if __name__ == "__main__":
    main()
