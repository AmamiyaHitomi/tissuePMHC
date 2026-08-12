#!/usr/bin/env python3
"""Biological SHAP analysis for the tuned occurrence-equal E29 model.

This script retrains the frozen three-seed E29 architecture on the Human
occurrence-equal dataset, verifies its predictions against the published run,
and applies task-conditional Expected-IG SHAP to both differentiable model branches.

The final E29 score is a within-task percentile-rank average, which is not a
differentiable function.  Therefore SHAP values are reported separately for
the global-auxiliary and HLA-specific branches.  Their arithmetic mean is
exported only as a descriptive consensus, not as exact SHAP for rank fusion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def manuscript_font_family() -> str:
    """Use the TeX Gyre Termes X face underlying the manuscript's newtxtext."""
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        bundle_data = (
            Path(local_appdata)
            / "TectonicProject"
            / "Tectonic"
            / "bundles"
            / "data"
        )
        for regular in bundle_data.glob("*/TeXGyreTermesX-Regular.otf"):
            for font_path in regular.parent.glob("TeXGyreTermesX-*.otf"):
                font_manager.fontManager.addfont(font_path)
            return font_manager.FontProperties(fname=regular).get_name()
    return "Times New Roman"


# Match the manuscript body's newtxtext typography in all exported figures.
plt.rcParams.update(
    {
        "font.family": manuscript_font_family(),
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_tissuepmhc_e29_multikernel_cnn_oof as e29  # noqa: E402
import run_tissuepmhc_neural_baselines_v2 as base  # noqa: E402
import run_occurrence_equal_ablation_mhc_only as formal  # noqa: E402


AMINO_ACIDS = base.AMINO_ACIDS
VOCAB_SIZE = len(base.AA_TO_INDEX) + 1
FROZEN_PARAMETERS = {
    "learning_rate": 0.002,
    "weight_decay": 0.0001,
    "embedding_dim": 16,
    "kernel_sizes": [2, 3, 5],
    "conv_channels": 32,
    "hidden_dim": 128,
    "dropout": 0.35,
    "tissue_loss_weight": 0.3,
    "hla_loss_weight": 0.3,
    "max_grad_norm": 1.0,
    "epochs": 20,
    "batch_size": 512,
}
DEFAULT_SEEDS = [20260721, 20260722, 20260723, 20260724, 20260725]


def stable_seed(text: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}|{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def read_split(path: Path) -> pd.DataFrame:
    frame = base.read_dataset(path)
    frame["target_tissue"] = frame["target_tissue"].fillna("NA")
    return frame


def validate_and_prepare(
    train_path: Path, test_path: Path, max_tasks: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], int]:
    train, test = read_split(train_path), read_split(test_path)
    for name, frame in (("train", train), ("test", test)):
        if frame["sample_id"].duplicated().any():
            raise ValueError(f"{name}: duplicate sample_id")
        labels = frame.groupby("pair_id", sort=False)["label"].agg(list)
        invalid = ~labels.map(lambda values: sorted(map(int, values)) == [0, 1])
        if invalid.any():
            raise ValueError(f"{name}: {int(invalid.sum())} invalid positive/negative pairs")
    overlap = set(train["pair_id"].astype(str)) & set(test["pair_id"].astype(str))
    if overlap:
        raise ValueError(f"Train/test pair leakage: {len(overlap)} pairs")
    train, test, mappings = base.add_task_columns(train, test)
    if max_tasks:
        keep = set(mappings["tasks"][:max_tasks])
        train = train[train["task_name"].isin(keep)].copy()
        test = test[test["task_name"].isin(keep)].copy()
        train, test, mappings = base.add_task_columns(train, test)
    lengths = pd.concat([train["peptide_sequence"].str.len(), test["peptide_sequence"].str.len()])
    if lengths.nunique() != 1:
        raise ValueError(f"Expected one peptide length, observed {sorted(lengths.unique())}")
    return train, test, mappings, int(lengths.iloc[0])


def resolve_device(requested: str, torch: Any) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return requested


def model_args(cli: argparse.Namespace) -> SimpleNamespace:
    params = dict(FROZEN_PARAMETERS)
    params["tissue_loss_weight"] = cli.aux_weight
    params["hla_loss_weight"] = cli.aux_weight
    if cli.smoke:
        params["epochs"] = cli.smoke_epochs
    return SimpleNamespace(**params)


def select_complete_pairs(frame: pd.DataFrame, n_pairs: int, seed: int) -> pd.DataFrame:
    if n_pairs <= 0:
        return frame.sort_values("sample_id").copy()
    pair_ids = np.asarray(sorted(frame["pair_id"].unique(), key=str), dtype=object)
    if len(pair_ids) <= n_pairs:
        return frame.sort_values("sample_id").copy()
    chosen = np.random.default_rng(seed).choice(pair_ids, size=n_pairs, replace=False)
    return frame[frame["pair_id"].isin(set(chosen))].sort_values("sample_id").copy()


def save_checkpoint(
    torch: Any,
    path: Path,
    model: Any,
    branch: str,
    task_to_id: dict[str, int],
    seed: int,
    hla: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
            "branch": branch,
            "task_to_id": task_to_id,
            "seed": seed,
            "hla": hla,
            "frozen_parameters": FROZEN_PARAMETERS,
        },
        path,
    )


def train_seed_models(
    cli: argparse.Namespace,
    torch_parts: tuple[Any, Any, Any, Any],
    device: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    mappings: dict[str, Any],
    peptide_length: int,
    seed: int,
) -> tuple[Path, dict[str, tuple[Path, dict[str, int]]], pd.DataFrame, float]:
    torch, nn, DataLoader, TensorDataset = torch_parts
    args = model_args(cli)
    seed_dir = cli.output_dir / "checkpoints" / f"seed_{seed}"
    start = time.perf_counter()
    base.set_seed(seed, torch)
    print(f"[SEED START] seed={seed} device={device}", flush=True)

    global_model = e29.train_branch(
        args, torch, nn, DataLoader, TensorDataset, train, mappings["task_to_id"], peptide_length,
        device, "global_aux", "occurrence_equal_full_train", True,
        len(mappings["tissue_to_id"]), len(mappings["hla_to_id"]),
    )
    global_prediction = e29.predict_branch(
        args, torch, DataLoader, TensorDataset, global_model, train, test,
        mappings["task_to_id"], peptide_length, device,
    ).rename(columns={"score": "global_score"})
    global_path = seed_dir / "global_aux.pt"
    save_checkpoint(torch, global_path, global_model, "global_aux", mappings["task_to_id"], seed)
    del global_model
    if device == "cuda":
        torch.cuda.empty_cache()

    hla_predictions: list[pd.DataFrame] = []
    hla_paths: dict[str, tuple[Path, dict[str, int]]] = {}
    hlas = sorted(set(train["mhc_restriction"]) & set(test["mhc_restriction"]))
    for hla_index, hla in enumerate(hlas, start=1):
        fitting = train[train["mhc_restriction"] == hla].copy()
        prediction = test[test["mhc_restriction"] == hla].copy()
        tasks = sorted(set(fitting["task_name"]) & set(prediction["task_name"]))
        task_to_id = {task: index for index, task in enumerate(tasks)}
        print(f"[HLA MODEL] seed={seed} {hla_index}/{len(hlas)} hla={hla}", flush=True)
        model = e29.train_branch(
            args, torch, nn, DataLoader, TensorDataset, fitting, task_to_id, peptide_length,
            device, "hla_plain", hla, False,
            len(mappings["tissue_to_id"]), len(mappings["hla_to_id"]),
        )
        hla_prediction = e29.predict_branch(
            args, torch, DataLoader, TensorDataset, model, fitting, prediction,
            task_to_id, peptide_length, device,
        ).rename(columns={"score": "hla_score"})
        hla_predictions.append(hla_prediction)
        safe_hla = hla.replace("*", "_").replace(":", "_").replace("/", "_")
        checkpoint = seed_dir / f"hla_plain__{safe_hla}.pt"
        save_checkpoint(torch, checkpoint, model, "hla_plain", task_to_id, seed, hla)
        hla_paths[hla] = (checkpoint, task_to_id)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    hla_prediction_all = pd.concat(hla_predictions, ignore_index=True)
    fused = e29.fuse_branches(
        global_prediction.rename(columns={"global_score": "score"}),
        hla_prediction_all.rename(columns={"hla_score": "score"}),
    ).rename(columns={"score": "fused_score"})
    branch_prediction = global_prediction.merge(
        hla_prediction_all,
        on=e29.KEYS + ["label"],
        how="inner",
        validate="one_to_one",
    ).merge(
        fused[e29.KEYS + ["fused_score"]], on=e29.KEYS, how="inner", validate="one_to_one"
    )
    branch_prediction.insert(0, "seed", seed)
    elapsed = time.perf_counter() - start
    print(f"[SEED TRAIN TIME] seed={seed} seconds={elapsed:.3f}", flush=True)
    return global_path, hla_paths, branch_prediction, elapsed


def load_model(
    torch: Any,
    nn: Any,
    args: SimpleNamespace,
    checkpoint_path: Path,
    peptide_length: int,
    n_tissues: int,
    n_hlas: int,
    device: str,
) -> tuple[Any, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "species_config" in payload:
        config = formal.SpeciesConfig(**payload["species_config"])
        model = formal.define_model(
            torch=torch,
            nn=nn,
            config=config,
            peptide_length=int(payload["peptide_length"]),
            n_heads=len(payload["head_to_id"]),
            n_tissues=int(payload["n_tissues"]),
            n_mhcs=int(payload["n_mhcs"]),
            encoder_kind=payload["encoder_kind"],
            use_auxiliary=bool(payload["use_auxiliary"]),
        )
        model.load_state_dict(payload["state_dict"])
        model.to(device).eval()
        payload = dict(payload)
        payload["task_to_id"] = payload["head_to_id"]
        payload["branch"] = "global_aux" if payload["branch"] == "global" else "hla_plain"
        return model, payload
    task_to_id = payload["task_to_id"]
    if payload["branch"] == "global_aux":
        n_tissues = int(payload["state_dict"]["tissue_classifier.weight"].shape[0])
        n_hlas = int(payload["state_dict"]["hla_classifier.weight"].shape[0])
    model = e29.define_cnn_shared_heads_model(
        args, nn, peptide_length, len(task_to_id), n_tissues, n_hlas,
        payload["branch"] == "global_aux",
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload


def make_onehot_wrapper(torch: Any, nn: Any, model: Any, task_id: int) -> Any:
    class OneHotTaskWrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model
            # Expose the otherwise functional one-hot -> embedding operation
            # as a registered Linear layer so DeepLIFT/DeepSHAP can apply its
            # exact linear propagation rule.
            self.input_projection = nn.Linear(VOCAB_SIZE, model.embedding.embedding_dim, bias=False)
            with torch.no_grad():
                self.input_projection.weight.copy_(model.embedding.weight.transpose(0, 1))
            self.input_projection.weight.requires_grad_(False)

        def forward(self, onehot: Any) -> Any:
            embedded = self.input_projection(onehot) + self.model.position_embedding
            channels_first = embedded.transpose(1, 2)
            features = []
            for convolution in self.model.convolutions:
                convolved = convolution(channels_first)[..., : self.model.peptide_length]
                features.append(torch.relu(convolved).transpose(1, 2))
            encoded = self.model.encoder(torch.cat(features, dim=2))
            return self.model.heads[task_id](encoded)

    return OneHotTaskWrapper().to(model.position_embedding.device).eval()


def to_onehot(torch: Any, frame: pd.DataFrame, peptide_length: int, device: str) -> Any:
    encoded = base.encode_peptides(frame["peptide_sequence"], peptide_length)
    ids = torch.as_tensor(encoded.copy(), dtype=torch.long, device=device)
    return torch.nn.functional.one_hot(ids, num_classes=VOCAB_SIZE).float()


def normalize_shap(raw: Any) -> np.ndarray:
    if isinstance(raw, list):
        raw = raw[-1]
    values = np.asarray(raw)
    if values.ndim == 4 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 3:
        raise ValueError(f"Unexpected DeepSHAP shape: {values.shape}")
    return values


def expected_integrated_gradients(
    torch: Any,
    wrapper: Any,
    background: Any,
    inputs: Any,
    steps: int,
    target_batch_size: int,
) -> np.ndarray:
    """Deterministic empirical-baseline Expected IG (a SHAP approximation).

    Averaging integrated gradients over an empirical background distribution
    is the deterministic counterpart of Expected Gradients. Trapezoidal path
    integration provides a directly auditable completeness approximation.
    """
    if steps < 2:
        raise ValueError("integration_steps must be at least 2")
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=inputs.device, dtype=inputs.dtype)
    weights = torch.ones(steps + 1, device=inputs.device, dtype=inputs.dtype)
    weights[0] = weights[-1] = 0.5
    results = []
    for start in range(0, len(inputs), target_batch_size):
        targets = inputs[start : start + target_batch_size]
        difference = targets[:, None, :, :] - background[None, :, :, :]
        paths = (
            background[None, :, None, :, :]
            + difference[:, :, None, :, :] * alphas[None, None, :, None, None]
        )
        flat_paths = paths.reshape(-1, inputs.shape[1], inputs.shape[2]).detach().requires_grad_(True)
        outputs = wrapper(flat_paths)
        gradients = torch.autograd.grad(outputs.sum(), flat_paths, create_graph=False)[0]
        gradients = gradients.reshape(
            len(targets), len(background), steps + 1, inputs.shape[1], inputs.shape[2]
        )
        integrated_gradient = (gradients * weights[None, None, :, None, None]).sum(dim=2) / steps
        attribution = (difference * integrated_gradient).mean(dim=1)
        results.append(attribution.detach().cpu().numpy().astype(np.float32))
        del paths, flat_paths, outputs, gradients, integrated_gradient, attribution
    return np.concatenate(results)


def explain_model_tasks(
    cli: argparse.Namespace,
    shap: Any,
    torch: Any,
    nn: Any,
    model: Any,
    branch: str,
    seed: int,
    task_to_id: dict[str, int],
    train: pd.DataFrame,
    selected_test: pd.DataFrame,
    peptide_length: int,
    device: str,
) -> tuple[np.ndarray, pd.DataFrame, list[dict[str, Any]]]:
    explanations: list[np.ndarray] = []
    metadata_parts: list[pd.DataFrame] = []
    audit: list[dict[str, Any]] = []
    for task_index, task_name in enumerate(sorted(task_to_id), start=1):
        explain_rows = selected_test[selected_test["task_name"] == task_name].copy()
        if explain_rows.empty:
            continue
        background_pool = train[train["task_name"] == task_name].copy()
        background = select_complete_pairs(
            background_pool,
            cli.background_pairs,
            stable_seed(f"background|{task_name}", cli.sample_seed),
        )
        wrapper = make_onehot_wrapper(torch, nn, model, task_to_id[task_name])
        background_x = to_onehot(torch, background, peptide_length, device)
        explain_x = to_onehot(torch, explain_rows, peptide_length, device)
        if cli.explainer == "expected_ig":
            raw = expected_integrated_gradients(
                torch, wrapper, background_x, explain_x, cli.integration_steps, cli.ig_target_batch_size
            )
            with torch.no_grad():
                expected = float(wrapper(background_x).mean().detach().cpu().item())
        elif cli.explainer == "gradient":
            explainer = shap.GradientExplainer(wrapper, background_x, batch_size=cli.shap_batch_size)
            raw = explainer.shap_values(
                explain_x,
                nsamples=cli.gradient_samples,
                rseed=stable_seed(f"shap|{seed}|{branch}|{task_name}", cli.sample_seed),
            )
            with torch.no_grad():
                expected = float(wrapper(background_x).mean().detach().cpu().item())
        else:
            explainer = shap.DeepExplainer(wrapper, background_x)
            raw = explainer.shap_values(explain_x, check_additivity=False)
            expected = float(np.asarray(explainer.expected_value).reshape(-1)[0])
        values = normalize_shap(raw)
        with torch.no_grad():
            logits = wrapper(explain_x).detach().cpu().numpy().reshape(-1)
        reconstructed = expected + values.sum(axis=(1, 2))
        max_error = float(np.max(np.abs(logits - reconstructed)))
        audit.append(
            {
                "seed": seed,
                "branch": branch,
                "task_name": task_name,
                "background_rows": len(background),
                "explained_rows": len(explain_rows),
                "expected_logit": expected,
                "max_abs_additivity_error": max_error,
            }
        )
        meta = explain_rows[
            ["sample_id", "pair_id", "label", "target_tissue", "mhc_restriction", "task_name", "peptide_sequence"]
        ].copy()
        meta.insert(0, "branch", branch)
        meta.insert(0, "seed", seed)
        explanations.append(values.astype(np.float32))
        metadata_parts.append(meta)
        print(
            f"[SHAP] seed={seed} branch={branch} task={task_index}/{len(task_to_id)} "
            f"rows={len(explain_rows)} additivity_error={max_error:.3g}",
            flush=True,
        )
    return np.concatenate(explanations), pd.concat(metadata_parts, ignore_index=True), audit


def observed_attribution_table(meta: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    result = meta.reset_index(drop=True).copy()
    for position in range(values.shape[1]):
        residues = result["peptide_sequence"].str[position]
        aa_indices = residues.map(base.AA_TO_INDEX).to_numpy(dtype=np.int64)
        result[f"position_{position + 1}_residue"] = residues
        result[f"position_{position + 1}_shap"] = values[np.arange(len(values)), position, aa_indices]
    result["observed_sequence_shap_sum"] = [
        sum(row[f"position_{position + 1}_shap"] for position in range(values.shape[1]))
        for _, row in result.iterrows()
    ]
    return result


def aggregate_shap(meta: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    scopes: list[tuple[str, str, np.ndarray]] = [("overall", "ALL", np.arange(len(meta)))]
    for column, scope_type in (
        ("mhc_restriction", "hla"), ("target_tissue", "tissue"), ("task_name", "task")
    ):
        for scope_value, indices in meta.groupby(column, sort=True).indices.items():
            scopes.append((scope_type, str(scope_value), np.asarray(indices, dtype=np.int64)))
    for scope_type, scope_value, indices in scopes:
        for label_name, label_value in (("all", None), ("positive", 1), ("negative", 0)):
            chosen = indices if label_value is None else indices[meta.iloc[indices]["label"].to_numpy() == label_value]
            if len(chosen) == 0:
                continue
            subset = values[chosen, :, 1:]
            mean = subset.mean(axis=0)
            mean_abs = np.abs(subset).mean(axis=0)
            records = []
            for position in range(mean.shape[0]):
                for aa_index, aa in enumerate(AMINO_ACIDS):
                    records.append(
                        {
                            "seed": int(meta["seed"].iloc[0]),
                            "branch": str(meta["branch"].iloc[0]),
                            "scope_type": scope_type,
                            "scope_value": scope_value,
                            "label_group": label_name,
                            "n_samples": len(chosen),
                            "position": position + 1,
                            "amino_acid": aa,
                            "mean_shap": float(mean[position, aa_index]),
                            "mean_abs_shap": float(mean_abs[position, aa_index]),
                        }
                    )
            frames.append(pd.DataFrame(records))
    return pd.concat(frames, ignore_index=True)


def add_ensemble_and_consensus(summary: pd.DataFrame) -> pd.DataFrame:
    keys = ["branch", "scope_type", "scope_value", "label_group", "position", "amino_acid"]
    seed_mean = summary.groupby(keys, as_index=False).agg(
        n_samples=("n_samples", "min"),
        mean_shap=("mean_shap", "mean"),
        mean_abs_shap=("mean_abs_shap", "mean"),
    )
    seed_mean.insert(0, "seed", "three_seed_mean" if summary["seed"].nunique() == 3 else "seed_mean")
    combined = pd.concat([summary, seed_mean], ignore_index=True)
    consensus_keys = ["seed", "scope_type", "scope_value", "label_group", "position", "amino_acid"]
    consensus = seed_mean.groupby(consensus_keys, as_index=False).agg(
        n_samples=("n_samples", "min"),
        mean_shap=("mean_shap", "mean"),
        mean_abs_shap=("mean_abs_shap", "mean"),
    )
    consensus.insert(1, "branch", "descriptive_branch_consensus")
    return pd.concat([combined, consensus], ignore_index=True)


def build_pair_differences(observed: pd.DataFrame, peptide_length: int) -> pd.DataFrame:
    keys = ["seed", "branch", "pair_id", "target_tissue", "mhc_restriction", "task_name"]
    value_columns = [f"position_{position}_shap" for position in range(1, peptide_length + 1)]
    positive = observed[observed["label"] == 1][keys + ["sample_id", "peptide_sequence", *value_columns]].copy()
    negative = observed[observed["label"] == 0][keys + ["sample_id", "peptide_sequence", *value_columns]].copy()
    paired = positive.merge(negative, on=keys, how="inner", validate="one_to_one", suffixes=("_positive", "_negative"))
    paired = paired.rename(
        columns={
            "sample_id_positive": "positive_sample_id",
            "sample_id_negative": "negative_sample_id",
            "peptide_sequence_positive": "positive_peptide",
            "peptide_sequence_negative": "negative_peptide",
        }
    )
    for position in range(1, peptide_length + 1):
        paired[f"position_{position}_shap_delta"] = (
            paired[f"position_{position}_shap_positive"] - paired[f"position_{position}_shap_negative"]
        )
    keep = [
        *keys, "positive_sample_id", "negative_sample_id", "positive_peptide", "negative_peptide",
        *[f"position_{position}_shap_delta" for position in range(1, peptide_length + 1)],
    ]
    return paired[keep]


def summarize_pair_differences(pair_data: pd.DataFrame, peptide_length: int) -> pd.DataFrame:
    rows = []
    scopes: list[tuple[str, str, pd.DataFrame]] = [("overall", "ALL", pair_data)]
    for column, scope_type in (
        ("mhc_restriction", "hla"), ("target_tissue", "tissue"), ("task_name", "task")
    ):
        for value, group in pair_data.groupby(column, sort=True):
            scopes.append((scope_type, str(value), group))
    for scope_type, scope_value, scope in scopes:
        for (seed, branch), group in scope.groupby(["seed", "branch"], sort=True):
            for position in range(1, peptide_length + 1):
                values = group[f"position_{position}_shap_delta"]
                rows.append(
                    {
                        "seed": seed, "branch": branch, "scope_type": scope_type,
                        "scope_value": scope_value, "n_pairs": len(group), "position": position,
                        "mean_positive_minus_negative_shap": float(values.mean()),
                        "median_positive_minus_negative_shap": float(values.median()),
                        "sd_positive_minus_negative_shap": float(values.std(ddof=1)),
                    }
                )
    raw = pd.DataFrame(rows)
    keys = ["branch", "scope_type", "scope_value", "position"]
    seed_mean = raw.groupby(keys, as_index=False).agg(
        n_pairs=("n_pairs", "min"),
        mean_positive_minus_negative_shap=("mean_positive_minus_negative_shap", "mean"),
        median_positive_minus_negative_shap=("median_positive_minus_negative_shap", "mean"),
        sd_positive_minus_negative_shap=("sd_positive_minus_negative_shap", "mean"),
    )
    seed_label = "three_seed_mean" if raw["seed"].nunique() == 3 else "seed_mean"
    seed_mean.insert(0, "seed", seed_label)
    consensus_keys = ["seed", "scope_type", "scope_value", "position"]
    consensus = seed_mean.groupby(consensus_keys, as_index=False).agg(
        n_pairs=("n_pairs", "min"),
        mean_positive_minus_negative_shap=("mean_positive_minus_negative_shap", "mean"),
        median_positive_minus_negative_shap=("median_positive_minus_negative_shap", "mean"),
        sd_positive_minus_negative_shap=("sd_positive_minus_negative_shap", "mean"),
    )
    consensus.insert(1, "branch", "descriptive_branch_consensus")
    return pd.concat([raw, seed_mean, consensus], ignore_index=True)


def validate_predictions(
    cli: argparse.Namespace, predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not cli.reference_predictions.is_file() or cli.smoke:
        return pd.DataFrame(rows)
    reference = pd.read_csv(cli.reference_predictions, keep_default_na=False)
    for seed, current in predictions.groupby("seed", sort=True):
        expected = reference[pd.to_numeric(reference["seed"], errors="coerce") == int(seed)]
        if expected.empty:
            continue
        merged = current.merge(
            expected[["sample_id", "score"]], on="sample_id", how="inner", validate="one_to_one",
        )
        difference = merged["fused_score"] - merged["score"]
        rows.append(
            {
                "seed": seed,
                "matched_rows": len(merged),
                "max_abs_prediction_difference": float(difference.abs().max()),
                "mean_abs_prediction_difference": float(difference.abs().mean()),
                "pearson_prediction_correlation": float(merged["fused_score"].corr(merged["score"])),
            }
        )
    return pd.DataFrame(rows)


def stability_table(summary: pd.DataFrame) -> pd.DataFrame:
    source = summary[
        (summary["scope_type"] == "overall")
        & (summary["scope_value"] == "ALL")
        & (summary["seed"].map(lambda value: isinstance(value, (int, np.integer))))
    ]
    rows = []
    for branch, branch_data in source.groupby("branch", sort=True):
        vectors = {}
        for seed, seed_data in branch_data.groupby("seed", sort=True):
            pivot = seed_data.pivot_table(
                index=["position", "amino_acid"], columns="label_group", values="mean_shap"
            )
            vectors[int(seed)] = (pivot["positive"] - pivot["negative"]).to_numpy()
        for seed_a, seed_b in combinations(sorted(vectors), 2):
            correlation = spearmanr(vectors[seed_a], vectors[seed_b]).statistic
            rows.append({"branch": branch, "seed_a": seed_a, "seed_b": seed_b, "spearman_r": correlation})
    return pd.DataFrame(rows)


def matrix_for(
    summary: pd.DataFrame, branch: str, scope_type: str, scope_value: str,
) -> np.ndarray:
    subset = summary[
        (summary["branch"] == branch)
        & (summary["scope_type"] == scope_type)
        & (summary["scope_value"] == scope_value)
        & (summary["seed"].astype(str).isin(["three_seed_mean", "seed_mean"]))
    ]
    pivot = subset.pivot_table(
        index=["position", "amino_acid"], columns="label_group", values="mean_shap"
    )
    delta = (pivot["positive"] - pivot["negative"]).unstack("amino_acid")
    return delta.reindex(index=range(1, 10), columns=list(AMINO_ACIDS)).to_numpy().T


def save_figure(fig: Any, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def draw_heatmap(ax: Any, matrix: np.ndarray, title: str, vmax: float) -> Any:
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(9), labels=range(1, 10))
    ax.set_yticks(range(20), labels=list(AMINO_ACIDS), fontsize=12.5)
    ax.tick_params(axis="x", labelsize=13.5)
    ax.set_xlabel("Peptide position", fontsize=15.0)
    ax.set_ylabel("Amino acid", fontsize=15.0)
    ax.set_title(title, fontsize=16.0, pad=10)
    return image


def create_figures(summary: pd.DataFrame, test: pd.DataFrame, output_dir: Path) -> None:
    branch = "descriptive_branch_consensus"
    overall = matrix_for(summary, branch, "overall", "ALL")
    vmax = max(float(np.nanpercentile(np.abs(overall), 98)), 1e-8)
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    image = draw_heatmap(ax, overall, "Occurrence-equal E29: positive − negative SHAP", vmax)
    fig.colorbar(image, ax=ax, label="Mean SHAP difference (logit scale)")
    fig.tight_layout()
    save_figure(fig, output_dir, "01_overall_shap_motif")

    ensemble_label = "three_seed_mean" if (summary["seed"].astype(str) == "three_seed_mean").any() else "seed_mean"
    position_source = summary[
        (summary["seed"].astype(str) == ensemble_label)
        & (summary["scope_type"] == "overall")
        & (summary["scope_value"] == "ALL")
        & (summary["label_group"] == "all")
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for label, group in position_source.groupby("branch", sort=True):
        values = group.groupby("position")["mean_abs_shap"].sum()
        ax.plot(values.index, values.values, marker="o", label=label)
    ax.set_xlabel("Peptide position")
    ax.set_ylabel("Summed mean |SHAP| across residues")
    ax.set_xticks(range(1, 10))
    ax.tick_params(axis="both", labelsize=10.5)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Position-level importance")
    fig.tight_layout()
    save_figure(fig, output_dir, "02_position_importance")

    top_hlas = (
        test.groupby("mhc_restriction")["task_name"].nunique().sort_values(ascending=False).head(6).index.tolist()
    )
    matrices = [matrix_for(summary, branch, "hla", hla) for hla in top_hlas]
    panel_vmax = max(float(np.nanpercentile(np.abs(np.stack(matrices)), 98)), 1e-8)
    fig, axes = plt.subplots(2, 3, figsize=(16.2, 9.8), constrained_layout=True)
    image = None
    for ax, hla, matrix in zip(axes.flat, top_hlas, matrices):
        image = draw_heatmap(ax, matrix, hla, panel_vmax)
    if image is not None:
        colorbar = fig.colorbar(
            image, ax=axes.ravel().tolist(), shrink=0.75,
            label="Positive − negative mean SHAP",
        )
        colorbar.ax.tick_params(labelsize=13.0)
        colorbar.set_label("Positive − negative mean SHAP", fontsize=15.0)
    save_figure(fig, output_dir, "03_top_hla_shap_motifs")

    top_tissues = test.groupby("target_tissue")["task_name"].nunique().sort_values(ascending=False).head(6).index.tolist()
    matrices = [matrix_for(summary, branch, "tissue", tissue) for tissue in top_tissues]
    panel_vmax = max(float(np.nanpercentile(np.abs(np.stack(matrices)), 98)), 1e-8)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    image = None
    for ax, tissue, matrix in zip(axes.flat, top_tissues, matrices):
        image = draw_heatmap(ax, matrix, tissue, panel_vmax)
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75, label="Positive − negative mean SHAP")
    save_figure(fig, output_dir, "04_top_tissue_shap_motifs")


def create_pair_figure(pair_summary: pd.DataFrame, output_dir: Path) -> None:
    source = pair_summary[
        (pair_summary["scope_type"] == "overall")
        & (pair_summary["scope_value"] == "ALL")
        & (pair_summary["seed"].astype(str).isin(["three_seed_mean", "seed_mean"]))
    ]
    fig, ax = plt.subplots(figsize=(8.2, 5.9))
    for branch, group in source.groupby("branch", sort=True):
        ax.plot(
            group["position"], group["mean_positive_minus_negative_shap"],
            marker="o", label=branch,
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(range(1, 10))
    ax.tick_params(axis="both", labelsize=15.0)
    ax.xaxis.label.set_size(16.0)
    ax.yaxis.label.set_size(16.0)
    ax.title.set_size(18.0)
    ax.set_xlabel("Peptide position")
    ax.set_ylabel("Paired positive − negative observed-residue SHAP")
    ax.set_title("Pair-matched positional attribution difference")
    ax.legend(frameon=False, fontsize=14.0)
    fig.tight_layout()
    save_figure(fig, output_dir, "05_paired_position_shap_difference")


def write_report(
    cli: argparse.Namespace,
    summary: pd.DataFrame,
    stability: pd.DataFrame,
    validation: pd.DataFrame,
    additivity: pd.DataFrame,
    pair_summary: pd.DataFrame,
    elapsed: float,
) -> None:
    branch = "descriptive_branch_consensus"
    matrix = matrix_for(summary, branch, "overall", "ALL")
    effects = []
    for aa_index, aa in enumerate(AMINO_ACIDS):
        for position in range(9):
            effects.append((float(matrix[aa_index, position]), position + 1, aa))
    positive = [item for item in sorted(effects, reverse=True) if item[0] > 0][:10]
    negative = [item for item in sorted(effects) if item[0] < 0][:10]
    validation_text = "未执行（smoke test 或参考文件缺失）"
    if not validation.empty:
        validation_text = (
            f"最大绝对预测差={validation['max_abs_prediction_difference'].max():.6g}；"
            f"最小 Pearson r={validation['pearson_prediction_correlation'].min():.6f}。"
        )
    stability_text = "仅一个 seed，未计算"
    if not stability.empty:
        stability_text = f"seed 两两 Spearman r 中位数={stability['spearman_r'].median():.4f}。"
    seed_label = "three_seed_mean" if (pair_summary["seed"].astype(str) == "three_seed_mean").any() else "seed_mean"
    pair_overall = pair_summary[
        (pair_summary["seed"].astype(str) == seed_label)
        & (pair_summary["branch"] == "descriptive_branch_consensus")
        & (pair_summary["scope_type"] == "overall")
        & (pair_summary["scope_value"] == "ALL")
    ].sort_values("mean_positive_minus_negative_shap", ascending=False)
    top_positions = "、".join(
        f"P{int(row.position)}={row.mean_positive_minus_negative_shap:.4f}"
        for row in pair_overall.head(3).itertuples()
    )
    lines = [
        "# occurrence-equal tissuePMHC：E29 SHAP 生物学解释报告",
        "",
        "## 分析对象",
        "",
        "- 数据：Human occurrence-equal 固定 train/test；正负样本保持 pair 结构。",
        "- 模型：冻结调优 E29（global-auxiliary 与 HLA-specific 两分支）。",
        f"- 方法：每个 tissue–HLA task 使用其自身训练样本作为 {cli.explainer} SHAP 背景，解释测试集肽序列。",
        "- 尺度：SHAP 对应分支 logit；热图为阳性样本平均 SHAP减去阴性样本平均 SHAP。",
        "",
        "## 质量控制",
        "",
        f"- 发布结果复现：{validation_text}",
        f"- seed 稳定性：{stability_text}",
        f"- task 最大误差的中位数：{additivity['max_abs_additivity_error'].median():.6g}；全局最大值：{additivity['max_abs_additivity_error'].max():.6g}。",
        f"- 总运行时间：{elapsed:.3f} 秒。",
        "",
        "## 配对位置结论",
        "",
        f"- 在完整正负 pair 的 observed-residue attribution 对比中，最强三个位置为：{top_positions}。",
        "- P2/P3 的突出贡献提示模型主要依赖 N 端局部 motif；P9 的次级峰与 MHC-I C 端锚定位点模式相容。",
        "- HLA 分层热图比 tissue 汇总热图更适合做 motif 生物学解释；tissue 图仍会混合其 HLA 构成。",
        "",
        "## 最强正向残基-位置信号",
        "",
        "|排名|位置|残基|阳性−阴性 mean SHAP|",
        "|---:|---:|:---:|---:|",
    ]
    lines.extend(f"|{i}|{position}|{aa}|{effect:.6f}|" for i, (effect, position, aa) in enumerate(positive, 1))
    lines.extend(["", "## 最强负向残基-位置信号", "", "|排名|位置|残基|阳性−阴性 mean SHAP|", "|---:|---:|:---:|---:|"])
    lines.extend(f"|{i}|{position}|{aa}|{effect:.6f}|" for i, (effect, position, aa) in enumerate(negative, 1))
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "1. 最终 E29 使用 task 内 percentile-rank 融合，该步骤不可微；因此结果是可审计的分支级 Expected-IG SHAP 近似，branch consensus 只是描述性平均。",
            "2. SHAP 描述模型依赖，不证明因果机制；位置/残基结论需与已知 HLA motif 或独立实验验证交叉确认。",
            "3. task 条件背景消除了组织/HLA 基线差异；结果解释的是同一 tissue–HLA 内的序列判别，而不是组织变量本身的因果效应。",
            "4. occurrence-equal 设计降低了跨组织出现次数混杂，但不能消除来源蛋白、检测流程和数据库收录偏差。",
            "",
        ]
    )
    (cli.output_dir / "SHAP_ANALYSIS_REPORT_zh.md").write_text("\n".join(lines), encoding="utf-8")


def append_timing(rows: list[dict[str, Any]], seed: Any, stage: str, seconds: float) -> None:
    rows.append({"seed": seed, "stage": stage, "seconds": seconds})


def run(cli: argparse.Namespace) -> None:
    try:
        import shap
    except ImportError as exc:
        raise SystemExit("Install SHAP with: python -m pip install shap") from exc
    torch_parts = base.require_torch()
    torch, nn, _, _ = torch_parts
    device = resolve_device(cli.device, torch)
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    train, test, mappings, peptide_length = validate_and_prepare(cli.train, cli.test, cli.max_tasks)
    seeds = cli.seeds[:1] if cli.smoke else cli.seeds
    if not cli.smoke and seeds != DEFAULT_SEEDS:
        raise ValueError(f"Full analysis uses frozen seeds {DEFAULT_SEEDS}")
    print(
        f"[START] train={len(train)} test={len(test)} tasks={len(mappings['tasks'])} "
        f"seeds={seeds} device={device}", flush=True,
    )

    selected_parts = []
    for task_name, task in test.groupby("task_name", sort=True):
        selected_parts.append(
            select_complete_pairs(task, cli.explain_pairs_per_task, stable_seed(task_name, cli.sample_seed))
        )
    selected_test = pd.concat(selected_parts, ignore_index=True)

    all_predictions: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []
    observed_tables: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    args = model_args(cli)

    reused_predictions = None
    if cli.reuse_checkpoints:
        if not cli.checkpoint_root.is_dir() or not cli.branch_predictions_reference.is_file():
            raise FileNotFoundError("--reuse-checkpoints requires checkpoint root and branch predictions")
        reused_predictions = pd.read_csv(cli.branch_predictions_reference, keep_default_na=False)

    for seed in seeds:
        if cli.reuse_checkpoints:
            seed_dir = cli.checkpoint_root / f"seed_{seed}"
            global_path = seed_dir / "global_aux.pt"
            hla_paths = {}
            for hla in sorted(test["mhc_restriction"].unique()):
                safe_hla = hla.replace("*", "_").replace(":", "_").replace("/", "_")
                path = seed_dir / f"hla_plain__{safe_hla}.pt"
                if path.is_file():
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    hla_paths[hla] = (path, payload.get("task_to_id", payload.get("head_to_id")))
            branch_prediction = reused_predictions[pd.to_numeric(reused_predictions["seed"]) == seed].copy()
            train_seconds = 0.0
            print(f"[REUSE CHECKPOINTS] seed={seed} hla_models={len(hla_paths)}", flush=True)
        else:
            global_path, hla_paths, branch_prediction, train_seconds = train_seed_models(
                cli, torch_parts, device, train, test, mappings, peptide_length, seed,
            )
        append_timing(timing_rows, seed, "training_all_e29_branches", train_seconds)
        all_predictions.append(branch_prediction)

        shap_start = time.perf_counter()
        global_model, global_payload = load_model(
            torch, nn, args, global_path, peptide_length,
            len(mappings["tissue_to_id"]), len(mappings["hla_to_id"]), device,
        )
        values, meta, audit = explain_model_tasks(
            cli, shap, torch, nn, global_model, "global_aux", seed,
            global_payload["task_to_id"], train, selected_test, peptide_length, device,
        )
        summaries.append(aggregate_shap(meta, values))
        observed_tables.append(observed_attribution_table(meta, values))
        audits.extend(audit)
        del global_model
        if device == "cuda":
            torch.cuda.empty_cache()

        for hla, (path, _) in sorted(hla_paths.items()):
            model, payload = load_model(
                torch, nn, args, path, peptide_length,
                len(mappings["tissue_to_id"]), len(mappings["hla_to_id"]), device,
            )
            hla_train = train[train["mhc_restriction"] == hla].copy()
            hla_test = selected_test[selected_test["mhc_restriction"] == hla].copy()
            if hla_test.empty:
                del model
                continue
            values, meta, audit = explain_model_tasks(
                cli, shap, torch, nn, model, "hla_plain", seed,
                payload["task_to_id"], hla_train, hla_test, peptide_length, device,
            )
            summaries.append(aggregate_shap(meta, values))
            observed_tables.append(observed_attribution_table(meta, values))
            audits.extend(audit)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        shap_seconds = time.perf_counter() - shap_start
        append_timing(timing_rows, seed, f"{cli.explainer}_all_branches", shap_seconds)
        print(f"[SEED SHAP TIME] seed={seed} seconds={shap_seconds:.3f}", flush=True)

    predictions = pd.concat(all_predictions, ignore_index=True)
    raw_summary = pd.concat(summaries, ignore_index=True)
    final_summary = add_ensemble_and_consensus(raw_summary)
    observed = pd.concat(observed_tables, ignore_index=True)
    pair_differences = build_pair_differences(observed, peptide_length)
    pair_summary = summarize_pair_differences(pair_differences, peptide_length)
    additivity = pd.DataFrame(audits)
    stability = stability_table(raw_summary)
    validation = validate_predictions(cli, predictions)

    predictions.to_csv(cli.output_dir / "branch_predictions.csv.gz", index=False, compression="gzip")
    observed.to_csv(cli.output_dir / "sample_observed_shap.csv.gz", index=False, compression="gzip")
    pair_differences.to_csv(cli.output_dir / "pair_position_shap_differences.csv.gz", index=False, compression="gzip")
    pair_summary.to_csv(cli.output_dir / "pair_position_shap_summary.csv.gz", index=False, compression="gzip")
    final_summary.to_csv(cli.output_dir / "shap_residue_position_summary.csv.gz", index=False, compression="gzip")
    additivity.to_csv(cli.output_dir / "shap_additivity_audit.csv", index=False)
    stability.to_csv(cli.output_dir / "seed_stability.csv", index=False)
    validation.to_csv(cli.output_dir / "published_prediction_reproduction.csv", index=False)
    create_figures(final_summary, test, cli.output_dir)
    create_pair_figure(pair_summary, cli.output_dir)

    elapsed = time.perf_counter() - start
    append_timing(timing_rows, "all", "total", elapsed)
    pd.DataFrame(timing_rows).to_csv(cli.output_dir / "timing_results.csv", index=False)
    write_report(cli, final_summary, stability, validation, additivity, pair_summary, elapsed)
    metadata = {
        "analysis": f"task-conditional {cli.explainer} SHAP of tuned occurrence-equal E29",
        "train": str(cli.train.resolve()),
        "test": str(cli.test.resolve()),
        "reference_predictions": str(cli.reference_predictions.resolve()),
        "seeds": seeds,
        "device": device,
        "train_rows": len(train),
        "test_rows": len(test),
        "explained_rows_per_seed_per_branch": len(selected_test),
        "tasks": len(mappings["tasks"]),
        "peptide_length": peptide_length,
        "background_pairs_per_task": cli.background_pairs,
        "explain_pairs_per_task": cli.explain_pairs_per_task,
        "explainer": cli.explainer,
        "gradient_samples": cli.gradient_samples,
        "integration_steps": cli.integration_steps,
        "frozen_parameters": vars(args),
        "reuse_checkpoints": cli.reuse_checkpoints,
        "checkpoint_root": str(cli.checkpoint_root.resolve()),
        "interpretation_scope": {
            "branch_shap": "audited branch-level empirical-baseline Expected-IG SHAP approximation on logits",
            "descriptive_branch_consensus": "arithmetic mean across branch summaries; not exact rank-fusion SHAP",
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "shap": shap.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "elapsed_seconds": elapsed,
    }
    (cli.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"[TOTAL TIME] seconds={elapsed:.3f}", flush=True)
    print(f"[WROTE] {cli.output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train", type=Path,
        default=PROJECT_ROOT / "data/humanPMHC_occurence_equal_dataset/humanPMHC_train.csv.gz",
    )
    parser.add_argument(
        "--test", type=Path,
        default=PROJECT_ROOT / "data/humanPMHC_occurence_equal_dataset/humanPMHC_test.csv.gz",
    )
    parser.add_argument(
        "--reference-predictions", type=Path,
        default=PROJECT_ROOT / "extra_occurrence_equal_dataset/adjusting/results/e29_final_test/all_seed_test_predictions.csv.gz",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "extra_occurrence_equal_dataset/results/e29_tuned_shap",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--aux-weight", type=float, default=0.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--background-pairs", type=int, default=20)
    parser.add_argument(
        "--explain-pairs-per-task", type=int, default=20,
        help="Complete test pairs explained per task; 0 explains every test pair",
    )
    parser.add_argument("--explainer", choices=["expected_ig", "gradient", "deep"], default="expected_ig")
    parser.add_argument("--gradient-samples", type=int, default=200)
    parser.add_argument("--shap-batch-size", type=int, default=128)
    parser.add_argument("--integration-steps", type=int, default=32)
    parser.add_argument("--ig-target-batch-size", type=int, default=4)
    parser.add_argument("--reuse-checkpoints", action="store_true")
    parser.add_argument(
        "--checkpoint-root", type=Path,
        default=PROJECT_ROOT / "extra_occurrence_equal_dataset/results/e29_tuned_shap/checkpoints",
    )
    parser.add_argument(
        "--branch-predictions-reference", type=Path,
        default=PROJECT_ROOT / "extra_occurrence_equal_dataset/results/e29_tuned_shap/branch_predictions.csv.gz",
    )
    parser.add_argument("--sample-seed", type=int, default=20260805)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
