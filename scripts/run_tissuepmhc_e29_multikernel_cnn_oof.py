#!/usr/bin/env python3
"""Run E29: multi-kernel CNN E14a with a leakage-safe OOF screen.

E29 preserves E14a's two complementary branches:

* global branch: shared task heads plus tissue/HLA auxiliary supervision;
* HLA branch: one plain shared-head model per HLA allele;
* fusion: mean of the two within-task percentile ranks.

Only the peptide encoder changes from E14a's Flatten-MLP to a positional,
multi-kernel 1D CNN.  The default invocation runs three seeds on three
pair-grouped OOF folds, then continues to full-train test prediction only if
all pre-registered OOF criteria pass.  ``--no-run-test`` keeps an OOF-only
execution path.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_tissuepmhc_auxiliary_soft_ensemble as e14
import run_tissuepmhc_e15_fusion_ablation as e15
import run_tissuepmhc_e26_greedy_ensemble_selection as e26
import run_tissuepmhc_neural_baselines_v2 as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KEYS = ["sample_id", "target_tissue", "mhc_restriction"]
PREDICTION_COLUMNS = ["split", "candidate", "seed", *KEYS, "label", "score"]
TASK_COMPARISON_COLUMNS = [
    "target_tissue", "mhc_restriction", "rows", "baseline_auroc", "cnn_auroc",
    "fused_auroc", "cnn_minus_baseline_auroc", "fused_minus_baseline_auroc",
]
PER_TASK_COLUMNS = [
    "experiment_name", "seed", "model", "n_member_seeds", "member_seeds",
    "target_tissue", "mhc_restriction", "test_rows", "test_positive", "test_negative",
    *e14.METRICS, "fusion_formula",
]
E17_COMPARISON_COLUMNS = [
    "target_tissue", "mhc_restriction", "e29_auroc", "e17_auroc", "delta_auroc",
    "e29_auprc", "e17_auprc", "delta_auprc", "e29_accuracy", "e17_accuracy",
    "delta_accuracy", "e29_mcc", "e17_mcc", "delta_mcc",
]


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def format_duration(seconds: float) -> str:
    minutes, remainder = divmod(seconds, 60.0)
    return f"{int(minutes)}m {remainder:04.1f}s" if minutes >= 1 else f"{remainder:.1f}s"


def make_pair_grouped_folds(train_df: pd.DataFrame, folds: int, split_seed: int) -> pd.Series:
    """Assign every pair_id to one deterministic OOF fold within its task."""
    if folds < 2:
        raise ValueError("oof_folds must be at least two.")
    rng = np.random.default_rng(split_seed)
    assignment = pd.Series(index=train_df.index, dtype="int64")
    for task_name, task in train_df.groupby("task_name", sort=True):
        pairs = np.asarray(sorted(task["pair_id"].unique()))
        if len(pairs) < folds:
            raise ValueError(f"Task {task_name} has {len(pairs)} pairs, fewer than oof_folds={folds}.")
        shuffled = rng.permutation(pairs)
        pair_to_fold = {pair: index % folds for index, pair in enumerate(shuffled)}
        assignment.loc[task.index] = task["pair_id"].map(pair_to_fold).astype(int)
    if assignment.isna().any():
        raise AssertionError("Some training rows were not assigned an OOF fold.")
    for fold in range(folds):
        fitting, held_out = train_df[assignment != fold], train_df[assignment == fold]
        if set(fitting["pair_id"]) & set(held_out["pair_id"]):
            raise AssertionError(f"pair_id leakage in OOF fold {fold}.")
    return assignment.astype(int)


def define_cnn_shared_heads_model(
    args: argparse.Namespace, nn: Any, peptide_length: int, n_tasks: int, n_tissues: int, n_hlas: int,
    use_aux: bool,
) -> Any:
    """Create a position-preserving multi-kernel replacement for E14a's MLP encoder."""

    class MultiKernelCnnSharedHeads(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.peptide_length = peptide_length
            self.embedding = nn.Embedding(len(base.AA_TO_INDEX) + 1, args.embedding_dim, padding_idx=base.PAD_INDEX)
            self.position_embedding = nn.Parameter(
                np_to_tensor(np.zeros((1, peptide_length, args.embedding_dim), dtype=np.float32))
            )
            self.convolutions = nn.ModuleList([
                nn.Conv1d(args.embedding_dim, args.conv_channels, kernel_size=kernel, padding=kernel // 2)
                for kernel in args.kernel_sizes
            ])
            encoded_dim = peptide_length * args.conv_channels * len(args.kernel_sizes)
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(encoded_dim, args.hidden_dim),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_dim, args.hidden_dim),
                nn.ReLU(),
            )
            self.heads = nn.ModuleList([nn.Linear(args.hidden_dim, 1) for _ in range(n_tasks)])
            self.use_aux = use_aux
            if use_aux:
                self.tissue_classifier = nn.Linear(args.hidden_dim, n_tissues)
                self.hla_classifier = nn.Linear(args.hidden_dim, n_hlas)

        def encode(self, peptide_ids: Any) -> Any:
            # Conv1d with an even kernel and symmetric padding has one extra
            # position.  Cropping restores exactly nine position-aware slots.
            embedded = self.embedding(peptide_ids) + self.position_embedding
            channels_first = embedded.transpose(1, 2)
            features = []
            for convolution in self.convolutions:
                convolved = convolution(channels_first)[..., :self.peptide_length]
                features.append(torch_relu(convolved).transpose(1, 2))
            return self.encoder(torch_cat(features, dim=2))

        def forward(self, peptide_ids: Any, task_ids: Any) -> Any:
            encoded = self.encode(peptide_ids)
            logits = encoded.new_empty(encoded.shape[0])
            for task_id in torch_unique(task_ids):
                mask = task_ids == task_id
                logits[mask] = self.heads[int(task_id.item())](encoded[mask]).squeeze(-1)
            return logits

        def auxiliary_logits(self, peptide_ids: Any) -> tuple[Any, Any]:
            if not self.use_aux:
                raise RuntimeError("Plain HLA branch has no auxiliary classifiers.")
            encoded = self.encode(peptide_ids)
            return self.tissue_classifier(encoded), self.hla_classifier(encoded)

    # The helpers avoid relying on a globally imported torch module, matching
    # the project convention of loading torch lazily through base.require_torch.
    import torch

    def np_to_tensor(values: np.ndarray) -> Any:
        return torch.from_numpy(values)

    def torch_relu(values: Any) -> Any:
        return torch.relu(values)

    def torch_cat(values: list[Any], dim: int) -> Any:
        return torch.cat(values, dim=dim)

    def torch_unique(values: Any) -> Any:
        return torch.unique(values)

    return MultiKernelCnnSharedHeads()


def mapped_arrays(df: pd.DataFrame, task_to_id: dict[str, int], peptide_length: int) -> dict[str, np.ndarray]:
    task_ids = df["task_name"].map(task_to_id)
    if task_ids.isna().any():
        missing = sorted(df.loc[task_ids.isna(), "task_name"].unique())
        raise ValueError(f"Task mapping misses task(s): {missing}")
    return {
        "peptides": base.encode_peptides(df["peptide_sequence"], peptide_length),
        "task_ids": task_ids.to_numpy(dtype=np.int64),
        "tissue_ids": df["tissue_id"].to_numpy(dtype=np.int64),
        "hla_ids": df["hla_id"].to_numpy(dtype=np.int64),
        "labels": df["label"].to_numpy(dtype=np.int64),
    }


def build_loader(
    torch: Any, DataLoader: Any, TensorDataset: Any, arrays: dict[str, np.ndarray], batch_size: int, shuffle: bool,
) -> Any:
    dataset = TensorDataset(
        # encode_peptides may return a read-only view; TensorDataset training
        # requires writable backing arrays to avoid undefined PyTorch behavior.
        torch.as_tensor(arrays["peptides"].copy()), torch.as_tensor(arrays["task_ids"].copy()),
        torch.as_tensor(arrays["tissue_ids"].copy()), torch.as_tensor(arrays["hla_ids"].copy()),
        torch.as_tensor(arrays["labels"].copy()),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_branch(
    args: argparse.Namespace, torch: Any, nn: Any, DataLoader: Any, TensorDataset: Any,
    train_df: pd.DataFrame, task_to_id: dict[str, int], peptide_length: int, device: str,
    branch: str, group_name: str, use_aux: bool, n_tissues: int, n_hlas: int,
) -> Any:
    model = define_cnn_shared_heads_model(
        args, nn, peptide_length, len(task_to_id), n_tissues, n_hlas, use_aux,
    ).to(device)
    arrays = mapped_arrays(train_df, task_to_id, peptide_length)
    loader = build_loader(torch, DataLoader, TensorDataset, arrays, args.batch_size, True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()
    cross_entropy = nn.CrossEntropyLoss()
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        for peptides, task_ids, tissue_ids, hla_ids, labels in loader:
            peptides, task_ids = peptides.to(device), task_ids.to(device)
            tissue_ids, hla_ids, labels = tissue_ids.to(device), hla_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(peptides, task_ids)
            loss = bce(logits, labels.float())
            if use_aux:
                tissue_logits, hla_logits = model.auxiliary_logits(peptides)
                loss = loss + args.tissue_loss_weight * cross_entropy(tissue_logits, tissue_ids)
                loss = loss + args.hla_loss_weight * cross_entropy(hla_logits, hla_ids)
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
        print(
            f"    epoch branch={branch} group={group_name} {epoch}/{args.epochs} "
            f"duration={format_duration(time.perf_counter() - started)}",
            flush=True,
        )
    return model


def predict_branch(
    args: argparse.Namespace, torch: Any, DataLoader: Any, TensorDataset: Any, model: Any,
    train_df: pd.DataFrame, prediction_df: pd.DataFrame, task_to_id: dict[str, int], peptide_length: int, device: str,
) -> pd.DataFrame:
    # ``train_df`` is accepted to make the call site explicitly show which data
    # trained the branch; only prediction_df is encoded and scored here.
    del train_df
    arrays = mapped_arrays(prediction_df, task_to_id, peptide_length)
    loader = build_loader(torch, DataLoader, TensorDataset, arrays, args.batch_size, False)
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for peptides, task_ids, _, _, _ in loader:
            logits = model(peptides.to(device), task_ids.to(device))
            probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
    result = prediction_df[KEYS + ["label"]].copy().reset_index(drop=True)
    result["score"] = np.concatenate(probabilities)
    if len(result) != len(prediction_df):
        raise AssertionError("Prediction length mismatch.")
    return result


def fuse_branches(global_predictions: pd.DataFrame, hla_predictions: pd.DataFrame) -> pd.DataFrame:
    merged = global_predictions.merge(
        hla_predictions, on=KEYS, how="inner", validate="one_to_one", suffixes=("_global", "_hla"),
    )
    if len(merged) != len(global_predictions) or len(merged) != len(hla_predictions):
        raise ValueError("Global/HLA branch predictions do not cover identical samples.")
    if not np.array_equal(merged["label_global"].to_numpy(), merged["label_hla"].to_numpy()):
        raise ValueError("Global/HLA branch labels disagree.")
    output = []
    for _, task in merged.groupby(["target_tissue", "mhc_restriction"], sort=True):
        score = e15.fusion_scores(pd.DataFrame({
            "probability_global_aux": task["score_global"].to_numpy(),
            "probability_hla_plain": task["score_hla"].to_numpy(),
            "logit_global_aux": np.zeros(len(task)),
            "logit_hla_plain": np.zeros(len(task)),
        }))["e15_task_rank_average"]
        item = task[KEYS + ["label_global"]].copy()
        item = item.rename(columns={"label_global": "label"})
        item["score"] = score
        output.append(item)
    return pd.concat(output, ignore_index=True).sort_values(KEYS).reset_index(drop=True)


def predict_seed(
    args: argparse.Namespace, torch: Any, nn: Any, DataLoader: Any, TensorDataset: Any,
    fitting: pd.DataFrame, prediction: pd.DataFrame, mappings: dict[str, Any], peptide_length: int,
    device: str, seed: int, context: str,
) -> pd.DataFrame:
    e14.set_seed(seed, torch)
    print(f"  train E29 seed={seed} context={context} global_aux", flush=True)
    global_model = train_branch(
        args, torch, nn, DataLoader, TensorDataset, fitting, mappings["task_to_id"], peptide_length, device,
        "global_aux", context, True, len(mappings["tissue_to_id"]), len(mappings["hla_to_id"]),
    )
    global_predictions = predict_branch(
        args, torch, DataLoader, TensorDataset, global_model, fitting, prediction,
        mappings["task_to_id"], peptide_length, device,
    )
    hla_parts = []
    hlas = sorted(set(fitting["mhc_restriction"]) & set(prediction["mhc_restriction"]))
    for index, hla in enumerate(hlas, start=1):
        hla_fit = fitting[fitting["mhc_restriction"] == hla].copy()
        hla_prediction = prediction[prediction["mhc_restriction"] == hla].copy()
        tasks = sorted(set(hla_fit["task_name"]) & set(hla_prediction["task_name"]))
        if not tasks:
            continue
        task_to_id = {task: task_index for task_index, task in enumerate(tasks)}
        print(f"  train E29 seed={seed} context={context} hla={index:02d}/{len(hlas)} {hla}", flush=True)
        model = train_branch(
            args, torch, nn, DataLoader, TensorDataset, hla_fit, task_to_id, peptide_length, device,
            "hla_plain", f"{context}_{hla}", False, len(mappings["tissue_to_id"]), len(mappings["hla_to_id"]),
        )
        hla_parts.append(predict_branch(
            args, torch, DataLoader, TensorDataset, model, hla_fit, hla_prediction, task_to_id, peptide_length, device,
        ))
    if not hla_parts:
        raise RuntimeError("No HLA branch predictions were produced.")
    return fuse_branches(global_predictions, pd.concat(hla_parts, ignore_index=True))


def candidate_rows(split: str, candidate: str, frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "split", split)
    result.insert(1, "candidate", candidate)
    result.insert(2, "seed", 0)
    return result[PREDICTION_COLUMNS]


def append_seed_mean(predictions: pd.DataFrame, split: str, seeds: list[int], candidate_prefix: str = "e29_cnn") -> pd.DataFrame:
    if len(seeds) < 2:
        return predictions
    members = [f"{candidate_prefix}_seed_{seed}" for seed in seeds]
    subset = predictions[predictions["candidate"].isin(members)]
    pivot = subset.pivot(index=["split", "seed", *KEYS, "label"], columns="candidate", values="score")
    if pivot.isna().any().any():
        raise AssertionError("E29 seed predictions are not aligned for their mean.")
    mean = pivot.mean(axis=1).rename("score").reset_index()
    mean.insert(1, "candidate", f"{candidate_prefix}_{len(seeds)}seed_mean")
    return pd.concat([predictions, mean[PREDICTION_COLUMNS]], ignore_index=True)


def oof_metrics(labels: pd.DataFrame, scores: np.ndarray) -> dict[str, float]:
    return e26.metric_summary(labels, scores)


def candidate_scores_from_long(frame: pd.DataFrame, candidate: str) -> tuple[pd.DataFrame, np.ndarray]:
    subset = frame[frame["candidate"] == candidate].copy()
    if subset.empty:
        raise ValueError(f"Candidate {candidate!r} is absent.")
    labels, candidates, matrix = e26.aligned_matrix(subset)
    if candidates != [candidate]:
        raise AssertionError("Single-candidate matrix construction failed.")
    return labels, matrix[:, 0]


def screen_against_baselines(args: argparse.Namespace, cnn_oof: pd.DataFrame) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Evaluate pre-registered OOF criteria without opening test predictions."""
    primary = f"{args.candidate_prefix}_seed_{args.seeds[0]}" if len(args.seeds) == 1 else f"{args.candidate_prefix}_{len(args.seeds)}seed_mean"
    cnn_labels, cnn_scores = candidate_scores_from_long(cnn_oof, primary)
    baseline = e26.read_predictions(args.baseline_oof_predictions, "oof")
    # ``--max-tasks`` is useful for a fast model-path smoke test.  Restrict the
    # baseline to exactly the E29 keys before building its aligned matrix; for
    # a formal run this is a no-op because E29 covers all 44 tasks.
    baseline = baseline.merge(cnn_labels[KEYS], on=KEYS, how="inner", validate="many_to_one")
    baseline_labels, baseline_candidates, baseline_matrix = e26.aligned_matrix(baseline)
    for requested in (args.matching_baseline_candidate, args.fusion_baseline_candidate):
        if requested not in baseline_candidates:
            raise ValueError(f"Baseline candidate {requested!r} is absent from {args.baseline_oof_predictions}.")
    merged = cnn_labels.merge(
        baseline_labels, on=e26.KEY_COLUMNS, how="inner", validate="one_to_one", suffixes=("_cnn", "_baseline"),
    )
    if len(merged) != len(cnn_labels) or len(merged) != len(baseline_labels):
        raise ValueError("E29 and E26 OOF samples do not align exactly.")
    if not np.array_equal(merged["label_cnn"].to_numpy(), merged["label_baseline"].to_numpy()):
        raise ValueError("E29 and E26 OOF labels disagree.")
    # Matrices are sorted on the same key order, but reindex explicitly to make
    # the alignment invariant to CSV row order.
    baseline_index = pd.MultiIndex.from_frame(baseline_labels[e26.KEY_COLUMNS])
    cnn_index = pd.MultiIndex.from_frame(cnn_labels[e26.KEY_COLUMNS])
    order = baseline_index.get_indexer(cnn_index)
    if (order < 0).any():
        raise AssertionError("Could not align E26 baseline predictions to E29 OOF keys.")
    matching_scores = baseline_matrix[order, baseline_candidates.index(args.matching_baseline_candidate)]
    fusion_scores = baseline_matrix[order, baseline_candidates.index(args.fusion_baseline_candidate)]
    ranked_cnn = pd.DataFrame({**{column: cnn_labels[column].to_numpy() for column in e26.KEY_COLUMNS}, "score": cnn_scores})
    ranked_match = pd.DataFrame({**{column: cnn_labels[column].to_numpy() for column in e26.KEY_COLUMNS}, "score": matching_scores})
    ranked_fusion = pd.DataFrame({**{column: cnn_labels[column].to_numpy() for column in e26.KEY_COLUMNS}, "score": fusion_scores})
    cnn_rank = ranked_cnn.groupby(e26.TASK_COLUMNS, sort=False)["score"].rank(method="average", pct=True).to_numpy()
    matching_rank = ranked_match.groupby(e26.TASK_COLUMNS, sort=False)["score"].rank(method="average", pct=True).to_numpy()
    fusion_rank = ranked_fusion.groupby(e26.TASK_COLUMNS, sort=False)["score"].rank(method="average", pct=True).to_numpy()
    fused_rank = 0.5 * (cnn_rank + fusion_rank)
    cnn_metric = oof_metrics(cnn_labels, cnn_scores)
    matching_metric = oof_metrics(cnn_labels, matching_scores)
    fusion_metric = oof_metrics(cnn_labels, fusion_scores)
    fused_metric = oof_metrics(cnn_labels, fused_rank)
    correlation = float(np.corrcoef(cnn_rank, matching_rank)[0, 1])
    screen = {
        "primary_candidate": primary,
        "matching_baseline_candidate": args.matching_baseline_candidate,
        "fusion_baseline_candidate": args.fusion_baseline_candidate,
        "cnn_oof": cnn_metric,
        "matching_baseline_oof": matching_metric,
        "fusion_baseline_oof": fusion_metric,
        "equal_rank_fusion_oof": fused_metric,
        "task_rank_correlation_with_matching_baseline": correlation,
        "thresholds": {
            "maximum_standalone_auroc_drop": args.maximum_standalone_auroc_drop,
            "maximum_task_rank_correlation": args.maximum_task_rank_correlation,
            "minimum_fusion_auroc_gain": args.minimum_fusion_auroc_gain,
            "maximum_worst10_auroc_drop": args.maximum_worst10_auroc_drop,
        },
    }
    standalone_drop = matching_metric["mean_auroc"] - cnn_metric["mean_auroc"]
    fusion_gain = fused_metric["mean_auroc"] - fusion_metric["mean_auroc"]
    worst10_drop = fusion_metric["worst_10_mean_auroc"] - fused_metric["worst_10_mean_auroc"]
    checks = {
        "standalone_not_too_weak": standalone_drop <= args.maximum_standalone_auroc_drop,
        "representation_not_too_correlated": correlation < args.maximum_task_rank_correlation,
        "fusion_improves_oof": fusion_gain >= args.minimum_fusion_auroc_gain,
        "worst10_not_too_weak": worst10_drop <= args.maximum_worst10_auroc_drop,
    }
    screen["deltas"] = {
        "matching_baseline_minus_cnn_mean_auroc": standalone_drop,
        "fusion_minus_fusion_baseline_mean_auroc": fusion_gain,
        "fusion_baseline_minus_fusion_worst10_auroc": worst10_drop,
    }
    screen["checks"] = checks
    screen["passed"] = bool(all(checks.values()))

    rows: list[dict[str, object]] = []
    task_frame = cnn_labels.copy()
    task_frame["baseline_score"] = fusion_scores
    task_frame["cnn_score"] = cnn_scores
    task_frame["fused_score"] = fused_rank
    for (tissue, hla), task in task_frame.groupby(["target_tissue", "mhc_restriction"], sort=True):
        y_true = task["label"].to_numpy(dtype=int)
        baseline_auroc = base.evaluate(y_true, task["baseline_score"].to_numpy()) ["auroc"]
        cnn_auroc = base.evaluate(y_true, task["cnn_score"].to_numpy()) ["auroc"]
        fused_auroc = base.evaluate(y_true, task["fused_score"].to_numpy()) ["auroc"]
        rows.append({
            "target_tissue": tissue, "mhc_restriction": hla, "rows": len(task),
            "baseline_auroc": baseline_auroc, "cnn_auroc": cnn_auroc, "fused_auroc": fused_auroc,
            "cnn_minus_baseline_auroc": cnn_auroc - baseline_auroc,
            "fused_minus_baseline_auroc": fused_auroc - baseline_auroc,
        })
    return screen, rows


def generate_oof(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    train = base.read_dataset(args.train)
    # Deliberately do not read args.test here.  OOF selection is finalized
    # before the optional, conditionally gated test stage starts.
    train, _, mappings = base.add_task_columns(train, train.copy())
    if args.max_tasks:
        keep = set(mappings["tasks"][:args.max_tasks])
        train = train[train["task_name"].isin(keep)].copy()
        train, _, mappings = base.add_task_columns(train, train.copy())
    peptide_length = int(train["peptide_sequence"].str.len().max())
    folds = make_pair_grouped_folds(train, args.oof_folds, args.oof_split_seed)
    parts = []
    for fold in range(args.oof_folds):
        fitting, held_out = train[folds != fold].copy(), train[folds == fold].copy()
        print(f"OOF fold={fold + 1}/{args.oof_folds} fit_rows={len(fitting)} holdout_rows={len(held_out)}", flush=True)
        for seed in args.seeds:
            prediction = predict_seed(
                args, torch, nn, DataLoader, TensorDataset, fitting, held_out, mappings, peptide_length,
                device, seed, f"oof_fold_{fold}",
            )
            parts.append(candidate_rows("oof", f"{args.candidate_prefix}_seed_{seed}", prediction))
    oof = append_seed_mean(pd.concat(parts, ignore_index=True), "oof", args.seeds, args.candidate_prefix)
    return oof, {"device": device, "n_tasks": len(mappings["tasks"]), "peptide_length": peptide_length}


def generate_test(args: argparse.Namespace) -> pd.DataFrame:
    """Generate full-train test predictions only after a passing OOF screen."""
    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    train, test = base.read_dataset(args.train), base.read_dataset(args.test)
    train, test, mappings = base.add_task_columns(train, test)
    if args.max_tasks:
        keep = set(mappings["tasks"][:args.max_tasks])
        train, test = train[train["task_name"].isin(keep)].copy(), test[test["task_name"].isin(keep)].copy()
        train, test, mappings = base.add_task_columns(train, test)
    peptide_length = int(max(train["peptide_sequence"].str.len().max(), test["peptide_sequence"].str.len().max()))
    parts = []
    for seed in args.seeds:
        print(f"Full-train test E29 seed={seed}", flush=True)
        prediction = predict_seed(
            args, torch, nn, DataLoader, TensorDataset, train, test, mappings, peptide_length,
            device, seed, "full_train_test",
        )
        parts.append(candidate_rows("test", f"{args.candidate_prefix}_seed_{seed}", prediction))
    return append_seed_mean(pd.concat(parts, ignore_index=True), "test", args.seeds, args.candidate_prefix)


def evaluate_existing_test_predictions(args: argparse.Namespace) -> None:
    """Create formal E29 metric tables from an already fixed test prediction file."""
    test = e26.read_predictions(args.test_predictions_output, "test")
    labels, candidates, matrix = e26.aligned_matrix(test)
    rows: list[dict[str, object]] = []
    single_seed_prefix = f"{args.candidate_prefix}_seed_"
    seed_mean_prefix = f"{args.candidate_prefix}_"
    for candidate_index, candidate in enumerate(candidates):
        if candidate.startswith(single_seed_prefix):
            member_seed = int(candidate.rsplit("_", 1)[1])
            model, summary_seed, n_members, members = "e29_cnn_single_seed", member_seed, 1, str(member_seed)
        elif candidate.startswith(seed_mean_prefix) and candidate.endswith("seed_mean"):
            n_members = int(candidate.removeprefix(seed_mean_prefix).split("seed_", 1)[0])
            model, summary_seed = candidate, 0
            members = ",".join(str(seed) for seed in args.seeds)
        else:
            raise ValueError(f"Unexpected E29 candidate name: {candidate}")
        working = labels.copy()
        working["score"] = matrix[:, candidate_index]
        for (_, tissue, hla), task in working.groupby(e26.TASK_COLUMNS, sort=True):
            y_true = task["label"].to_numpy(dtype=int)
            rows.append({
                "experiment_name": "E29_multikernel_cnn", "seed": summary_seed, "model": model,
                "n_member_seeds": n_members, "member_seeds": members,
                "target_tissue": tissue, "mhc_restriction": hla, "test_rows": len(task),
                "test_positive": int(y_true.sum()), "test_negative": int(len(task) - y_true.sum()),
                **base.evaluate(y_true, task["score"].to_numpy(dtype=float)),
                "fusion_formula": "task_percentile_rank(mean_seed_e29_branch_rank_fusion)" if n_members > 1
                else "task_percentile_rank(e29_global_hla_branch_rank_fusion)",
            })
    summary = base.summarize_results(rows)
    stability = base.summarize_seed_stability(summary)
    base.write_csv(args.per_task_output, PER_TASK_COLUMNS, rows)
    base.write_csv(args.summary_output, base.SUMMARY_COLUMNS, summary)
    base.write_csv(args.stability_output, base.STABILITY_COLUMNS, stability)

    main_model = f"{args.candidate_prefix}_{len(args.seeds)}seed_mean"
    e29_main = pd.DataFrame([row for row in rows if row["model"] == main_model])
    comparison_rows: list[dict[str, object]] = []
    if args.e17_per_task.is_file():
        e17 = pd.read_csv(args.e17_per_task)
        e17_model = f"e17_{len(args.seeds)}seed_rank_average"
        e17 = e17[e17["model"] == e17_model].copy()
        merged = e29_main.merge(
            e17, on=["target_tissue", "mhc_restriction"], how="inner", validate="one_to_one",
            suffixes=("_e29", "_e17"),
        )
        if len(merged) != len(e29_main):
            raise ValueError(
                f"E29 and {e17_model} per-task results do not align on all tasks."
            )
        for item in merged.itertuples(index=False):
            comparison_rows.append({
                "target_tissue": item.target_tissue, "mhc_restriction": item.mhc_restriction,
                "e29_auroc": item.auroc_e29, "e17_auroc": item.auroc_e17,
                "delta_auroc": item.auroc_e29 - item.auroc_e17,
                "e29_auprc": item.auprc_e29, "e17_auprc": item.auprc_e17,
                "delta_auprc": item.auprc_e29 - item.auprc_e17,
                "e29_accuracy": item.accuracy_e29, "e17_accuracy": item.accuracy_e17,
                "delta_accuracy": item.accuracy_e29 - item.accuracy_e17,
                "e29_mcc": item.mcc_e29, "e17_mcc": item.mcc_e17,
                "delta_mcc": item.mcc_e29 - item.mcc_e17,
            })
        base.write_csv(args.e17_comparison_output, E17_COMPARISON_COLUMNS, comparison_rows)
    print(f"wrote: {args.per_task_output}", flush=True)
    print(f"wrote: {args.summary_output}", flush=True)
    print(f"wrote: {args.stability_output}", flush=True)
    if comparison_rows:
        print(f"wrote: {args.e17_comparison_output}", flush=True)


def run(args: argparse.Namespace) -> None:
    if args.oof_folds < 2:
        raise ValueError("--oof-folds must be at least two.")
    if not args.seeds:
        raise ValueError("--seeds must not be empty.")
    if not args.kernel_sizes or any(kernel < 1 for kernel in args.kernel_sizes):
        raise ValueError("--kernel-sizes must contain positive integers.")
    if args.evaluate_existing_test_only:
        evaluate_existing_test_predictions(args)
        return
    started = time.perf_counter()
    oof, details = generate_oof(args)
    args.oof_predictions_output.parent.mkdir(parents=True, exist_ok=True)
    oof.to_csv(args.oof_predictions_output, index=False)
    screen, task_rows = screen_against_baselines(args, oof)
    base.write_csv(args.task_comparison_output, TASK_COMPARISON_COLUMNS, task_rows)
    args.screen_summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.screen_summary_output.write_text(json.dumps({
        "experiment_name": args.experiment_name, "candidate_prefix": args.candidate_prefix, "seeds": args.seeds,
        "oof_folds": args.oof_folds, "oof_split_seed": args.oof_split_seed,
        "architecture": {
            "encoder": "position-preserving multi-kernel Conv1d then Flatten/MLP",
            "embedding_dim": args.embedding_dim, "kernel_sizes": args.kernel_sizes,
            "conv_channels": args.conv_channels, "hidden_dim": args.hidden_dim, "dropout": args.dropout,
        },
        "training": {
            "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay, "tissue_loss_weight": args.tissue_loss_weight,
            "hla_loss_weight": args.hla_loss_weight,
        },
        "test_policy": "Test CSV is read only after the 3-seed OOF screen passes; --no-run-test forces OOF-only execution.",
        "screen": screen, **details,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"OOF screen passed={screen['passed']}; details={args.screen_summary_output}", flush=True)
    if args.run_test:
        if not screen["passed"]:
            raise RuntimeError("E29 OOF screen did not pass; refusing to read or generate test predictions.")
        test_predictions = generate_test(args)
        args.test_predictions_output.parent.mkdir(parents=True, exist_ok=True)
        test_predictions.to_csv(args.test_predictions_output, index=False)
        print(f"wrote: {args.test_predictions_output}", flush=True)
        evaluate_existing_test_predictions(args)
    print(f"run total time: {format_duration(time.perf_counter() - started)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = project_path("results/tissuePMHC_e29_multikernel_cnn_3seed")
    e26_root = project_path("results/tissuePMHC_e26_greedy_ensemble_selection")
    parser.add_argument("--train", type=Path, default=project_path("data/tissuePMHC/tissuePMHC_train.csv.gz"))
    parser.add_argument("--test", type=Path, default=project_path("data/tissuePMHC/tissuePMHC_test.csv.gz"))
    parser.add_argument("--baseline-oof-predictions", type=Path, default=e26_root / "oof_predictions.csv")
    parser.add_argument("--matching-baseline-candidate", default="e14_final_3seed_mean")
    parser.add_argument("--fusion-baseline-candidate", default="e14_final_3seed_mean")
    parser.add_argument("--experiment-name", default="E29_multikernel_cnn_oof_screen")
    parser.add_argument("--candidate-prefix", default="e29_cnn")
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260704, 20260705, 20260706])
    parser.add_argument("--oof-folds", type=int, default=3)
    parser.add_argument("--oof-split-seed", type=int, default=20260711)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--kernel-sizes", nargs="+", type=int, default=[2, 3, 5])
    parser.add_argument("--conv-channels", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--tissue-loss-weight", type=float, default=0.1)
    parser.add_argument("--hla-loss-weight", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--maximum-standalone-auroc-drop", type=float, default=0.005)
    parser.add_argument("--maximum-task-rank-correlation", type=float, default=0.97)
    parser.add_argument("--minimum-fusion-auroc-gain", type=float, default=0.001)
    parser.add_argument("--maximum-worst10-auroc-drop", type=float, default=0.001)
    parser.add_argument("--max-tasks", type=int, default=0, help="Optional smoke-test task limit.")
    parser.add_argument(
        "--run-test", action=argparse.BooleanOptionalAction, default=True,
        help="After a passed OOF screen, train full data and write test predictions; use --no-run-test for OOF only.",
    )
    parser.add_argument(
        "--evaluate-existing-test-only", action="store_true",
        help="Skip all training and create formal metrics from the existing --test-predictions-output file.",
    )
    parser.add_argument("--oof-predictions-output", type=Path, default=root / "oof_predictions.csv")
    parser.add_argument("--test-predictions-output", type=Path, default=root / "test_predictions.csv")
    parser.add_argument("--task-comparison-output", type=Path, default=root / "oof_task_comparison.csv")
    parser.add_argument("--screen-summary-output", type=Path, default=root / "oof_screen_summary.json")
    parser.add_argument("--per-task-output", type=Path, default=root / "per_task_metrics.csv")
    parser.add_argument("--summary-output", type=Path, default=root / "summary_metrics.csv")
    parser.add_argument("--stability-output", type=Path, default=root / "stability_metrics.csv")
    parser.add_argument("--e17-per-task", type=Path, default=project_path("results/tissuePMHC_e17_seed_ensemble/per_task_metrics.csv"))
    parser.add_argument("--e17-comparison-output", type=Path, default=root / "e17_5seed_comparison_metrics.csv")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
