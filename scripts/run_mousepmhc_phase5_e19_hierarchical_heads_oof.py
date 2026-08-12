#!/usr/bin/env python3
"""Run Phase 5 E19: hierarchical tissue-H2 heads on the E3b MMoE.

The E3b encoder, experts, conditional gate, task-balanced sampler, and training
budget are frozen.  Only the 24 independent heads are replaced by a global
head, centered tissue and H2 main effects, and a centered rank-4 bilinear
tissue-H2 interaction.  A matched E3b is rerun on identical folds, seeds, and
batches.  The fixed test split is never read.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_mousepmhc_phase3_e3_factorized_mmoe_oof as e3
import run_mousepmhc_phase3_e5_famo_mmoe_oof as e5
import run_tissuepmhc_e26_all_in_one as folds
import run_tissuepmhc_neural_baselines_v2 as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "mousePMHC_phase5_e19_hierarchical_heads_oof"
BASELINE = "e3b_matched_control"
CANDIDATE = "e19_hierarchical_tissue_h2_head"
KEYS = ["sample_id", "target_tissue", "mhc_restriction"]
METRICS = ["accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "mcc", "brier"]
COMPONENTS = ["global", "tissue", "h2", "interaction", "total"]


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def define_hierarchical_model(
    torch: Any, nn: Any, peptide_length: int, n_tasks: int, n_tissues: int, n_h2: int,
    args: argparse.Namespace,
) -> Any:
    class HierarchicalHeadMMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.amino_embedding = nn.Embedding(
                len(base.AA_TO_INDEX) + 1, args.embedding_dim, padding_idx=base.PAD_INDEX,
            )
            self.peptide_encoder = nn.Sequential(
                nn.Flatten(), nn.Linear(peptide_length * args.embedding_dim, args.hidden_dim),
                nn.ReLU(), nn.Dropout(args.dropout),
            )
            self.tissue_embedding = nn.Embedding(n_tissues, args.condition_dim)
            self.mhc_embedding = nn.Embedding(n_h2, args.condition_dim)
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(args.hidden_dim, args.expert_dim), nn.ReLU(), nn.Dropout(args.dropout),
                    nn.Linear(args.expert_dim, args.expert_dim), nn.ReLU(),
                )
                for _ in range(args.n_experts)
            ])
            self.gate = nn.Sequential(
                nn.Linear(args.hidden_dim + 2 * args.condition_dim, args.gate_hidden_dim), nn.ReLU(),
                nn.Linear(args.gate_hidden_dim, args.n_experts),
            )
            self.global_head = nn.Linear(args.expert_dim, 1)
            self.tissue_weight = nn.Parameter(torch.zeros(n_tissues, args.expert_dim))
            self.tissue_bias = nn.Parameter(torch.zeros(n_tissues))
            self.h2_weight = nn.Parameter(torch.zeros(n_h2, args.expert_dim))
            self.h2_bias = nn.Parameter(torch.zeros(n_h2))
            # A/B can be nonzero because the zero projection makes the initial
            # interaction logit exactly zero while preserving a learning path.
            self.interaction_tissue_factor = nn.Parameter(torch.empty(n_tissues, args.interaction_rank))
            self.interaction_h2_factor = nn.Parameter(torch.empty(n_h2, args.interaction_rank))
            factor_scale = args.interaction_rank ** -0.5
            nn.init.normal_(self.interaction_tissue_factor, mean=0.0, std=factor_scale)
            nn.init.normal_(self.interaction_h2_factor, mean=0.0, std=factor_scale)
            self.interaction_projection = nn.Parameter(torch.zeros(args.interaction_rank, args.expert_dim))
            self.interaction_bias_projection = nn.Parameter(torch.zeros(args.interaction_rank))

        @staticmethod
        def center_main(values: Any) -> Any:
            return values - values.mean(dim=0, keepdim=True)

        def centered_interaction_coefficients(self) -> Any:
            coefficients = (
                self.interaction_tissue_factor[:, None, :] * self.interaction_h2_factor[None, :, :]
            )
            return (
                coefficients - coefficients.mean(dim=0, keepdim=True)
                - coefficients.mean(dim=1, keepdim=True)
                + coefficients.mean(dim=(0, 1), keepdim=True)
            )

        def effective_parameters(self) -> dict[str, Any]:
            coefficients = self.centered_interaction_coefficients()
            return {
                "tissue_weight": self.center_main(self.tissue_weight),
                "tissue_bias": self.center_main(self.tissue_bias),
                "h2_weight": self.center_main(self.h2_weight),
                "h2_bias": self.center_main(self.h2_bias),
                "interaction_weight": torch.einsum("thr,rd->thd", coefficients, self.interaction_projection),
                "interaction_bias": torch.einsum("thr,r->th", coefficients, self.interaction_bias_projection),
            }

        def regularization_terms(self) -> tuple[Any, Any]:
            main = (
                self.tissue_weight.square().sum() + self.tissue_bias.square().sum()
                + self.h2_weight.square().sum() + self.h2_bias.square().sum()
            )
            interaction = (
                self.interaction_projection.square().sum() + self.interaction_bias_projection.square().sum()
            )
            return main, interaction

        def forward(
            self, peptide_ids: Any, task_ids: Any, tissue_ids: Any, h2_ids: Any,
            return_gates: bool = False, return_components: bool = False,
        ) -> Any:
            del task_ids  # hierarchy is addressed directly by tissue/H2 ids
            peptide = self.peptide_encoder(self.amino_embedding(peptide_ids))
            expert_outputs = torch.stack([expert(peptide) for expert in self.experts], dim=1)
            gate_input = torch.cat([
                peptide, self.tissue_embedding(tissue_ids), self.mhc_embedding(h2_ids),
            ], dim=1)
            gates = torch.softmax(self.gate(gate_input), dim=1)
            mixed = (expert_outputs * gates.unsqueeze(-1)).sum(dim=1)
            effective = self.effective_parameters()
            global_logit = self.global_head(mixed).squeeze(-1)
            tissue_logit = (
                mixed * effective["tissue_weight"][tissue_ids]
            ).sum(dim=1) + effective["tissue_bias"][tissue_ids]
            h2_logit = (
                mixed * effective["h2_weight"][h2_ids]
            ).sum(dim=1) + effective["h2_bias"][h2_ids]
            interaction_weight = effective["interaction_weight"][tissue_ids, h2_ids]
            interaction_logit = (
                mixed * interaction_weight
            ).sum(dim=1) + effective["interaction_bias"][tissue_ids, h2_ids]
            total = global_logit + tissue_logit + h2_logit + interaction_logit
            if return_components:
                components = {
                    "global": global_logit, "tissue": tissue_logit, "h2": h2_logit,
                    "interaction": interaction_logit, "total": total,
                }
                return total, gates, components
            return (total, gates) if return_gates else total

    return HierarchicalHeadMMoE()


def train_fold(
    args: argparse.Namespace, torch: Any, nn: Any, fitting: pd.DataFrame, mappings: dict[str, Any],
    peptide_length: int, device: str, seed: int, fold: int, candidate: str,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    if candidate == BASELINE:
        model = e3.define_model(
            torch, nn, peptide_length, len(mappings["tasks"]), len(mappings["tissue_to_id"]),
            len(mappings["hla_to_id"]), args,
        ).to(device)
    else:
        model = define_hierarchical_model(
            torch, nn, peptide_length, len(mappings["tasks"]), len(mappings["tissue_to_id"]),
            len(mappings["hla_to_id"]), args,
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    arrays = e5.task_arrays(fitting, mappings, peptide_length)
    n_tasks = len(arrays)
    steps = args.steps_per_epoch or int(np.ceil(max(len(item["label"]) for item in arrays) / args.task_batch_size))
    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    fold_started = time.perf_counter()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train(); bces: list[float] = []; entropies: list[float] = []
        main_penalties: list[float] = []; interaction_penalties: list[float] = []
        for _ in range(steps):
            batch = e5.sample_balanced_batch(rng, arrays, args.task_batch_size)
            peptide, task, tissue, h2, label = [torch.as_tensor(value, device=device) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            losses, gates = e5.task_loss_vector(
                torch, model, peptide, task, tissue, h2, label, n_tasks, args.task_batch_size,
            )
            entropy = -(gates * torch.log(gates.clamp_min(1e-12))).sum(dim=1).mean()
            if candidate == CANDIDATE:
                main_raw, interaction_raw = model.regularization_terms()
                main_penalty = args.main_effect_l2 * main_raw
                interaction_penalty = args.interaction_l2 * interaction_raw
            else:
                main_penalty = losses.new_zeros(()); interaction_penalty = losses.new_zeros(())
            objective = losses.mean() - args.gate_entropy_weight * entropy + main_penalty + interaction_penalty
            objective.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            bces.append(float(losses.mean().detach().cpu())); entropies.append(float(entropy.detach().cpu()))
            main_penalties.append(float(main_penalty.detach().cpu()))
            interaction_penalties.append(float(interaction_penalty.detach().cpu()))
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        epoch_seconds = time.perf_counter() - epoch_started
        history.append({
            "experiment_name": EXPERIMENT, "candidate": candidate, "seed": seed, "fold": fold,
            "epoch": epoch, "mean_task_bce": float(np.mean(bces)),
            "mean_gate_entropy": float(np.mean(entropies)),
            "main_effect_penalty": float(np.mean(main_penalties)),
            "interaction_penalty": float(np.mean(interaction_penalties)),
        })
        print(
            f"E19 candidate={candidate} seed={seed} fold={fold + 1}/{args.oof_folds} "
            f"epoch={epoch}/{args.epochs} bce={np.mean(bces):.4f} "
            f"main_penalty={np.mean(main_penalties):.6f} interaction_penalty={np.mean(interaction_penalties):.6f} "
            f"epoch_seconds={epoch_seconds:.3f}", flush=True,
        )
    resource = {
        "candidate": candidate, "seed": seed, "fold": fold,
        "wall_seconds": time.perf_counter() - fold_started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0,
    }
    return model, history, resource


def collect_mechanism_diagnostics(
    args: argparse.Namespace, torch: Any, DataLoader: Any, TensorDataset: Any, model: Any,
    fitting: pd.DataFrame, peptide_length: int, device: str, seed: int, fold: int,
) -> list[dict[str, Any]]:
    loader = e3.build_loader(torch, DataLoader, TensorDataset, fitting, peptide_length, args.batch_size, False)
    sums = {task_id: {name: 0.0 for name in COMPONENTS} for task_id in sorted(fitting.task_id.unique())}
    counts = {task_id: 0 for task_id in sums}
    model.eval()
    with torch.no_grad():
        for peptide, task, tissue, h2, _ in loader:
            peptide, task, tissue, h2 = [value.to(device) for value in (peptide, task, tissue, h2)]
            _, _, components = model(peptide, task, tissue, h2, return_components=True)
            for task_id in torch.unique(task):
                numeric_id = int(task_id.item()); mask = task == task_id
                counts[numeric_id] += int(mask.sum().item())
                for name in COMPONENTS:
                    sums[numeric_id][name] += float(components[name][mask].square().sum().cpu())
        effective = {key: value.detach().cpu() for key, value in model.effective_parameters().items()}
        global_parameter_norm = float(torch.sqrt(
            model.global_head.weight.detach().cpu().square().sum()
            + model.global_head.bias.detach().cpu().square().sum()
        ))
    lookup = fitting[[
        "task_id", "task_name", "target_tissue", "mhc_restriction", "tissue_id", "hla_id",
    ]].drop_duplicates().set_index("task_id")
    pair_counts = fitting.groupby("task_id").pair_id.nunique()
    small_cutoff = float(pair_counts.quantile(0.25))
    rows: list[dict[str, Any]] = []
    for task_id in sorted(sums):
        info = lookup.loc[task_id]; tissue_id = int(info.tissue_id); h2_id = int(info.hla_id)
        tissue_norm = float(torch.sqrt(
            effective["tissue_weight"][tissue_id].square().sum() + effective["tissue_bias"][tissue_id].square()
        ))
        h2_norm = float(torch.sqrt(
            effective["h2_weight"][h2_id].square().sum() + effective["h2_bias"][h2_id].square()
        ))
        interaction_norm = float(torch.sqrt(
            effective["interaction_weight"][tissue_id, h2_id].square().sum()
            + effective["interaction_bias"][tissue_id, h2_id].square()
        ))
        row: dict[str, Any] = {
            "experiment_name": EXPERIMENT, "candidate": CANDIDATE, "seed": seed, "fold": fold,
            "task_name": info.task_name, "target_tissue": info.target_tissue,
            "mhc_restriction": info.mhc_restriction, "fitting_pairs": int(pair_counts.loc[task_id]),
            "small_sample_task": bool(pair_counts.loc[task_id] <= small_cutoff),
            "global_parameter_norm": global_parameter_norm,
            "tissue_parameter_norm": tissue_norm, "h2_parameter_norm": h2_norm,
            "interaction_parameter_norm": interaction_norm,
            "non_global_parameter_norm": float(np.sqrt(tissue_norm ** 2 + h2_norm ** 2 + interaction_norm ** 2)),
        }
        for name in COMPONENTS:
            row[f"{name}_logit_rms"] = float(np.sqrt(sums[task_id][name] / counts[task_id]))
        row["interaction_to_global_logit_ratio"] = row["interaction_logit_rms"] / max(row["global_logit_rms"], 1e-12)
        rows.append(row)
    return rows


def evaluate_with_brier(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    result = base.evaluate(labels, scores)
    result["brier"] = float(np.mean((scores - labels) ** 2))
    return result


def metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for (candidate, seed, tissue, h2), task in predictions.groupby(
        ["candidate", "seed", "target_tissue", "mhc_restriction"], sort=True,
    ):
        rows.append({
            "experiment_name": EXPERIMENT, "candidate": candidate, "seed": int(seed),
            "target_tissue": tissue, "mhc_restriction": h2, "oof_rows": len(task),
            **evaluate_with_brier(task.label.to_numpy(dtype=int), task.score.to_numpy(dtype=float)),
        })
    per_task = pd.DataFrame(rows)
    summaries: list[dict[str, Any]] = []
    for (candidate, seed), subset in per_task.groupby(["candidate", "seed"], sort=True):
        row: dict[str, Any] = {"experiment_name": EXPERIMENT, "candidate": candidate, "seed": int(seed)}
        for metric in METRICS:
            row[f"mean_task_{metric}"] = float(subset[metric].mean())
        row["worst_task_auroc"] = float(subset.auroc.min())
        row["worst6_task_auroc"] = float(subset.nsmallest(6, "auroc").auroc.mean())
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    stability_rows: list[dict[str, Any]] = []
    numeric = [column for column in summary if column.startswith("mean_task_") or column.startswith("worst")]
    for candidate, subset in summary.groupby("candidate", sort=True):
        for metric in numeric:
            stability_rows.append({
                "experiment_name": EXPERIMENT, "candidate": candidate, "metric": metric,
                "seed_count": len(subset), "seed_mean": float(subset[metric].mean()),
                "seed_sd": float(subset[metric].std(ddof=1)) if len(subset) > 1 else float("nan"),
                "seed_min": float(subset[metric].min()), "seed_max": float(subset[metric].max()),
            })
    return per_task, summary, pd.DataFrame(stability_rows)


def paired_deltas(per_task: pd.DataFrame) -> pd.DataFrame:
    keys = ["seed", "target_tissue", "mhc_restriction"]
    baseline = per_task[per_task.candidate == BASELINE].set_index(keys)
    candidate = per_task[per_task.candidate == CANDIDATE].set_index(keys)
    rows: list[dict[str, Any]] = []
    for key in baseline.index.intersection(candidate.index):
        row: dict[str, Any] = {
            "candidate": CANDIDATE, "baseline_candidate": BASELINE, "seed": int(key[0]),
            "target_tissue": key[1], "mhc_restriction": key[2],
        }
        for metric in METRICS:
            row[f"delta_{metric}"] = float(candidate.loc[key, metric] - baseline.loc[key, metric])
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, iterations: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    means = np.empty(iterations, dtype=float)
    for start in range(0, iterations, 1000):
        stop = min(start + 1000, iterations)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def comparison_tables(
    per_task: pd.DataFrame, summary: pd.DataFrame, args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    deltas = paired_deltas(per_task)
    averaged = deltas.groupby(["target_tissue", "mhc_restriction"], sort=True)[
        [f"delta_{metric}" for metric in METRICS]
    ].mean().reset_index()
    rng = np.random.default_rng(args.bootstrap_seed)
    bootstrap_rows: list[dict[str, Any]] = []
    for metric in METRICS:
        lower, upper = bootstrap_ci(averaged[f"delta_{metric}"].to_numpy(dtype=float), rng, args.bootstrap_iterations)
        bootstrap_rows.append({
            "candidate": CANDIDATE, "baseline_candidate": BASELINE, "metric": metric,
            "task_mean_delta": float(averaged[f"delta_{metric}"].mean()),
            "ci95_lower": lower, "ci95_upper": upper, "bootstrap_iterations": args.bootstrap_iterations,
        })
    bootstrap = pd.DataFrame(bootstrap_rows)
    h2 = averaged.groupby("mhc_restriction", sort=True)[[f"delta_{metric}" for metric in METRICS]].mean().reset_index()
    candidate_summary = summary[summary.candidate == CANDIDATE].set_index("seed")
    baseline_summary = summary[summary.candidate == BASELINE].set_index("seed")
    common = candidate_summary.index.intersection(baseline_summary.index)
    auroc_ci = bootstrap.set_index("metric").loc["auroc"]
    mcc_ci = bootstrap.set_index("metric").loc["mcc"]
    decision = {
        "candidate": CANDIDATE, "matched_seeds": [int(value) for value in common],
        "mean_task_auroc_delta": float((candidate_summary.loc[common, "mean_task_auroc"] - baseline_summary.loc[common, "mean_task_auroc"]).mean()),
        "mean_task_auprc_delta": float((candidate_summary.loc[common, "mean_task_auprc"] - baseline_summary.loc[common, "mean_task_auprc"]).mean()),
        "worst6_task_auroc_delta": float((candidate_summary.loc[common, "worst6_task_auroc"] - baseline_summary.loc[common, "worst6_task_auroc"]).mean()),
        "worst_task_auroc_delta": float((candidate_summary.loc[common, "worst_task_auroc"] - baseline_summary.loc[common, "worst_task_auroc"]).mean()),
        "mean_task_mcc_delta": float(averaged.delta_mcc.mean()),
        "mean_task_brier_delta": float(averaged.delta_brier.mean()),
        "task_auroc_wins": int((averaged.delta_auroc > 0).sum()),
        "task_auroc_losses": int((averaged.delta_auroc < 0).sum()),
        "minimum_h2_auroc_delta": float(h2.delta_auroc.min()),
        "auroc_bootstrap_ci95": [float(auroc_ci.ci95_lower), float(auroc_ci.ci95_upper)],
        "mcc_bootstrap_ci95": [float(mcc_ci.ci95_lower), float(mcc_ci.ci95_upper)],
    }
    return deltas, h2, bootstrap, decision


def mechanism_decision(diagnostics: pd.DataFrame) -> dict[str, Any]:
    if diagnostics.empty:
        return {"mechanism_pass": False, "reason": "no candidate diagnostics"}
    interaction_global_ratio = float(diagnostics.interaction_logit_rms.mean() / max(diagnostics.global_logit_rms.mean(), 1e-12))
    interaction_norm_median = float(diagnostics.interaction_parameter_norm.median())
    interaction_norm_ratio = float(
        diagnostics.interaction_parameter_norm.max() / max(interaction_norm_median, 1e-12)
    )
    small = diagnostics[diagnostics.small_sample_task]
    other = diagnostics[~diagnostics.small_sample_task]
    small_dominance = float(
        small.non_global_parameter_norm.max() / max(other.non_global_parameter_norm.median(), 1e-12)
    )
    checks = {
        "interaction_logit_not_dominant": interaction_global_ratio <= 1.0,
        "small_sample_task_not_dominant": small_dominance <= 3.0,
    }
    return {
        "mean_interaction_to_global_logit_ratio": interaction_global_ratio,
        "max_to_median_interaction_parameter_norm_ratio": interaction_norm_ratio,
        "small_task_max_to_other_median_non_global_norm_ratio": small_dominance,
        "checks": checks, "mechanism_pass": all(checks.values()),
    }


def screening_pass(decision: dict[str, Any]) -> tuple[bool, dict[str, bool]]:
    checks = {
        "auroc_noninferiority": decision["mean_task_auroc_delta"] >= -0.001,
        "auprc_protection": decision["mean_task_auprc_delta"] >= -0.001,
        "worst6_protection": decision["worst6_task_auroc_delta"] >= -0.003,
        "mcc_not_significantly_worse": decision["mcc_bootstrap_ci95"][1] >= 0.0,
    }
    return all(checks.values()), checks


def formal_pass(decision: dict[str, Any], mechanism: dict[str, Any]) -> tuple[bool, dict[str, bool]]:
    checks = {
        "auroc_gain": decision["mean_task_auroc_delta"] >= 0.003,
        "auprc_protection": decision["mean_task_auprc_delta"] >= -0.001,
        "worst6_protection": decision["worst6_task_auroc_delta"] >= -0.003,
        "h2_protection": decision["minimum_h2_auroc_delta"] >= -0.010,
        "task_wins": decision["task_auroc_wins"] >= 14,
        "auroc_ci_lower": decision["auroc_bootstrap_ci95"][0] > -0.001,
        "mcc_not_significantly_worse": decision["mcc_bootstrap_ci95"][1] >= 0.0,
        "mechanism_pass": bool(mechanism["mechanism_pass"]),
    }
    return all(checks.values()), checks


def run_candidate_seed(
    args: argparse.Namespace, torch: Any, nn: Any, DataLoader: Any, TensorDataset: Any,
    train: pd.DataFrame, assignments: pd.Series, mappings: dict[str, Any], peptide_length: int,
    device: str, seed: int, candidate: str,
) -> tuple[list[pd.DataFrame], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[pd.DataFrame] = []; history: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []; resources: list[dict[str, Any]] = []
    for fold in range(args.oof_folds):
        fitting, held_out = train[assignments != fold].copy(), train[assignments == fold].copy()
        base.set_seed(seed, torch)
        print(f"E19 candidate={candidate} seed={seed} fold={fold + 1}/{args.oof_folds} fit={len(fitting)} holdout={len(held_out)}", flush=True)
        model, fold_history, resource = train_fold(
            args, torch, nn, fitting, mappings, peptide_length, device, seed, fold, candidate,
        )
        scores = e5.predict(torch, DataLoader, TensorDataset, model, held_out, peptide_length, args.batch_size, device)
        output = held_out[KEYS + ["label"]].copy()
        output.insert(0, "split", "oof"); output.insert(1, "candidate", candidate); output.insert(2, "seed", seed)
        output["score"] = scores
        predictions.append(output[["split", "candidate", "seed", *KEYS, "label", "score"]])
        history.extend(fold_history); resources.append(resource)
        if candidate == CANDIDATE:
            diagnostics.extend(collect_mechanism_diagnostics(
                args, torch, DataLoader, TensorDataset, model, fitting, peptide_length, device, seed, fold,
            ))
    return predictions, history, diagnostics, resources


def validate_protocol(args: argparse.Namespace) -> None:
    if not args.seeds:
        raise ValueError("At least one seed is required.")
    if args.interaction_rank != 4:
        raise ValueError("E19 interaction rank is preregistered as 4.")
    if args.interaction_l2 <= args.main_effect_l2:
        raise ValueError("E19 interaction regularization must be stronger than main-effect regularization.")
    if args.n_experts != 3:
        raise ValueError("E19 is frozen to the three-expert E3b architecture.")


def run(args: argparse.Namespace) -> None:
    validate_protocol(args)
    total_started = time.perf_counter()
    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    raw = base.read_dataset(args.train); e3.validate_input(raw)
    train, _, mappings = base.add_task_columns(raw, raw.copy())
    if len(mappings["tasks"]) != 24:
        raise ValueError(f"E19 expects 24 tasks, found {len(mappings['tasks'])}.")
    peptide_length = int(train.peptide_sequence.str.len().max())
    assignments = folds.make_pair_grouped_folds(train, args.oof_folds, args.oof_split_seed)
    prediction_parts: list[pd.DataFrame] = []; history: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []; resources: list[dict[str, Any]] = []
    screen_seed = int(args.seeds[0]); seed_started = time.perf_counter()
    for candidate in [BASELINE, CANDIDATE]:
        parts, records, mechanism_rows, resource_rows = run_candidate_seed(
            args, torch, nn, DataLoader, TensorDataset, train, assignments, mappings,
            peptide_length, device, screen_seed, candidate,
        )
        prediction_parts.extend(parts); history.extend(records)
        diagnostics.extend(mechanism_rows); resources.extend(resource_rows)
    print(f"E19 seed={screen_seed} seed_seconds={time.perf_counter() - seed_started:.3f}", flush=True)
    screen_predictions = pd.concat(prediction_parts, ignore_index=True)
    screen_per_task, screen_summary, _ = metric_tables(screen_predictions)
    _, _, _, screen_decision = comparison_tables(screen_per_task, screen_summary, args)
    passed_screen, screen_checks = screening_pass(screen_decision)
    print(f"E19 screening seed={screen_seed} delta_auroc={screen_decision['mean_task_auroc_delta']:.6f} pass={passed_screen}", flush=True)
    if passed_screen:
        for seed_value in args.seeds[1:]:
            seed = int(seed_value); seed_started = time.perf_counter()
            for candidate in [BASELINE, CANDIDATE]:
                parts, records, mechanism_rows, resource_rows = run_candidate_seed(
                    args, torch, nn, DataLoader, TensorDataset, train, assignments, mappings,
                    peptide_length, device, seed, candidate,
                )
                prediction_parts.extend(parts); history.extend(records)
                diagnostics.extend(mechanism_rows); resources.extend(resource_rows)
            print(f"E19 seed={seed} seed_seconds={time.perf_counter() - seed_started:.3f}", flush=True)
    else:
        print("E19 screening failed; additional seeds are not run by the preregistered stopping rule.", flush=True)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    if predictions.duplicated(["candidate", "seed", "sample_id"]).any():
        raise AssertionError("Duplicate E19 candidate/seed/sample prediction.")
    if not (predictions.groupby(["candidate", "seed"]).size() == len(train)).all():
        raise AssertionError("Each executed E19 candidate/seed must cover all train rows once.")
    per_task, summary, stability = metric_tables(predictions)
    deltas, h2, bootstrap, decision = comparison_tables(per_task, summary, args)
    diagnostic_table = pd.DataFrame(diagnostics)
    mechanism = mechanism_decision(diagnostic_table)
    decision["screen_seed"] = screen_seed; decision["screen_pass"] = passed_screen
    decision["screen_checks"] = screen_checks; decision["mechanism"] = mechanism
    if passed_screen and len(args.seeds) >= 3:
        passed_formal, formal_checks = formal_pass(decision, mechanism)
        decision["formal_three_seed_pass"] = passed_formal; decision["formal_checks"] = formal_checks
        decision["final_status"] = "advance_to_e24_candidate_pool" if passed_formal else "stop_after_three_seed_confirmation"
    elif passed_screen:
        decision["formal_three_seed_pass"] = None; decision["formal_checks"] = {}
        decision["final_status"] = "screen_passed_but_three_seed_confirmation_not_requested"
    else:
        decision["formal_three_seed_pass"] = False; decision["formal_checks"] = {}
        decision["final_status"] = "stopped_after_single_seed_screen"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "mousePMHC_phase5_e19"
    predictions.to_csv(args.output_dir / f"{prefix}_oof_predictions.csv", index=False)
    per_task.to_csv(args.output_dir / f"{prefix}_oof_per_task_metrics.csv", index=False)
    summary.to_csv(args.output_dir / f"{prefix}_oof_summary_metrics.csv", index=False)
    stability.to_csv(args.output_dir / f"{prefix}_oof_stability_metrics.csv", index=False)
    pd.DataFrame(history).to_csv(args.output_dir / f"{prefix}_training_history.csv", index=False)
    diagnostic_table.to_csv(args.output_dir / f"{prefix}_mechanism_diagnostics.csv", index=False)
    pd.DataFrame(resources).to_csv(args.output_dir / f"{prefix}_resource_usage.csv", index=False)
    deltas.to_csv(args.output_dir / f"{prefix}_paired_task_deltas.csv", index=False)
    h2.to_csv(args.output_dir / f"{prefix}_h2_macro_deltas.csv", index=False)
    bootstrap.to_csv(args.output_dir / f"{prefix}_task_paired_bootstrap.csv", index=False)
    summary[summary.candidate == BASELINE].to_csv(args.output_dir / f"{prefix}_matched_e3b_baseline.csv", index=False)
    (args.output_dir / f"{prefix}_decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    baseline_probe = e3.define_model(
        torch, nn, peptide_length, len(mappings["tasks"]), len(mappings["tissue_to_id"]),
        len(mappings["hla_to_id"]), args,
    )
    candidate_probe = define_hierarchical_model(
        torch, nn, peptide_length, len(mappings["tasks"]), len(mappings["tissue_to_id"]),
        len(mappings["hla_to_id"]), args,
    )
    metadata = {
        "experiment_name": EXPERIMENT, "test_data_read": False, "train": str(args.train),
        "n_rows": len(train), "n_pairs": int(train.pair_id.nunique()), "n_tasks": len(mappings["tasks"]),
        "n_tissues": len(mappings["tissue_to_id"]), "n_h2": len(mappings["hla_to_id"]),
        "seeds_requested": args.seeds,
        "seeds_executed": sorted(int(value) for value in predictions[predictions.candidate == CANDIDATE].seed.unique()),
        "oof_folds": args.oof_folds, "oof_split_seed": args.oof_split_seed, "epochs": args.epochs,
        "backbone": "frozen E3b Factorized MMoE; only final head differs",
        "head_formula": "global + centered tissue + centered H2 + double-centered rank-4 bilinear interaction",
        "zero_initialization": "tissue/H2 main weights and biases plus interaction output projection are zero",
        "interaction_rank": args.interaction_rank, "main_effect_l2": args.main_effect_l2,
        "interaction_l2": args.interaction_l2,
        "baseline_parameter_count": sum(parameter.numel() for parameter in baseline_probe.parameters()),
        "candidate_parameter_count": sum(parameter.numel() for parameter in candidate_probe.parameters()),
        "device": device, "git_commit": git_commit(), "cli": [sys.executable, *sys.argv],
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                        "numpy": np.__version__, "pandas": pd.__version__, "cuda_available": bool(torch.cuda.is_available()),
                        "cuda_version": torch.version.cuda, "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV")},
        "final_status": decision["final_status"], "total_wall_seconds": time.perf_counter() - total_started,
    }
    (args.output_dir / f"{prefix}_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(decision, indent=2, ensure_ascii=False), flush=True)
    print(f"E19 total_seconds={time.perf_counter() - total_started:.3f}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=project_path("data/mousePMHC/mousePMHC_train.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=project_path("results/mousePMHC_phase5_e19_hierarchical_heads_oof"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260704, 20260705, 20260706])
    parser.add_argument("--oof-folds", type=int, default=3); parser.add_argument("--oof-split-seed", type=int, default=20260711)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=25); parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--task-batch-size", type=int, default=16); parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3); parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=16); parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--expert-dim", type=int, default=64); parser.add_argument("--condition-dim", type=int, default=16)
    parser.add_argument("--gate-hidden-dim", type=int, default=64); parser.add_argument("--n-experts", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2); parser.add_argument("--gate-entropy-weight", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--interaction-rank", type=int, default=4)
    parser.add_argument("--main-effect-l2", type=float, default=1e-4)
    parser.add_argument("--interaction-l2", type=float, default=5e-4)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000); parser.add_argument("--bootstrap-seed", type=int, default=20260719)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
