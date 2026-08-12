#!/usr/bin/env python3
"""Run E23: ForkMerge for E14a's global auxiliary branch.

At each merge interval, copies of the global model are trained with several
auxiliary-loss weights.  A pair-grouped, train-internal validation split scores
only primary BCE.  Branches no worse than the fixed-weight reference are merged
by their validation improvement; the independent test set is touched once only.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_tissuepmhc_auxiliary_soft_ensemble as e14
import run_tissuepmhc_e15_fusion_ablation as e15
import run_tissuepmhc_e18_global_weight_selection as e18
import run_tissuepmhc_e21_gradient_similarity_auxiliary as e21
import run_tissuepmhc_neural_baselines_v2 as base

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "E23_forkmerge"
MODEL = "e23_forkmerge_hla_plain_rank_average"
GLOBAL_MODEL = "e23_global_forkmerge"
DIAGNOSTIC_COLUMNS = [
    "experiment_name", "seed", "merge_round", "start_epoch", "end_epoch", "candidate",
    "tissue_weight", "hla_weight", "validation_primary_bce", "reference_primary_bce",
    "improvement_over_reference", "selected", "merge_weight", "train_primary_bce",
    "train_tissue_loss", "train_hla_loss",
]


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def candidate_weights() -> list[tuple[str, float, float]]:
    return [
        ("fixed_0.10_0.10", 0.10, 0.10), ("primary_only", 0.0, 0.0),
        ("tissue_heavy", 0.20, 0.0), ("hla_heavy", 0.0, 0.20),
        ("both_heavy", 0.20, 0.20),
    ]


def task_logits(torch: Any, model: Any, encoded: Any, task_ids: Any) -> Any:
    return e21.task_logits_from_encoded(torch, model, encoded, task_ids)


def train_segment(args: argparse.Namespace, torch: Any, model: Any, loader: Any, device: str,
                  tissue_weight: float, hla_weight: float, epochs: int) -> dict[str, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    values = {"bce": [], "tissue": [], "hla": []}
    model.train()
    for _ in range(epochs):
        for batch in loader:
            peptide_ids, task_ids, tissue_ids, hla_ids, labels = [item.to(device) for item in batch]
            optimizer.zero_grad(set_to_none=True)
            encoded = model.encode(peptide_ids)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(task_logits(torch, model, encoded, task_ids), labels.float())
            tissue = torch.nn.functional.cross_entropy(model.tissue_classifier(encoded), tissue_ids)
            hla = torch.nn.functional.cross_entropy(model.hla_classifier(encoded), hla_ids)
            loss = bce + tissue_weight * tissue + hla_weight * hla
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            values["bce"].append(float(bce.detach().cpu())); values["tissue"].append(float(tissue.detach().cpu())); values["hla"].append(float(hla.detach().cpu()))
    return {key: float(np.mean(value)) for key, value in values.items()}


def validation_primary_bce(args: argparse.Namespace, torch: Any, DataLoader: Any, TensorDataset: Any,
                           model: Any, validation_df: pd.DataFrame, task_to_id: dict[str, int],
                           peptide_length: int, device: str) -> float:
    mapped = e14.e7.prepare_with_mapping(validation_df, task_to_id)
    loader = e14.build_aux_loader(args, torch, DataLoader, TensorDataset, mapped, peptide_length, False)
    model.eval(); per_task: dict[int, list[float]] = {}
    with torch.no_grad():
        for batch in loader:
            peptide_ids, task_ids, _, _, labels = [item.to(device) for item in batch]
            logits = task_logits(torch, model, model.encode(peptide_ids), task_ids)
            losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
            for task_id in torch.unique(task_ids):
                mask = task_ids == task_id
                per_task.setdefault(int(task_id.item()), []).extend(losses[mask].detach().cpu().tolist())
    return float(np.mean([np.mean(losses) for losses in per_task.values()]))


def merged_state(models: list[Any], weights: np.ndarray) -> dict[str, Any]:
    states = [model.state_dict() for model in models]
    result: dict[str, Any] = {}
    for key, value in states[0].items():
        if value.is_floating_point():
            result[key] = sum(float(weight) * state[key] for weight, state in zip(weights, states))
        else:
            result[key] = value.clone()
    return result


def train_forkmerge(args: argparse.Namespace, torch: Any, nn: Any, DataLoader: Any, TensorDataset: Any,
                    fitting_df: pd.DataFrame, validation_df: pd.DataFrame, mappings: dict[str, Any],
                    peptide_length: int, device: str, seed: int) -> tuple[Any, list[dict[str, object]]]:
    mapped = e14.e7.prepare_with_mapping(fitting_df, mappings["task_to_id"])
    loader = e14.build_aux_loader(args, torch, DataLoader, TensorDataset, mapped, peptide_length, True)
    model = e14.define_aux_shared_heads_model(args, torch, nn, peptide_length, len(mappings["task_to_id"]), len(mappings["tissue_to_id"]), len(mappings["hla_to_id"])).to(device)
    diagnostics: list[dict[str, object]] = []
    candidates = candidate_weights()
    for round_index, start_epoch in enumerate(range(1, args.epochs + 1, args.merge_interval), start=1):
        segment_epochs = min(args.merge_interval, args.epochs - start_epoch + 1)
        anchor = copy.deepcopy(model.state_dict())
        rng_state = e21.capture_rng_state(torch)
        branches, rows = [], []
        for name, tissue_weight, hla_weight in candidates:
            model.load_state_dict(anchor)
            e21.restore_rng_state(torch, rng_state)
            stats = train_segment(args, torch, model, loader, device, tissue_weight, hla_weight, segment_epochs)
            score = validation_primary_bce(args, torch, DataLoader, TensorDataset, model, validation_df, mappings["task_to_id"], peptide_length, device)
            branches.append(copy.deepcopy(model))
            rows.append({"candidate": name, "tissue_weight": tissue_weight, "hla_weight": hla_weight, "score": score, **stats})
        reference = next(row["score"] for row in rows if row["candidate"] == "fixed_0.10_0.10")
        eligible = [row for row in rows if row["score"] <= reference + args.selection_tolerance]
        if not eligible:
            eligible = [min(rows, key=lambda row: row["score"])]
        improvements = np.array([reference - row["score"] for row in eligible], dtype=float)
        shifted = improvements - improvements.max()
        merge_weights = np.exp(shifted / args.merge_temperature); merge_weights /= merge_weights.sum()
        index_by_name = {row["candidate"]: index for index, row in enumerate(rows)}
        model.load_state_dict(merged_state([branches[index_by_name[row["candidate"]]] for row in eligible], merge_weights))
        for row in rows:
            selected_index = next((index for index, selected in enumerate(eligible) if selected["candidate"] == row["candidate"]), None)
            merge_weight = float(merge_weights[selected_index]) if selected_index is not None else 0.0
            diagnostics.append({
                "experiment_name": EXPERIMENT, "seed": seed, "merge_round": round_index,
                "start_epoch": start_epoch, "end_epoch": start_epoch + segment_epochs - 1,
                "candidate": row["candidate"], "tissue_weight": row["tissue_weight"], "hla_weight": row["hla_weight"],
                "validation_primary_bce": row["score"], "reference_primary_bce": reference,
                "improvement_over_reference": reference - row["score"], "selected": selected_index is not None,
                "merge_weight": merge_weight, "train_primary_bce": row["bce"], "train_tissue_loss": row["tissue"], "train_hla_loss": row["hla"],
            })
        selected_text = ",".join(f"{row['candidate']}:{weight:.2f}" for row, weight in zip(eligible, merge_weights))
        print(f"  E23 seed={seed} round={round_index} epochs={start_epoch}-{start_epoch + segment_epochs - 1} reference_bce={reference:.5f} merged={selected_text}", flush=True)
    return model, diagnostics


def prediction_frame(args: argparse.Namespace, torch: Any, DataLoader: Any, TensorDataset: Any, model: Any,
                     train_df: pd.DataFrame, test_df: pd.DataFrame, mappings: dict[str, Any], peptide_length: int,
                     device: str, seed: int) -> pd.DataFrame:
    predictions, _ = e14.predict_branch(args, torch, DataLoader, TensorDataset, model, train_df, test_df, mappings["task_to_id"], peptide_length, device, seed, "global_forkmerge", "all_tasks", True)
    frames = []
    for _, task in sorted(predictions.items()):
        test = task["test_task"]
        frames.append(pd.DataFrame({"experiment_name": EXPERIMENT, "seed": seed, "branch": "global_forkmerge", "branch_model": "aux_shared_heads_forkmerge", "sample_id": test["sample_id"].to_numpy(), "target_tissue": test["target_tissue"].to_numpy(), "mhc_restriction": test["mhc_restriction"].to_numpy(), "label": task["y_true"], "probability": task["y_score"], "logit": task["y_logit"]}))
    return pd.concat(frames, ignore_index=True)


def make_rows(reference: pd.DataFrame, predictions: pd.DataFrame) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    keys = ["seed", "sample_id", "target_tissue", "mhc_restriction"]
    candidate = predictions.rename(columns={"label": "label_candidate", "probability": "probability_candidate"})
    merged = reference.merge(candidate[keys + ["label_candidate", "probability_candidate"]], on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(reference) or not np.array_equal(merged.label_global_aux, merged.label_candidate): raise ValueError("E23 prediction alignment failed.")
    rows, global_rows = [], []
    for (seed, tissue, hla), task in merged.groupby(["seed", "target_tissue", "mhc_restriction"], sort=True):
        y = task.label_candidate.to_numpy(dtype=int); probability = task.probability_candidate.to_numpy(dtype=float)
        baseline = e15.fusion_scores(task)["e15_task_rank_average"]
        score = .5 * (pd.Series(probability).rank(method="average", pct=True).to_numpy() + pd.Series(task.probability_hla_plain).rank(method="average", pct=True).to_numpy())
        common = {"experiment_name": EXPERIMENT, "seed": int(seed), "target_tissue": tissue, "mhc_restriction": hla, "test_rows": len(task), "test_positive": int(y.sum()), "test_negative": int(len(y)-y.sum()), "global_branch": "global_forkmerge", "hla_branch": "hla_plain", "fusion_formula": "0.5 * (within_task_rank(global) + within_task_rank(hla))"}
        rows.append({**common, "model": "e14_reference_task_rank_average", **base.evaluate(y, baseline)})
        rows.append({**common, "model": MODEL, **base.evaluate(y, score)})
        global_rows.append({"experiment_name": EXPERIMENT, "seed": int(seed), "model": GLOBAL_MODEL, "target_tissue": tissue, "mhc_restriction": hla, "test_rows": len(task), "test_positive": int(y.sum()), "test_negative": int(len(y)-y.sum()), **base.evaluate(y, probability)})
    return rows, global_rows


def run(args: argparse.Namespace) -> None:
    if not 0 < args.validation_fraction < .5 or args.merge_interval < 1 or args.merge_temperature <= 0: raise ValueError("Invalid validation_fraction, merge_interval, or merge_temperature.")
    torch, nn, DataLoader, TensorDataset = base.require_torch(); device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    e14.set_seed(args.rng_preflight_seed, torch); e21.preflight_rng_roundtrip(torch)
    train_df, test_df = base.read_dataset(args.train), base.read_dataset(args.test); train_df, test_df, mappings = base.add_task_columns(train_df, test_df)
    if args.max_tasks:
        keep=set(mappings["tasks"][:args.max_tasks]); train_df=train_df[train_df.task_name.isin(keep)].copy(); test_df=test_df[test_df.task_name.isin(keep)].copy(); train_df,test_df,mappings=base.add_task_columns(train_df,test_df)
    reference=e15.require_aligned_e14a_predictions(args.e14_branch_predictions); reference=reference[reference.seed.isin(args.seeds)].copy()
    if sorted(reference.seed.astype(int).unique()) != sorted(args.seeds): raise ValueError("Saved E14 predictions lack an E23 seed.")
    reference=reference.merge(test_df[["sample_id","target_tissue","mhc_restriction"]],on=["sample_id","target_tissue","mhc_restriction"],how="inner")
    peptide_length=int(max(train_df.peptide_sequence.str.len().max(),test_df.peptide_sequence.str.len().max()))
    diagnostics=[]; frames=[]
    for seed in args.seeds:
        e14.set_seed(seed, torch); fitting, validation=e18.split_train_validation(train_df,args.validation_fraction,args.validation_split_seed + seed)
        print(f"device={device}; seed={seed}; fit_rows={len(fitting)}; validation_rows={len(validation)}",flush=True)
        model, rows=train_forkmerge(args,torch,nn,DataLoader,TensorDataset,fitting,validation,mappings,peptide_length,device,seed); diagnostics.extend(rows)
        frames.append(prediction_frame(args,torch,DataLoader,TensorDataset,model,fitting,test_df,mappings,peptide_length,device,seed))
    predictions=pd.concat(frames,ignore_index=True); rows,global_rows=make_rows(reference,predictions); summary=base.summarize_results(rows)
    base.write_csv(args.per_task_output,e21.PER_TASK_COLUMNS,rows); base.write_csv(args.summary_output,base.SUMMARY_COLUMNS,summary); base.write_csv(args.stability_output,base.STABILITY_COLUMNS,base.summarize_seed_stability(summary)); base.write_csv(args.global_per_task_output,e21.GLOBAL_PER_TASK_COLUMNS,global_rows); base.write_csv(args.global_summary_output,base.SUMMARY_COLUMNS,base.summarize_results(global_rows)); base.write_csv(args.diagnostics_output,DIAGNOSTIC_COLUMNS,diagnostics); base.write_csv(args.predictions_output,e21.PREDICTION_COLUMNS,predictions.to_dict("records"))
    args.metadata_output.parent.mkdir(parents=True,exist_ok=True); args.metadata_output.write_text(json.dumps({"experiment_name":EXPERIMENT,"device":device,"seeds":args.seeds,"epochs":args.epochs,"validation_fraction":args.validation_fraction,"validation_split":"pair-grouped within task; excluded from updates","merge_interval":args.merge_interval,"candidate_weights":[{"name":n,"tissue":t,"hla":h} for n,t,h in candidate_weights()],"selection":"primary BCE on train-internal validation only; retain candidates no worse than fixed 0.1/0.1 plus tolerance","fixed_hla_source":str(args.e14_branch_predictions)},indent=2,ensure_ascii=False),encoding="utf-8")
    for path in [args.per_task_output,args.summary_output,args.stability_output,args.global_per_task_output,args.global_summary_output,args.diagnostics_output,args.predictions_output,args.metadata_output]: print(f"wrote: {path}",flush=True)


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--train",type=Path,default=project_path("data/tissuePMHC/tissuePMHC_train.csv.gz")); p.add_argument("--test",type=Path,default=project_path("data/tissuePMHC/tissuePMHC_test.csv.gz")); p.add_argument("--e14-branch-predictions",type=Path,default=project_path("results/tissuePMHC_auxiliary_soft_ensemble/branch_predictions.csv")); p.add_argument("--seeds",nargs="+",type=int,default=[20260704]); p.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    p.add_argument("--epochs",type=int,default=25); p.add_argument("--batch-size",type=int,default=512); p.add_argument("--learning-rate",type=float,default=1e-3); p.add_argument("--weight-decay",type=float,default=1e-4); p.add_argument("--embedding-dim",type=int,default=16); p.add_argument("--hidden-dim",type=int,default=128); p.add_argument("--dropout",type=float,default=.2); p.add_argument("--max-grad-norm",type=float,default=1.0)
    p.add_argument("--validation-fraction",type=float,default=.2); p.add_argument("--validation-split-seed",type=int,default=20260723); p.add_argument("--merge-interval",type=int,default=5); p.add_argument("--selection-tolerance",type=float,default=0.0); p.add_argument("--merge-temperature",type=float,default=.002); p.add_argument("--max-tasks",type=int,default=0); p.add_argument("--rng-preflight-seed",type=int,default=20260711)
    root=project_path("results/tissuePMHC_e23_forkmerge"); p.add_argument("--per-task-output",type=Path,default=root/"per_task_metrics.csv"); p.add_argument("--summary-output",type=Path,default=root/"summary_metrics.csv"); p.add_argument("--stability-output",type=Path,default=root/"stability_metrics.csv"); p.add_argument("--global-per-task-output",type=Path,default=root/"global_per_task_metrics.csv"); p.add_argument("--global-summary-output",type=Path,default=root/"global_summary_metrics.csv"); p.add_argument("--diagnostics-output",type=Path,default=root/"forkmerge_diagnostics.csv"); p.add_argument("--predictions-output",type=Path,default=root/"global_branch_predictions.csv"); p.add_argument("--metadata-output",type=Path,default=root/"metadata.json"); return p.parse_args()

if __name__ == "__main__": run(parse_args())
