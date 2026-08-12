#!/usr/bin/env python3
"""Run E24: Auto-Lambda auxiliary weighting for E14a's global branch.

Primary BCE always has weight 1.  Tissue/HLA weights are sigmoid-parameterised
and updated from a one-step virtual model using only pair-grouped, train-inner
validation primary BCE.  Test labels never enter weight updates.
"""
from __future__ import annotations

import argparse
import json
import math
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
EXPERIMENT = "E24_auto_lambda"
MODEL = "e24_auto_lambda_hla_plain_rank_average"
GLOBAL_MODEL = "e24_global_auto_lambda"
DIAGNOSTIC_COLUMNS = [
    "experiment_name", "seed", "epoch", "mean_total_loss", "mean_bce_loss", "mean_tissue_loss", "mean_hla_loss",
    "mean_tissue_weight", "mean_hla_weight", "last_tissue_weight", "last_hla_weight", "mean_meta_validation_bce",
    "meta_updates", "min_tissue_weight", "max_tissue_weight", "min_hla_weight", "max_hla_weight",
]


def project_path(relative: str) -> Path: return PROJECT_ROOT / relative


def inverse_sigmoid(value: float) -> float:
    return math.log(value / (1.0 - value))


def auxiliary_weights(torch: Any, logits: Any, maximum: float) -> Any:
    return maximum * torch.sigmoid(logits)


def losses_for_batch(torch: Any, model: Any, batch: Any, device: str) -> tuple[Any, Any, Any]:
    peptide_ids, task_ids, tissue_ids, hla_ids, labels = [item.to(device) for item in batch]
    encoded = model.encode(peptide_ids)
    logits = e21.task_logits_from_encoded(torch, model, encoded, task_ids)
    return (
        torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float()),
        torch.nn.functional.cross_entropy(model.tissue_classifier(encoded), tissue_ids),
        torch.nn.functional.cross_entropy(model.hla_classifier(encoded), hla_ids),
    )


def next_batch(iterator: Any, loader: Any) -> tuple[Any, Any]:
    try: return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def meta_update(args: argparse.Namespace, torch: Any, model: Any, train_batch: Any, validation_batch: Any,
                alpha: Any, alpha_optimizer: Any, device: str) -> tuple[float, np.ndarray]:
    """One differentiable SGD virtual step, then validation-primary meta update."""
    model.train()
    bce, tissue, hla = losses_for_batch(torch, model, train_batch, device)
    weights = auxiliary_weights(torch, alpha, args.max_auxiliary_weight)
    inner_loss = bce + weights[0] * tissue + weights[1] * hla
    parameters = dict(model.named_parameters())
    gradients = torch.autograd.grad(inner_loss, tuple(parameters.values()), create_graph=True, allow_unused=True)
    virtual = {name: parameter - args.meta_inner_lr * gradient if gradient is not None else parameter
               for (name, parameter), gradient in zip(parameters.items(), gradients)}
    val_peptide, val_task, _, _, val_label = [item.to(device) for item in validation_batch]
    was_training = model.training; model.eval()
    try:
        try:
            from torch.func import functional_call
        except ImportError:
            from torch.nn.utils.stateless import functional_call
        val_logits = functional_call(model, virtual, (val_peptide, val_task))
        meta_loss = torch.nn.functional.binary_cross_entropy_with_logits(val_logits, val_label.float())
    finally:
        model.train(was_training)
    alpha_optimizer.zero_grad(set_to_none=True)
    meta_gradient = torch.autograd.grad(meta_loss, alpha, allow_unused=False)[0]
    alpha.grad = meta_gradient.detach()
    alpha_optimizer.step()
    return float(meta_loss.detach().cpu()), weights.detach().cpu().numpy()


def train_auto_lambda(args: argparse.Namespace, torch: Any, nn: Any, DataLoader: Any, TensorDataset: Any,
                      fitting_df: pd.DataFrame, validation_df: pd.DataFrame, mappings: dict[str, Any],
                      peptide_length: int, device: str, seed: int) -> tuple[Any, list[dict[str, object]]]:
    fitting = e14.e7.prepare_with_mapping(fitting_df, mappings["task_to_id"])
    validation = e14.e7.prepare_with_mapping(validation_df, mappings["task_to_id"])
    loader = e14.build_aux_loader(args, torch, DataLoader, TensorDataset, fitting, peptide_length, True)
    validation_loader = e14.build_aux_loader(args, torch, DataLoader, TensorDataset, validation, peptide_length, True)
    model = e14.define_aux_shared_heads_model(args, torch, nn, peptide_length, len(mappings["task_to_id"]), len(mappings["tissue_to_id"]), len(mappings["hla_to_id"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    initial_fraction = args.initial_auxiliary_weight / args.max_auxiliary_weight
    alpha = torch.nn.Parameter(torch.full((2,), inverse_sigmoid(initial_fraction), device=device))
    alpha_optimizer = torch.optim.Adam([alpha], lr=args.auto_lambda_lr)
    diagnostics: list[dict[str, object]] = []; validation_iterator = iter(validation_loader)
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter(); values = {key: [] for key in ["total", "bce", "tissue", "hla", "tissue_w", "hla_w", "meta"]}; updates = 0
        model.train()
        for batch_index, batch in enumerate(loader):
            if batch_index % args.meta_interval == 0:
                validation_batch, validation_iterator = next_batch(validation_iterator, validation_loader)
                meta_loss, _ = meta_update(args, torch, model, batch, validation_batch, alpha, alpha_optimizer, device)
                values["meta"].append(meta_loss); updates += 1
            optimizer.zero_grad(set_to_none=True)
            bce, tissue, hla = losses_for_batch(torch, model, batch, device)
            weights = auxiliary_weights(torch, alpha, args.max_auxiliary_weight).detach()
            total = bce + weights[0] * tissue + weights[1] * hla
            total.backward()
            if args.max_grad_norm > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            values["total"].append(float(total.detach().cpu())); values["bce"].append(float(bce.detach().cpu())); values["tissue"].append(float(tissue.detach().cpu())); values["hla"].append(float(hla.detach().cpu())); values["tissue_w"].append(float(weights[0].cpu())); values["hla_w"].append(float(weights[1].cpu()))
        mean = lambda key: float(np.mean(values[key])) if values[key] else float("nan")
        row = {"experiment_name": EXPERIMENT, "seed": seed, "epoch": epoch, "mean_total_loss": mean("total"), "mean_bce_loss": mean("bce"), "mean_tissue_loss": mean("tissue"), "mean_hla_loss": mean("hla"), "mean_tissue_weight": mean("tissue_w"), "mean_hla_weight": mean("hla_w"), "last_tissue_weight": float(auxiliary_weights(torch, alpha, args.max_auxiliary_weight)[0].detach().cpu()), "last_hla_weight": float(auxiliary_weights(torch, alpha, args.max_auxiliary_weight)[1].detach().cpu()), "mean_meta_validation_bce": mean("meta"), "meta_updates": updates, "min_tissue_weight": min(values["tissue_w"]), "max_tissue_weight": max(values["tissue_w"]), "min_hla_weight": min(values["hla_w"]), "max_hla_weight": max(values["hla_w"])}
        diagnostics.append(row)
        print(f"  E24 seed={seed} epoch={epoch}/{args.epochs} tissue_w={row['mean_tissue_weight']:.4f} hla_w={row['mean_hla_weight']:.4f} meta_bce={row['mean_meta_validation_bce']:.5f} duration={e14.format_duration(time.perf_counter()-started)}", flush=True)
    return model, diagnostics


def prediction_frame(args: argparse.Namespace, torch: Any, DataLoader: Any, TensorDataset: Any, model: Any, train_df: pd.DataFrame, test_df: pd.DataFrame, mappings: dict[str, Any], peptide_length: int, device: str, seed: int) -> pd.DataFrame:
    predictions, _ = e14.predict_branch(args, torch, DataLoader, TensorDataset, model, train_df, test_df, mappings["task_to_id"], peptide_length, device, seed, "global_auto_lambda", "all_tasks", True)
    frames=[]
    for _, task in sorted(predictions.items()):
        test=task["test_task"]
        frames.append(pd.DataFrame({"experiment_name":EXPERIMENT,"seed":seed,"branch":"global_auto_lambda","branch_model":"aux_shared_heads_auto_lambda","sample_id":test.sample_id.to_numpy(),"target_tissue":test.target_tissue.to_numpy(),"mhc_restriction":test.mhc_restriction.to_numpy(),"label":task["y_true"],"probability":task["y_score"],"logit":task["y_logit"]}))
    return pd.concat(frames,ignore_index=True)


def make_rows(reference: pd.DataFrame, predictions: pd.DataFrame) -> tuple[list[dict[str, object]],list[dict[str, object]]]:
    keys=["seed","sample_id","target_tissue","mhc_restriction"]; candidate=predictions.rename(columns={"label":"label_candidate","probability":"probability_candidate"})
    merged=reference.merge(candidate[keys+["label_candidate","probability_candidate"]],on=keys,how="inner",validate="one_to_one")
    if len(merged)!=len(reference) or not np.array_equal(merged.label_global_aux,merged.label_candidate): raise ValueError("E24 prediction alignment failed.")
    rows=[]; global_rows=[]
    for (seed,tissue,hla),task in merged.groupby(["seed","target_tissue","mhc_restriction"],sort=True):
        y=task.label_candidate.to_numpy(dtype=int); probability=task.probability_candidate.to_numpy(dtype=float); baseline=e15.fusion_scores(task)["e15_task_rank_average"]; score=.5*(pd.Series(probability).rank(method="average",pct=True).to_numpy()+pd.Series(task.probability_hla_plain).rank(method="average",pct=True).to_numpy())
        common={"experiment_name":EXPERIMENT,"seed":int(seed),"target_tissue":tissue,"mhc_restriction":hla,"test_rows":len(task),"test_positive":int(y.sum()),"test_negative":int(len(y)-y.sum()),"global_branch":"global_auto_lambda","hla_branch":"hla_plain","fusion_formula":"0.5 * (within_task_rank(global) + within_task_rank(hla))"}
        rows.append({**common,"model":"e14_reference_task_rank_average",**base.evaluate(y,baseline)}); rows.append({**common,"model":MODEL,**base.evaluate(y,score)}); global_rows.append({"experiment_name":EXPERIMENT,"seed":int(seed),"model":GLOBAL_MODEL,"target_tissue":tissue,"mhc_restriction":hla,"test_rows":len(task),"test_positive":int(y.sum()),"test_negative":int(len(y)-y.sum()),**base.evaluate(y,probability)})
    return rows,global_rows


def run(args: argparse.Namespace) -> None:
    if not 0 < args.validation_fraction < .5 or args.meta_interval < 1 or not 0 < args.initial_auxiliary_weight < args.max_auxiliary_weight: raise ValueError("Invalid validation or Auto-Lambda settings.")
    torch,nn,DataLoader,TensorDataset=base.require_torch(); device="cuda" if args.device=="auto" and torch.cuda.is_available() else ("cpu" if args.device=="auto" else args.device)
    e14.set_seed(args.rng_preflight_seed,torch); e21.preflight_rng_roundtrip(torch); train_df,test_df=base.read_dataset(args.train),base.read_dataset(args.test); train_df,test_df,mappings=base.add_task_columns(train_df,test_df)
    if args.max_tasks:
        keep=set(mappings["tasks"][:args.max_tasks]); train_df=train_df[train_df.task_name.isin(keep)].copy(); test_df=test_df[test_df.task_name.isin(keep)].copy(); train_df,test_df,mappings=base.add_task_columns(train_df,test_df)
    reference=e15.require_aligned_e14a_predictions(args.e14_branch_predictions); reference=reference[reference.seed.isin(args.seeds)].copy()
    if sorted(reference.seed.astype(int).unique())!=sorted(args.seeds): raise ValueError("Saved E14 predictions lack an E24 seed.")
    reference=reference.merge(test_df[["sample_id","target_tissue","mhc_restriction"]],on=["sample_id","target_tissue","mhc_restriction"],how="inner"); peptide_length=int(max(train_df.peptide_sequence.str.len().max(),test_df.peptide_sequence.str.len().max()))
    diagnostics=[]; frames=[]
    for seed in args.seeds:
        e14.set_seed(seed,torch); fitting,validation=e18.split_train_validation(train_df,args.validation_fraction,args.validation_split_seed+seed); print(f"device={device}; seed={seed}; fit_rows={len(fitting)}; validation_rows={len(validation)}",flush=True)
        model,rows=train_auto_lambda(args,torch,nn,DataLoader,TensorDataset,fitting,validation,mappings,peptide_length,device,seed); diagnostics.extend(rows); frames.append(prediction_frame(args,torch,DataLoader,TensorDataset,model,fitting,test_df,mappings,peptide_length,device,seed))
    predictions=pd.concat(frames,ignore_index=True); rows,global_rows=make_rows(reference,predictions); summary=base.summarize_results(rows)
    base.write_csv(args.per_task_output,e21.PER_TASK_COLUMNS,rows); base.write_csv(args.summary_output,base.SUMMARY_COLUMNS,summary); base.write_csv(args.stability_output,base.STABILITY_COLUMNS,base.summarize_seed_stability(summary)); base.write_csv(args.global_per_task_output,e21.GLOBAL_PER_TASK_COLUMNS,global_rows); base.write_csv(args.global_summary_output,base.SUMMARY_COLUMNS,base.summarize_results(global_rows)); base.write_csv(args.diagnostics_output,DIAGNOSTIC_COLUMNS,diagnostics); base.write_csv(args.predictions_output,e21.PREDICTION_COLUMNS,predictions.to_dict("records"))
    args.metadata_output.parent.mkdir(parents=True,exist_ok=True); args.metadata_output.write_text(json.dumps({"experiment_name":EXPERIMENT,"device":device,"seeds":args.seeds,"epochs":args.epochs,"primary_loss_weight":1.0,"auxiliary_weight_parameterisation":"max_auxiliary_weight * sigmoid(alpha)","initial_auxiliary_weight":args.initial_auxiliary_weight,"meta_algorithm":"one-step differentiable SGD virtual update; validation primary BCE updates only auxiliary alpha","meta_interval":args.meta_interval,"validation_split":"pair-grouped within task; excluded from ordinary training updates","fixed_hla_source":str(args.e14_branch_predictions)},indent=2,ensure_ascii=False),encoding="utf-8")
    for path in [args.per_task_output,args.summary_output,args.stability_output,args.global_per_task_output,args.global_summary_output,args.diagnostics_output,args.predictions_output,args.metadata_output]: print(f"wrote: {path}",flush=True)


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--train",type=Path,default=project_path("data/tissuePMHC/tissuePMHC_train.csv.gz")); p.add_argument("--test",type=Path,default=project_path("data/tissuePMHC/tissuePMHC_test.csv.gz")); p.add_argument("--e14-branch-predictions",type=Path,default=project_path("results/tissuePMHC_auxiliary_soft_ensemble/branch_predictions.csv")); p.add_argument("--seeds",nargs="+",type=int,default=[20260704]); p.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    p.add_argument("--epochs",type=int,default=25); p.add_argument("--batch-size",type=int,default=512); p.add_argument("--learning-rate",type=float,default=1e-3); p.add_argument("--weight-decay",type=float,default=1e-4); p.add_argument("--embedding-dim",type=int,default=16); p.add_argument("--hidden-dim",type=int,default=128); p.add_argument("--dropout",type=float,default=.2); p.add_argument("--max-grad-norm",type=float,default=1.0)
    p.add_argument("--validation-fraction",type=float,default=.2); p.add_argument("--validation-split-seed",type=int,default=20260724); p.add_argument("--meta-interval",type=int,default=10); p.add_argument("--meta-inner-lr",type=float,default=1e-3); p.add_argument("--auto-lambda-lr",type=float,default=1e-3); p.add_argument("--initial-auxiliary-weight",type=float,default=.1); p.add_argument("--max-auxiliary-weight",type=float,default=.3); p.add_argument("--max-tasks",type=int,default=0); p.add_argument("--rng-preflight-seed",type=int,default=20260711)
    root=project_path("results/tissuePMHC_e24_auto_lambda"); p.add_argument("--per-task-output",type=Path,default=root/"per_task_metrics.csv"); p.add_argument("--summary-output",type=Path,default=root/"summary_metrics.csv"); p.add_argument("--stability-output",type=Path,default=root/"stability_metrics.csv"); p.add_argument("--global-per-task-output",type=Path,default=root/"global_per_task_metrics.csv"); p.add_argument("--global-summary-output",type=Path,default=root/"global_summary_metrics.csv"); p.add_argument("--diagnostics-output",type=Path,default=root/"auto_lambda_diagnostics.csv"); p.add_argument("--predictions-output",type=Path,default=root/"global_branch_predictions.csv"); p.add_argument("--metadata-output",type=Path,default=root/"metadata.json"); return p.parse_args()

if __name__=="__main__": run(parse_args())
