#!/usr/bin/env python3
"""Run E25: HLA-Structured PLE as a standalone global model.

Two narrow global experts are combined with exactly one lightweight private
expert selected by a sample's HLA.  Each tissue-HLA task has a gate over the
two global outputs plus its corresponding HLA-private output; this avoids
creating 44 full private experts.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_tissuepmhc_neural_baselines_v2 as base

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "E25_hla_structured_ple"; MODEL = "e25_hla_structured_ple"
GATE_COLUMNS = ["experiment_name","seed","epoch","task_name","target_tissue","mhc_restriction","source","expert_id","gate_weight"]
PREDICTION_COLUMNS = ["experiment_name","seed","model","sample_id","target_tissue","mhc_restriction","label","probability","logit"]
COMPARISON_COLUMNS = ["seed","target_tissue","mhc_restriction","baseline_model","candidate_model","baseline_source",*[f"delta_{m}" for m in ["accuracy","balanced_accuracy","auroc","auprc","f1","mcc"]]]

def project_path(relative: str) -> Path: return PROJECT_ROOT / relative

def define_model(args: argparse.Namespace, torch: Any, nn: Any, peptide_length: int, n_tasks: int, n_hlas: int) -> Any:
    class HLAStructuredPLE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding=nn.Embedding(len(base.AA_TO_INDEX)+1,args.embedding_dim,padding_idx=base.PAD_INDEX)
            self.shared_input=nn.Sequential(nn.Flatten(),nn.Linear(peptide_length*args.embedding_dim,args.hidden_dim),nn.ReLU(),nn.Dropout(args.dropout))
            def global_expert() -> Any: return nn.Sequential(nn.Linear(args.hidden_dim,args.expert_dim),nn.ReLU(),nn.Dropout(args.dropout),nn.Linear(args.expert_dim,args.expert_dim),nn.ReLU())
            self.global_experts=nn.ModuleList([global_expert() for _ in range(args.n_global_experts)])
            self.hla_private_experts=nn.ModuleList([nn.Sequential(nn.Linear(args.hidden_dim,args.expert_dim),nn.ReLU()) for _ in range(n_hlas)])
            self.task_gates=nn.Embedding(n_tasks,args.n_global_experts+1)
            self.heads=nn.ModuleList([nn.Linear(args.expert_dim,1) for _ in range(n_tasks)])
        def features(self, peptide_ids: Any, hla_ids: Any) -> Any:
            shared=self.shared_input(self.embedding(peptide_ids)); global_outputs=torch.stack([expert(shared) for expert in self.global_experts],dim=1)
            private=shared.new_empty((shared.shape[0],args.expert_dim))
            for hla_id in torch.unique(hla_ids):
                mask=hla_ids==hla_id; private[mask]=self.hla_private_experts[int(hla_id.item())](shared[mask])
            return torch.cat([global_outputs,private.unsqueeze(1)],dim=1)
        def gate_weights(self,task_ids: Any) -> Any: return torch.softmax(self.task_gates(task_ids),dim=1)
        def forward(self,peptide_ids: Any,task_ids: Any,hla_ids: Any) -> Any:
            mixed=(self.features(peptide_ids,hla_ids)*self.gate_weights(task_ids).unsqueeze(-1)).sum(dim=1); logits=mixed.new_empty(mixed.shape[0])
            for task_id in torch.unique(task_ids):
                mask=task_ids==task_id; logits[mask]=self.heads[int(task_id.item())](mixed[mask]).squeeze(-1)
            return logits
        def all_gates(self) -> Any: return self.gate_weights(torch.arange(n_tasks,device=self.task_gates.weight.device))
    return HLAStructuredPLE()

def build_loader(args: argparse.Namespace,torch: Any,DataLoader: Any,TensorDataset: Any,df: pd.DataFrame,length: int,shuffle: bool) -> Any:
    return base.build_loader(torch,DataLoader,TensorDataset,[base.encode_peptides(df.peptide_sequence,length),df.task_id.to_numpy(dtype=np.int64).copy(),df.hla_id.to_numpy(dtype=np.int64).copy(),df.label.to_numpy(dtype=np.int64).copy()],args.batch_size,shuffle)

def gate_rows(args: argparse.Namespace,model: Any,mappings: dict[str,Any],seed: int,epoch: int) -> list[dict[str,object]]:
    gates=model.all_gates().detach().cpu().numpy(); rows=[]
    for task_name in mappings["tasks"]:
        tissue,hla=task_name.split("||",1); task_id=mappings["task_to_id"][task_name]
        for expert_id,weight in enumerate(gates[task_id][:-1]): rows.append({"experiment_name":EXPERIMENT,"seed":seed,"epoch":epoch,"task_name":task_name,"target_tissue":tissue,"mhc_restriction":hla,"source":"global","expert_id":expert_id,"gate_weight":float(weight)})
        rows.append({"experiment_name":EXPERIMENT,"seed":seed,"epoch":epoch,"task_name":task_name,"target_tissue":tissue,"mhc_restriction":hla,"source":"hla_private","expert_id":mappings["hla_to_id"][hla],"gate_weight":float(gates[task_id][-1])})
    return rows

def train(args: argparse.Namespace,torch: Any,nn: Any,DataLoader: Any,TensorDataset: Any,train_df: pd.DataFrame,mappings: dict[str,Any],length: int,device: str,seed: int) -> tuple[Any,list[dict[str,object]]]:
    model=define_model(args,torch,nn,length,len(mappings["tasks"]),len(mappings["hla_to_id"])).to(device); loader=build_loader(args,torch,DataLoader,TensorDataset,train_df,length,True); optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay); history=[]
    for epoch in range(1,args.epochs+1):
        started=time.perf_counter(); model.train(); losses=[]; entropies=[]
        for peptide,task,hla,label in loader:
            peptide,task,hla,label=[x.to(device) for x in [peptide,task,hla,label]]; optimizer.zero_grad(set_to_none=True); logits=model(peptide,task,hla); loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,label.float())
            gates=model.gate_weights(task); entropy=-(gates*torch.log(gates.clamp_min(1e-12))).sum(dim=1).mean()
            loss=loss-args.gate_entropy_weight*entropy; loss.backward()
            if args.max_grad_norm>0: torch.nn.utils.clip_grad_norm_(model.parameters(),args.max_grad_norm)
            optimizer.step(); losses.append(float(loss.detach().cpu())); entropies.append(float(entropy.detach().cpu()))
        history.extend(gate_rows(args,model,mappings,seed,epoch)); print(f"  E25 seed={seed} epoch={epoch}/{args.epochs} loss={np.mean(losses):.4f} gate_entropy={np.mean(entropies):.4f} duration={time.perf_counter()-started:.2f}s",flush=True)
    return model,history

def evaluate(args: argparse.Namespace,torch: Any,DataLoader: Any,TensorDataset: Any,model: Any,train_df: pd.DataFrame,test_df: pd.DataFrame,mappings: dict[str,Any],length: int,device: str,seed: int) -> tuple[list[dict[str,object]],list[dict[str,object]]]:
    rows=[]; predictions=[]; model.eval()
    for task_name in mappings["tasks"]:
        train_task=train_df[train_df.task_name==task_name]; test_task=test_df[test_df.task_name==task_name]; loader=build_loader(args,torch,DataLoader,TensorDataset,test_task,length,False); scores=[]; logits=[]; labels=[]
        with torch.no_grad():
            for peptide,task,hla,label in loader:
                peptide,task,hla=[x.to(device) for x in [peptide,task,hla]]; out=model(peptide,task,hla); logits.append(out.cpu().numpy()); scores.append(torch.sigmoid(out).cpu().numpy()); labels.append(label.numpy())
        y=np.concatenate(labels); probability=np.concatenate(scores); logit=np.concatenate(logits); metric=base.make_metric_row(MODEL,train_task,test_task,base.evaluate(y,probability)); metric.update({"experiment_name":EXPERIMENT,"seed":seed}); rows.append(metric)
        predictions.append(pd.DataFrame({"experiment_name":EXPERIMENT,"seed":seed,"model":MODEL,"sample_id":test_task.sample_id.to_numpy(),"target_tissue":test_task.target_tissue.to_numpy(),"mhc_restriction":test_task.mhc_restriction.to_numpy(),"label":y,"probability":probability,"logit":logit}))
    return rows,predictions

def comparison(rows: list[dict[str,object]],path: Path) -> list[dict[str,object]]:
    if not path.exists(): return []
    baseline=pd.read_csv(path); baseline=baseline[baseline.branch.eq("global_aux")]; indexed={(int(r.seed),r.target_tissue,r.mhc_restriction):r for r in baseline.itertuples()}; output=[]
    for row in rows:
        ref=indexed.get((int(row["seed"]),row["target_tissue"],row["mhc_restriction"]));
        if ref is None: continue
        item={"seed":int(row["seed"]),"target_tissue":row["target_tissue"],"mhc_restriction":row["mhc_restriction"],"baseline_model":"e14a_global_aux","candidate_model":MODEL,"baseline_source":str(path)}
        for metric in ["accuracy","balanced_accuracy","auroc","auprc","f1","mcc"]: item[f"delta_{metric}"]=float(row[metric])-float(getattr(ref,metric))
        output.append(item)
    return output

def run(args: argparse.Namespace) -> None:
    torch,nn,DataLoader,TensorDataset=base.require_torch(); device="cuda" if args.device=="auto" and torch.cuda.is_available() else ("cpu" if args.device=="auto" else args.device); train_df,test_df=base.read_dataset(args.train),base.read_dataset(args.test); train_df,test_df,mappings=base.add_task_columns(train_df,test_df)
    if args.max_tasks:
        keep=set(mappings["tasks"][:args.max_tasks]); train_df=train_df[train_df.task_name.isin(keep)].copy(); test_df=test_df[test_df.task_name.isin(keep)].copy(); train_df,test_df,mappings=base.add_task_columns(train_df,test_df)
    length=int(max(train_df.peptide_sequence.str.len().max(),test_df.peptide_sequence.str.len().max())); all_rows=[]; all_gates=[]; frames=[]
    print(f"device={device}; tasks={len(mappings['tasks'])}; HLA-private experts={len(mappings['hla_to_id'])}; global experts={args.n_global_experts}",flush=True)
    for seed in args.seeds:
        base.set_seed(seed,torch); model,gates=train(args,torch,nn,DataLoader,TensorDataset,train_df,mappings,length,device,seed); rows,predictions=evaluate(args,torch,DataLoader,TensorDataset,model,train_df,test_df,mappings,length,device,seed); all_rows.extend(rows); all_gates.extend(gates); frames.extend(predictions)
    summary=base.summarize_results(all_rows); base.write_csv(args.per_task_output,base.METRIC_COLUMNS,all_rows); base.write_csv(args.summary_output,base.SUMMARY_COLUMNS,summary); base.write_csv(args.stability_output,base.STABILITY_COLUMNS,base.summarize_seed_stability(summary)); base.write_csv(args.gate_output,GATE_COLUMNS,all_gates); base.write_csv(args.predictions_output,PREDICTION_COLUMNS,pd.concat(frames,ignore_index=True).to_dict("records")); base.write_csv(args.comparison_output,COMPARISON_COLUMNS,comparison(all_rows,args.e14_global_candidate_metrics))
    args.metadata_output.parent.mkdir(parents=True,exist_ok=True); args.metadata_output.write_text(json.dumps({"experiment_name":EXPERIMENT,"model":MODEL,"device":device,"seeds":args.seeds,"n_global_experts":args.n_global_experts,"n_hla_private_experts":len(mappings["hla_to_id"]),"private_routing":"one HLA-private expert selected per sample","task_gate":"two global experts plus the matching HLA-private expert","expert_dim":args.expert_dim,"gate_entropy_weight":args.gate_entropy_weight,"e14_global_candidate_metrics":str(args.e14_global_candidate_metrics)},indent=2,ensure_ascii=False),encoding="utf-8")
    for path in [args.per_task_output,args.summary_output,args.stability_output,args.gate_output,args.predictions_output,args.comparison_output,args.metadata_output]: print(f"wrote: {path}",flush=True)

def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--train",type=Path,default=project_path("data/tissuePMHC/tissuePMHC_train.csv.gz")); p.add_argument("--test",type=Path,default=project_path("data/tissuePMHC/tissuePMHC_test.csv.gz")); p.add_argument("--e14-global-candidate-metrics",type=Path,default=project_path("results/tissuePMHC_auxiliary_soft_ensemble/candidate_metrics.csv")); p.add_argument("--seeds",nargs="+",type=int,default=[20260704,20260705,20260706]); p.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    p.add_argument("--epochs",type=int,default=25); p.add_argument("--batch-size",type=int,default=512); p.add_argument("--learning-rate",type=float,default=1e-3); p.add_argument("--weight-decay",type=float,default=1e-4); p.add_argument("--embedding-dim",type=int,default=16); p.add_argument("--hidden-dim",type=int,default=128); p.add_argument("--expert-dim",type=int,default=64); p.add_argument("--n-global-experts",type=int,default=2); p.add_argument("--dropout",type=float,default=.2); p.add_argument("--gate-entropy-weight",type=float,default=.01); p.add_argument("--max-grad-norm",type=float,default=1.0); p.add_argument("--max-tasks",type=int,default=0)
    root=project_path("results/tissuePMHC_e25_hla_structured_ple"); p.add_argument("--per-task-output",type=Path,default=root/"per_task_metrics.csv"); p.add_argument("--summary-output",type=Path,default=root/"summary_metrics.csv"); p.add_argument("--stability-output",type=Path,default=root/"stability_metrics.csv"); p.add_argument("--gate-output",type=Path,default=root/"gate_weight_history.csv"); p.add_argument("--predictions-output",type=Path,default=root/"branch_predictions.csv"); p.add_argument("--comparison-output",type=Path,default=root/"e14_global_comparison_metrics.csv"); p.add_argument("--metadata-output",type=Path,default=root/"metadata.json"); return p.parse_args()

if __name__=="__main__": run(parse_args())
