#!/usr/bin/env python3
"""Run Phase 6 E27: zero-init low-rank H2-Kd adapter on E3b MMoE OOF."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import run_mousepmhc_phase3_e3_factorized_mmoe_oof as e3
import run_mousepmhc_phase3_e5_famo_mmoe_oof as e5
import run_tissuepmhc_e26_all_in_one as folds
import run_tissuepmhc_neural_baselines_v2 as base

ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT="mousePMHC_phase6_e27_kd_adapter_oof"; CANDIDATE="mousePMHC_phase6_e27_e3b_zero_init_h2_kd_rank8_adapter"; KEYS=["sample_id","target_tissue","mhc_restriction","label"]
def path(x:str)->Path:return ROOT/x

def model(torch:Any,nn:Any,args:argparse.Namespace,m:dict[str,Any])->Any:
    target_ids={name:m["hla_to_id"].get(name) for name in args.adapter_h2s}
    if any(value is None for value in target_ids.values()): raise ValueError(f"Unknown adapter H2: {target_ids}")
    class KdAdapter(nn.Module):
        def __init__(self):
            super().__init__(); self.core=e3.define_model(torch,nn,9,len(m["tasks"]),len(m["tissue_to_id"]),len(m["hla_to_id"]),args)
            self.down=nn.ModuleDict({name:nn.Linear(args.hidden_dim,args.adapter_rank) for name in target_ids})
            self.up=nn.ModuleDict({name:nn.Linear(args.adapter_rank,args.hidden_dim,bias=False) for name in target_ids})
            for layer in self.up.values(): nn.init.zeros_(layer.weight)
        def forward(self,p,t,ti,h,return_gates=False):
            encoded=self.core.peptide_encoder(self.core.amino_embedding(p)); encoded=encoded.clone()
            for name,h2_id in target_ids.items():
                mask=(h==h2_id)
                if mask.any(): encoded[mask]=encoded[mask]+self.up[name](torch.relu(self.down[name](encoded[mask])))
            experts=torch.stack([x(encoded) for x in self.core.experts],1); gates=torch.softmax(self.core.gate(torch.cat([encoded,self.core.tissue_embedding(ti),self.core.mhc_embedding(h)],1)),1); mixed=(experts*gates.unsqueeze(-1)).sum(1); logits=torch.empty(len(t),device=t.device)
            for task_id in t.unique():
                hit=t==task_id; logits[hit]=self.core.heads[int(task_id.item())](mixed[hit]).squeeze(-1)
            return (logits,gates) if return_gates else logits
    return KdAdapter()

def train(args,torch,nn,fitting,m,seed,device):
    net=model(torch,nn,args,m).to(device); opt=torch.optim.AdamW(net.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay); arrays=e5.task_arrays(fitting,m,9); rng=np.random.default_rng(seed); steps=args.steps_per_epoch or int(np.ceil(max(len(x['label']) for x in arrays)/args.task_batch_size))
    for epoch in range(1,args.epochs+1):
        losses=[]; net.train()
        for _ in range(steps):
            p,t,ti,h,y=[torch.as_tensor(x,device=device) for x in e5.sample_balanced_batch(rng,arrays,args.task_batch_size)]; opt.zero_grad(set_to_none=True); ls,g= e5.task_loss_vector(torch,net,p,t,ti,h,y,len(arrays),args.task_batch_size); loss=ls.mean()-args.gate_entropy_weight*(-(g*torch.log(g.clamp_min(1e-12))).sum(1).mean()); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),args.max_grad_norm);opt.step();losses.append(float(loss.detach().cpu()))
        print(f"E27 seed={seed} epoch={epoch}/{args.epochs} loss={np.mean(losses):.5f}",flush=True)
    return net

def run(a):
    torch,nn,DL,TD=base.require_torch(); device="cuda" if a.device=="auto" and torch.cuda.is_available() else ("cpu" if a.device=="auto" else a.device); raw=base.read_dataset(a.train); e3.validate_input(raw); train_df,_,m=base.add_task_columns(raw,raw.copy()); folds_id=folds.make_pair_grouped_folds(train_df,a.oof_folds,a.oof_split_seed); out=[]
    for seed in a.seeds:
        for fold in range(a.oof_folds):
            base.set_seed(seed,torch); fit,hold=train_df[folds_id!=fold].copy(),train_df[folds_id==fold].copy(); print(f"E27 seed={seed} fold={fold+1}/{a.oof_folds}",flush=True); net=train(a,torch,nn,fit,m,seed,device); scores=e5.predict(torch,DL,TD,net,hold,9,a.batch_size,device); z=hold[KEYS].copy();z.insert(0,"split","oof");z.insert(1,"candidate",CANDIDATE);z.insert(2,"seed",seed);z["fold"]=fold;z["score"]=scores;out.append(z)
    pred=pd.concat(out,ignore_index=True)
    if len(pred)!=len(train_df)*len(a.seeds) or pred.duplicated(["seed","sample_id"]).any():raise AssertionError("Incomplete E27 OOF coverage")
    rows=[]
    for (seed,tissue,h2),x in pred.groupby(["seed","target_tissue","mhc_restriction"]):rows.append({"experiment_name":EXPERIMENT,"candidate":CANDIDATE,"seed":seed,"target_tissue":tissue,"mhc_restriction":h2,"oof_rows":len(x),**base.evaluate(x.label.to_numpy(int),x.score.to_numpy(float))})
    task=pd.DataFrame(rows); summ=task.groupby("seed")[["accuracy","balanced_accuracy","auroc","auprc","f1","mcc"]].mean().add_prefix("mean_task_").reset_index();summ.insert(0,"candidate",CANDIDATE);summ.insert(0,"experiment_name",EXPERIMENT);summ["worst6_task_auroc"]=[task[task.seed==s].nsmallest(6,"auroc").auroc.mean() for s in summ.seed]
    a.output_dir.mkdir(parents=True,exist_ok=True);pred.to_csv(a.output_dir/"mousePMHC_phase6_e27_oof_predictions.csv",index=False);task.to_csv(a.output_dir/"mousePMHC_phase6_e27_oof_per_task_metrics.csv",index=False);summ.to_csv(a.output_dir/"mousePMHC_phase6_e27_oof_summary_metrics.csv",index=False);task.groupby(["seed","mhc_restriction"])[["auroc","auprc"]].mean().reset_index().to_csv(a.output_dir/"mousePMHC_phase6_e27_oof_h2_metrics.csv",index=False)
    (a.output_dir/"mousePMHC_phase6_e27_metadata.json").write_text(json.dumps({"experiment_name":EXPERIMENT,"candidate":CANDIDATE,"test_data_read":False,"train":str(a.train),"seeds":a.seeds,"epochs":a.epochs,"adapter_rank":a.adapter_rank,"adapter_h2s":a.adapter_h2s,"note":"Same E3b task-balanced training protocol; each adapter up-projection is zero initialized."},ensure_ascii=False,indent=2),encoding="utf-8");print(summ.to_string(index=False),flush=True)

def args():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--train",type=Path,default=path("data/mousePMHC/mousePMHC_train.csv.gz"));p.add_argument("--output-dir",type=Path,default=path("results/mousePMHC_phase6_e27_kd_adapter_oof"));p.add_argument("--seeds",nargs="+",type=int,default=[20260704,20260705,20260706]);p.add_argument("--oof-folds",type=int,default=3);p.add_argument("--oof-split-seed",type=int,default=20260711);p.add_argument("--device",choices=["auto","cpu","cuda"],default="auto");p.add_argument("--epochs",type=int,default=25);p.add_argument("--steps-per-epoch",type=int,default=0);p.add_argument("--task-batch-size",type=int,default=16);p.add_argument("--batch-size",type=int,default=512);p.add_argument("--learning-rate",type=float,default=1e-3);p.add_argument("--weight-decay",type=float,default=1e-4);p.add_argument("--embedding-dim",type=int,default=16);p.add_argument("--hidden-dim",type=int,default=128);p.add_argument("--expert-dim",type=int,default=64);p.add_argument("--condition-dim",type=int,default=16);p.add_argument("--gate-hidden-dim",type=int,default=64);p.add_argument("--n-experts",type=int,default=3);p.add_argument("--dropout",type=float,default=.2);p.add_argument("--gate-entropy-weight",type=float,default=.01);p.add_argument("--max-grad-norm",type=float,default=1.);p.add_argument("--adapter-rank",type=int,default=8);p.add_argument("--adapter-h2s",nargs="+",default=["H2-Kd"]);return p.parse_args()
if __name__=="__main__":run(args())
