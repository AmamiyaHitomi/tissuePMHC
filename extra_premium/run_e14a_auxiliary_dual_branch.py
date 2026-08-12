#!/usr/bin/env python3
"""One-seed premium test of original human E14a global-aux/HLA-plain."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import common


MODEL_NAME = "e14a_auxiliary_dual_branch"


def prediction_frame(
    global_aux: dict[tuple[str, str], dict[str, object]],
    hla_plain: dict[tuple[str, str], dict[str, object]],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for key in sorted(set(global_aux) & set(hla_plain)):
        global_item = global_aux[key]
        hla_item = hla_plain[key]
        if not np.array_equal(global_item["y_true"], hla_item["y_true"]):
            raise ValueError(f"E14a branch labels disagree for task {key}.")
        global_test = global_item["test_task"]
        hla_test = hla_item["test_task"]
        if not global_test["sample_id"].reset_index(drop=True).equals(
            hla_test["sample_id"].reset_index(drop=True)
        ):
            raise ValueError(f"E14a branch sample order disagrees for task {key}.")
        rows.append(
            pd.DataFrame(
                {
                    "sample_id": global_test["sample_id"].to_numpy(),
                    "score": 0.5 * global_item["y_score"] + 0.5 * hla_item["y_score"],
                }
            )
        )
    if not rows:
        raise RuntimeError("E14a produced no test predictions.")
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    cli = common.basic_parser(__doc__).parse_args()
    if cli.epochs < 1:
        raise ValueError("--epochs must be positive.")

    common.enable_original_modules()
    import run_tissuepmhc_auxiliary_soft_ensemble as e14
    import run_tissuepmhc_neural_baselines_v2 as base

    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = common.resolve_device(cli.device, torch)
    train, test, mappings, peptide_length = common.load_premium_data(base)
    args = common.model_args(cli)

    e14.set_seed(common.SEED, torch)
    print(
        f"{MODEL_NAME}: device={device}, seed={common.SEED}, "
        f"train_rows={len(train)}, test_rows={len(test)}, tasks={len(mappings['tasks'])}",
        flush=True,
    )
    started = time.perf_counter()

    global_aux, _, _ = e14.train_and_predict_global_branch(
        args,
        torch,
        nn,
        DataLoader,
        TensorDataset,
        train,
        test,
        mappings,
        peptide_length,
        device,
        common.SEED,
        True,
    )
    hla_plain, _, _ = e14.train_and_predict_hla_branches(
        args,
        torch,
        nn,
        DataLoader,
        TensorDataset,
        train,
        test,
        mappings,
        peptide_length,
        device,
        common.SEED,
        False,
    )

    predicted = prediction_frame(global_aux, hla_plain)
    aligned = test[["sample_id"]].merge(
        predicted,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if aligned["score"].isna().any():
        raise ValueError("E14a did not produce a score for every premium test row.")

    common.save_basic_test_results(
        MODEL_NAME,
        train,
        test,
        aligned["score"].to_numpy(),
        base,
        {
            "source_model": "scripts/run_tissuepmhc_auxiliary_soft_ensemble.py",
            "architecture": "E14a global auxiliary MLP plus per-HLA plain MLP, 0.5 probability fusion",
            "device": device,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "tissue_loss_weight": args.tissue_loss_weight,
            "hla_loss_weight": args.hla_loss_weight,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()

