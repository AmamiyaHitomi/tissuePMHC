#!/usr/bin/env python3
"""One-seed premium test of the original human E29 multi-kernel CNN."""

from __future__ import annotations

import time

import common


MODEL_NAME = "e29_multikernel_cnn"


def main() -> None:
    cli = common.basic_parser(__doc__).parse_args()
    if cli.epochs < 1:
        raise ValueError("--epochs must be positive.")

    common.enable_original_modules()
    import run_tissuepmhc_e29_multikernel_cnn_oof as e29
    import run_tissuepmhc_neural_baselines_v2 as base

    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = common.resolve_device(cli.device, torch)
    train, test, mappings, peptide_length = common.load_premium_data(base)
    args = common.model_args(cli)

    print(
        f"{MODEL_NAME}: device={device}, seed={common.SEED}, "
        f"train_rows={len(train)}, test_rows={len(test)}, tasks={len(mappings['tasks'])}",
        flush=True,
    )
    started = time.perf_counter()
    prediction = e29.predict_seed(
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
        "premium_full_train_test",
    )

    aligned = test[["sample_id"]].merge(
        prediction[["sample_id", "score"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if aligned["score"].isna().any():
        raise ValueError("E29 did not produce a score for every premium test row.")

    common.save_basic_test_results(
        MODEL_NAME,
        train,
        test,
        aligned["score"].to_numpy(),
        base,
        {
            "source_model": "scripts/run_tissuepmhc_e29_multikernel_cnn_oof.py",
            "architecture": "E29 multi-kernel CNN global-aux/HLA-plain branches with task-rank fusion",
            "device": device,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "kernel_sizes": args.kernel_sizes,
            "conv_channels": args.conv_channels,
            "tissue_loss_weight": args.tissue_loss_weight,
            "hla_loss_weight": args.hla_loss_weight,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()

