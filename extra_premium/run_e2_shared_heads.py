#!/usr/bin/env python3
"""One-seed premium test of the original human E2 shared-head MLP."""

from __future__ import annotations

import time

import common


MODEL_NAME = "e2_shared_heads"


def main() -> None:
    cli = common.basic_parser(__doc__).parse_args()
    if cli.epochs < 1:
        raise ValueError("--epochs must be positive.")

    common.enable_original_modules()
    import run_tissuepmhc_neural_baselines_v2 as base

    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = common.resolve_device(cli.device, torch)
    train, test, mappings, peptide_length = common.load_premium_data(base)
    args = common.model_args(cli)

    base.set_seed(common.SEED, torch)
    _, SharedTaskHeadsModel, _ = base.define_models(nn)
    model = SharedTaskHeadsModel(
        peptide_length,
        len(mappings["tasks"]),
        args.embedding_dim,
        args.hidden_dim,
        args.dropout,
    ).to(device)

    train_loader = base.build_loader(
        torch,
        DataLoader,
        TensorDataset,
        [
            base.encode_peptides(train["peptide_sequence"], peptide_length),
            train["task_id"].to_numpy(dtype="int64"),
            train["label"].to_numpy(dtype="int64"),
        ],
        args.batch_size,
        True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    print(
        f"{MODEL_NAME}: device={device}, seed={common.SEED}, "
        f"train_rows={len(train)}, test_rows={len(test)}, tasks={len(mappings['tasks'])}",
        flush=True,
    )
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        base.train_binary_model(
            torch,
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            "task_heads",
            1,
        )
        print(
            f"epoch {epoch:02d}/{args.epochs:02d} "
            f"time={time.perf_counter() - epoch_started:.1f}s",
            flush=True,
        )

    test_loader = base.build_loader(
        torch,
        DataLoader,
        TensorDataset,
        [
            base.encode_peptides(test["peptide_sequence"], peptide_length),
            test["task_id"].to_numpy(dtype="int64"),
            test["label"].to_numpy(dtype="int64"),
        ],
        args.batch_size,
        False,
    )
    labels, scores = base.predict_scores(torch, model, test_loader, device, "task_heads")
    if not (labels == test["label"].to_numpy(dtype="int64")).all():
        raise AssertionError("E2 prediction labels are not aligned with the premium test rows.")

    common.save_basic_test_results(
        MODEL_NAME,
        train,
        test,
        scores,
        base,
        {
            "source_model": "scripts/run_tissuepmhc_neural_baselines_v2.py",
            "architecture": "E2 shared peptide MLP encoder with task-specific heads",
            "device": device,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()

