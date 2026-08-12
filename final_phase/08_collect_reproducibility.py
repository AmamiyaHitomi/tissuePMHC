#!/usr/bin/env python3
"""Collect environment, hashes, configurations, timing and parameter counts."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from common import EXPERIMENTS, ROOT, ensure_output, read_train, sha256, write_json


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def package_versions() -> dict[str, str | None]:
    names = ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "torch"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def human_parameter_count() -> dict[str, int]:
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import torch
    import run_tissuepmhc_e29_multikernel_cnn_oof as e29

    train = read_train(EXPERIMENTS["human"])
    args = SimpleNamespace(
        embedding_dim=16, kernel_sizes=[2, 3, 5], conv_channels=32,
        hidden_dim=128, dropout=0.2,
    )
    global_model = e29.define_cnn_shared_heads_model(
        args, torch.nn, 9, train.task_name.nunique(), train.target_tissue.nunique(),
        train.mhc_restriction.nunique(), True,
    )
    global_count = sum(parameter.numel() for parameter in global_model.parameters())
    branch_counts = {}
    for mhc, group in train.groupby("mhc_restriction", sort=True):
        model = e29.define_cnn_shared_heads_model(
            args, torch.nn, 9, group.task_name.nunique(), train.target_tissue.nunique(),
            train.mhc_restriction.nunique(), False,
        )
        branch_counts[mhc] = sum(parameter.numel() for parameter in model.parameters())
    return {
        "global_aux_parameters": int(global_count),
        "hla_plain_models": len(branch_counts),
        "hla_plain_parameters_total": int(sum(branch_counts.values())),
        "all_separately_trained_models_parameters_total": int(global_count + sum(branch_counts.values())),
        "largest_hla_plain_model_parameters": int(max(branch_counts.values())),
        "smallest_hla_plain_model_parameters": int(min(branch_counts.values())),
    }


def main() -> None:
    output = ensure_output("08_reproducibility")
    try:
        import torch
        torch_info = {
            "version": torch.__version__, "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        torch_info = {"available": False}
    nvidia = command_output([
        "nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"
    ])
    files = []
    for experiment in EXPERIMENTS.values():
        for path in (
            experiment.train, experiment.standard_predictions,
            experiment.strict_predictions, experiment.strict_assignments,
        ):
            files.append({"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size})
    metadata_paths = [
        ROOT / "results/tissuePMHC_phase7_min200_e31_peptide_disjoint_oof/metadata.json",
        ROOT / "results/mousePMHC_phase6_e33_peptide_disjoint_oof/mousePMHC_phase6_e33_metadata.json",
    ]
    run_metadata = {}
    for path in metadata_paths:
        if path.is_file():
            run_metadata[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "python": {"version": sys.version, "executable": sys.executable, "platform": platform.platform()},
        "packages": package_versions(), "torch": torch_info, "nvidia_smi": nvidia,
        "git_commit": command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "files": files, "run_metadata": run_metadata,
        "parameter_counts": {
            "human_e29": human_parameter_count(),
            "mouse_e33_reported": run_metadata.get("mousePMHC_phase6_e33_metadata", {}).get("parameter_count"),
        },
        "peak_gpu_memory": {
            "value": None,
            "status": "not recorded during completed runs; cannot be reconstructed exactly without profiling a rerun",
        },
    }
    write_json(output / "reproducibility_manifest.json", payload)
    pd.DataFrame(files).to_csv(output / "file_hashes.csv", index=False)
    Path(output / "pip_freeze.txt").write_text(
        command_output([sys.executable, "-m", "pip", "freeze"]) or "unavailable", encoding="utf-8"
    )
    print(f"wrote reproducibility manifest to {output}")


if __name__ == "__main__":
    main()

