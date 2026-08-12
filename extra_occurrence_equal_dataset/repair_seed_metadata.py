"""Repair seed labels written by the legacy saver's definition-time default."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent / "results" / "seed_runs"
SEEDS = (20260705, 20260706)
EXPERIMENTS = (
    "e2_shared_heads",
    "e14a_auxiliary_dual_branch",
    "e29_multikernel_cnn",
)


def main() -> None:
    for seed in SEEDS:
        for experiment in EXPERIMENTS:
            directory = ROOT / str(seed) / experiment
            for filename in (
                "test_predictions.csv",
                "per_task_metrics.csv",
                "summary_metrics.csv",
            ):
                path = directory / filename
                frame = pd.read_csv(path)
                observed = set(frame["seed"].astype(int))
                if observed not in ({20260704}, {seed}):
                    raise ValueError(f"Unexpected seed labels {observed} in {path}")
                frame["seed"] = seed
                frame.to_csv(path, index=False)
            settings_path = directory / "run_settings.json"
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
            if int(settings["seed"]) not in (20260704, seed):
                raise ValueError(f"Unexpected settings seed in {settings_path}")
            settings["seed"] = seed
            settings_path.write_text(
                json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"repaired seed={seed} experiment={experiment}")


if __name__ == "__main__":
    main()
