#!/usr/bin/env python3
"""Run Phase 6 E28: independent zero-init rank-8 adapters for H2-Kb and H2-Kd.

E28 deliberately reuses the E27 training implementation, but locks the
adapter set to Kb/Kd and writes into an isolated experiment directory.  It
never opens the fixed test split.
"""
from __future__ import annotations

from pathlib import Path

import run_mousepmhc_phase6_e27_kd_adapter_oof as shared


EXPERIMENT = "mousePMHC_phase6_e28_kb_kd_adapters_oof"
CANDIDATE = "mousePMHC_phase6_e28_e3b_zero_init_h2_kb_kd_rank8_adapters"
DEFAULT_OUTPUT = shared.path("results/mousePMHC_phase6_e28_kb_kd_adapters_oof")
E27_DEFAULT_OUTPUT = shared.path("results/mousePMHC_phase6_e27_kd_adapter_oof")


def main() -> None:
    shared.EXPERIMENT = EXPERIMENT
    shared.CANDIDATE = CANDIDATE
    args = shared.args()
    if args.output_dir == E27_DEFAULT_OUTPUT:
        args.output_dir = DEFAULT_OUTPUT
    if args.adapter_h2s != ["H2-Kd"]:
        raise ValueError("E28 fixes --adapter-h2s to H2-Kb H2-Kd; do not tune the target set.")
    args.adapter_h2s = ["H2-Kb", "H2-Kd"]
    shared.run(args)
    for source in args.output_dir.glob("mousePMHC_phase6_e27_*"):
        source.rename(args.output_dir / source.name.replace("mousePMHC_phase6_e27_", "mousePMHC_phase6_e28_", 1))


if __name__ == "__main__":
    main()
