#!/usr/bin/env python3
"""Run frozen NetMHCpan-4.1b through WSL on premium test-only queries."""

from __future__ import annotations

import argparse
import os

import common


MANIFEST_PATH = (
    common.EXTERNAL_ROOT / "queries" / "premium_test_netmhcpan_manifest.csv"
)
RAW_DIR = common.EXTERNAL_ROOT / "raw_outputs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executable",
        default=os.environ.get("NETMHCPAN_EXECUTABLE", "netMHCpan"),
        help="Absolute NetMHCpan executable path inside WSL.",
    )
    parser.add_argument("--wsl-distro", default="Ubuntu")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute allele outputs that already exist.",
    )
    args = parser.parse_args()

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {MANIFEST_PATH}. Run build_external_queries.py first."
        )

    common.enable_project_package_imports()
    from extra1.issue5.run_predictors import run_netmhcpan

    run_netmhcpan(
        args.executable,
        MANIFEST_PATH,
        RAW_DIR / "premium_test_netmhcpan.metadata.json",
        args.force,
        args.wsl_distro,
    )


if __name__ == "__main__":
    main()
