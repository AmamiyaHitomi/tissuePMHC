"""Run frozen NetMHCpan-4.1b through WSL on the mouse test queries."""

from __future__ import annotations

import argparse
import os

import common
from _runner import run_callable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executable",
        default=os.environ.get("NETMHCPAN_EXECUTABLE", "netMHCpan"),
    )
    parser.add_argument("--wsl-distro", default="Ubuntu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = common.EXTERNAL_ROOT / "queries" / "mouse_test_netmhcpan_manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing {manifest}; build queries first.")
    common.enable_project_package_imports()
    from extra1.issue5.run_predictors import run_netmhcpan

    run_netmhcpan(
        args.executable,
        manifest,
        common.EXTERNAL_ROOT / "raw_outputs" / "mouse_test_netmhcpan.metadata.json",
        args.force,
        args.wsl_distro,
    )


run_callable("run_netmhcpan.py", main)
