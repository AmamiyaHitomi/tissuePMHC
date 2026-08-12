from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .common import DEFAULT_RESULTS, atomic_json, sha256
except ImportError:
    from common import DEFAULT_RESULTS, atomic_json, sha256


def run_command(command: list[str], log_path: Path) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}; see {log_path}"
        )
    return completed


def command_text(command: list[str]) -> str:
    return shlex.join(command)


def wsl_path(path: Path, distro: str) -> str:
    del distro  # The standard /mnt/<drive> mapping is distro-independent.
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    if len(drive) != 1 or not drive.isalpha():
        raise ValueError(f"Cannot map non-drive Windows path into WSL: {resolved}")
    relative_parts = resolved.parts[1:]
    return f"/mnt/{drive}/{'/'.join(relative_parts)}"


def netmhcpan_allele(allele: str) -> str:
    """Convert project HLA notation to the standalone NetMHCpan spelling."""
    return allele.replace("*", "") if allele.startswith("HLA-") else allele


def run_mhcflurry(
    executable: str,
    input_path: Path,
    output_path: Path,
    metadata_path: Path,
    legacy_cli: bool,
    model_dir: Path | None,
    num_jobs: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if legacy_cli:
        command = [executable, str(input_path.resolve()), "--out", str(output_path.resolve())]
    else:
        command = [
            executable,
            "predict",
            str(input_path.resolve()),
            "--out",
            str(output_path.resolve()),
        ]
    command.extend(["--no-throw", "--no-flanking"])
    if num_jobs.lower() != "none":
        command.extend(["--num-jobs", num_jobs])
    if model_dir is not None:
        command.extend(["--models", str(model_dir.resolve())])
    run_command(command, output_path.with_suffix(".log"))
    if not output_path.exists():
        raise RuntimeError("MHCflurry completed without creating the requested output.")
    atomic_json(
        metadata_path,
        {
            "tool": "MHCflurry",
            "run_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "command": command_text(command),
            "input": str(input_path.resolve()),
            "input_sha256": sha256(input_path),
            "output": str(output_path.resolve()),
            "output_sha256": sha256(output_path),
            "model_dir": str(model_dir.resolve()) if model_dir else None,
            "model_files": (
                {
                    str(path.relative_to(model_dir)): sha256(path)
                    for path in sorted(model_dir.rglob("*"))
                    if path.is_file()
                }
                if model_dir
                else "default bundle; resolve and record its path before formal run"
            ),
            "flanking_policy": "disabled",
        },
    )


def run_netmhcpan(
    executable: str,
    manifest_path: Path,
    metadata_path: Path,
    force: bool,
    wsl_distro: str | None,
) -> None:
    manifest = pd.read_csv(manifest_path)
    required = {"tool_allele", "peptide_file", "expected_output"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"NetMHCpan manifest misses columns: {missing}")
    runs: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        peptide_file = Path(row.peptide_file)
        output = Path(row.expected_output)
        if output.exists() and not force:
            runs.append(
                {
                    "allele": row.tool_allele,
                    "status": "reused",
                    "output": str(output.resolve()),
                    "output_sha256": sha256(output),
                }
            )
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        predictor_args = [
            executable,
            "-p",
            (
                wsl_path(peptide_file, wsl_distro)
                if wsl_distro
                else str(peptide_file.resolve())
            ),
            "-a",
            netmhcpan_allele(str(row.tool_allele)),
            "-BA",
            "-xls",
            "-xlsfile",
            wsl_path(output, wsl_distro) if wsl_distro else str(output.resolve()),
        ]
        command = (
            ["wsl.exe", "-d", wsl_distro, "--", *predictor_args]
            if wsl_distro
            else predictor_args
        )
        run_command(command, output.with_suffix(".log"))
        if not output.exists():
            raise RuntimeError(f"NetMHCpan did not create {output}")
        runs.append(
            {
                "allele": row.tool_allele,
                "status": "completed",
                "command": command_text(command),
                "input": str(peptide_file.resolve()),
                "input_sha256": sha256(peptide_file),
                "output": str(output.resolve()),
                "output_sha256": sha256(output),
            }
        )
    atomic_json(
        metadata_path,
        {
            "tool": "NetMHCpan",
            "declared_version": "4.1",
            "run_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "executable": executable,
            "wsl_distro": wsl_distro,
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": sha256(manifest_path),
            "runs": runs,
            "formal_run_requirement": (
                "Verify the executable/data package version and academic license; "
                "archive the package data-file hashes outside the repository if redistribution "
                "is not permitted."
            ),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="predictor", required=True)

    mhcflurry = subparsers.add_parser("mhcflurry")
    mhcflurry.add_argument("--executable", default="mhcflurry")
    mhcflurry.add_argument("--input", type=Path, required=True)
    mhcflurry.add_argument("--output", type=Path, required=True)
    mhcflurry.add_argument("--metadata", type=Path, required=True)
    mhcflurry.add_argument("--legacy-cli", action="store_true")
    mhcflurry.add_argument("--model-dir", type=Path)
    mhcflurry.add_argument(
        "--num-jobs",
        default="none",
        help="Pass a worker count only for MHCflurry versions whose predict CLI supports it.",
    )

    netmhcpan = subparsers.add_parser("netmhcpan")
    netmhcpan.add_argument("--executable", required=True)
    netmhcpan.add_argument("--manifest", type=Path, required=True)
    netmhcpan.add_argument("--metadata", type=Path, required=True)
    netmhcpan.add_argument("--force", action="store_true")
    netmhcpan.add_argument(
        "--wsl-distro",
        help="Run the Linux executable through this WSL distro and translate paths.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.predictor == "mhcflurry":
        run_mhcflurry(
            args.executable,
            args.input,
            args.output,
            args.metadata,
            args.legacy_cli,
            args.model_dir,
            args.num_jobs,
        )
    else:
        run_netmhcpan(
            args.executable,
            args.manifest,
            args.metadata,
            args.force,
            args.wsl_distro,
        )
