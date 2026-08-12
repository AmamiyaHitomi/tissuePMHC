from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .common import DEFAULT_RESULTS, SPECS, atomic_json, build_query_frame, safe_name, sha256
except ImportError:
    from common import DEFAULT_RESULTS, SPECS, atomic_json, build_query_frame, safe_name, sha256


def build(output_dir: Path) -> None:
    query_dir = output_dir / "queries"
    query_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {
        "purpose": "Deduplicated frozen external-predictor queries for Issue 5",
        "species": {},
    }
    for species, spec in SPECS.items():
        queries = build_query_frame(species)
        query_path = query_dir / f"{species}_unique_peptide_mhc.csv.gz"
        queries.to_csv(query_path, index=False)

        mhcflurry = queries.rename(
            columns={"tool_allele": "allele", "peptide_sequence": "peptide"}
        )[["query_id", "allele", "peptide"]]
        mhcflurry_path = query_dir / f"{species}_mhcflurry_input.csv"
        mhcflurry.to_csv(mhcflurry_path, index=False)

        allele_rows: list[dict[str, object]] = []
        netmhc_dir = query_dir / "netmhcpan" / species
        netmhc_dir.mkdir(parents=True, exist_ok=True)
        for mhc, group in queries.groupby("mhc_restriction", sort=True):
            stem = safe_name(mhc)
            peptide_path = netmhc_dir / f"{stem}.pep"
            peptide_path.write_text(
                "\n".join(group["peptide_sequence"].astype(str)) + "\n", encoding="ascii"
            )
            allele_rows.append(
                {
                    "species": species,
                    "mhc_restriction": mhc,
                    "tool_allele": group["tool_allele"].iloc[0],
                    "safe_name": stem,
                    "n_queries": len(group),
                    "peptide_file": str(peptide_path.resolve()),
                    "expected_output": str((netmhc_dir / f"{stem}.xls").resolve()),
                    "command_template": (
                        f"netMHCpan -p {peptide_path.resolve()} "
                        f"-a {group['tool_allele'].iloc[0]} -BA -xls "
                        f"-xlsfile {(netmhc_dir / f'{stem}.xls').resolve()}"
                    ),
                }
            )
        import pandas as pd

        allele_manifest = pd.DataFrame(allele_rows)
        allele_path = query_dir / f"{species}_netmhcpan_manifest.csv"
        allele_manifest.to_csv(allele_path, index=False)
        metadata["species"][species] = {
            "train": str(spec.train.resolve()),
            "train_sha256": sha256(spec.train),
            "test": str(spec.test.resolve()),
            "test_sha256": sha256(spec.test),
            "n_queries": int(len(queries)),
            "n_alleles": int(queries["mhc_restriction"].nunique()),
            "query_file": str(query_path.resolve()),
            "query_sha256": sha256(query_path),
            "mhcflurry_input": str(mhcflurry_path.resolve()),
            "netmhcpan_manifest": str(allele_path.resolve()),
        }
    atomic_json(query_dir / "metadata.json", metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(args.output_dir)
