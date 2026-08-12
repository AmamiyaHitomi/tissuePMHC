#!/usr/bin/env python3
"""Build test-only MHCflurry and NetMHCpan queries for humanPMHC premium."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import common


QUERY_DIR = common.EXTERNAL_ROOT / "queries"
NETMHCPAN_DIR = QUERY_DIR / "netmhcpan"


def query_id(peptide: str, hla: str) -> str:
    payload = f"humanPMHC_premium_test\t{peptide}\t{hla}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def safe_name(hla: str) -> str:
    return (
        hla.replace("*", "_")
        .replace(":", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def validate_test(test: pd.DataFrame) -> None:
    required = {
        "dataset",
        "split",
        "sample_id",
        "pair_id",
        "label",
        "target_tissue",
        "mhc_restriction",
        "peptide_sequence",
    }
    missing = required - set(test.columns)
    if missing:
        raise ValueError(f"Premium test is missing columns: {sorted(missing)}")
    if set(test["dataset"].astype(str)) != {"humanPMHC"}:
        raise ValueError("Premium test contains an unexpected dataset value.")
    if set(test["split"].astype(str)) != {"test"}:
        raise ValueError("Premium test contains an unexpected split value.")
    if test["sample_id"].duplicated().any():
        raise ValueError("Premium test sample_id values must be unique.")
    if not test["mhc_restriction"].astype(str).str.startswith("HLA-").all():
        raise ValueError("Premium test contains a non-HLA restriction.")
    if not test["peptide_sequence"].astype(str).str.fullmatch(
        r"[ACDEFGHIKLMNPQRSTVWY]{9}"
    ).all():
        raise ValueError("Premium test must contain canonical 9-mer peptides.")
    pairs = test.groupby("pair_id")["label"].agg(["size", "sum", "nunique"])
    if not ((pairs["size"] == 2) & (pairs["sum"] == 1) & (pairs["nunique"] == 2)).all():
        raise ValueError("Every premium test pair must contain one positive and one negative.")


def main() -> None:
    test = pd.read_csv(common.TEST_PATH)
    validate_test(test)
    queries = test[["peptide_sequence", "mhc_restriction"]].drop_duplicates().copy()
    queries.insert(
        0,
        "query_id",
        [
            query_id(peptide, hla)
            for peptide, hla in zip(
                queries["peptide_sequence"],
                queries["mhc_restriction"],
                strict=True,
            )
        ],
    )
    if queries["query_id"].duplicated().any():
        raise AssertionError("External query_id collision detected.")
    queries.insert(0, "species", "human")
    queries["tool_allele"] = queries["mhc_restriction"]
    queries = queries.sort_values(
        ["mhc_restriction", "peptide_sequence"], kind="stable"
    ).reset_index(drop=True)

    QUERY_DIR.mkdir(parents=True, exist_ok=True)
    query_path = QUERY_DIR / "premium_test_unique_peptide_hla.csv.gz"
    queries.to_csv(query_path, index=False)

    mhcflurry_input = queries.rename(
        columns={"tool_allele": "allele", "peptide_sequence": "peptide"}
    )[["query_id", "allele", "peptide"]]
    mhcflurry_path = QUERY_DIR / "premium_test_mhcflurry_input.csv"
    mhcflurry_input.to_csv(mhcflurry_path, index=False)

    NETMHCPAN_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    for hla, group in queries.groupby("mhc_restriction", sort=True):
        stem = safe_name(hla)
        peptide_path = NETMHCPAN_DIR / f"{stem}.pep"
        peptide_path.write_text(
            "\n".join(group["peptide_sequence"].astype(str)) + "\n",
            encoding="ascii",
        )
        output_path = NETMHCPAN_DIR / f"{stem}.xls"
        manifest_rows.append(
            {
                "species": "human",
                "mhc_restriction": hla,
                "tool_allele": hla,
                "safe_name": stem,
                "n_queries": int(len(group)),
                "peptide_file": str(peptide_path.resolve()),
                "expected_output": str(output_path.resolve()),
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = QUERY_DIR / "premium_test_netmhcpan_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    metadata = {
        "purpose": "Test-only frozen external predictor queries for humanPMHC premium",
        "test": str(common.TEST_PATH),
        "test_rows": int(len(test)),
        "test_pairs": int(test["pair_id"].nunique()),
        "n_unique_peptide_hla_queries": int(len(queries)),
        "n_hla_alleles": int(queries["mhc_restriction"].nunique()),
        "query_file": str(query_path),
        "mhcflurry_input": str(mhcflurry_path),
        "netmhcpan_manifest": str(manifest_path),
    }
    (QUERY_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

