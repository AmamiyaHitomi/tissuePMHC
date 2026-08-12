"""Build test-only MHCflurry and NetMHCpan queries for mouse occurrence-equal."""

from __future__ import annotations

import hashlib
import json

import pandas as pd

import common
from _runner import run_callable


QUERY_DIR = common.EXTERNAL_ROOT / "queries"
NETMHCPAN_DIR = QUERY_DIR / "netmhcpan"


def query_id(peptide: str, mhc: str) -> str:
    return hashlib.sha256(
        f"mousePMHC_occurrence_equal_test\t{peptide}\t{mhc}".encode("utf-8")
    ).hexdigest()[:24]


def tool_allele(mhc: str) -> str:
    if not mhc.startswith("H2-"):
        raise ValueError(f"Unsupported mouse MHC name: {mhc}")
    return "H-2-" + mhc[3:]


def safe_name(mhc: str) -> str:
    return tool_allele(mhc).replace("-", "_").replace("/", "_")


def main() -> None:
    test = pd.read_csv(common.TEST_PATH, keep_default_na=False)
    required = common.REQUIRED_COLUMNS
    missing = required - set(test.columns)
    if missing:
        raise ValueError(f"Mouse test misses columns: {sorted(missing)}")
    if set(test["dataset"].astype(str)) != {"mousePMHC"} or set(test["split"]) != {"test"}:
        raise ValueError("Unexpected dataset or split in mouse test.")
    if not test["mhc_restriction"].astype(str).str.startswith("H2-").all():
        raise ValueError("Mouse test contains a non-H2 restriction.")
    if not test["peptide_sequence"].astype(str).str.fullmatch(
        r"[ACDEFGHIKLMNPQRSTVWY]{9}"
    ).all():
        raise ValueError("Mouse test contains a non-canonical 9-mer.")
    pair_check = test.groupby("pair_id")["label"].agg(["size", "sum", "nunique"])
    if not (
        pair_check["size"].eq(2)
        & pair_check["sum"].eq(1)
        & pair_check["nunique"].eq(2)
    ).all():
        raise ValueError("Mouse test pairs are incomplete.")

    queries = test[["peptide_sequence", "mhc_restriction"]].drop_duplicates().copy()
    queries.insert(
        0,
        "query_id",
        [query_id(peptide, mhc) for peptide, mhc in zip(
            queries["peptide_sequence"], queries["mhc_restriction"], strict=True
        )],
    )
    if queries["query_id"].duplicated().any():
        raise AssertionError("External query_id collision detected.")
    queries.insert(0, "species", "mouse")
    queries["tool_allele"] = queries["mhc_restriction"].map(tool_allele)
    queries = queries.sort_values(
        ["mhc_restriction", "peptide_sequence"], kind="stable"
    ).reset_index(drop=True)

    QUERY_DIR.mkdir(parents=True, exist_ok=True)
    query_path = QUERY_DIR / "mouse_test_unique_peptide_h2.csv.gz"
    queries.to_csv(query_path, index=False)
    mhcflurry_path = QUERY_DIR / "mouse_test_mhcflurry_input.csv"
    queries.rename(columns={"tool_allele": "allele", "peptide_sequence": "peptide"})[
        ["query_id", "allele", "peptide"]
    ].to_csv(mhcflurry_path, index=False)

    NETMHCPAN_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for mhc, group in queries.groupby("mhc_restriction", sort=True):
        stem = safe_name(mhc)
        peptide_path = NETMHCPAN_DIR / f"{stem}.pep"
        peptide_path.write_text(
            "\n".join(group["peptide_sequence"].astype(str)) + "\n", encoding="ascii"
        )
        manifest_rows.append(
            {
                "species": "mouse",
                "mhc_restriction": mhc,
                "tool_allele": tool_allele(mhc),
                "safe_name": stem,
                "n_queries": int(len(group)),
                "peptide_file": str(peptide_path.resolve()),
                "expected_output": str((NETMHCPAN_DIR / f"{stem}.xls").resolve()),
            }
        )
    manifest_path = QUERY_DIR / "mouse_test_netmhcpan_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    metadata = {
        "purpose": "Test-only frozen external predictions for mouse occurrence-equal",
        "test": str(common.TEST_PATH),
        "test_rows": int(len(test)),
        "test_pairs": int(test["pair_id"].nunique()),
        "n_unique_peptide_h2_queries": int(len(queries)),
        "n_h2_alleles": int(queries["mhc_restriction"].nunique()),
        "query_file": str(query_path),
        "mhcflurry_input": str(mhcflurry_path),
        "netmhcpan_manifest": str(manifest_path),
    }
    (QUERY_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


run_callable("build_external_queries.py", main)
