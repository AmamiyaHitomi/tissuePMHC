#!/usr/bin/env python3
"""Download parent protein sequences and UniProt-to-Ensembl mappings for E.

Only UniProt accessions present in premium train are submitted. Results are
cached as a FASTA and a two-column mapping table under extra_premium/external.
The script uses the documented UniProt ID-mapping REST workflow and never reads
premium test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests


EXPERIMENTS_DIR = Path(__file__).resolve().parent
EXTRA_PREMIUM_DIR = EXPERIMENTS_DIR.parent
if str(EXTRA_PREMIUM_DIR) not in sys.path:
    sys.path.insert(0, str(EXTRA_PREMIUM_DIR))

import common


OUTPUT_DIR = EXTRA_PREMIUM_DIR / "external" / "mechanism"
DEFAULT_FASTA = OUTPUT_DIR / "premium_train_parent_uniprot.fasta"
DEFAULT_MAPPING = OUTPUT_DIR / "premium_train_uniprot_to_ensembl.csv"
DEFAULT_METADATA = OUTPUT_DIR / "premium_train_uniprot_download_metadata.json"
API = "https://rest.uniprot.org"
ENSG_PATTERN = re.compile(r"ENSG\d{11}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-fasta", type=Path, default=DEFAULT_FASTA)
    result.add_argument("--output-mapping", type=Path, default=DEFAULT_MAPPING)
    result.add_argument("--metadata-json", type=Path, default=DEFAULT_METADATA)
    result.add_argument("--poll-seconds", type=float, default=2.0)
    result.add_argument("--timeout-seconds", type=float, default=1800.0)
    result.add_argument("--request-timeout", type=float, default=120.0)
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pick_column(frame: pd.DataFrame, options: tuple[str, ...]) -> str:
    normalized = {str(column).lower().strip(): str(column) for column in frame}
    for option in options:
        if option.lower() in normalized:
            return normalized[option.lower()]
    raise ValueError(f"UniProt result lacks any of columns {options}; got {list(frame)}")


def main() -> None:
    args = parser().parse_args()
    if args.poll_seconds <= 0 or args.timeout_seconds <= 0 or args.request_timeout <= 0:
        raise ValueError("Timeout and polling values must be positive.")
    train = pd.read_csv(common.TRAIN_PATH, usecols=["molecule_parent_uniprot_id"])
    accessions = sorted(
        set(train["molecule_parent_uniprot_id"].dropna().astype(str).str.strip())
    )
    if not accessions:
        raise ValueError("No parent UniProt accessions found in premium train.")
    session = requests.Session()
    session.headers["User-Agent"] = "tissuePMHC-premium-E/1.0"
    response = session.post(
        f"{API}/idmapping/run",
        data={
            "from": "UniProtKB_AC-ID",
            "to": "UniProtKB",
            "ids": ",".join(accessions),
        },
        timeout=args.request_timeout,
    )
    response.raise_for_status()
    job_id = response.json()["jobId"]
    started = time.monotonic()
    while True:
        status_response = session.get(
            f"{API}/idmapping/status/{job_id}", timeout=args.request_timeout
        )
        status_response.raise_for_status()
        status = status_response.json()
        if status.get("jobStatus") == "RUNNING":
            if time.monotonic() - started > args.timeout_seconds:
                raise TimeoutError(f"UniProt mapping job timed out: {job_id}")
            time.sleep(args.poll_seconds)
            continue
        if status.get("jobStatus") in {"FAILED", "ERROR"}:
            raise RuntimeError(f"UniProt mapping failed: {status}")
        break
    details_response = session.get(
        f"{API}/idmapping/details/{job_id}", timeout=args.request_timeout
    )
    details_response.raise_for_status()
    redirect = details_response.json()["redirectURL"]
    stream_url = redirect.replace("/results/", "/results/stream/")
    separator = "&" if "?" in stream_url else "?"
    fields = "accession,sequence,xref_ensembl,gene_primary,reviewed"
    result_response = session.get(
        f"{stream_url}{separator}format=tsv&fields={fields}",
        timeout=args.request_timeout,
    )
    result_response.raise_for_status()
    args.output_fasta.parent.mkdir(parents=True, exist_ok=True)
    args.output_mapping.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_fasta.parent / "premium_train_uniprot_mapping_raw.tsv"
    raw_path.write_bytes(result_response.content)
    result = pd.read_csv(raw_path, sep="\t", dtype=str).fillna("")
    from_column = pick_column(result, ("From",))
    entry_column = pick_column(result, ("Entry", "accession"))
    sequence_column = pick_column(result, ("Sequence",))
    ensembl_column = pick_column(result, ("Ensembl",))
    gene_column = next(
        (column for column in result if str(column).lower().startswith("gene names")),
        None,
    )

    fasta_lines: list[str] = []
    mapping_rows: list[dict[str, str]] = []
    mapped_accessions: set[str] = set()
    sequence_accessions: set[str] = set()
    for row in result.itertuples(index=False, name=None):
        record = dict(zip(result.columns, row))
        requested = str(record[from_column]).strip()
        primary = str(record[entry_column]).strip()
        sequence = str(record[sequence_column]).strip().upper()
        gene_name = str(record.get(gene_column, "")).split()[0] if gene_column else ""
        if requested and sequence:
            fasta_lines.extend([
                f">sp|{requested}|mapped_primary={primary}",
                *[sequence[index:index + 60] for index in range(0, len(sequence), 60)],
            ])
            sequence_accessions.add(requested)
        gene_ids = sorted(set(ENSG_PATTERN.findall(str(record[ensembl_column]))))
        for gene_id in gene_ids:
            mapping_rows.append({
                "uniprot_id": requested,
                "gene_id": gene_id,
                "gene_name": gene_name,
                "primary_uniprot_id": primary,
            })
            mapped_accessions.add(requested)
    if not fasta_lines:
        raise ValueError("UniProt returned no protein sequences.")
    args.output_fasta.write_text("\n".join(fasta_lines) + "\n", encoding="utf-8")
    pd.DataFrame(
        mapping_rows,
        columns=["uniprot_id", "gene_id", "gene_name", "primary_uniprot_id"],
    ).drop_duplicates().to_csv(args.output_mapping, index=False)
    missing_sequence = sorted(set(accessions) - sequence_accessions)
    missing_gene = sorted(set(accessions) - mapped_accessions)
    metadata = {
        "schema_version": 1,
        "test_data_read": False,
        "uniprot_api": API,
        "download_date": date.today().isoformat(),
        "uniprot_release": result_response.headers.get("x-uniprot-release", "unknown"),
        "job_id": job_id,
        "requested_accessions": len(accessions),
        "sequence_accessions": len(sequence_accessions),
        "ensembl_mapped_accessions": len(mapped_accessions),
        "missing_sequence_accessions": missing_sequence,
        "missing_ensembl_mapping_accessions": missing_gene,
        "files": {
            str(args.output_fasta): sha256(args.output_fasta),
            str(args.output_mapping): sha256(args.output_mapping),
            str(raw_path): sha256(raw_path),
        },
    }
    args.metadata_json.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
