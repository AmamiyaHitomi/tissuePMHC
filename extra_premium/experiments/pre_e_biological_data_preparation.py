#!/usr/bin/env python3
"""Pre-E: prepare and audit biological expression inputs for experiment E.

Pre-E0 downloads or reads Human Protein Atlas consensus tissue RNA and converts
it to the exact long-table schema consumed by E. Pre-E1 maps unique premium parent UniProt accessions
to Ensembl genes through the official UniProt ID Mapping API (or an supplied
offline TSV). Pre-E2 writes a tissue-mapping review template but never approves
biologically ambiguous mappings automatically. Pre-E3 runs row-level coverage and
missingness audits after a user-reviewed tissue mapping is supplied.

F is deterministic data preparation, so it has no training seed. The aligned
E0-E4 model screen is single-seed by default (20260704).

Elapsed time is printed only. No elapsed/duration value is persisted.
The fixed premium test set is never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


EXPERIMENTS_DIR = Path(__file__).resolve().parent
EXTRA_PREMIUM_DIR = EXPERIMENTS_DIR.parent
if str(EXTRA_PREMIUM_DIR) not in sys.path:
    sys.path.insert(0, str(EXTRA_PREMIUM_DIR))

import common


HPA_VERSION = "25.1"
HPA_ENSEMBL_VERSION = "109"
HPA_URL = "https://www.proteinatlas.org/download/tsv/rna_tissue_consensus.tsv.zip"
HPA_PAGE = "https://www.proteinatlas.org/humanproteome/tissue/data"
HPA_LICENSE = "CC BY 4.0"
UNIPROT_API = "https://rest.uniprot.org"
UNIPROT_HELP = "https://www.uniprot.org/help/id_mapping_prog"
DEFAULT_OUTPUT_DIR = (
    common.PROJECT_ROOT / "data" / "expression" / "hpa_v25_1"
)
APPROVED_STATUSES = {"approved", "reviewed"}


TISSUE_SUGGESTIONS: dict[str, tuple[str, str, str]] = {
    "blood": (
        "",
        "unresolved",
        "HPA consensus has no single whole-blood tissue; do not substitute blood vessel or bone marrow.",
    ),
    "bone": (
        "",
        "unresolved",
        "Bone marrow is not automatically equivalent to bone.",
    ),
    "brain": (
        "",
        "unresolved",
        "HPA contains multiple brain regions; an aggregation rule needs biological review.",
    ),
    "breast": ("breast", "exact", "Exact name match."),
    "colon": ("colon", "exact", "Exact name match."),
    "kidney": ("kidney", "exact", "Exact name match."),
    "lung": ("lung", "exact", "Exact name match."),
    "lymph node": ("lymph node", "exact", "Exact name match."),
    "lymphoid": (
        "",
        "unresolved",
        "Lymphoid is a broad category, not one HPA tissue.",
    ),
    "ovary": ("ovary", "exact", "Exact name match."),
    "spleen": ("spleen", "exact", "Exact name match."),
    "thymus": ("thymus", "exact", "Exact name match."),
    "umbilical cord blood": (
        "",
        "unresolved",
        "Not represented in HPA consensus tissue RNA.",
    ),
    "uterine cervix": (
        "cervix",
        "synonym_candidate",
        "Likely anatomical synonym, but still requires human review.",
    ),
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Run Pre-E biological-expression data preparation."
    )
    result.add_argument(
        "--train-csv",
        type=Path,
        default=common.TRAIN_PATH,
        help="Premium train file. The fixed test file is never read.",
    )
    result.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    result.add_argument(
        "--hpa-zip",
        type=Path,
        help="Use an existing HPA zip instead of downloading it.",
    )
    result.add_argument("--hpa-url", default=HPA_URL)
    result.add_argument(
        "--uniprot-mapping-tsv",
        type=Path,
        help=(
            "Use an existing UniProt ID-mapping TSV with From/To columns "
            "instead of calling the API."
        ),
    )
    result.add_argument(
        "--approved-tissue-mapping-csv",
        type=Path,
        help=(
            "Human-reviewed target_tissue/expression_tissue/review_status "
            "table. If omitted, Pre-E stops after writing the review template."
        ),
    )
    result.add_argument(
        "--refresh",
        action="store_true",
        help="Redownload/requery files that already exist in output-dir.",
    )
    result.add_argument("--http-timeout", type=float, default=60.0)
    result.add_argument("--mapping-timeout", type=float, default=900.0)
    result.add_argument("--poll-seconds", type=float, default=3.0)
    result.add_argument("--uniprot-chunk-size", type=int, default=50000)
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request(
    url: str,
    *,
    data: bytes | None = None,
    timeout: float,
    accept: str = "application/json",
    retries: int = 4,
    extra_headers: dict[str, str] | None = None,
) -> bytes:
    headers = {
        # HPA's CDN currently stalls custom non-browser user agents on ZIP GETs.
        "User-Agent": "Mozilla/5.0",
        "Accept": accept,
    }
    if extra_headers:
        headers.update(extra_headers)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url, data=data, headers=headers, method="POST" if data else "GET"
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                content_length = response.headers.get("Content-Length")
                if content_length is not None:
                    return response.read(int(content_length))
                return response.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt + 1 == retries:
                break
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"HTTP request failed after {retries} attempts: {url}") from last_error


def download_file(url: str, target: Path, timeout: float) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    started = time.perf_counter()
    # HPA's current download endpoint can keep a normal GET response open after
    # sending the advertised payload. Fixed byte ranges avoid waiting for EOF
    # and also make the download visibly progress instead of appearing frozen.
    head = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        method="HEAD",
    )
    with urllib.request.urlopen(head, timeout=timeout) as response:
        total = int(response.headers.get("Content-Length", "0"))
    if total <= 0:
        raise RuntimeError(f"HPA server did not report a valid Content-Length: {url}")
    # The HPA CDN reliably serves small ranges but may stall on megabyte ranges.
    chunk_size = 64 * 1024
    downloaded = temporary.stat().st_size if temporary.exists() else 0
    if downloaded > total:
        temporary.unlink()
        downloaded = 0
    if downloaded:
        print(
            f"Pre-E0 HPA download: resuming at {downloaded:,}/{total:,} bytes",
            flush=True,
        )
    with temporary.open("ab") as handle:
        for start in range(downloaded, total, chunk_size):
            end = min(start + chunk_size, total) - 1
            payload = _request(
                url,
                timeout=timeout,
                accept="application/zip,application/octet-stream",
                extra_headers={"Range": f"bytes={start}-{end}"},
            )
            expected = end - start + 1
            if len(payload) != expected:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Incomplete HPA byte range {start}-{end}: "
                    f"expected {expected}, received {len(payload)}"
                )
            handle.write(payload)
            downloaded += len(payload)
            print(
                f"Pre-E0 HPA download: {downloaded:,}/{total:,} bytes",
                flush=True,
            )
    if not zipfile.is_zipfile(temporary):
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Downloaded HPA file is not a valid zip archive: {url}")
    temporary.replace(target)
    print(
        f"Pre-E0 HPA download complete: {downloaded:,} bytes, "
        f"elapsed={time.perf_counter() - started:.2f}s",
        flush=True,
    )
    return target


def _find_column(frame: pd.DataFrame, aliases: set[str], label: str) -> str:
    normalized = {
        str(column).strip().lower().replace("_", " "): str(column)
        for column in frame.columns
    }
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    raise ValueError(
        f"Could not find {label} column. Available columns={list(frame.columns)}"
    )


def read_hpa_zip(path: Path) -> pd.DataFrame:
    if not path.is_file() or not zipfile.is_zipfile(path):
        raise FileNotFoundError(f"HPA zip is missing or invalid: {path}")
    with zipfile.ZipFile(path) as archive:
        members = [
            name
            for name in archive.namelist()
            if not name.endswith("/") and name.lower().endswith((".tsv", ".txt"))
        ]
        if len(members) != 1:
            raise ValueError(
                f"Expected one TSV/TXT inside HPA zip, found {members}."
            )
        with archive.open(members[0]) as raw:
            frame = pd.read_csv(io.TextIOWrapper(raw, encoding="utf-8"), sep="\t")
    if frame.empty:
        raise ValueError("HPA expression table is empty.")
    gene = _find_column(frame, {"gene", "gene id", "ensembl"}, "gene")
    gene_name = _find_column(
        frame, {"gene name", "gene_name", "genesymbol", "gene symbol"}, "gene name"
    )
    tissue = _find_column(frame, {"tissue", "analysed tissue"}, "tissue")
    value = _find_column(frame, {"ntpm", "n tpm"}, "nTPM")
    output = frame[[gene, gene_name, tissue, value]].rename(
        columns={
            gene: "gene_id",
            gene_name: "gene_name",
            tissue: "expression_tissue",
            value: "expression_value",
        }
    )
    output["gene_id"] = (
        output["gene_id"].astype(str).str.strip().str.split(".").str[0]
    )
    output["expression_tissue"] = (
        output["expression_tissue"].astype(str).str.strip().str.lower()
    )
    output["gene_name"] = output["gene_name"].fillna("").astype(str).str.strip().str.upper()
    output["expression_value"] = pd.to_numeric(
        output["expression_value"], errors="raise"
    )
    if not output["gene_id"].str.fullmatch(r"ENSG\d+").all():
        bad = output.loc[
            ~output["gene_id"].str.fullmatch(r"ENSG\d+"), "gene_id"
        ].head(5).tolist()
        raise ValueError(f"HPA contains invalid Ensembl gene IDs: {bad}")
    if not np.isfinite(output["expression_value"]).all():
        raise ValueError("HPA expression contains NaN or infinite values.")
    if (output["expression_value"] < 0).any():
        raise ValueError("HPA nTPM values must be non-negative.")
    if output.duplicated(["gene_id", "expression_tissue"]).any():
        raise ValueError("HPA has duplicate gene/tissue expression rows.")
    return output.sort_values(
        ["gene_id", "gene_name", "expression_tissue"], ignore_index=True
    )


def load_train(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Premium train file does not exist: {path}")
    frame = pd.read_csv(path)
    required = {
        "sample_id",
        "label",
        "target_tissue",
        "molecule_parent_uniprot_id",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Premium train is missing columns: {sorted(missing)}")
    if "split" in frame and set(frame["split"].astype(str)) != {"train"}:
        raise ValueError("Pre-E preparation may read premium train rows only.")
    if frame["sample_id"].duplicated().any():
        raise ValueError("Premium train sample_id values must be unique.")
    if not set(frame["label"].astype(int)).issubset({0, 1}):
        raise ValueError("Premium train labels must be binary.")
    return frame


def unique_uniprot_ids(train: pd.DataFrame) -> list[str]:
    values = (
        train["molecule_parent_uniprot_id"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    values = values[values.ne("") & values.str.lower().ne("nan")]
    return sorted(values.unique().tolist())


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _mapping_stream_url(redirect_url: str) -> str:
    if "/idmapping/results/" in redirect_url:
        stream = redirect_url.replace("/idmapping/results/", "/idmapping/stream/")
    elif "/results/" in redirect_url:
        stream = redirect_url.replace("/results/", "/results/stream/")
    else:
        raise ValueError(f"Unexpected UniProt redirect URL: {redirect_url}")
    separator = "&" if "?" in stream else "?"
    return f"{stream}{separator}format=tsv"


def query_uniprot_chunk(
    identifiers: list[str],
    *,
    http_timeout: float,
    mapping_timeout: float,
    poll_seconds: float,
) -> pd.DataFrame:
    if not identifiers:
        return pd.DataFrame(columns=["From", "To"])
    payload = urllib.parse.urlencode(
        {
            "from": "UniProtKB_AC-ID",
            "to": "Ensembl",
            "ids": ",".join(identifiers),
        }
    ).encode("utf-8")
    submission = json.loads(
        _request(
            f"{UNIPROT_API}/idmapping/run",
            data=payload,
            timeout=http_timeout,
        ).decode("utf-8")
    )
    job_id = submission.get("jobId")
    if not job_id:
        raise RuntimeError(f"UniProt submission returned no jobId: {submission}")
    deadline = time.monotonic() + mapping_timeout
    while True:
        if time.monotonic() > deadline:
            raise TimeoutError(f"UniProt mapping job timed out: {job_id}")
        status = json.loads(
            _request(
                f"{UNIPROT_API}/idmapping/status/{job_id}",
                timeout=http_timeout,
            ).decode("utf-8")
        )
        job_status = str(status.get("jobStatus", "")).upper()
        if job_status in {"FAILED", "ERROR"}:
            raise RuntimeError(f"UniProt mapping failed: {status}")
        if job_status not in {"RUNNING", "NEW"}:
            break
        time.sleep(poll_seconds)
    details = json.loads(
        _request(
            f"{UNIPROT_API}/idmapping/details/{job_id}", timeout=http_timeout
        ).decode("utf-8")
    )
    redirect_url = details.get("redirectURL")
    if not redirect_url:
        raise RuntimeError(f"UniProt mapping details lack redirectURL: {details}")
    content = _request(
        _mapping_stream_url(str(redirect_url)),
        timeout=http_timeout,
        accept="text/tab-separated-values,text/plain",
    ).decode("utf-8")
    if not content.strip():
        return pd.DataFrame(columns=["From", "To"])
    return pd.read_csv(io.StringIO(content), sep="\t", dtype=str)


def normalize_uniprot_mapping(
    raw: pd.DataFrame, hpa_genes: set[str]
) -> pd.DataFrame:
    source = _find_column(raw, {"from", "uniprot", "uniprot id"}, "UniProt From")
    target = _find_column(raw, {"to", "ensembl", "gene id"}, "Ensembl To")
    output = raw[[source, target]].rename(
        columns={source: "uniprot_id", target: "gene_id"}
    )
    output = output.dropna().copy()
    output["uniprot_id"] = output["uniprot_id"].astype(str).str.strip()
    output["gene_id"] = (
        output["gene_id"].astype(str).str.strip().str.split(".").str[0]
    )
    output = output[
        output["uniprot_id"].ne("")
        & output["gene_id"].str.fullmatch(r"ENSG\d+")
    ]
    output["present_in_hpa"] = output["gene_id"].isin(hpa_genes)
    return output.drop_duplicates().sort_values(
        ["uniprot_id", "gene_id"], ignore_index=True
    )


def obtain_uniprot_mapping(
    identifiers: list[str],
    hpa_genes: set[str],
    raw_path: Path,
    provided_path: Path | None,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if provided_path is not None:
        if not provided_path.is_file():
            raise FileNotFoundError(f"UniProt mapping TSV is missing: {provided_path}")
        raw = pd.read_csv(provided_path, sep="\t", dtype=str)
    elif raw_path.exists() and not args.refresh:
        raw = pd.read_csv(raw_path, sep="\t", dtype=str)
    else:
        started = time.perf_counter()
        parts: list[pd.DataFrame] = []
        for index, chunk in enumerate(
            _chunks(identifiers, args.uniprot_chunk_size), start=1
        ):
            print(
                f"Pre-E1 UniProt mapping chunk {index}: {len(chunk):,} IDs",
                flush=True,
            )
            parts.append(
                query_uniprot_chunk(
                    chunk,
                    http_timeout=args.http_timeout,
                    mapping_timeout=args.mapping_timeout,
                    poll_seconds=args.poll_seconds,
                )
            )
        raw = (
            pd.concat(parts, ignore_index=True)
            if parts
            else pd.DataFrame(columns=["From", "To"])
        )
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(raw_path, sep="\t", index=False)
        print(
            f"Pre-E1 UniProt API mapping complete: elapsed="
            f"{time.perf_counter() - started:.2f}s",
            flush=True,
        )
    if raw.empty:
        return pd.DataFrame(
            columns=["uniprot_id", "gene_id", "present_in_hpa"]
        )
    return normalize_uniprot_mapping(raw, hpa_genes)


def tissue_review_template(
    premium_tissues: list[str], hpa_tissues: set[str]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for tissue in premium_tissues:
        suggestion, mapping_type, note = TISSUE_SUGGESTIONS.get(
            tissue,
            (
                tissue if tissue in hpa_tissues else "",
                "exact" if tissue in hpa_tissues else "unresolved",
                "Automatically detected name match; still requires human review."
                if tissue in hpa_tissues
                else "No automatic suggestion.",
            ),
        )
        available = bool(suggestion and suggestion in hpa_tissues)
        rows.append(
            {
                "target_tissue": tissue,
                "expression_tissue": suggestion,
                "review_status": "pending_review",
                "mapping_type": mapping_type,
                "suggestion_available_in_hpa": available,
                "review_note": note,
                "reviewer": "",
                "review_date": "",
            }
        )
    return pd.DataFrame(rows)


def validate_approved_tissue_mapping(
    path: Path,
    premium_tissues: set[str],
    hpa_tissues: set[str],
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Approved tissue mapping is missing: {path}")
    frame = pd.read_csv(path, dtype=str).fillna("")
    required = {"target_tissue", "expression_tissue", "review_status"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Approved tissue mapping lacks columns: {sorted(missing)}")
    for column in required:
        frame[column] = frame[column].astype(str).str.strip()
    if set(frame["target_tissue"]) != premium_tissues:
        raise ValueError(
            "Approved tissue mapping must exactly cover premium tissues; "
            f"missing={sorted(premium_tissues - set(frame['target_tissue']))}, "
            f"extra={sorted(set(frame['target_tissue']) - premium_tissues)}"
        )
    bad_status = ~frame["review_status"].str.lower().isin(APPROVED_STATUSES)
    if bad_status.any():
        raise ValueError(
            "Every tissue mapping must be approved/reviewed before Pre-E3; "
            f"pending={frame.loc[bad_status, 'target_tissue'].tolist()}"
        )
    bad_tissue = ~frame["expression_tissue"].isin(hpa_tissues)
    if bad_tissue.any():
        raise ValueError(
            "Approved mappings reference tissues absent from HPA: "
            f"{frame.loc[bad_tissue, ['target_tissue', 'expression_tissue']].to_dict('records')}"
        )
    if "weight" not in frame:
        frame["weight"] = 1.0
    frame["weight"] = pd.to_numeric(frame["weight"], errors="raise")
    if (frame["weight"] <= 0).any() or not np.isfinite(frame["weight"]).all():
        raise ValueError("Tissue mapping weights must be finite and positive.")
    if "mapping_quality" not in frame:
        frame["mapping_quality"] = "unspecified"
    frame["mapping_quality"] = frame["mapping_quality"].astype(str).str.strip()
    return frame


def mapping_audit(
    train: pd.DataFrame,
    identifiers: list[str],
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    mapped = set(mapping["uniprot_id"])
    in_hpa = set(mapping.loc[mapping["present_in_hpa"], "uniprot_id"])
    one_to_many = mapping.groupby("uniprot_id")["gene_id"].nunique().gt(1)
    row_ids = train["molecule_parent_uniprot_id"].astype("string")
    rows = [
        {"dimension": "overall", "group": "all", "metric": "train_rows", "value": len(train)},
        {"dimension": "overall", "group": "all", "metric": "unique_uniprot", "value": len(identifiers)},
        {"dimension": "overall", "group": "all", "metric": "mapped_uniprot", "value": len(set(identifiers) & mapped)},
        {"dimension": "overall", "group": "all", "metric": "hpa_gene_mapped_uniprot", "value": len(set(identifiers) & in_hpa)},
        {"dimension": "overall", "group": "all", "metric": "one_to_many_uniprot", "value": int(one_to_many.sum())},
        {"dimension": "overall", "group": "all", "metric": "uniprot_mapping_coverage", "value": len(set(identifiers) & mapped) / len(identifiers) if identifiers else 0.0},
        {"dimension": "overall", "group": "all", "metric": "hpa_usable_uniprot_coverage", "value": len(set(identifiers) & in_hpa) / len(identifiers) if identifiers else 0.0},
        {"dimension": "overall", "group": "all", "metric": "train_row_hpa_gene_mapping_rate", "value": float(row_ids.isin(in_hpa).mean())},
    ]
    return pd.DataFrame(rows)


def row_expression_audit(
    train: pd.DataFrame,
    expression: pd.DataFrame,
    mapping: pd.DataFrame,
    tissue_mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = mapping[mapping["present_in_hpa"]][["uniprot_id", "gene_id"]]
    protein_expression = usable.merge(
        expression, on="gene_id", how="left", validate="many_to_many"
    )
    protein_expression = (
        protein_expression.groupby(
            ["uniprot_id", "expression_tissue"], as_index=False, sort=True
        )["expression_value"]
        .mean()
    )
    weighted = tissue_mapping[["target_tissue", "expression_tissue", "weight"]].merge(
        protein_expression,
        on="expression_tissue",
        how="left",
        validate="many_to_many",
    )
    weighted["weighted_expression"] = weighted["expression_value"] * weighted["weight"]
    numerator = weighted.groupby(
        ["target_tissue", "uniprot_id"], as_index=False, dropna=False
    )["weighted_expression"].sum(min_count=1)
    denominator = weighted.groupby(
        ["target_tissue", "uniprot_id"], as_index=False, dropna=False
    )["weight"].sum()
    target_expression = numerator.merge(
        denominator,
        on=["target_tissue", "uniprot_id"],
        validate="one_to_one",
    )
    target_expression["expression_value"] = (
        target_expression["weighted_expression"] / target_expression["weight"]
    )
    target_expression = target_expression[
        ["target_tissue", "uniprot_id", "expression_value"]
    ]

    rows = train[
        [
            "sample_id",
            "label",
            "target_tissue",
            "molecule_parent_uniprot_id",
        ]
    ].copy()
    rows["molecule_parent_uniprot_id"] = rows[
        "molecule_parent_uniprot_id"
    ].astype("string")
    rows = rows.merge(
        target_expression.rename(columns={"uniprot_id": "molecule_parent_uniprot_id"}),
        on=["molecule_parent_uniprot_id", "target_tissue"],
        how="left",
        validate="many_to_one",
    )
    rows["expression_missing"] = rows["expression_value"].isna().astype(int)
    audit_rows: list[dict[str, object]] = []
    groups: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", rows)]
    groups.extend(
        ("label", str(label), group)
        for label, group in rows.groupby("label", sort=True)
    )
    groups.extend(
        ("tissue", str(tissue), group)
        for tissue, group in rows.groupby("target_tissue", sort=True)
    )
    for dimension, group_name, group in groups:
        audit_rows.append(
            {
                "dimension": dimension,
                "group": group_name,
                "n_rows": len(group),
                "mapped_expression_rows": int((group["expression_missing"] == 0).sum()),
                "expression_missing_rate": float(group["expression_missing"].mean()),
            }
        )
    return rows, pd.DataFrame(audit_rows)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_all_pre_e(cli_args: argparse.Namespace | None = None) -> None:
    args = cli_args or parser().parse_args()
    if args.http_timeout <= 0 or args.mapping_timeout <= 0:
        raise ValueError("HTTP and mapping timeouts must be positive.")
    if args.poll_seconds <= 0 or not (1 <= args.uniprot_chunk_size <= 100000):
        raise ValueError("Invalid poll interval or UniProt chunk size.")
    args.train_csv = args.train_csv.resolve()
    args.output_dir = args.output_dir.resolve()
    output = args.output_dir
    raw_dir = output / "raw"
    processed_dir = output / "processed"
    audit_dir = output / "audit"
    e_ready_dir = output / "e_ready"
    for directory in (raw_dir, processed_dir, audit_dir):
        directory.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    train = load_train(args.train_csv)
    identifiers = unique_uniprot_ids(train)
    pd.DataFrame({"uniprot_id": identifiers}).to_csv(
        processed_dir / "premium_unique_uniprot_ids.csv", index=False
    )

    if args.hpa_zip is not None:
        hpa_zip = args.hpa_zip.resolve()
    else:
        hpa_zip = raw_dir / "rna_tissue_consensus.tsv.zip"
        if args.refresh or not hpa_zip.exists():
            download_file(args.hpa_url, hpa_zip, args.http_timeout)
    expression = read_hpa_zip(hpa_zip)
    expression_path = processed_dir / "hpa_consensus_expression.csv.gz"
    expression.to_csv(expression_path, index=False, compression="gzip")
    print(
        f"Pre-E0 HPA conversion: genes={expression['gene_id'].nunique():,}, "
        f"tissues={expression['expression_tissue'].nunique():,}, "
        f"rows={len(expression):,}",
        flush=True,
    )

    raw_mapping_path = raw_dir / "uniprot_to_ensembl_raw.tsv"
    provided_mapping = (
        args.uniprot_mapping_tsv.resolve()
        if args.uniprot_mapping_tsv is not None
        else None
    )
    mapping = obtain_uniprot_mapping(
        identifiers,
        set(expression["gene_id"]),
        raw_mapping_path,
        provided_mapping,
        args,
    )
    mapping_path = processed_dir / "uniprot_to_gene.csv"
    mapping[["uniprot_id", "gene_id"]].to_csv(mapping_path, index=False)
    unmapped = sorted(set(identifiers) - set(mapping["uniprot_id"]))
    pd.DataFrame({"uniprot_id": unmapped}).to_csv(
        audit_dir / "unmapped_uniprot_ids.csv", index=False
    )
    mapping_audit(train, identifiers, mapping).to_csv(
        audit_dir / "protein_mapping_audit.csv", index=False
    )

    premium_tissues = sorted(train["target_tissue"].astype(str).unique())
    hpa_tissues = set(expression["expression_tissue"])
    template = tissue_review_template(premium_tissues, hpa_tissues)
    template_path = processed_dir / "tissue_mapping_review_template.csv"
    template.to_csv(template_path, index=False)

    status = "blocked_pending_manual_tissue_review"
    approved_mapping_path: Path | None = None
    if args.approved_tissue_mapping_csv is not None:
        approved_mapping_path = args.approved_tissue_mapping_csv.resolve()
        approved = validate_approved_tissue_mapping(
            approved_mapping_path, set(premium_tissues), hpa_tissues
        )
        e_ready_dir.mkdir(parents=True, exist_ok=True)
        approved[
            [
                "target_tissue", "expression_tissue", "review_status",
                "weight", "mapping_quality",
            ]
        ].to_csv(
            e_ready_dir / "tissue_mapping.csv", index=False
        )
        expression.to_csv(
            e_ready_dir / "expression.csv.gz", index=False, compression="gzip"
        )
        mapping.loc[
            mapping["present_in_hpa"], ["uniprot_id", "gene_id"]
        ].to_csv(e_ready_dir / "protein_mapping.csv", index=False)
        sample_coverage, expression_audit = row_expression_audit(
            train, expression, mapping, approved
        )
        sample_coverage.to_csv(
            audit_dir / "sample_expression_coverage.csv.gz",
            index=False,
            compression="gzip",
        )
        expression_audit.to_csv(
            audit_dir / "expression_missingness_audit.csv", index=False
        )
        label_rates = expression_audit[
            expression_audit["dimension"] == "label"
        ].set_index("group")["expression_missing_rate"]
        missing_gap = (
            abs(float(label_rates.get("1", np.nan)) - float(label_rates.get("0", np.nan)))
            if {"0", "1"}.issubset(label_rates.index)
            else np.nan
        )
        metadata = {
            "dataset_name": "Human Protein Atlas consensus RNA",
            "version": HPA_VERSION,
            "download_date": date.today().isoformat(),
            "license": HPA_LICENSE,
            "measurement_level": "gene",
            "normalization": "nTPM",
            "ensembl_version": HPA_ENSEMBL_VERSION,
            "source_url": args.hpa_url,
            "source_page": HPA_PAGE,
            "uniprot_mapping_api": UNIPROT_API,
            "uniprot_mapping_help": UNIPROT_HELP,
            "protein_mapping_aggregation": "mean across mapped genes per UniProt",
            "positive_negative_missing_rate_absolute_gap": missing_gap,
            "raw_hpa_sha256": sha256(hpa_zip),
            "train_sha256": sha256(args.train_csv),
        }
        write_json(e_ready_dir / "expression_metadata.json", metadata)
        status = "e_ready_inputs_created"

    settings = {
        "experiment": "Pre_E_biological_expression_data_preparation",
        "parts": [
            "Pre_E0_HPA", "Pre_E1_UniProt_mapping",
            "Pre_E2_tissue_review", "Pre_E3_row_audit",
        ],
        "status": status,
        "seed_policy": "deterministic data preparation; no seed; aligned E defaults to seed 20260704",
        "test_data_read": False,
        "train": str(args.train_csv),
        "train_rows": len(train),
        "unique_uniprot": len(identifiers),
        "hpa_version": HPA_VERSION,
        "hpa_ensembl_version": HPA_ENSEMBL_VERSION,
        "hpa_url": args.hpa_url,
        "hpa_zip": str(hpa_zip),
        "hpa_zip_sha256": sha256(hpa_zip),
        "uniprot_mapping_source": str(provided_mapping or raw_mapping_path),
        "approved_tissue_mapping": str(approved_mapping_path) if approved_mapping_path else None,
        "tissue_review_policy": "all automatic suggestions remain pending_review until a human-approved file is supplied",
        "isoform_policy": "submit exact UniProt strings; do not strip isoform suffixes automatically",
        "non_human_policy": "retain as unmapped/missing unless mapping lands on an HPA ENSG gene",
        "direct_uniprot_feature_used": False,
        "output_files": {
            "expression": str(expression_path),
            "protein_mapping": str(mapping_path),
            "tissue_review_template": str(template_path),
        },
    }
    write_json(output / "run_settings.json", settings)
    print(
        f"\nPre-E complete: status={status}\noutput={output}\n"
        f"Pre-E total elapsed: {time.perf_counter() - started:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    run_all_pre_e()
