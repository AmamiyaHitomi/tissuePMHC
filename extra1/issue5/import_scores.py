from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .common import DEFAULT_RESULTS, atomic_json, sha256
except ImportError:
    from common import DEFAULT_RESULTS, atomic_json, sha256


def normalized_columns(frame: pd.DataFrame) -> dict[str, str]:
    def normalize(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value).lower())

    return {normalize(column): column for column in frame.columns}


def find_column(frame: pd.DataFrame, aliases: list[str], required: bool = True) -> str | None:
    columns = normalized_columns(frame)
    for alias in aliases:
        key = re.sub(r"[^a-z0-9]+", "", alias.lower())
        if key in columns:
            return columns[key]
    if required:
        raise ValueError(f"None of {aliases} found in columns {frame.columns.tolist()}")
    return None


def cache_rows(
    queries: pd.DataFrame,
    predictor: str,
    scoring_mode: str,
    raw_values: pd.Series,
    transform: str,
    source_file: Path,
    raw_column: str,
) -> pd.DataFrame:
    numeric = pd.to_numeric(raw_values, errors="coerce")
    if transform == "negate":
        scores = -numeric
        direction = "raw lower is better; cached score = -raw"
    elif transform == "neg_log10":
        scores = -np.log10(numeric.clip(lower=1e-12))
        direction = "raw lower is better; cached score = -log10(raw)"
    elif transform == "identity":
        scores = numeric
        direction = "raw and cached score higher is better"
    else:
        raise ValueError(f"Unknown transform: {transform}")
    result = queries[
        ["query_id", "species", "peptide_sequence", "mhc_restriction"]
    ].copy()
    result["predictor"] = predictor
    result["scoring_mode"] = scoring_mode
    result["score"] = scores.to_numpy(float)
    result["raw_score"] = numeric.to_numpy(float)
    result["raw_column"] = raw_column
    result["score_direction"] = direction
    result["is_supported"] = result["score"].notna()
    result["missing_reason"] = np.where(result["is_supported"], "", "missing_or_unsupported")
    result["source_file"] = str(source_file.resolve())
    return result


def import_mhcflurry(
    queries_path: Path, predictions_path: Path, output_path: Path, version: str
) -> None:
    queries = pd.read_csv(queries_path)
    predictions = pd.read_csv(predictions_path)
    if "query_id" not in predictions.columns:
        peptide_col = find_column(predictions, ["peptide"])
        allele_col = find_column(predictions, ["allele"])
        keyed = queries.merge(
            predictions,
            left_on=["tool_allele", "peptide_sequence"],
            right_on=[allele_col, peptide_col],
            how="left",
            validate="one_to_one",
        )
    else:
        if predictions["query_id"].duplicated().any():
            raise ValueError("MHCflurry output contains duplicate query_id rows.")
        keyed = queries.merge(predictions, on="query_id", how="left", validate="one_to_one")

    modes = [
        (
            "affinity_percentile",
            ["mhcflurry_affinity_percentile", "affinity_percentile"],
            "negate",
        ),
        ("affinity_nm", ["mhcflurry_affinity", "affinity"], "neg_log10"),
        (
            "presentation_score",
            ["mhcflurry_presentation_score", "presentation_score"],
            "identity",
        ),
    ]
    rows: list[pd.DataFrame] = []
    for mode, aliases, transform in modes:
        column = find_column(keyed, aliases, required=False)
        if column is None:
            continue
        rows.append(
            cache_rows(
                keyed,
                f"mhcflurry_{version}",
                mode,
                keyed[column],
                transform,
                predictions_path,
                column,
            )
        )
    if not rows:
        raise ValueError("MHCflurry output contains no recognized score columns.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache = pd.concat(rows, ignore_index=True)
    cache.to_csv(output_path, index=False)
    atomic_json(
        output_path.with_suffix(".metadata.json"),
        {
            "format": "mhcflurry",
            "version": version,
            "queries": str(queries_path.resolve()),
            "queries_sha256": sha256(queries_path),
            "predictions": str(predictions_path.resolve()),
            "predictions_sha256": sha256(predictions_path),
            "output": str(output_path.resolve()),
            "modes": [mode for mode, _, _ in modes if mode in set(cache["scoring_mode"])],
            "n_queries": int(queries["query_id"].nunique()),
            "coverage_by_mode": cache.groupby("scoring_mode")["is_supported"]
            .mean()
            .astype(float)
            .to_dict(),
            "required_runtime_note": "Record exact model bundle path and SHA256 separately.",
        },
    )


def read_netmhcpan_table(path: Path) -> pd.DataFrame:
    attempts: list[Exception] = []
    for skiprows in (0, 1):
        try:
            frame = pd.read_csv(path, sep="\t", skiprows=skiprows, comment="#")
            if find_column(frame, ["Peptide"], required=False) is not None:
                return frame
        except Exception as error:  # pragma: no cover - diagnostic path
            attempts.append(error)
    raise ValueError(
        f"Could not find a Peptide header in NetMHCpan xls table {path}: {attempts}"
    )


def import_netmhcpan(
    queries_path: Path,
    manifest_path: Path,
    output_path: Path,
    version: str,
) -> None:
    queries = pd.read_csv(queries_path)
    manifest = pd.read_csv(manifest_path)
    scored_parts: list[pd.DataFrame] = []
    source_hashes: dict[str, str] = {}
    for row in manifest.itertuples(index=False):
        output = Path(row.expected_output)
        subset = queries[queries["mhc_restriction"] == row.mhc_restriction].copy()
        if not output.exists():
            subset["_source"] = str(output.resolve())
            scored_parts.append(subset)
            continue
        table = read_netmhcpan_table(output)
        peptide_col = find_column(table, ["Peptide"])
        if table[peptide_col].duplicated().any():
            raise ValueError(f"{output} contains duplicate peptide rows.")
        subset = subset.merge(
            table,
            left_on="peptide_sequence",
            right_on=peptide_col,
            how="left",
            validate="one_to_one",
        )
        subset["_source"] = str(output.resolve())
        source_hashes[str(output.resolve())] = sha256(output)
        scored_parts.append(subset)
    keyed = pd.concat(scored_parts, ignore_index=True)
    modes = [
        ("ba_rank", ["BA_Rank", "%Rank_BA", "Rank_BA"], "negate"),
        ("el_rank", ["EL_Rank", "%Rank_EL", "Rank_EL"], "negate"),
        ("affinity_nm", ["Aff(nM)", "Affinity", "BA_Affinity"], "neg_log10"),
        ("el_score", ["EL-score", "Score_EL", "EL_Score"], "identity"),
    ]
    rows: list[pd.DataFrame] = []
    for mode, aliases, transform in modes:
        column = find_column(keyed, aliases, required=False)
        if column is None:
            continue
        # Preserve per-row source paths even though cache_rows accepts one provenance path.
        result = cache_rows(
            keyed,
            f"netmhcpan_{version}",
            mode,
            keyed[column],
            transform,
            manifest_path,
            column,
        )
        result["source_file"] = keyed["_source"].to_numpy(str)
        rows.append(result)
    if not rows:
        raise ValueError("NetMHCpan outputs contain no recognized BA/EL score columns.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache = pd.concat(rows, ignore_index=True)
    cache.to_csv(output_path, index=False)
    atomic_json(
        output_path.with_suffix(".metadata.json"),
        {
            "format": "netmhcpan",
            "version": version,
            "queries": str(queries_path.resolve()),
            "queries_sha256": sha256(queries_path),
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": sha256(manifest_path),
            "source_hashes": source_hashes,
            "output": str(output_path.resolve()),
            "n_queries": int(queries["query_id"].nunique()),
            "coverage_by_mode": cache.groupby("scoring_mode")["is_supported"]
            .mean()
            .astype(float)
            .to_dict(),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="format", required=True)

    mhcflurry = subparsers.add_parser("mhcflurry")
    mhcflurry.add_argument("--queries", type=Path, required=True)
    mhcflurry.add_argument("--predictions", type=Path, required=True)
    mhcflurry.add_argument("--version", required=True)
    mhcflurry.add_argument("--output", type=Path, required=True)

    netmhcpan = subparsers.add_parser("netmhcpan")
    netmhcpan.add_argument("--queries", type=Path, required=True)
    netmhcpan.add_argument("--manifest", type=Path, required=True)
    netmhcpan.add_argument("--version", default="4.1")
    netmhcpan.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.format == "mhcflurry":
        import_mhcflurry(args.queries, args.predictions, args.output, args.version)
    else:
        import_netmhcpan(args.queries, args.manifest, args.output, args.version)
