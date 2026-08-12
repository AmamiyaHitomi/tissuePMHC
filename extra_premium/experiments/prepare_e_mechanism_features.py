#!/usr/bin/env python3
"""Prepare train-only protein-expression and antigen-processing features for E.

The script deliberately reads only ``humanPMHC_train.csv.gz``.  It joins an
audited external expression matrix to the premium parent UniProt IDs, locates
each peptide in a supplied protein FASTA, extracts real N/C flanks, and can run
the locally installed MHCflurry processing predictor with those flanks.

No approximate peptide-to-protein matching is performed.  Missing, ambiguous,
and proxy tissue mappings remain explicit features so they cannot silently turn
into biological facts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENTS_DIR = Path(__file__).resolve().parent
EXTRA_PREMIUM_DIR = EXPERIMENTS_DIR.parent
if str(EXTRA_PREMIUM_DIR) not in sys.path:
    sys.path.insert(0, str(EXTRA_PREMIUM_DIR))

import common


MECHANISM_DIR = EXTRA_PREMIUM_DIR / "external" / "mechanism"
DEFAULT_OUTPUT = MECHANISM_DIR / "e_mechanism_features.csv.gz"
DEFAULT_AUDIT = MECHANISM_DIR / "e_mechanism_feature_audit.json"
DEFAULT_MHCFLURRY_EXECUTABLE = Path(
    os.environ.get("MHCFLURRY_EXECUTABLE")
    or shutil.which("mhcflurry-predict")
    or "mhcflurry-predict"
)
MACHINERY_GENES = (
    "PSMB5", "PSMB6", "PSMB7", "PSMB8", "PSMB9", "PSMB10",
    "TAP1", "TAP2", "TAPBP", "ERAP1", "ERAP2", "B2M",
    "HLA-A", "HLA-B", "HLA-C",
)
# UniProt proteins can contain ambiguity/rare-residue codes even though the
# premium 9-mers themselves are canonical. They remain explicit in the cached
# FASTA; the experiment encoder maps unsupported flank symbols to padding.
AA_PATTERN = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBXZJUO]+$")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--protein-fasta", type=Path, required=True)
    result.add_argument(
        "--uniprot-gene-map",
        type=Path,
        required=True,
        help="CSV/TSV with uniprot_id and gene_id; gene_name is optional.",
    )
    result.add_argument(
        "--expression-table",
        type=Path,
        required=True,
        help=(
            "Long CSV/TSV expression table. Accepted HPA-style columns are "
            "Gene, Gene name, Tissue, nTPM; normalized names are also accepted."
        ),
    )
    result.add_argument(
        "--tissue-mapping",
        type=Path,
        required=True,
        help=(
            "Audited CSV/TSV with target_tissue, expression_tissue, "
            "review_status and mapping_quality; weight is optional. Multiple "
            "source tissues per premium tissue are allowed."
        ),
    )
    result.add_argument(
        "--expression-metadata-json",
        type=Path,
        required=True,
        help=(
            "Reviewed provenance with dataset_name, version, download_date, "
            "license, measurement_level, and normalization."
        ),
    )
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT)
    result.add_argument("--flank-length", type=int, default=15)
    result.add_argument("--expression-threshold", type=float, default=1.0)
    result.add_argument(
        "--low-quality-policy",
        choices=("missing", "keep"),
        default="missing",
        help=(
            "How low_proxy tissue mappings enter the primary E features. "
            "The formal default is missing; raw proxy values are retained in "
            "proxy_enabled__ columns for inference-time sensitivity checks."
        ),
    )
    result.add_argument(
        "--mhcflurry-executable",
        type=Path,
        default=DEFAULT_MHCFLURRY_EXECUTABLE,
    )
    result.add_argument(
        "--mhcflurry-model-dir",
        type=Path,
        default=None,
        help=(
            "Optional ASCII-safe presentation bundle path. By default the "
            "installed MHCflurry release bundle is resolved automatically."
        ),
    )
    result.add_argument(
        "--skip-mhcflurry",
        action="store_true",
        help="Prepare all other features and leave processing score missing.",
    )
    result.add_argument(
        "--allow-unreviewed-tissue-map",
        action="store_true",
        help="Only for smoke tests. Formal runs must use reviewed/approved rows.",
    )
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_table(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    compression = "infer"
    try:
        frame = pd.read_csv(path, sep=None, engine="python", compression=compression)
    except (UnicodeDecodeError, pd.errors.ParserError):
        frame = pd.read_csv(path, sep="\t", compression=compression)
    if frame.empty:
        raise ValueError(f"{label} is empty: {path}")
    return frame


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "Gene": "gene_id",
        "Gene name": "gene_name",
        "Tissue": "expression_tissue",
        "nTPM": "expression_value",
        "Expression": "expression_value",
        "Uniprot": "uniprot_id",
        "UniProt": "uniprot_id",
    }
    result = frame.rename(columns={key: value for key, value in aliases.items() if key in frame})
    result.columns = [str(column).strip() for column in result.columns]
    return result


def read_expression_metadata(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Expression metadata does not exist: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "dataset_name", "version", "download_date", "license",
        "measurement_level", "normalization",
    }
    missing = required - set(metadata)
    if missing:
        raise ValueError(f"Expression metadata missing fields: {sorted(missing)}")
    for key in required:
        value = str(metadata[key]).strip()
        if not value or value.upper().startswith("REPLACE"):
            raise ValueError(f"Expression metadata field {key!r} is not finalized.")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(metadata["download_date"])):
        raise ValueError("Expression metadata download_date must use YYYY-MM-DD.")
    return metadata


def clean_gene_id(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.replace(r"\.\d+$", "", regex=True)


def parse_fasta(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    sequences: dict[str, str] = {}
    aliases: dict[str, str] = {}
    current: str | None = None
    chunks: list[str] = []

    def commit() -> None:
        nonlocal current, chunks
        if current is None:
            return
        sequence = "".join(chunks).replace(" ", "").upper()
        if not sequence or not AA_PATTERN.fullmatch(sequence):
            raise ValueError(f"Invalid protein sequence for FASTA entry {current!r}.")
        if current in sequences and sequences[current] != sequence:
            raise ValueError(f"Conflicting duplicate FASTA accession: {current}")
        sequences[current] = sequence

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                commit()
                header = line[1:].split()[0]
                tokens = header.split("|")
                current = tokens[1] if len(tokens) >= 3 else tokens[0]
                aliases[header] = current
                aliases[current] = current
                chunks = []
            else:
                if current is None:
                    raise ValueError("FASTA sequence encountered before first header.")
                chunks.append(line)
    commit()
    if not sequences:
        raise ValueError(f"No protein sequences found in {path}")
    return sequences, aliases


def all_occurrences(sequence: str, peptide: str) -> list[int]:
    positions: list[int] = []
    start = 0
    while True:
        found = sequence.find(peptide, start)
        if found < 0:
            return positions
        positions.append(found)
        start = found + 1


def extract_flanks(
    train: pd.DataFrame,
    sequences: dict[str, str],
    flank_length: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in train[[
        "sample_id", "molecule_parent_uniprot_id", "peptide_sequence"
    ]].itertuples(index=False):
        accession = str(row.molecule_parent_uniprot_id).strip()
        peptide = str(row.peptide_sequence).strip().upper()
        sequence = sequences.get(accession)
        positions = all_occurrences(sequence, peptide) if sequence else []
        exact = len(positions) > 0
        unique = len(positions) == 1
        position = positions[0] if unique else None
        if position is None:
            n_flank = ""
            c_flank = ""
            relative_position = np.nan
        else:
            n_flank = sequence[max(0, position - flank_length):position]
            c_start = position + len(peptide)
            c_flank = sequence[c_start:c_start + flank_length]
            denominator = max(1, len(sequence) - len(peptide))
            relative_position = position / denominator
        rows.append({
            "sample_id": row.sample_id,
            "parent_sequence_found": float(sequence is not None),
            "peptide_exact_match": float(exact),
            "peptide_unique_match": float(unique),
            "peptide_occurrence_count": float(len(positions)),
            "peptide_start": float(position) if position is not None else np.nan,
            "relative_position": relative_position,
            "n_flank_sequence": n_flank,
            "c_flank_sequence": c_flank,
            "flank_missing": float(not unique),
        })
    return pd.DataFrame(rows).set_index("sample_id")


def audited_tissue_map(
    train: pd.DataFrame,
    path: Path,
    expression_tissues: set[str],
    allow_unreviewed: bool,
) -> pd.DataFrame:
    mapping = normalize_columns(read_table(path, "tissue mapping"))
    required = {"target_tissue", "expression_tissue", "review_status"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"tissue mapping missing columns: {sorted(missing)}")
    mapping = mapping.copy()
    mapping["target_tissue"] = mapping["target_tissue"].astype(str).str.strip()
    mapping["expression_tissue"] = mapping["expression_tissue"].astype(str).str.strip()
    mapping["review_status"] = mapping["review_status"].astype(str).str.lower().str.strip()
    if "weight" not in mapping:
        mapping["weight"] = 1.0
    mapping["weight"] = pd.to_numeric(mapping["weight"], errors="raise")
    if (mapping["weight"] <= 0).any() or not np.isfinite(mapping["weight"]).all():
        raise ValueError("tissue mapping weights must be finite and positive.")
    if "mapping_quality" not in mapping:
        if not allow_unreviewed:
            raise ValueError(
                "Formal E preparation requires mapping_quality for every tissue mapping."
            )
        mapping["mapping_quality"] = "unspecified"
    mapping["mapping_quality"] = (
        mapping["mapping_quality"].astype(str).str.lower().str.strip()
    )
    if mapping["mapping_quality"].eq("").any():
        raise ValueError("tissue mapping contains empty mapping_quality values.")
    if not allow_unreviewed:
        allowed = {"reviewed", "approved"}
        bad = mapping.loc[~mapping["review_status"].isin(allowed)]
        if not bad.empty:
            raise ValueError(
                "Formal E preparation requires reviewed/approved tissue mappings; "
                f"unreviewed targets={sorted(bad.target_tissue.unique())}"
            )
    premium = set(train["target_tissue"].astype(str))
    mapped = set(mapping["target_tissue"])
    if premium != mapped:
        raise ValueError(
            "Tissue mapping must exactly cover premium tissues; "
            f"missing={sorted(premium - mapped)}, extra={sorted(mapped - premium)}"
        )
    unavailable = sorted(set(mapping["expression_tissue"]) - expression_tissues)
    if unavailable:
        raise ValueError(f"Mapped tissues absent from expression table: {unavailable}")
    return mapping


def weighted_expression(
    values: pd.DataFrame,
    mapping: pd.DataFrame,
    value_columns: list[str],
) -> pd.DataFrame:
    joined = mapping.merge(
        values,
        on="expression_tissue",
        how="left",
        validate="many_to_many",
    )
    for column in value_columns:
        joined[f"weighted__{column}"] = joined[column] * joined["weight"]
    groups = ["target_tissue"] + [
        column for column in values.columns if column != "expression_tissue" and column not in value_columns
    ]
    numerator = joined.groupby(groups, as_index=False, dropna=False)[
        [f"weighted__{column}" for column in value_columns]
    ].sum(min_count=1)
    denominator = joined.groupby(groups, as_index=False, dropna=False)["weight"].sum()
    result = numerator.merge(denominator, on=groups, validate="one_to_one")
    for column in value_columns:
        result[column] = result.pop(f"weighted__{column}") / result["weight"]
    return result.drop(columns="weight")


def build_expression_features(
    train: pd.DataFrame,
    expression_path: Path,
    protein_map_path: Path,
    tissue_map_path: Path,
    threshold: float,
    allow_unreviewed: bool,
    low_quality_policy: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    expression = normalize_columns(read_table(expression_path, "expression table"))
    required = {"gene_id", "gene_name", "expression_tissue", "expression_value"}
    missing = required - set(expression.columns)
    if missing:
        raise ValueError(f"expression table missing columns: {sorted(missing)}")
    expression = expression[list(required)].copy()
    expression["gene_id"] = clean_gene_id(expression["gene_id"])
    expression["gene_name"] = expression["gene_name"].astype(str).str.strip().str.upper()
    expression["expression_tissue"] = expression["expression_tissue"].astype(str).str.strip()
    expression["expression_value"] = pd.to_numeric(
        expression["expression_value"], errors="raise"
    )
    if (expression["expression_value"] < 0).any() or not np.isfinite(
        expression["expression_value"]
    ).all():
        raise ValueError("expression values must be finite and non-negative.")
    expression = expression.groupby(
        ["gene_id", "gene_name", "expression_tissue"], as_index=False
    )["expression_value"].mean()

    protein_map = normalize_columns(read_table(protein_map_path, "UniProt-gene mapping"))
    required_map = {"uniprot_id", "gene_id"}
    missing = required_map - set(protein_map.columns)
    if missing:
        raise ValueError(f"UniProt-gene mapping missing columns: {sorted(missing)}")
    protein_map = protein_map.copy()
    protein_map["uniprot_id"] = protein_map["uniprot_id"].astype(str).str.strip()
    protein_map["gene_id"] = clean_gene_id(protein_map["gene_id"])
    protein_map = protein_map[["uniprot_id", "gene_id"]].drop_duplicates()

    tissue_map = audited_tissue_map(
        train,
        tissue_map_path,
        set(expression["expression_tissue"]),
        allow_unreviewed,
    )
    quality_by_tissue = tissue_map.groupby("target_tissue")["mapping_quality"].apply(
        lambda values: ";".join(sorted(set(map(str, values))))
    )

    def quality_tier(raw: str) -> str:
        values = set(raw.split(";"))
        if "low_proxy" in values or "unspecified" in values:
            return "low_proxy"
        if "aggregate_proxy" in values:
            return "aggregate_proxy"
        if values.issubset({"exact", "synonym_exact"}):
            return "exact_or_synonym"
        return "other_proxy"

    tier_by_tissue = quality_by_tissue.map(quality_tier)
    gene_expression = expression[["gene_id", "expression_tissue", "expression_value"]]
    target_gene_expression = weighted_expression(
        gene_expression, tissue_map, ["expression_value"]
    ).rename(columns={"expression_value": "query_expression"})
    cross_mean = expression.groupby("gene_id")["expression_value"].mean().rename(
        "cross_tissue_mean_expression"
    )
    protein_expression = protein_map.merge(
        target_gene_expression, on="gene_id", how="left", validate="many_to_many"
    ).merge(cross_mean, on="gene_id", how="left", validate="many_to_one")
    protein_expression = protein_expression.groupby(
        ["uniprot_id", "target_tissue"], as_index=False
    )[["query_expression", "cross_tissue_mean_expression"]].mean()

    rows = train[["sample_id", "molecule_parent_uniprot_id", "target_tissue"]].merge(
        protein_expression,
        left_on=["molecule_parent_uniprot_id", "target_tissue"],
        right_on=["uniprot_id", "target_tissue"],
        how="left",
        validate="many_to_one",
    )
    rows["expression_missing"] = rows["query_expression"].isna().astype(float)
    rows["is_expressed"] = (
        rows["query_expression"].fillna(0.0) >= threshold
    ).astype(float)
    rows["relative_expression"] = np.log1p(rows["query_expression"]) - np.log1p(
        rows["cross_tissue_mean_expression"]
    )
    feature_columns = [
        "query_expression", "cross_tissue_mean_expression",
        "relative_expression", "is_expressed", "expression_missing",
    ]
    row_features = rows.set_index("sample_id")[feature_columns]
    row_quality = train[["sample_id", "target_tissue"]].copy()
    row_quality["tissue_mapping_quality"] = row_quality["target_tissue"].map(
        quality_by_tissue
    )
    row_quality["tissue_mapping_tier"] = row_quality["target_tissue"].map(
        tier_by_tissue
    )
    row_quality["low_proxy"] = row_quality["tissue_mapping_tier"].eq(
        "low_proxy"
    ).astype(float)
    row_quality = row_quality.set_index("sample_id").drop(columns="target_tissue")
    for column in feature_columns:
        row_features[f"proxy_enabled__{column}"] = row_features[column]
    row_features = row_features.join(row_quality, how="left", validate="one_to_one")
    if low_quality_policy == "missing":
        low_proxy_mask = row_features["low_proxy"].eq(1.0)
        row_features.loc[
            low_proxy_mask, ["query_expression", "relative_expression"]
        ] = np.nan
        row_features.loc[low_proxy_mask, "is_expressed"] = 0.0
        row_features.loc[low_proxy_mask, "expression_missing"] = 1.0

    machinery = expression[expression["gene_name"].isin(MACHINERY_GENES)]
    machinery = machinery.groupby(
        ["gene_name", "expression_tissue"], as_index=False
    )["expression_value"].mean()
    machinery_wide = machinery.pivot(
        index="expression_tissue", columns="gene_name", values="expression_value"
    ).reset_index()
    for gene in MACHINERY_GENES:
        if gene not in machinery_wide:
            machinery_wide[gene] = np.nan
    target_machinery = weighted_expression(
        machinery_wide[["expression_tissue", *MACHINERY_GENES]],
        tissue_map,
        list(MACHINERY_GENES),
    )
    target_machinery = target_machinery.rename(
        columns={gene: f"machinery__{gene}" for gene in MACHINERY_GENES}
    )
    target_machinery["machinery_missing_fraction"] = target_machinery[
        [f"machinery__{gene}" for gene in MACHINERY_GENES]
    ].isna().mean(axis=1)
    row_machinery = train[["sample_id", "target_tissue"]].merge(
        target_machinery, on="target_tissue", how="left", validate="many_to_one"
    ).set_index("sample_id").drop(columns="target_tissue")
    machinery_value_columns = [f"machinery__{gene}" for gene in MACHINERY_GENES]
    for column in [*machinery_value_columns, "machinery_missing_fraction"]:
        row_machinery[f"proxy_enabled__{column}"] = row_machinery[column]
    if low_quality_policy == "missing":
        low_proxy_ids = row_quality.index[row_quality["low_proxy"].eq(1.0)]
        row_machinery.loc[low_proxy_ids, machinery_value_columns] = np.nan
        row_machinery.loc[low_proxy_ids, "machinery_missing_fraction"] = 1.0

    metadata = {
        "expression_threshold": threshold,
        "low_quality_policy": low_quality_policy,
        "mapping_tier_train_rows": row_quality["tissue_mapping_tier"].value_counts(
            dropna=False
        ).to_dict(),
        "machinery_genes": list(MACHINERY_GENES),
        "mapped_parent_uniprot_fraction": float(
            train["molecule_parent_uniprot_id"].isin(protein_map["uniprot_id"]).mean()
        ),
        "row_expression_missing_fraction": float(row_features["expression_missing"].mean()),
        "tissue_mapping_rows": int(len(tissue_map)),
        "tissue_mapping_reviewed": bool(
            tissue_map["review_status"].isin({"reviewed", "approved"}).all()
        ),
        "tissue_mapping_quality": quality_by_tissue.to_dict(),
        "tissue_mapping_tier": tier_by_tissue.to_dict(),
    }
    return row_features, row_machinery, metadata


def run_mhcflurry(
    train: pd.DataFrame,
    flank_features: pd.DataFrame,
    executable: Path,
    model_dir: Path | None,
    mechanism_dir: Path,
) -> tuple[pd.Series, dict[str, object]]:
    if not executable.is_file():
        raise FileNotFoundError(f"MHCflurry executable not found: {executable}")
    if model_dir is not None and not model_dir.is_dir():
        raise FileNotFoundError(f"MHCflurry model directory not found: {model_dir}")
    queries = train[["sample_id", "mhc_restriction", "peptide_sequence"]].join(
        flank_features, on="sample_id"
    )
    valid = queries["flank_missing"].eq(0)
    unique_columns = [
        "mhc_restriction", "peptide_sequence", "n_flank_sequence", "c_flank_sequence"
    ]
    unique = queries.loc[valid, unique_columns].drop_duplicates().reset_index(drop=True)
    unique = unique.rename(columns={
        "mhc_restriction": "allele",
        "peptide_sequence": "peptide",
        "n_flank_sequence": "n_flank",
        "c_flank_sequence": "c_flank",
    })
    # The MHCFlurry CLI reads empty CSV fields as NaN under current pandas,
    # although its processing encoder requires strings. A single X is exactly
    # equivalent to an empty terminal flank because the encoder pads absent
    # flank positions with X before slicing to its fixed flank length.
    for column in ("n_flank", "c_flank"):
        unique[column] = unique[column].fillna("").astype(str).replace("", "X")
    mechanism_dir.mkdir(parents=True, exist_ok=True)
    query_path = mechanism_dir / "premium_train_mhcflurry_with_flanks.csv"
    raw_path = mechanism_dir / "premium_train_mhcflurry_with_flanks_output.csv"
    unique.to_csv(query_path, index=False)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    command = [
        str(executable), str(query_path), "--out", str(raw_path), "--no-throw"
    ]
    if model_dir is not None:
        command.extend(["--models", str(model_dir)])
    subprocess.run(
        command,
        check=True,
        env=environment,
    )
    scored = pd.read_csv(raw_path)
    processing_columns = [column for column in scored if column.endswith("processing_score")]
    if len(processing_columns) != 1:
        raise ValueError(
            "Expected one MHCflurry processing score column; "
            f"found {processing_columns}"
        )
    score_column = processing_columns[0]
    merge_columns = {
        "allele": "mhc_restriction",
        "peptide": "peptide_sequence",
        "n_flank": "n_flank_sequence",
        "c_flank": "c_flank_sequence",
    }
    scored = scored.rename(columns=merge_columns)
    for column in unique_columns:
        scored[column] = scored[column].fillna("").astype(str)
        queries[column] = queries[column].fillna("").astype(str)
        if column in {"n_flank_sequence", "c_flank_sequence"}:
            scored[column] = scored[column].replace("", "X")
            queries[column] = queries[column].replace("", "X")
    scored = scored[[*unique_columns, score_column]].drop_duplicates(unique_columns)
    aligned = queries.reset_index(drop=True).merge(
        scored, on=unique_columns, how="left", validate="many_to_one"
    ).set_index("sample_id")[score_column]
    aligned.name = "mhcflurry_processing_score"
    return aligned, {
        "mhcflurry_query_rows": int(len(unique)),
        "mhcflurry_scored_fraction": float(aligned.notna().mean()),
        "mhcflurry_executable": str(executable),
        "mhcflurry_model_dir": str(model_dir) if model_dir else "installed default bundle",
        "mhcflurry_raw_output": str(raw_path),
    }


def assert_pair_invariants(train: pd.DataFrame, features: pd.DataFrame) -> dict[str, object]:
    joined = train[["sample_id", "pair_id"]].merge(
        features.reset_index(), on="sample_id", validate="one_to_one"
    )
    expression_columns = [
        "query_expression", "cross_tissue_mean_expression", "relative_expression",
        "is_expressed", "expression_missing",
    ]
    violations = 0
    for _, pair in joined.groupby("pair_id", sort=False):
        first = pair.iloc[0][expression_columns]
        second = pair.iloc[1][expression_columns]
        equal = (first.eq(second) | (first.isna() & second.isna())).all()
        violations += int(not equal)
    if violations:
        raise AssertionError(
            f"Parent-expression features vary inside {violations} premium pairs."
        )
    return {
        "pair_count": int(joined["pair_id"].nunique()),
        "expression_pair_invariance_violations": violations,
    }


def main() -> None:
    args = parser().parse_args()
    if args.flank_length < 1:
        raise ValueError("--flank-length must be positive.")
    if args.expression_threshold < 0:
        raise ValueError("--expression-threshold must be non-negative.")
    input_paths = [
        args.protein_fasta.resolve(), args.uniprot_gene_map.resolve(),
        args.expression_table.resolve(), args.tissue_mapping.resolve(),
        args.expression_metadata_json.resolve(),
    ]
    expression_source_metadata = read_expression_metadata(input_paths[4])
    train = pd.read_csv(common.TRAIN_PATH)
    required_train = {
        "sample_id", "pair_id", "label", "target_tissue", "mhc_restriction",
        "peptide_sequence", "molecule_parent_uniprot_id",
    }
    missing = required_train - set(train.columns)
    if missing:
        raise ValueError(f"Premium train missing columns: {sorted(missing)}")
    if train["sample_id"].duplicated().any():
        raise ValueError("Premium train sample_id must be unique.")

    sequences, _ = parse_fasta(input_paths[0])
    flanks = extract_flanks(train, sequences, args.flank_length)
    expression, machinery, expression_metadata = build_expression_features(
        train,
        input_paths[2],
        input_paths[1],
        input_paths[3],
        args.expression_threshold,
        args.allow_unreviewed_tissue_map,
        args.low_quality_policy,
    )
    features = flanks.join(expression, how="left", validate="one_to_one").join(
        machinery, how="left", validate="one_to_one"
    )
    mhc_metadata: dict[str, object]
    if args.skip_mhcflurry:
        features["mhcflurry_processing_score"] = np.nan
        mhc_metadata = {
            "mhcflurry_skipped": True,
            "mhcflurry_scored_fraction": 0.0,
        }
    else:
        processing, mhc_metadata = run_mhcflurry(
            train,
            flanks,
            args.mhcflurry_executable.resolve(),
            args.mhcflurry_model_dir.resolve() if args.mhcflurry_model_dir else None,
            args.output.resolve().parent,
        )
        features = features.join(processing, how="left", validate="one_to_one")
    features["processing_score_missing"] = features[
        "mhcflurry_processing_score"
    ].isna().astype(float)
    features = features.reindex(train["sample_id"])
    if len(features) != len(train) or features.index.has_duplicates:
        raise AssertionError("Prepared mechanism features are not one-to-one with train.")
    pair_audit = assert_pair_invariants(train, features)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.reset_index().to_csv(args.output, index=False)
    audit = {
        "schema_version": 1,
        "test_data_read": False,
        "train_path": str(common.TRAIN_PATH),
        "train_rows": int(len(train)),
        "train_pairs": int(train["pair_id"].nunique()),
        "flank_length": args.flank_length,
        "protein_fasta_entries": len(sequences),
        "parent_sequence_coverage": float(features["parent_sequence_found"].mean()),
        "exact_peptide_match_coverage": float(features["peptide_exact_match"].mean()),
        "unique_peptide_match_coverage": float(features["peptide_unique_match"].mean()),
        **pair_audit,
        **expression_metadata,
        "expression_source_metadata": expression_source_metadata,
        **mhc_metadata,
        "input_sha256": {str(path): sha256(path) for path in input_paths},
        "output_sha256": sha256(args.output),
    }
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"Prepared E mechanism features: {args.output}")


if __name__ == "__main__":
    main()
