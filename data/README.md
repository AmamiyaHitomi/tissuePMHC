# Data

This directory contains the compact input and processed datasets used by the
tissuePMHC project. Two large IEDB source files are intentionally not stored in
this GitHub repository:

```text
data/raw/mhc_ligand_full_single_file.zip
data/raw/mhc_ligand_full.csv
```

## Download the IEDB MHC ligand export

The omitted archive is the official `mhc_ligand_full (single_file.zip)` CSV
metric export provided by the Immune Epitope Database (IEDB).

- Official export page: <https://www.iedb.org/database_export_v3.php>
- Direct download: <https://www.iedb.org/downloader.php?file_name=doc/mhc_ligand_full_single_file.zip>
- IEDB license: Creative Commons Attribution 4.0 International (CC BY 4.0)

The experiments reported in the manuscript used the export downloaded on
2026-07-04. Its local archive had the following SHA-256 checksum:

```text
26DAFE07F782D3CE29B076105A47AB41EF1BF017F1974F53D5A407838141402D
```

IEDB updates its exports regularly. A newly downloaded archive may therefore
have a different checksum and may not reproduce the exact dataset snapshot used
in the manuscript.

### Download with the bundled script

From the repository root in PowerShell, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_iedb_mouse_source.ps1
```

Alternatively, download it directly:

```powershell
New-Item -ItemType Directory -Force data/raw | Out-Null
Invoke-WebRequest `
  -Uri "https://www.iedb.org/downloader.php?file_name=doc/mhc_ligand_full_single_file.zip" `
  -OutFile "data/raw/mhc_ligand_full_single_file.zip"
```

Verify the downloaded archive when reproducing the manuscript snapshot:

```powershell
Get-FileHash -Algorithm SHA256 data/raw/mhc_ligand_full_single_file.zip
```

The processing scripts accept the official ZIP archive directly where noted;
the extracted CSV remains ignored because it is approximately 8.2 GiB.

## Attribution

IEDB data are manually curated from published experiments and submitted
datasets. When using these data, cite IEDB and retain attribution to the
original publishing or submitting authors as required by CC BY 4.0.
