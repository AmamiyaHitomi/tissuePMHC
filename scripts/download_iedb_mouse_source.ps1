param([switch]$Force)
$ErrorActionPreference = 'Stop'
$out = 'data/raw/mhc_ligand_full_single_file.zip'
if ((Test-Path $out) -and -not $Force) { Write-Host "Already present: $out"; exit 0 }
New-Item -ItemType Directory -Force (Split-Path $out) | Out-Null
Invoke-WebRequest -Uri 'https://www.iedb.org/downloader.php?file_name=doc/mhc_ligand_full_single_file.zip' -OutFile $out
Get-FileHash $out -Algorithm SHA256 | Format-List
