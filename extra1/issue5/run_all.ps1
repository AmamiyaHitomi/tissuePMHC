param(
    [ValidateSet("prepare", "evaluate", "mhc-only", "all")]
    [string]$Mode = "prepare",
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",
    [string[]]$ScoreCaches = @()
)

$ErrorActionPreference = "Stop"
$Python = "E:\ancd\envs\my_pytorch\python.exe"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root

& $Python extra\issue5\run_tests.py

if ($Mode -in @("prepare", "all")) {
    & $Python extra\issue5\build_queries.py
    Write-Host "Queries prepared. Run the frozen external predictors, then import their outputs."
}

if ($Mode -in @("evaluate", "all")) {
    if ($ScoreCaches.Count -eq 0) {
        throw "Mode=$Mode requires -ScoreCaches with one or more imported cache CSV files."
    }
    $Arguments = @("extra\issue5\evaluate_external.py")
    foreach ($Cache in $ScoreCaches) {
        $Arguments += @("--score-cache", $Cache)
    }
    & $Python @Arguments
    & $Python extra\issue5\stack_increment.py `
        --row-predictions results\issue5_general_pmhc\external_evaluation\row_predictions.csv.gz
}

if ($Mode -in @("mhc-only", "all")) {
    & $Python extra\issue5\run_mhc_only.py --species human --device $Device
    & $Python extra\issue5\run_mhc_only.py --species mouse --device $Device
    & $Python extra\issue5\analyze_mhc_only.py `
        --predictions results\issue5_general_pmhc\human_mhc_only\ensemble_predictions.csv.gz `
        --predictions results\issue5_general_pmhc\mouse_mhc_only\ensemble_predictions.csv.gz
}
