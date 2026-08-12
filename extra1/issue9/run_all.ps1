param(
    [ValidateSet("full", "smoke")]
    [string]$Mode = "full",
    [string]$Python = "E:\ancd\envs\my_pytorch\python.exe",
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"
$Issue9Dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Resolve-Path (Join-Path $Issue9Dir "..\..")

& $Python (Join-Path $Issue9Dir "audit_inputs.py")

if ($Mode -eq "smoke") {
    $HumanOutput = Join-Path $Root "tmp\issue9_human_smoke"
    $MouseOutput = Join-Path $Root "tmp\issue9_mouse_smoke"
    & $Python (Join-Path $Issue9Dir "run_human.py") `
        --output-dir $HumanOutput `
        --models human_shared_heads human_tissuepmhc_net `
        --seeds 20260704 `
        --epochs 1 `
        --device $Device
    & $Python (Join-Path $Issue9Dir "run_mouse.py") `
        --output-dir $MouseOutput `
        --models mouse_shared_heads mouse_factorized_mmoe `
        --seeds 20260704 `
        --epochs 1 `
        --device $Device
    Write-Host "Smoke runs completed. Full paired analysis requires the full model sets."
    exit 0
}

& $Python (Join-Path $Issue9Dir "run_human.py") --device $Device
& $Python (Join-Path $Issue9Dir "run_mouse.py") --device $Device
& $Python (Join-Path $Issue9Dir "analyze.py")
