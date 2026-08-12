param(
    [int]$MaxPasses = 6
)

$ErrorActionPreference = 'Stop'
$paperDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$engine = Get-Command xelatex -ErrorAction SilentlyContinue
if (-not $engine) {
    $localMiKTeX = Join-Path $env:LOCALAPPDATA 'Programs\MiKTeX\miktex\bin\x64\xelatex.exe'
    if (Test-Path $localMiKTeX) {
        $enginePath = $localMiKTeX
    }
    else {
        throw 'xelatex was not found. Install MiKTeX or TeX Live and add xelatex to PATH.'
    }
}
else {
    $enginePath = $engine.Source
}

Push-Location $paperDir
try {
    # The two documents import one another's labels through xr-hyper.
    # Alternating them prevents otherwise valid references becoming "??".
    $documents = @('supplementary_main', 'main')
    $previousHashes = @{}

    for ($pass = 1; $pass -le $MaxPasses; $pass++) {
        Write-Host "Cross-document pass $pass/$MaxPasses"
        foreach ($document in $documents) {
            Write-Host "  Compiling $document.tex"
            & $enginePath -interaction=nonstopmode -halt-on-error "$document.tex"
            if ($LASTEXITCODE -ne 0) {
                throw "XeLaTeX failed for $document.tex (exit code $LASTEXITCODE)."
            }
        }

        $currentHashes = @{}
        foreach ($document in $documents) {
            $currentHashes[$document] = (Get-FileHash -Algorithm SHA256 "$document.aux").Hash
        }

        $stable = $pass -gt 1
        foreach ($document in $documents) {
            if ($previousHashes[$document] -ne $currentHashes[$document]) {
                $stable = $false
            }
        }
        $previousHashes = $currentHashes
        if ($stable) {
            Write-Host "Cross-document references stabilized after $pass passes."
            break
        }
        if ($pass -eq $MaxPasses) {
            throw "Auxiliary files did not stabilize after $MaxPasses alternating passes."
        }
    }

    $badLogPatterns = @(
        'undefined references',
        'Reference .* undefined',
        'Citation .* undefined',
        'LABELS NOT IMPORTED',
        'multiply defined'
    )
    foreach ($document in $documents) {
        $logText = Get-Content -Raw "$document.log"
        foreach ($pattern in $badLogPatterns) {
            if ($logText -match $pattern) {
                throw "$document.log still contains a reference error matching: $pattern"
            }
        }
    }

    Write-Host 'Both PDFs were built with no unresolved or duplicate references.'
}
finally {
    Pop-Location
}
