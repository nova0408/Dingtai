param(
    [Parameter(Mandatory = $false)]
    [string]$Target = "."
)

$ErrorActionPreference = "Stop"

$RuffScript = Join-Path $PSScriptRoot "run_ruff.ps1"
$PyrightScript = Join-Path $PSScriptRoot "run_pyright.ps1"

if (-not (Test-Path -LiteralPath $RuffScript)) {
    throw "script not found: $RuffScript"
}
if (-not (Test-Path -LiteralPath $PyrightScript)) {
    throw "script not found: $PyrightScript"
}

Write-Host "[check] start ruff"
powershell -ExecutionPolicy Bypass -File $RuffScript -Target $Target
if ($LASTEXITCODE -ne 0) {
    Write-Host "[check] ruff failed: $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "[check] start pyright"
powershell -ExecutionPolicy Bypass -File $PyrightScript -Target $Target
if ($LASTEXITCODE -ne 0) {
    Write-Host "[check] pyright failed: $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "[check] all passed"
exit 0
