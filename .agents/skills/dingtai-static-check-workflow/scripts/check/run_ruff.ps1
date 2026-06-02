param(
    [Parameter(Mandatory = $false)]
    [string]$Target = ".",

    [Parameter(Mandatory = $false)]
    [switch]$Fix
)

$ErrorActionPreference = "Stop"

$RuffPath = "C:\Users\ICO\anaconda3\envs\DingTai\Scripts\ruff.exe"
if (-not (Test-Path -LiteralPath $RuffPath)) {
    throw "ruff not found: $RuffPath"
}

Write-Host "[ruff] executable: $RuffPath"
Write-Host "[ruff] target: $Target"

if ($Fix) {
    & $RuffPath check $Target --fix
} else {
    & $RuffPath check $Target
}

exit $LASTEXITCODE
