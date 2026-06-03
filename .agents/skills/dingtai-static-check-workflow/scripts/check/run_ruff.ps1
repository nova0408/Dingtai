param(
    [Parameter(Mandatory = $false)]
    [string]$Target = ".",

    [Parameter(Mandatory = $false)]
    [switch]$Fix
)

$ErrorActionPreference = "Stop"

function Get-DingTaiEnvRoot {
    if ($env:CONDA_PREFIX -and (Split-Path -Leaf $env:CONDA_PREFIX) -eq "DingTai") {
        return $env:CONDA_PREFIX
    }

    $defaultRoot = Join-Path $env:USERPROFILE "anaconda3\envs\DingTai"
    if (Test-Path -LiteralPath $defaultRoot) {
        return $defaultRoot
    }

    return "C:\Users\ICO\anaconda3\envs\DingTai"
}

$EnvRoot = Get-DingTaiEnvRoot
$RuffPath = Join-Path $EnvRoot "Scripts\ruff.exe"
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
