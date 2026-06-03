param(
    [Parameter(Mandatory = $false)]
    [string]$Target = "."
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
$PyrightPath = Join-Path $EnvRoot "Scripts\pyright.exe"
if (-not (Test-Path -LiteralPath $PyrightPath)) {
    throw "pyright not found: $PyrightPath"
}

Write-Host "[pyright] executable: $PyrightPath"
Write-Host "[pyright] target: $Target"

& $PyrightPath $Target
exit $LASTEXITCODE
