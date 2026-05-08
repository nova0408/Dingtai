param(
    [Parameter(Mandatory = $false)]
    [string]$Target = "."
)

$ErrorActionPreference = "Stop"

$PyrightPath = "C:\Users\ICO\anaconda3\envs\DingTai\Scripts\pyright.exe"
if (-not (Test-Path -LiteralPath $PyrightPath)) {
    throw "pyright not found: $PyrightPath"
}

Write-Host "[pyright] executable: $PyrightPath"
Write-Host "[pyright] target: $Target"

& $PyrightPath $Target
exit $LASTEXITCODE
