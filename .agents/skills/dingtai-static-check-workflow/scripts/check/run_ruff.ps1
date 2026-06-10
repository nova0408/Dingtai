param(
    [Parameter(Mandatory = $false)]
    [string]$Target = ".",

    [Parameter(Mandatory = $false)]
    [switch]$Fix
)

$ErrorActionPreference = "Stop"

function Test-GeneratedQtUiFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    return [System.IO.Path]::GetFileName($Path) -like "*_ui.py"
}

function Resolve-RuffTargets {
    param(
        [Parameter(Mandatory = $true)]
        [string]$InputTarget
    )

    if (-not (Test-Path -LiteralPath $InputTarget)) {
        return @($InputTarget)
    }

    $item = Get-Item -LiteralPath $InputTarget
    if ($item.PSIsContainer) {
        $pythonFiles = Get-ChildItem -LiteralPath $item.FullName -Recurse -File -Filter *.py |
            Where-Object { -not (Test-GeneratedQtUiFile -Path $_.Name) } |
            ForEach-Object { Resolve-Path -LiteralPath $_.FullName -Relative }
        return @($pythonFiles)
    }

    if (Test-GeneratedQtUiFile -Path $item.Name) {
        return @()
    }

    return @($InputTarget)
}

function Get-DingTaiEnvRoot {
    if ($env:CONDA_PREFIX -and (Split-Path -Leaf $env:CONDA_PREFIX) -eq "DingTai") {
        return $env:CONDA_PREFIX
    }

    $condaExe = $env:CONDA_EXE
    if (-not $condaExe) {
        $condaCommand = Get-Command conda.exe -ErrorAction SilentlyContinue
        if ($null -ne $condaCommand) {
            $condaExe = $condaCommand.Source
        }
    }

    if (-not $condaExe) {
        throw "conda executable not found, cannot resolve environment by name: DingTai"
    }

    $envListJson = & $condaExe env list --json 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($envListJson)) {
        throw "failed to query conda environments via: $condaExe"
    }

    $envList = $envListJson | ConvertFrom-Json
    foreach ($envPath in $envList.envs) {
        if ((Split-Path -Leaf $envPath) -eq "DingTai") {
            return $envPath
        }
    }

    throw "conda environment not found by name: DingTai"
}

$EnvRoot = Get-DingTaiEnvRoot
$RuffPath = Join-Path $EnvRoot "Scripts\ruff.exe"
if (-not (Test-Path -LiteralPath $RuffPath)) {
    throw "ruff not found: $RuffPath"
}

Write-Host "[ruff] executable: $RuffPath"
Write-Host "[ruff] target: $Target"

$ResolvedTargets = Resolve-RuffTargets -InputTarget $Target
$ResolvedTargets = @($ResolvedTargets)
if ($ResolvedTargets.Count -eq 0) {
    Write-Host "[ruff] skipped generated Qt UI file target"
    exit 0
}

if ($Fix) {
    if ($ResolvedTargets.Count -eq 1) {
        & $RuffPath check $ResolvedTargets[0] --fix
    } else {
        & $RuffPath check $ResolvedTargets --fix
    }
} else {
    if ($ResolvedTargets.Count -eq 1) {
        & $RuffPath check $ResolvedTargets[0]
    } else {
        & $RuffPath check $ResolvedTargets
    }
}

exit $LASTEXITCODE
