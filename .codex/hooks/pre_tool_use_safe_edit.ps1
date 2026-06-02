param()

$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    $root = git rev-parse --show-toplevel 2>$null
    if ($LASTEXITCODE -eq 0 -and $root) {
        return $root.Trim()
    }
    return (Resolve-Path ".").Path
}

function Get-PatchPaths {
    param([string]$Command)

    $paths = New-Object System.Collections.Generic.List[string]
    $pattern = '(?m)^\*\*\* (?:Update|Delete) File: (?<path>.+)$'
    foreach ($match in [System.Text.RegularExpressions.Regex]::Matches($Command, $pattern)) {
        $path = $match.Groups["path"].Value.Trim()
        if ($path.Length -gt 0) {
            $paths.Add($path)
        }
    }
    return $paths
}

function New-Snapshot {
    param(
        [string]$RepoRoot,
        [string]$RelativePath
    )

    $target = Join-Path $RepoRoot $RelativePath
    if (-not (Test-Path -LiteralPath $target)) {
        return
    }

    $resolved = (Resolve-Path -LiteralPath $target).Path
    if (-not $resolved.StartsWith($RepoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "refuse snapshot outside repo: $RelativePath"
    }

    $relativeDir = Split-Path $RelativePath -Parent
    $name = Split-Path $RelativePath -Leaf
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $snapshotDir = Join-Path $RepoRoot (Join-Path ".archive" $relativeDir)
    New-Item -ItemType Directory -Path $snapshotDir -Force | Out-Null
    Copy-Item -LiteralPath $resolved -Destination (Join-Path $snapshotDir "$name.$timestamp.bak") -Force
}

$stdin = [Console]::In.ReadToEnd()
if ([string]::IsNullOrWhiteSpace($stdin)) {
    exit 0
}

$event = $stdin | ConvertFrom-Json
if ($event.tool_name -ne "apply_patch") {
    exit 0
}

$command = [string]$event.tool_input.command
if ([string]::IsNullOrWhiteSpace($command)) {
    exit 0
}

$repoRoot = Get-RepoRoot
$paths = Get-PatchPaths -Command $command
foreach ($path in $paths) {
    New-Snapshot -RepoRoot $repoRoot -RelativePath $path
}

exit 0
