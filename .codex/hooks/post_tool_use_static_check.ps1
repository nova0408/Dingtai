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
    $pattern = '(?m)^\*\*\* (?:Add|Update|Delete) File: (?<path>.+)$'
    foreach ($match in [System.Text.RegularExpressions.Regex]::Matches($Command, $pattern)) {
        $path = $match.Groups["path"].Value.Trim()
        if ($path.Length -gt 0) {
            $paths.Add($path)
        }
    }
    return $paths
}

function Write-Block {
    param([string]$Reason)

    $payload = @{
        decision = "block"
        reason = $Reason
        hookSpecificOutput = @{
            hookEventName = "PostToolUse"
            additionalContext = $Reason
        }
    }
    $payload | ConvertTo-Json -Depth 5 -Compress
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
$relativePaths = Get-PatchPaths -Command $command
$existingPaths = New-Object System.Collections.Generic.List[string]
$pythonPaths = New-Object System.Collections.Generic.List[string]

foreach ($relative in $relativePaths) {
    $full = Join-Path $repoRoot $relative
    if (-not (Test-Path -LiteralPath $full)) {
        continue
    }
    $existingPaths.Add($full)
    if ([System.IO.Path]::GetExtension($full) -eq ".py") {
        $pythonPaths.Add($full)
    }
}

try {
    if ($existingPaths.Count -gt 0) {
        foreach ($path in $existingPaths) {
            powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $repoRoot ".codex/hooks/scan_text_integrity.ps1") -Path $path -Fix
            if ($LASTEXITCODE -ne 0) {
                throw "text integrity scan failed: $path"
            }
        }
    }

    if ($pythonPaths.Count -gt 0) {
        $ruff = Join-Path $repoRoot ".agents/skills/dingtai-static-check-workflow/scripts/check/run_ruff.ps1"
        $pyright = Join-Path $repoRoot ".agents/skills/dingtai-static-check-workflow/scripts/check/run_pyright.ps1"
        foreach ($path in $pythonPaths) {
            powershell -NoProfile -ExecutionPolicy Bypass -File $ruff -Target $path -Fix
            if ($LASTEXITCODE -ne 0) {
                throw "ruff failed after edit: $path"
            }
            powershell -NoProfile -ExecutionPolicy Bypass -File $pyright -Target $path
            if ($LASTEXITCODE -ne 0) {
                throw "pyright failed after edit: $path"
            }
        }
    }
}
catch {
    Write-Block -Reason $_.Exception.Message
    exit 0
}

exit 0
