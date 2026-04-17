param(
    [Parameter(Mandatory = $true)]
    [string]$Path,

    [Parameter(Mandatory = $true)]
    [string]$Pattern,

    [Parameter(Mandatory = $true)]
    [string]$Replacement,

    [switch]$Multiline,
    [switch]$ValidatePython
)

$ErrorActionPreference = "Stop"
$utf8 = [System.Text.UTF8Encoding]::new($false)

$repoRoot = (Resolve-Path ".").Path
$target = (Resolve-Path $Path).Path
$relative = $target.Substring($repoRoot.Length).TrimStart("\", "/")
$relativeDir = Split-Path $relative -Parent
$name = Split-Path $relative -Leaf
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"

$snapshotDir = Join-Path $repoRoot (Join-Path ".archive" $relativeDir)
New-Item -ItemType Directory -Path $snapshotDir -Force | Out-Null
$backup = Join-Path $snapshotDir "$name.$timestamp.bak"
Copy-Item -LiteralPath $target -Destination $backup -Force

$old = [System.IO.File]::ReadAllText($target, $utf8)
$regexOpt = [System.Text.RegularExpressions.RegexOptions]::None
if ($Multiline) {
    $regexOpt = $regexOpt -bor [System.Text.RegularExpressions.RegexOptions]::Singleline
}

$match = [System.Text.RegularExpressions.Regex]::Match($old, $Pattern, $regexOpt)
if (-not $match.Success) {
    throw "replace failed: no match. pattern=$Pattern"
}

$new = [System.Text.RegularExpressions.Regex]::Replace($old, $Pattern, $Replacement, $regexOpt)
if ($new -eq $old) {
    throw "replace failed: unchanged after replace"
}

[System.IO.File]::WriteAllText($target, $new, $utf8)

if ($ValidatePython) {
    python -m py_compile $target
}

Write-Output "UPDATED: $relative"
Write-Output "SNAPSHOT: $backup"
