param(
    [Parameter(Mandatory = $true)]
    [string]$Path
)

$ErrorActionPreference = "Stop"

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

Write-Output "SNAPSHOT: $relative -> $backup"
