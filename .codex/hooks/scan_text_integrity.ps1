param(
    [Parameter(Mandatory = $true)]
    [string]$Path,

    [switch]$Fix
)

$ErrorActionPreference = "Stop"
$utf8Strict = [System.Text.UTF8Encoding]::new($false, $true)
$utf8Write = [System.Text.UTF8Encoding]::new($false)
$failed = New-Object System.Collections.Generic.List[string]

if (-not (Test-Path -LiteralPath $Path)) {
    exit 0
}

$bytes = [System.IO.File]::ReadAllBytes($Path)
try {
    $text = $utf8Strict.GetString($bytes)
}
catch {
    Write-Error "invalid utf-8: $Path"
    exit 1
}

$extension = [System.IO.Path]::GetExtension($Path).ToLowerInvariant()
$strictEscapes = $extension -in @(".py", ".ps1", ".json", ".toml", ".yaml", ".yml")

$backtickEscapedNewline = ([string][char]0x60) + "r" + ([string][char]0x60) + "n"
$slashEscapedNewline = ([string][char]0x5c) + "r" + ([string][char]0x5c) + "n"

if ($strictEscapes -and $Fix) {
    $updated = $text.Replace($backtickEscapedNewline, [Environment]::NewLine)
    $updated = $updated.Replace($slashEscapedNewline, [Environment]::NewLine)
    if ($updated -ne $text) {
        [System.IO.File]::WriteAllText($Path, $updated, $utf8Write)
        $text = $updated
    }
}

if ($strictEscapes -and $text.Contains($backtickEscapedNewline)) {
    $failed.Add("literal backtick newline remains: $Path")
}
if ($strictEscapes -and $text.Contains($slashEscapedNewline)) {
    $failed.Add("literal escaped newline remains: $Path")
}
if ($text.Contains([char]0xFFFD)) {
    $failed.Add("replacement character found: $Path")
}
if ($text.Contains([char]0x0000)) {
    $failed.Add("nul byte found: $Path")
}

if ($failed.Count -gt 0) {
    $failed -join [Environment]::NewLine | Write-Error
    exit 1
}

exit 0
