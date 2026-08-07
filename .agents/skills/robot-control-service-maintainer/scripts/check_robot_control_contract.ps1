#Requires -Version 7.0

[CmdletBinding()]
param(
    [string]$ProjectRoot = (Join-Path $PSScriptRoot "..\..\..\..")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)

function Read-Utf8Text {
    param([Parameter(Mandatory)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "缺少 RobotControl 契约文件：$Path"
    }
    return [System.IO.File]::ReadAllText($Path, [System.Text.Encoding]::UTF8)
}

function Get-MatchedValue {
    param(
        [Parameter(Mandatory)][string]$Text,
        [Parameter(Mandatory)][string]$Pattern,
        [Parameter(Mandatory)][string]$Label
    )

    $match = [regex]::Match($Text, $Pattern)
    if (-not $match.Success) {
        throw "无法从 $Label 读取 RobotControl 版本"
    }
    return $match.Groups[1].Value
}

$paths = @{
    Source = Join-Path $ProjectRoot "robot_control\__init__.py"
    Readme = Join-Path $ProjectRoot "robot_control\README.md"
    ApiReference = Join-Path $ProjectRoot "robot_control\API Reference.md"
    OpenApi = Join-Path $ProjectRoot "robot_control\openapi.yaml"
    Changelog = Join-Path $ProjectRoot "robot_control\CHANGELOG.md"
}

$sourceText = Read-Utf8Text $paths.Source
$readmeText = Read-Utf8Text $paths.Readme
$apiReferenceText = Read-Utf8Text $paths.ApiReference
$openApiText = Read-Utf8Text $paths.OpenApi
$changelogText = Read-Utf8Text $paths.Changelog

$version = Get-MatchedValue $sourceText '(?m)^\s*ROBOT_CONTROL_VERSION\s*=\s*["'']([^"'']+)["'']' $paths.Source
if ($version -notmatch '^\d+\.\d+\.\d+$') {
    throw "RobotControl 版本不是 a.b.c 形式：$version"
}

$openApiVersion = Get-MatchedValue $openApiText '(?ms)^info:\s*.*?^\s+version:\s*([^\s#]+)\s*$' $paths.OpenApi
$apiReferenceVersion = Get-MatchedValue $apiReferenceText '(?m)^当前契约版本：`([^`]+)`' $paths.ApiReference
$changelogVersion = Get-MatchedValue $changelogText '(?m)^当前版本：`([^`]+)`' $paths.Changelog

if ($openApiVersion -ne $version) {
    throw "OpenAPI 版本不一致：source=$version openapi=$openApiVersion"
}
if ($apiReferenceVersion -ne $version) {
    throw "API Reference 契约版本不一致：source=$version api_reference=$apiReferenceVersion"
}
if ($changelogVersion -ne $version) {
    throw "CHANGELOG 当前版本不一致：source=$version changelog=$changelogVersion"
}
if ($readmeText -notmatch 'API Reference\.md' -or $readmeText -notmatch 'openapi\.yaml') {
    throw "README 未同时声明 API Reference 和 OpenAPI 契约文档"
}
if ($changelogText -notmatch "(?m)^## $([regex]::Escape($version))(?:\s|$)") {
    throw "CHANGELOG 缺少当前版本章节：$version"
}

$headSourceText = (& git.exe -C $ProjectRoot show "HEAD:robot_control/__init__.py" 2>$null | Out-String)
if ($LASTEXITCODE -eq 0 -and $headSourceText) {
    $headVersion = Get-MatchedValue $headSourceText '(?m)^\s*ROBOT_CONTROL_VERSION\s*=\s*["'']([^"'']+)["'']' "HEAD:robot_control/__init__.py"
    if ($headVersion -ne $version) {
        $changedPaths = @(
            & git.exe -C $ProjectRoot diff --name-only HEAD --
            & git.exe -C $ProjectRoot ls-files --others --exclude-standard
        ) | ForEach-Object { $_.Trim().Replace("\", "/") } | Where-Object { $_ }

        $requiredPaths = @(
            "robot_control/README.md",
            "robot_control/API Reference.md",
            "robot_control/openapi.yaml",
            "robot_control/CHANGELOG.md"
        )
        $missingPaths = @($requiredPaths | Where-Object { $_ -notin $changedPaths })
        if ($missingPaths.Count -gt 0) {
            throw (
                "RobotControl 版本从 $headVersion 更新为 $version 时，以下强制文档未修改：" +
                ($missingPaths -join ", ")
            )
        }
        Write-Host "版本变更门禁通过：$headVersion -> $version；四份强制文档均已修改。"
    }
    else {
        Write-Host "版本未变化：$version；执行契约一致性检查，不触发版本文档变更门禁。"
    }
}
else {
    Write-Warning "无法读取 Git HEAD 版本，已跳过‘版本变化时四份文档必须修改’检查。"
}

Write-Host "RobotControl 契约检查通过：version=$version"
