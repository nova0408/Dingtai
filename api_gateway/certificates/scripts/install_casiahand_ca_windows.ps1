#Requires -Version 5.1

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$certificateDirectory = Join-Path (Split-Path -Parent $scriptDirectory) "client"
$localCertificatePath = Join-Path $certificateDirectory "casiahand-root-ca.cer"
$sshTarget = "orin"
$remoteCertificatePath = "/home/wuji-brain/casiahand-pki/orin/casiahand-root-ca.cer"
$sshOptions = @(
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=5",
    "-o", "ServerAliveInterval=5",
    "-o", "ServerAliveCountMax=2"
)

if (-not (Get-Command ssh.exe -ErrorAction SilentlyContinue)) {
    throw "找不到 ssh.exe，请先安装 Windows OpenSSH，并配置 ssh orin。"
}
if (-not (Get-Command scp.exe -ErrorAction SilentlyContinue)) {
    throw "找不到 scp.exe，请先安装 Windows OpenSSH，并配置 ssh orin。"
}

Write-Host "检查 SSH 连接：orin"
& ssh.exe @sshOptions orin "test -r '$remoteCertificatePath'"
if ($LASTEXITCODE -ne 0) {
    throw "无法通过 ssh orin 读取远端 CA。请首先配置好 ssh orin，并确认远端已完成证书注册：$remoteCertificatePath"
}

New-Item -ItemType Directory -Force -Path $certificateDirectory | Out-Null
Write-Host "下载 CasiaHand CA：orin`:$remoteCertificatePath"
& scp.exe @sshOptions "orin:$remoteCertificatePath" $localCertificatePath
if ($LASTEXITCODE -ne 0) {
    throw "CA 下载失败，请检查 ssh orin 配置和远端文件权限。"
}

$installerPath = Join-Path $scriptDirectory "install_ca_windows.ps1"
& $installerPath
if ($LASTEXITCODE -ne 0) {
    throw "Windows CA 安装失败。"
}

Write-Host "CasiaHand CA 已从 orin 下载并安装完成。"
Write-Host "本地证书文件：$localCertificatePath"
