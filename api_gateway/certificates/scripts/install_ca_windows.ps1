#Requires -Version 5.1

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$certificatePath = Join-Path (Join-Path (Split-Path -Parent $scriptDirectory) "client") "casiahand-root-ca.cer"
$resolvedPath = (Resolve-Path -LiteralPath $certificatePath).Path
$store = "Cert:\CurrentUser\Root"

$certificate = Import-Certificate -FilePath $resolvedPath -CertStoreLocation $store
Write-Host "CasiaHand CA 已安装到 $store，指纹：$($certificate.Thumbprint)"
