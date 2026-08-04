#Requires -Version 5.1

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$certificatePath = Join-Path (Join-Path (Split-Path -Parent $scriptDirectory) "client") "casiahand-root-ca.cer"
$resolvedPath = (Resolve-Path -LiteralPath $certificatePath).Path
& adb get-state | Out-Null
if ($LASTEXITCODE -ne 0) { throw "未检测到可用的 adb 设备。" }
& adb push $resolvedPath /sdcard/Download/CasiaHand-Root-CA.cer
if ($LASTEXITCODE -ne 0) { throw "证书复制到 Android 设备失败。" }
& adb shell am start -a android.settings.SECURITY_SETTINGS | Out-Null
Write-Host "证书已复制到 Download。请在系统设置中手动完成 CA 安装；Android 不允许脚本静默信任用户 CA。"
