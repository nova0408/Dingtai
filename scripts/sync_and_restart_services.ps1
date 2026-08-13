#Requires -Version 7.0

[CmdletBinding()]
param(
    [switch]$CameraPipelineOnly,
    [switch]$RecordReplayOnly,
    [switch]$RobotControlOnly,
    [switch]$ApiGatewayOnly,
    [switch]$CalibrationServiceOnly,
    [switch]$RestartOnly,
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# 请在本机手动填写 Orin 用户的 sudo 密码。不要把填写密码后的脚本提交到 Git。
$RemoteSudoPassword = "wuji-brain"

$SshTarget = "orin"
$RemoteWorkspace = "/home/wuji-brain/workspace"
$RemoteStageRoot = "/home/wuji-brain/workspace/.deploy_stage"
$RemoteDeployScript = "/tmp/dingtai_sync_and_restart.sh"
$SshOptions = @(
    "-o", "ConnectTimeout=10",
    "-o", "ServerAliveInterval=5",
    "-o", "ServerAliveCountMax=2"
)

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory)]
        [string]$FilePath,
        [Parameter(Mandatory)]
        [string[]]$ArgumentList
    )

    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "命令执行失败：$FilePath $($ArgumentList -join ' ')"
    }
}

function Test-DeployFile {
    param(
        [Parameter(Mandatory)]
        [System.IO.FileInfo]$File
    )

    if ($File.Extension -in @(".pyc", ".log") -or $File.Name -in @("ball_debug_overlay.jpg", "runtime_state.json")) {
        return $false
    }
    if (
        $File.FullName -match "[\\/]api_gateway[\\/]certificates[\\/]" -and
        ($File.Name -match "\.(key|csr)\.pem$" -or $File.Extension -eq ".srl")
    ) {
        return $false
    }
    return $File.FullName -notmatch "[\\/]__pycache__[\\/]" -and
        $File.FullName -notmatch "[\\/]\.archive[\\/]" -and
        $File.FullName -notmatch "[\\/]api_gateway[\\/]certificates[\\/]generated[\\/]"
}

function Test-RecordReplayDeployFile {
    param(
        [Parameter(Mandatory)]
        [System.IO.FileInfo]$File,
        [Parameter(Mandatory)]
        [string]$RecordReplayRoot
    )

    if (-not (Test-DeployFile -File $File)) {
        return $false
    }
    $relativePath = [System.IO.Path]::GetRelativePath(
        $RecordReplayRoot,
        $File.FullName
    ).Replace("\", "/")
    return -not (
        $relativePath.StartsWith("prior_data/", [System.StringComparison]::OrdinalIgnoreCase) -or
        $relativePath.StartsWith("records/", [System.StringComparison]::OrdinalIgnoreCase)
    )
}

function Get-SourceVersion {
    param(
        [Parameter(Mandatory)]
        [string]$Path,
        [Parameter(Mandatory)]
        [string]$Symbol
    )

    $content = [System.IO.File]::ReadAllText($Path, [System.Text.Encoding]::UTF8)
    $pattern = '(?m)^\s*' + [regex]::Escape($Symbol) + '\s*=\s*["'']([^"'']+)["'']'
    $match = [regex]::Match($content, $pattern)
    if (-not $match.Success) {
        throw "无法从 $Path 读取版本常量：$Symbol"
    }
    return $match.Groups[1].Value
}

if ([string]::IsNullOrWhiteSpace($RemoteSudoPassword)) {
    throw "请先在脚本顶部填写 `$RemoteSudoPassword，再手动运行本脚本。"
}

$projectRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot "..")
)
$cameraPipelinePath = Join-Path $projectRoot "camera_pipeline"
$recordReplayPath = Join-Path $projectRoot "record_replay"
$robotControlPath = Join-Path $projectRoot "robot_control"
$apiGatewayPath = Join-Path $projectRoot "api_gateway"
$calibrationServicePath = Join-Path $projectRoot "calibration_service"
$calibrationServiceServicePath = Join-Path $calibrationServicePath "service/calibration.service"
$apiGatewayServicePath = Join-Path $apiGatewayPath "service/api-gateway.service"
$robotControlServicePath = Join-Path $robotControlPath "service/robot-control.service"
$cameraPipelineVersionPath = Join-Path $cameraPipelinePath "service/protocol.py"
$recordReplayVersionPath = Join-Path $recordReplayPath "__init__.py"
$robotControlVersionPath = Join-Path $robotControlPath "__init__.py"
$apiGatewayVersionPath = Join-Path $apiGatewayPath "__init__.py"
$restartScriptPaths = @(
    (Join-Path $projectRoot "scripts/restart_camera_pipeline_service.sh"),
    (Join-Path $projectRoot "scripts/restart_record_replay_service.sh"),
    (Join-Path $projectRoot "scripts/restart_robot_control_service.sh"),
    (Join-Path $projectRoot "scripts/restart_calibration_service.sh"),
    (Join-Path $projectRoot "scripts/restart_api_gateway_service.sh")
)
if (-not (Test-Path -LiteralPath $cameraPipelinePath -PathType Container)) {
    throw "缺少本机 camera_pipeline 目录：$cameraPipelinePath"
}
if (-not (Test-Path -LiteralPath $recordReplayPath -PathType Container)) {
    throw "缺少本机 record_replay 目录：$recordReplayPath"
}
if (-not (Test-Path -LiteralPath $robotControlPath -PathType Container)) {
    throw "缺少本机 robot_control 目录：$robotControlPath"
}
if (-not (Test-Path -LiteralPath $apiGatewayPath -PathType Container)) {
    throw "缺少本机 API Gateway 目录：$apiGatewayPath"
}
if (-not (Test-Path -LiteralPath $calibrationServicePath -PathType Container)) {
    throw "缺少本机 Calibration Service 目录：$calibrationServicePath"
}
if (-not (Test-Path -LiteralPath $calibrationServiceServicePath -PathType Leaf)) {
    throw "缺少本机 Calibration Service systemd 服务文件：$calibrationServiceServicePath"
}
if (-not (Test-Path -LiteralPath $apiGatewayServicePath -PathType Leaf)) {
    throw "缺少本机 API Gateway systemd 服务文件：$apiGatewayServicePath"
}
if (-not (Test-Path -LiteralPath $robotControlServicePath -PathType Leaf)) {
    throw "缺少本机 RobotControl systemd 服务文件：$robotControlServicePath"
}
foreach ($restartScriptPath in $restartScriptPaths) {
    if (-not (Test-Path -LiteralPath $restartScriptPath -PathType Leaf)) {
        throw "缺少本机服务重启脚本：$restartScriptPath"
    }
}
$expectedCameraPipelineVersion = Get-SourceVersion -Path $cameraPipelineVersionPath -Symbol "SERVICE_VERSION"
$expectedRecordReplayVersion = Get-SourceVersion -Path $recordReplayVersionPath -Symbol "RECORD_REPLAY_VERSION"
$expectedRobotControlVersion = Get-SourceVersion -Path $robotControlVersionPath -Symbol "ROBOT_CONTROL_VERSION"
$expectedApiGatewayVersion = Get-SourceVersion -Path $apiGatewayVersionPath -Symbol "API_GATEWAY_VERSION"
$calibrationServiceVersionPath = Join-Path $calibrationServicePath "__init__.py"
$expectedCalibrationServiceVersion = Get-SourceVersion -Path $calibrationServiceVersionPath -Symbol "CALIBRATION_SERVICE_VERSION"
Write-Host "本地部署版本：CameraPipeline=$expectedCameraPipelineVersion RecordReplay=$expectedRecordReplayVersion RobotControl=$expectedRobotControlVersion Calibration=$expectedCalibrationServiceVersion ApiGateway=$expectedApiGatewayVersion"

if ($CameraPipelineOnly -and -not $RestartOnly) {
    Write-Host "-CameraPipelineOnly 执行 CameraPipeline 单服务同步部署。"
}
if ($CameraPipelineOnly -and ($RecordReplayOnly -or $RobotControlOnly -or $ApiGatewayOnly -or $CalibrationServiceOnly)) {
    throw "CameraPipelineOnly 不能与其它 *Only 选项同时使用。"
}
if ($CalibrationServiceOnly -and -not $RestartOnly) {
    Write-Host "-CalibrationServiceOnly 执行 Calibration Service 单服务同步部署。"
}
if ($RecordReplayOnly -and ($CameraPipelineOnly -or $RobotControlOnly -or $ApiGatewayOnly -or $CalibrationServiceOnly)) {
    throw "-RecordReplayOnly 不能与其他 *Only 选项同时使用。"
}
if ($ApiGatewayOnly -and ($CameraPipelineOnly -or $RecordReplayOnly -or $RobotControlOnly -or $CalibrationServiceOnly)) {
    throw "-ApiGatewayOnly 不能与其他 *Only 选项同时使用。"
}
if ($RobotControlOnly -and ($CameraPipelineOnly -or $RecordReplayOnly -or $ApiGatewayOnly -or $CalibrationServiceOnly)) {
    throw "-RobotControlOnly 不能与其他 *Only 选项同时使用。"
}
if ($Force -and (-not $RecordReplayOnly -or $RestartOnly)) {
    throw "-Force 只能与 -RecordReplayOnly 同时用于同步部署，不能用于全量、其它服务或 -RestartOnly。"
}
if ($Force) {
    Write-Warning (
        "已启用 RecordReplay 强制同步模式：仅允许越过 rapid_stop 状态检查；" +
        "busy、未知状态及其它部署安全检查仍会拒绝。"
    )
}
if ($RestartOnly) {
    # 下游服务依赖：CameraPipeline -> RecordReplay；RobotControl/Calibration Service -> API Gateway。
    # 只重启上游时必须显式重启受影响的下游；只重启下游时不反向重启无变化的上游。
    $selectedRestartScripts = if ($CameraPipelineOnly) {
        @($restartScriptPaths[0], $restartScriptPaths[1], $restartScriptPaths[3], $restartScriptPaths[4])
    }
    elseif ($RecordReplayOnly) {
        @($restartScriptPaths[1], $restartScriptPaths[3])
    }
    elseif ($RobotControlOnly) {
        @($restartScriptPaths[2], $restartScriptPaths[3], $restartScriptPaths[4])
    }
    elseif ($CalibrationServiceOnly) {
        @($restartScriptPaths[3], $restartScriptPaths[4])
    }
    elseif ($ApiGatewayOnly) {
        @($restartScriptPaths[4])
    }
    else {
        $restartScriptPaths
    }
    $selectedRestartNames = @(
        $selectedRestartScripts | ForEach-Object { [System.IO.Path]::GetFileName($_) }
    )
    Write-Warning (
        "按服务依赖顺序重启：" + ($selectedRestartNames -join " -> ") +
        "；仅执行 systemd 重启和只读就绪检查，不会发送 /start 请求。"
    )
    foreach ($restartScriptPath in $selectedRestartScripts) {
        $restartScriptName = [System.IO.Path]::GetFileName($restartScriptPath)
        $remoteRestartScript = "/tmp/dingtai-agent-$restartScriptName"
        Invoke-CheckedCommand -FilePath "scp.exe" -ArgumentList @(
            $SshOptions + @(
                $restartScriptPath,
                "${SshTarget}:$remoteRestartScript"
            )
        )
        try {
            $restartArguments = if (
                $restartScriptName -eq "restart_record_replay_service.sh"
            ) {
                " --non-interactive"
            }
            else {
                ""
            }
            $remoteCommand = (
                "sudo -S -p '' bash '$remoteRestartScript'$restartArguments"
            )
            $RemoteSudoPassword | & ssh.exe @SshOptions $SshTarget $remoteCommand
            if ($LASTEXITCODE -ne 0) {
                throw "远端服务重启失败：$restartScriptName"
            }
        }
        finally {
            & ssh.exe @SshOptions $SshTarget "rm -f '$remoteRestartScript'"
        }
    }
    return
}

if ($CameraPipelineOnly -or $CalibrationServiceOnly) {
    $isCameraPipelineDeployment = $CameraPipelineOnly
    $serviceLabel = if ($isCameraPipelineDeployment) { "CameraPipeline" } else { "Calibration Service" }
    $packageName = if ($isCameraPipelineDeployment) { "camera_pipeline" } else { "calibration_service" }
    $serviceUnit = if ($isCameraPipelineDeployment) { "camera-pipeline.service" } else { "calibration.service" }
    $serviceFileName = if ($isCameraPipelineDeployment) { "camera-pipeline.service" } else { "calibration.service" }
    $expectedVersion = if ($isCameraPipelineDeployment) {
        $expectedCameraPipelineVersion
    }
    else {
        $expectedCalibrationServiceVersion
    }
    $readinessMode = if ($isCameraPipelineDeployment) { "camera" } else { "calibration" }
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $localTempBase = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
    $localTempRoot = Join-Path $localTempBase "dingtai-$packageName-deploy-$timestamp"
    $archivePath = Join-Path $localTempRoot "$packageName.tar"
    $manifestPath = Join-Path $localTempRoot "$packageName.sha256"
    $remoteStagePath = "$RemoteStageRoot/$($packageName -replace '_', '-')-$timestamp"
    $remoteArchivePath = "$remoteStagePath/$packageName.tar"
    $remoteManifestPath = "$remoteStagePath/$packageName.sha256"
    $singleServiceDeployScript = Join-Path $projectRoot "scripts/deploy_single_service.sh"

    New-Item -ItemType Directory -Path $localTempRoot | Out-Null
    try {
        $packagePath = Join-Path $projectRoot $packageName
        $deployFiles = @(
            Get-ChildItem -LiteralPath $packagePath -Recurse -File |
                Where-Object { Test-DeployFile -File $_ }
        )
        if ($deployFiles.Count -eq 0) {
            throw "$serviceLabel 部署清单为空"
        }
        $manifestLines = foreach ($file in ($deployFiles | Sort-Object FullName)) {
            $relativePath = [System.IO.Path]::GetRelativePath(
                $projectRoot,
                $file.FullName
            ).Replace("\", "/")
            $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
            "$hash  $relativePath"
        }
        [System.IO.File]::WriteAllText(
            $manifestPath,
            (($manifestLines -join "`n") + "`n"),
            [System.Text.UTF8Encoding]::new($false)
        )
        Invoke-CheckedCommand -FilePath "tar.exe" -ArgumentList @(
            "-cf", $archivePath,
            "--exclude=*/__pycache__/*",
            "--exclude=*/.archive",
            "--exclude=*/.archive/*",
            "--exclude=*.pyc",
            "--exclude=*.log",
            "-C", $projectRoot, $packageName
        )

        Write-Host "即将同步 $($deployFiles.Count) 个 $serviceLabel 文件到 $SshTarget"
        Write-Host "远端旧版本由 Git 管理，部署时不生成 .archive 快照"
        Invoke-CheckedCommand -FilePath "ssh.exe" -ArgumentList @(
            $SshOptions + @($SshTarget, "mkdir -p '$remoteStagePath'")
        )
        Invoke-CheckedCommand -FilePath "scp.exe" -ArgumentList @(
            $SshOptions + @($archivePath, "${SshTarget}:$remoteArchivePath")
        )
        Invoke-CheckedCommand -FilePath "scp.exe" -ArgumentList @(
            $SshOptions + @($manifestPath, "${SshTarget}:$remoteManifestPath")
        )
        Invoke-CheckedCommand -FilePath "scp.exe" -ArgumentList @(
            $SshOptions + @($singleServiceDeployScript, "${SshTarget}:/tmp/dingtai_single_service_deploy.sh")
        )
        Invoke-CheckedCommand -FilePath "ssh.exe" -ArgumentList @(
            $SshOptions + @($SshTarget, "tar -xf '$remoteArchivePath' -C '$remoteStagePath'")
        )
        $remoteCommand = "sudo -S -p '' bash '/tmp/dingtai_single_service_deploy.sh' " +
            "'$RemoteWorkspace' '$remoteStagePath' '$remoteManifestPath' '$packageName' " +
            "'$serviceUnit' '$serviceFileName' '$expectedVersion' '$readinessMode'"
        $RemoteSudoPassword | & ssh.exe @SshOptions $SshTarget $remoteCommand
        if ($LASTEXITCODE -ne 0) {
            throw "$serviceLabel 同步或重启失败，请检查上方日志。"
        }
    }
    finally {
        if (Test-Path -LiteralPath $localTempRoot) {
            $resolvedTempRoot = [System.IO.Path]::GetFullPath($localTempRoot)
            if (
                -not $resolvedTempRoot.StartsWith(
                    $localTempBase,
                    [System.StringComparison]::OrdinalIgnoreCase
                ) -or
                [System.IO.Path]::GetFileName($resolvedTempRoot) -notlike "dingtai-$packageName-deploy-*"
            ) {
                throw "拒绝清理非预期临时目录：$resolvedTempRoot"
            }
            Remove-Item -LiteralPath $localTempRoot -Recurse -Force
        }
    }
    return
}

if ($ApiGatewayOnly) {
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $localTempBase = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
    $localTempRoot = Join-Path $localTempBase "dingtai-api-gateway-deploy-$timestamp"
    $archivePath = Join-Path $localTempRoot "api_gateway.tar"
    $manifestPath = Join-Path $localTempRoot "api_gateway.sha256"
    $remoteScriptPath = Join-Path $localTempRoot "deploy_api_gateway.sh"
    $remoteStagePath = "$RemoteStageRoot/api-gateway-$timestamp"
    $remoteArchivePath = "$remoteStagePath/api_gateway.tar"
    $remoteManifestPath = "$remoteStagePath/api_gateway.sha256"

    New-Item -ItemType Directory -Path $localTempRoot | Out-Null
    try {
        $deployFiles = @(
            Get-ChildItem -LiteralPath $apiGatewayPath -Recurse -File |
                Where-Object { Test-DeployFile -File $_ }
        )
        if ($deployFiles.Count -eq 0) {
            throw "API Gateway 部署清单为空"
        }
        $manifestLines = foreach ($file in ($deployFiles | Sort-Object FullName)) {
            $relativePath = [System.IO.Path]::GetRelativePath(
                $projectRoot,
                $file.FullName
            ).Replace("\", "/")
            $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
            "$hash  $relativePath"
        }
        [System.IO.File]::WriteAllText(
            $manifestPath,
            (($manifestLines -join "`n") + "`n"),
            [System.Text.UTF8Encoding]::new($false)
        )
        Invoke-CheckedCommand -FilePath "tar.exe" -ArgumentList @(
            "-cf", $archivePath,
            "--exclude=*/__pycache__/*", "--exclude=*/.archive", "--exclude=*/.archive/*", "--exclude=api_gateway/certificates/generated", "--exclude=api_gateway/certificates/generated/*", "--exclude=*.key.pem", "--exclude=*.csr.pem", "--exclude=*.srl", "--exclude=*.pyc", "--exclude=*.log",
            "-C", $projectRoot, "api_gateway"
        )

        $remoteScript = @'
#!/usr/bin/env bash
set -euo pipefail

workspace="$1"
stage_path="$2"
manifest_path="$3"
expected_gateway_version="$4"
unit="api-gateway.service"
deploy_log_since="$(date '+%Y-%m-%d %H:%M:%S')"

cleanup() {
  rm -rf -- "${stage_path}"
  rm -f -- "/tmp/dingtai_api_gateway_deploy.sh"
}
trap cleanup EXIT

case "${stage_path}" in
  "${workspace}/.deploy_stage/api-gateway-"*) ;;
  *) echo "[deploy] invalid API Gateway stage path: ${stage_path}"; exit 1 ;;
esac
expected_count="$(wc -l < "${manifest_path}")"
actual_stage_count="$(find "${stage_path}/api_gateway" -type f ! -name '*.pyc' ! -name '*.log' ! -path '*/__pycache__/*' | wc -l)"
if [[ "${actual_stage_count}" -ne "${expected_count}" ]]; then
  echo "[deploy] API Gateway staged file count mismatch expected=${expected_count} actual=${actual_stage_count}"
  exit 1
fi
(
  cd "${stage_path}"
  sha256sum --check "${manifest_path}"
)
json_field() {
  local field="$1"
  /home/wuji-brain/miniconda3/envs/wuji/bin/python -c \
    'import json, sys; print(json.load(sys.stdin)[sys.argv[1]])' "${field}"
}
for tls_path in \
  "/etc/dingtai/api-gateway/tls/casiahand-root-ca.crt.pem" \
  "/etc/dingtai/api-gateway/tls/api-gateway.fullchain.pem" \
  "/etc/dingtai/api-gateway/tls/api-gateway.key.pem"; do
  [[ -r "${tls_path}" ]] || { echo "[deploy] missing or unreadable API Gateway TLS file: ${tls_path}"; exit 1; }
done
systemctl stop --no-block "${unit}" || true
for attempt in $(seq 1 15); do
  active_state="$(systemctl show "${unit}" -p ActiveState --value 2>/dev/null || true)"
  if [[ -z "${active_state}" || "${active_state}" == "inactive" || "${active_state}" == "failed" ]]; then
    systemctl reset-failed "${unit}" || true
    break
  fi
  sleep 1
done
if [[ -d "${workspace}/api_gateway" ]]; then
  rm -rf -- "${workspace:?}/api_gateway"
fi
mv "${stage_path}/api_gateway" "${workspace}/api_gateway"
install -m 0644 "${workspace}/api_gateway/service/api-gateway.service" "/etc/systemd/system/${unit}"
if [[ -f "${workspace}/api_gateway/requirements.txt" ]]; then
  "/home/wuji-brain/miniconda3/envs/wuji/bin/python" -m pip install --disable-pip-version-check --no-input -r "${workspace}/api_gateway/requirements.txt"
else
  echo "[deploy] api_gateway/requirements.txt not present; skip API Gateway dependency installation"
fi
systemctl daemon-reload
systemctl enable "${unit}"
systemctl restart --no-block "${unit}"
ready=false
gateway_hostname="$(hostname)"
for attempt in $(seq 1 20); do
  if systemctl is-active --quiet "${unit}" &&
     ss -ltn '( sport = :443 )' | grep -q LISTEN &&
     curl -fsS --max-time 5 --cacert /etc/dingtai/api-gateway/tls/casiahand-root-ca.crt.pem --resolve "${gateway_hostname}:443:127.0.0.1" "https://${gateway_hostname}/api/v1/gateway/health"; then
    ready=true
    echo
    break
  fi
  sleep 1
done
if [[ "${ready}" != "true" ]]; then
  echo "[deploy] API Gateway was not HTTPS-ready within 20s"
  systemctl status "${unit}" --no-pager -l || true
  journalctl -u "${unit}" --since "${deploy_log_since}" --no-pager -o short-precise || true
  exit 1
fi
gateway_payload="$(curl -fsS --max-time 5 --cacert /etc/dingtai/api-gateway/tls/casiahand-root-ca.crt.pem --resolve "${gateway_hostname}:443:127.0.0.1" "https://${gateway_hostname}/api/v1/gateway/health")"
gateway_version="$(printf '%s' "${gateway_payload}" | json_field gateway_version)"
if [[ "${gateway_version}" != "${expected_gateway_version}" ]]; then
  echo "[deploy] API Gateway version mismatch expected=${expected_gateway_version} actual=${gateway_version}"
  exit 1
fi
echo "[deploy] API Gateway updated and restarted; version=${gateway_version}; GET /api/v1/gateway/health passed"
'@
        [System.IO.File]::WriteAllText(
            $remoteScriptPath,
            ($remoteScript.Replace([Environment]::NewLine, [string][char]10) + [char]10),
            [System.Text.UTF8Encoding]::new($false)
        )

        Write-Host "即将同步 $($deployFiles.Count) 个 API Gateway 文件到 $SshTarget"
        Invoke-CheckedCommand -FilePath "ssh.exe" -ArgumentList @($SshOptions + @($SshTarget, "mkdir -p '$remoteStagePath'"))
        Invoke-CheckedCommand -FilePath "scp.exe" -ArgumentList @($SshOptions + @($archivePath, "${SshTarget}:$remoteArchivePath"))
        Invoke-CheckedCommand -FilePath "scp.exe" -ArgumentList @($SshOptions + @($manifestPath, "${SshTarget}:$remoteManifestPath"))
        Invoke-CheckedCommand -FilePath "scp.exe" -ArgumentList @($SshOptions + @($remoteScriptPath, "${SshTarget}:/tmp/dingtai_api_gateway_deploy.sh"))
        Invoke-CheckedCommand -FilePath "ssh.exe" -ArgumentList @($SshOptions + @($SshTarget, "tar -xf '$remoteArchivePath' -C '$remoteStagePath'"))
        $remoteCommand = "sudo -S -p '' bash '/tmp/dingtai_api_gateway_deploy.sh' '$RemoteWorkspace' '$remoteStagePath' '$remoteManifestPath' '$expectedApiGatewayVersion'"
        $RemoteSudoPassword | & ssh.exe @SshOptions $SshTarget $remoteCommand
        if ($LASTEXITCODE -ne 0) {
            throw "远端 API Gateway 同步或重启失败，请检查上方日志。"
        }
    }
    finally {
        if (Test-Path -LiteralPath $localTempRoot) {
            Remove-Item -LiteralPath $localTempRoot -Recurse -Force
        }
    }
    return
}

if ($RobotControlOnly) {
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $localTempBase = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
    $localTempRoot = Join-Path $localTempBase "dingtai-robot-control-deploy-$timestamp"
    $archivePath = Join-Path $localTempRoot "robot_control.tar"
    $manifestPath = Join-Path $localTempRoot "robot_control.sha256"
    $remoteScriptPath = Join-Path $localTempRoot "deploy_robot_control.sh"
    $remoteStagePath = "$RemoteStageRoot/robot-control-$timestamp"
    $remoteArchivePath = "$remoteStagePath/robot_control.tar"
    $remoteManifestPath = "$remoteStagePath/robot_control.sha256"

    New-Item -ItemType Directory -Path $localTempRoot | Out-Null
    try {
        $deployFiles = @(
            Get-ChildItem -LiteralPath $robotControlPath -Recurse -File |
                Where-Object { Test-DeployFile -File $_ }
        )
        if ($deployFiles.Count -eq 0) {
            throw "RobotControl 部署清单为空"
        }

        $manifestLines = foreach ($file in ($deployFiles | Sort-Object FullName)) {
            $relativePath = [System.IO.Path]::GetRelativePath(
                $projectRoot,
                $file.FullName
            ).Replace("\", "/")
            $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
            "$hash  $relativePath"
        }
        [System.IO.File]::WriteAllText(
            $manifestPath,
            (($manifestLines -join "`n") + "`n"),
            [System.Text.UTF8Encoding]::new($false)
        )

        Invoke-CheckedCommand -FilePath "tar.exe" -ArgumentList @(
            "-cf",
            $archivePath,
            "--exclude=*/__pycache__/*",
            "--exclude=*/.archive",
            "--exclude=*/.archive/*",
            "--exclude=*.pyc",
            "--exclude=*.log",
            "-C",
            $projectRoot,
            "robot_control"
        )

        $remoteScript = @'
#!/usr/bin/env bash
set -euo pipefail

workspace="$1"
stage_path="$2"
manifest_path="$3"
expected_robot_version="$4"
unit="robot-control.service"
deploy_log_since="$(date '+%Y-%m-%d %H:%M:%S')"

cleanup() {
  rm -rf -- "${stage_path}"
  rm -f -- "/tmp/dingtai_robot_control_deploy.sh"
}
trap cleanup EXIT

case "${stage_path}" in
  "${workspace}/.deploy_stage/robot-control-"*) ;;
  *) echo "[deploy] invalid RobotControl stage path: ${stage_path}"; exit 1 ;;
esac

expected_count="$(wc -l < "${manifest_path}")"
actual_stage_count="$(find "${stage_path}/robot_control" -type f ! -name '*.pyc' ! -name '*.log' ! -path '*/__pycache__/*' | wc -l)"
if [[ "${actual_stage_count}" -ne "${expected_count}" ]]; then
  echo "[deploy] RobotControl staged file count mismatch expected=${expected_count} actual=${actual_stage_count}"
  exit 1
fi
(
  cd "${stage_path}"
  sha256sum --check "${manifest_path}"
)
json_field() {
  local field="$1"
  /home/wuji-brain/miniconda3/envs/wuji/bin/python -c \
    'import json, sys; print(json.load(sys.stdin)[sys.argv[1]])' "${field}"
}
echo "[deploy] stopping ${unit}"
systemctl stop --no-block "${unit}" || true
for attempt in $(seq 1 15); do
  active_state="$(systemctl show "${unit}" -p ActiveState --value 2>/dev/null || true)"
  if [[ -z "${active_state}" || "${active_state}" == "inactive" || "${active_state}" == "failed" ]]; then
    systemctl reset-failed "${unit}" || true
    break
  fi
  sleep 1
done

if [[ -d "${workspace}/robot_control" ]]; then
  rm -rf -- "${workspace:?}/robot_control"
fi
mv "${stage_path}/robot_control" "${workspace}/robot_control"

install -m 0644 \
  "${workspace}/robot_control/service/robot-control.service" \
  "/etc/systemd/system/${unit}"
systemctl daemon-reload
systemctl enable "${unit}"
systemctl restart --no-block "${unit}"

ready=false
for attempt in $(seq 1 20); do
  if systemctl is-active --quiet "${unit}" &&
     ss -ltn '( sport = :6500 )' | grep -q LISTEN &&
     curl -fsS --max-time 5 http://127.0.0.1:6500/api/v1/health; then
    ready=true
    echo
    break
  fi
  sleep 1
done
if [[ "${ready}" != "true" ]]; then
  echo "[deploy] RobotControl was not HTTP-ready within 20s"
  systemctl status "${unit}" --no-pager -l || true
  journalctl -u "${unit}" --since "${deploy_log_since}" --no-pager -o short-precise || true
  exit 1
fi
robot_payload="$(curl -fsS --max-time 5 http://127.0.0.1:6500/api/v1/health)"
robot_version="$(printf '%s' "${robot_payload}" | json_field service_version)"
if [[ "${robot_version}" != "${expected_robot_version}" ]]; then
  echo "[deploy] RobotControl version mismatch expected=${expected_robot_version} actual=${robot_version}"
  exit 1
fi
echo "[deploy] RobotControl updated and restarted; version=${robot_version}; GET /api/v1/health passed"

echo "[deploy] restoring calibration.service after RobotControl deployment"
systemctl start --no-block calibration.service
calibration_ready=false
for attempt in $(seq 1 10); do
  if systemctl is-active --quiet calibration.service &&
     ss -ltn '( sport = :6600 )' | grep -q LISTEN &&
     calibration_payload="$(curl -fsS --max-time 5 http://127.0.0.1:6600/api/v1/status)"; then
    calibration_state="$(printf '%s' "${calibration_payload}" | json_field state)"
    if [[ "${calibration_state}" == "idle" ]]; then
      calibration_ready=true
      break
    fi
  fi
  sleep 1
done
if [[ "${calibration_ready}" != "true" ]]; then
  echo "[deploy] Calibration Service was not restored after RobotControl deployment"
  systemctl status calibration.service --no-pager -l || true
  journalctl -u calibration.service --since "${deploy_log_since}" --no-pager -o short-precise || true
  exit 1
fi
echo "[deploy] Calibration Service restored; GET /api/v1/status state=idle"
'@

        [System.IO.File]::WriteAllText(
            $remoteScriptPath,
            ($remoteScript.Replace([Environment]::NewLine, [string][char]10) + [char]10),
            [System.Text.UTF8Encoding]::new($false)
        )

        Write-Host "即将同步 $($deployFiles.Count) 个 RobotControl 文件到 $SshTarget"
        Write-Host "远端旧版本由 Git 管理，部署时不生成 .archive 快照"
        Invoke-CheckedCommand -FilePath "ssh.exe" -ArgumentList @(
            $SshOptions + @($SshTarget, "mkdir -p '$remoteStagePath'")
        )
        Invoke-CheckedCommand -FilePath "scp.exe" -ArgumentList @(
            $SshOptions + @($archivePath, "${SshTarget}:$remoteArchivePath")
        )
        Invoke-CheckedCommand -FilePath "scp.exe" -ArgumentList @(
            $SshOptions + @($manifestPath, "${SshTarget}:$remoteManifestPath")
        )
        Invoke-CheckedCommand -FilePath "scp.exe" -ArgumentList @(
            $SshOptions + @($remoteScriptPath, "${SshTarget}:/tmp/dingtai_robot_control_deploy.sh")
        )
        Invoke-CheckedCommand -FilePath "ssh.exe" -ArgumentList @(
            $SshOptions + @($SshTarget, "tar -xf '$remoteArchivePath' -C '$remoteStagePath'")
        )

        $remoteCommand = "sudo -S -p '' bash '/tmp/dingtai_robot_control_deploy.sh' " +
            "'$RemoteWorkspace' '$remoteStagePath' '$remoteManifestPath' '$expectedRobotControlVersion'"
        $RemoteSudoPassword | & ssh.exe @SshOptions $SshTarget $remoteCommand
        if ($LASTEXITCODE -ne 0) {
            throw "远端 RobotControl 同步或重启失败，请检查上方日志。"
        }
    }
    finally {
        if (Test-Path -LiteralPath $localTempRoot) {
            $resolvedTempRoot = [System.IO.Path]::GetFullPath($localTempRoot)
            if (
                -not $resolvedTempRoot.StartsWith(
                    $localTempBase,
                    [System.StringComparison]::OrdinalIgnoreCase
                ) -or
                [System.IO.Path]::GetFileName($resolvedTempRoot) -notlike "dingtai-robot-control-deploy-*"
            ) {
                throw "拒绝清理非预期临时目录：$resolvedTempRoot"
            }
            Remove-Item -LiteralPath $localTempRoot -Recurse -Force
        }
    }
    return
}

if ($RecordReplayOnly) {
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $localTempBase = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
    $localTempRoot = Join-Path $localTempBase "dingtai-record-replay-deploy-$timestamp"
    $archivePath = Join-Path $localTempRoot "record_replay.tar"
    $manifestPath = Join-Path $localTempRoot "record_replay.sha256"
    $remoteScriptPath = Join-Path $localTempRoot "deploy_record_replay.sh"
    $remoteStagePath = "$RemoteStageRoot/record-replay-$timestamp"
    $remoteArchivePath = "$remoteStagePath/record_replay.tar"
    $remoteManifestPath = "$remoteStagePath/record_replay.sha256"

    New-Item -ItemType Directory -Path $localTempRoot | Out-Null
    try {
        $deployFiles = @(
            Get-ChildItem -LiteralPath $recordReplayPath -Recurse -File |
                Where-Object {
                    Test-RecordReplayDeployFile -File $_ -RecordReplayRoot $recordReplayPath
                }
        )
        if ($deployFiles.Count -eq 0) {
            throw "RecordReplay 部署清单为空"
        }

        $manifestLines = foreach ($file in ($deployFiles | Sort-Object FullName)) {
            $relativePath = [System.IO.Path]::GetRelativePath(
                $projectRoot,
                $file.FullName
            ).Replace("\", "/")
            $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
            "$hash  $relativePath"
        }
        [System.IO.File]::WriteAllText(
            $manifestPath,
            (($manifestLines -join "`n") + "`n"),
            [System.Text.UTF8Encoding]::new($false)
        )

        Invoke-CheckedCommand -FilePath "tar.exe" -ArgumentList @(
            "-cf",
            $archivePath,
            "--exclude=*/__pycache__/*",
            "--exclude=*/.archive",
            "--exclude=*/.archive/*",
            "--exclude=*.pyc",
            "--exclude=*.log",
            "--exclude=record_replay/prior_data",
            "--exclude=record_replay/prior_data/*",
            "--exclude=record_replay/records",
            "--exclude=record_replay/records/*",
            "-C",
            $projectRoot,
            "record_replay"
        )

        $remoteScript = @'
#!/usr/bin/env bash
set -euo pipefail

workspace="$1"
stage_path="$2"
manifest_path="$3"
expected_record_version="$4"
force_record_replay_state="$5"
unit="record-replay.service"
preserve_root="${stage_path}/preserve"
deploy_log_since="$(date '+%Y-%m-%d %H:%M:%S')"

cleanup() {
  rm -rf -- "${stage_path}"
  rm -f -- "/tmp/dingtai_record_replay_deploy.sh"
}
trap cleanup EXIT

case "${stage_path}" in
  "${workspace}/.deploy_stage/record-replay-"*) ;;
  *) echo "[deploy] invalid RecordReplay stage path: ${stage_path}"; exit 1 ;;
esac
if [[ "${force_record_replay_state}" != "true" && "${force_record_replay_state}" != "false" ]]; then
  echo "[deploy] invalid force_record_replay_state: ${force_record_replay_state}"
  exit 1
fi

for exempt_dir in prior_data records; do
  if [[ -e "${stage_path}/record_replay/${exempt_dir}" ]]; then
    echo "[deploy] staged RecordReplay package must not contain exempt directory: ${exempt_dir}"
    exit 1
  fi
done

expected_count="$(wc -l < "${manifest_path}")"
actual_stage_count="$(find "${stage_path}/record_replay" -type f ! -name '*.pyc' ! -name '*.log' ! -path '*/__pycache__/*' ! -path '*/.archive/*' | wc -l)"
if [[ "${actual_stage_count}" -ne "${expected_count}" ]]; then
  echo "[deploy] RecordReplay staged file count mismatch expected=${expected_count} actual=${actual_stage_count}"
  exit 1
fi
(
  cd "${stage_path}"
  sha256sum --check "${manifest_path}"
)

if ! systemctl is-active --quiet camera-pipeline.service ||
   ! ss -ltn '( sport = :6200 )' | grep -q LISTEN; then
  echo "[deploy] refused: CameraPipeline is not active and listening on 6200"
  systemctl status camera-pipeline.service --no-pager -l || true
  exit 1
fi

json_field() {
  local field="$1"
  /home/wuji-brain/miniconda3/envs/wuji/bin/python -c \
    'import json, sys; print(json.load(sys.stdin)[sys.argv[1]])' "${field}"
}

service_state="not_running"
if ss -ltn '( sport = :6300 )' | grep -q LISTEN; then
  status_payload="$(curl -fsS --max-time 5 http://127.0.0.1:6300/status)"
  service_state="$(printf '%s' "${status_payload}" | json_field state)"
  if [[ "${service_state}" == "rapid_stop" && "${force_record_replay_state}" == "true" ]]; then
    echo "[deploy] WARNING: explicitly forced RecordReplay deployment from rapid_stop"
  elif [[ "${service_state}" != "idle" && "${service_state}" != "waiting" ]]; then
    echo "[deploy] refused: RecordReplay current state is ${service_state}"
    exit 1
  fi
fi
if ss -ltn '( sport = :6600 )' | grep -q LISTEN; then
  calibration_payload="$(curl -fsS --max-time 5 http://127.0.0.1:6600/api/v1/status)"
  calibration_state="$(printf '%s' "${calibration_payload}" | /home/wuji-brain/miniconda3/envs/wuji/bin/python -c 'import json, sys; print(json.load(sys.stdin)["state"])')"
  if [[ "${calibration_state}" != "idle" ]]; then
    echo "[deploy] refused: Calibration Service current state is ${calibration_state}"
    exit 1
  fi
fi

echo "[deploy] stopping ${unit}"
systemctl stop --no-block "${unit}" || true
for attempt in $(seq 1 15); do
  active_state="$(systemctl show "${unit}" -p ActiveState --value 2>/dev/null || true)"
  if [[ -z "${active_state}" || "${active_state}" == "inactive" || "${active_state}" == "failed" ]]; then
    systemctl reset-failed "${unit}" || true
    break
  fi
  sleep 1
done

if [[ -e "${stage_path}/record_replay/.archive" ]]; then
  echo "[deploy] staged RecordReplay package must not contain runtime .archive"
  exit 1
fi
mkdir -p "${preserve_root}"
for preserved_name in prior_data records runtime_state.json; do
  if [[ -e "${workspace}/record_replay/${preserved_name}" ]]; then
    mv "${workspace}/record_replay/${preserved_name}" "${preserve_root}/${preserved_name}"
  fi
done
if [[ -d "${workspace}/record_replay" ]]; then
  rm -rf -- "${workspace:?}/record_replay"
fi
mv "${stage_path}/record_replay" "${workspace}/record_replay"
for exempt_dir in prior_data records; do
  preserved_dir="${preserve_root}/${exempt_dir}"
  target_dir="${workspace}/record_replay/${exempt_dir}"
  if [[ -e "${preserved_dir}" ]]; then
    mv "${preserved_dir}" "${target_dir}"
    echo "[deploy] preserved remote RecordReplay directory: ${exempt_dir}"
  else
    mkdir -p "${target_dir}"
    echo "[deploy] created missing remote RecordReplay directory: ${exempt_dir}"
  fi
done
runtime_state_path="${workspace}/record_replay/runtime_state.json"
preserved_runtime_state_path="${preserve_root}/runtime_state.json"
case "${service_state}" in
  idle|waiting)
    printf '{\n  "state": "idle"\n}\n' > "${runtime_state_path}"
    echo "[deploy] restored pre-deploy RecordReplay state: idle"
    ;;
  rapid_stop)
    printf '{\n  "state": "rapid_stop"\n}\n' > "${runtime_state_path}"
    echo "[deploy] preserved pre-deploy RecordReplay state: rapid_stop"
    ;;
  not_running)
    if [[ -f "${preserved_runtime_state_path}" ]]; then
      mv "${preserved_runtime_state_path}" "${runtime_state_path}"
      echo "[deploy] preserved existing RecordReplay runtime_state.json"
    fi
    ;;
  *)
    echo "[deploy] refused unexpected captured RecordReplay state: ${service_state}"
    exit 1
    ;;
esac

install -m 0644 \
  "${workspace}/record_replay/service/record-replay.service" \
  "/etc/systemd/system/${unit}"
systemctl daemon-reload
systemctl enable "${unit}"
systemctl restart --no-block "${unit}"

record_ready=false
for attempt in $(seq 1 20); do
  if systemctl is-active --quiet "${unit}" &&
     systemctl is-active --quiet camera-pipeline.service &&
     ss -ltn '( sport = :6200 )' | grep -q LISTEN &&
     ss -ltn '( sport = :6300 )' | grep -q LISTEN &&
     status_payload="$(curl -fsS --max-time 5 http://127.0.0.1:6300/status)"; then
    record_ready=true
    printf '%s\n' "${status_payload}"
    break
  fi
  sleep 1
done
if [[ "${record_ready}" != "true" ]]; then
  echo "[deploy] RecordReplay was not HTTP-ready within 20s"
  systemctl status "${unit}" --no-pager -l || true
  journalctl -u "${unit}" --since "${deploy_log_since}" --no-pager -o short-precise || true
  exit 1
fi

record_version="$(cd "${workspace}" && /home/wuji-brain/miniconda3/envs/wuji/bin/python -c 'from record_replay import RECORD_REPLAY_VERSION; print(RECORD_REPLAY_VERSION)')"
if [[ "${record_version}" != "${expected_record_version}" ]]; then
  echo "[deploy] RecordReplay version mismatch expected=${expected_record_version} actual=${record_version}"
  exit 1
fi
(
  cd "${workspace}"
  sha256sum --check "${manifest_path}"
)
echo "[deploy] RecordReplay updated and restarted; version=${record_version}; GET /status passed"
'@

        [System.IO.File]::WriteAllText(
            $remoteScriptPath,
            ($remoteScript.Replace([Environment]::NewLine, [string][char]10) + [char]10),
            [System.Text.UTF8Encoding]::new($false)
        )

        Write-Host "即将同步 $($deployFiles.Count) 个 RecordReplay 文件到 $SshTarget"
        Write-Host "远端旧版本由 Git 管理，部署时不生成 .archive 快照"
        $forceRecordReplayState = if ($Force) { "true" } else { "false" }
        Invoke-CheckedCommand -FilePath "ssh.exe" -ArgumentList @(
            $SshOptions + @($SshTarget, "mkdir -p '$remoteStagePath'")
        )
        Invoke-CheckedCommand -FilePath "scp.exe" -ArgumentList @(
            $SshOptions + @($archivePath, "${SshTarget}:$remoteArchivePath")
        )
        Invoke-CheckedCommand -FilePath "scp.exe" -ArgumentList @(
            $SshOptions + @($manifestPath, "${SshTarget}:$remoteManifestPath")
        )
        Invoke-CheckedCommand -FilePath "scp.exe" -ArgumentList @(
            $SshOptions + @($remoteScriptPath, "${SshTarget}:/tmp/dingtai_record_replay_deploy.sh")
        )
        Invoke-CheckedCommand -FilePath "ssh.exe" -ArgumentList @(
            $SshOptions + @($SshTarget, "tar -xf '$remoteArchivePath' -C '$remoteStagePath'")
        )

        $remoteCommand = "sudo -S -p '' bash '/tmp/dingtai_record_replay_deploy.sh' " +
            "'$RemoteWorkspace' '$remoteStagePath' '$remoteManifestPath' " +
            "'$expectedRecordReplayVersion' '$forceRecordReplayState'"
        $RemoteSudoPassword | & ssh.exe @SshOptions $SshTarget $remoteCommand
        if ($LASTEXITCODE -ne 0) {
            throw "远端 RecordReplay 同步或重启失败，请检查上方日志。"
        }
    }
    finally {
        if (Test-Path -LiteralPath $localTempRoot) {
            $resolvedTempRoot = [System.IO.Path]::GetFullPath($localTempRoot)
            if (
                -not $resolvedTempRoot.StartsWith(
                    $localTempBase,
                    [System.StringComparison]::OrdinalIgnoreCase
                ) -or
                [System.IO.Path]::GetFileName($resolvedTempRoot) -notlike "dingtai-record-replay-deploy-*"
            ) {
                throw "拒绝清理非预期临时目录：$resolvedTempRoot"
            }
            Remove-Item -LiteralPath $localTempRoot -Recurse -Force
        }
    }
    return
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$localTempBase = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
$localTempRoot = Join-Path $localTempBase "dingtai-deploy-$timestamp"
$archivePath = Join-Path $localTempRoot "services.tar"
$manifestPath = Join-Path $localTempRoot "services.sha256"
$remoteScriptPath = Join-Path $localTempRoot "dingtai_sync_and_restart.sh"
$remoteStagePath = "$RemoteStageRoot/$timestamp"
$remoteArchivePath = "$remoteStagePath/services.tar"
$remoteManifestPath = "$remoteStagePath/services.sha256"

New-Item -ItemType Directory -Path $localTempRoot | Out-Null
try {
    $deployFiles = @(
        Get-ChildItem -LiteralPath $cameraPipelinePath -Recurse -File |
            Where-Object { Test-DeployFile -File $_ }
        Get-ChildItem -LiteralPath $recordReplayPath -Recurse -File |
            Where-Object {
                Test-RecordReplayDeployFile -File $_ -RecordReplayRoot $recordReplayPath
            }
        Get-ChildItem -LiteralPath $calibrationServicePath -Recurse -File |
            Where-Object { Test-DeployFile -File $_ }
        Get-ChildItem -LiteralPath $robotControlPath -Recurse -File |
            Where-Object { Test-DeployFile -File $_ }
        Get-ChildItem -LiteralPath $apiGatewayPath -Recurse -File |
            Where-Object { Test-DeployFile -File $_ }
        $restartScriptPaths | ForEach-Object { Get-Item -LiteralPath $_ }
    )
    if ($deployFiles.Count -eq 0) {
        throw "部署清单为空"
    }

    $manifestLines = foreach ($file in ($deployFiles | Sort-Object FullName)) {
        $relativePath = [System.IO.Path]::GetRelativePath(
            $projectRoot,
            $file.FullName
        ).Replace("\", "/")
        $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        "$hash  $relativePath"
    }
    [System.IO.File]::WriteAllText(
        $manifestPath,
        (($manifestLines -join "`n") + "`n"),
        [System.Text.UTF8Encoding]::new($false)
    )

    Invoke-CheckedCommand -FilePath "tar.exe" -ArgumentList @(
        "-cf",
        $archivePath,
        "--exclude=*/__pycache__/*",
        "--exclude=*/.archive",
        "--exclude=*/.archive/*",
        "--exclude=record_replay/prior_data",
        "--exclude=record_replay/prior_data/*",
        "--exclude=record_replay/records",
        "--exclude=record_replay/records/*",
        "--exclude=api_gateway/certificates/generated",
        "--exclude=api_gateway/certificates/generated/*",
        "--exclude=*.key.pem",
        "--exclude=*.csr.pem",
        "--exclude=*.srl",
        "--exclude=*.pyc",
        "--exclude=*.log",
        "-C",
        $projectRoot,
        "camera_pipeline",
        "record_replay",
        "calibration_service",
        "robot_control",
        "api_gateway",
        "scripts/restart_camera_pipeline_service.sh",
        "scripts/restart_record_replay_service.sh",
        "scripts/restart_robot_control_service.sh",
        "scripts/restart_api_gateway_service.sh",
        "scripts/restart_calibration_service.sh"
    )

    $remoteScript = @'
#!/usr/bin/env bash
set -euo pipefail

workspace="$1"
stage_path="$2"
manifest_path="$3"
expected_camera_version="$4"
expected_record_version="$5"
expected_robot_version="$6"
expected_calibration_version="$7"
expected_gateway_version="$8"
legacy_camera_units=(
  "camera_pipeline_service.service"
  "orin-camera-pipeline.service"
)
camera_unit="camera-pipeline.service"
record_unit="record-replay.service"
robot_unit="robot-control.service"
calibration_unit="calibration.service"
gateway_unit="api-gateway.service"
preserve_root="${stage_path}/preserve"
deploy_log_since="$(date '+%Y-%m-%d %H:%M:%S')"

print_deploy_logs() {
  journalctl -u "${camera_unit}" -u "${record_unit}" -u "${robot_unit}" -u "${calibration_unit}" -u "${gateway_unit}" \
    --since "${deploy_log_since}" --no-pager -o short-precise || true
}

stop_service() {
  local unit="$1"
  local timeout_seconds="$2"
  local deadline=$((SECONDS + timeout_seconds))
  systemctl stop --no-block "${unit}" || true
  while ((SECONDS < deadline)); do
    active_state="$(systemctl show "${unit}" -p ActiveState --value)"
    if [[ "${active_state}" == "inactive" || "${active_state}" == "failed" ]]; then
      systemctl reset-failed "${unit}" || true
      return 0
    fi
    sleep 1
  done
  echo "[deploy] ${unit} did not stop within ${timeout_seconds}s; killing its cgroup"
  systemctl kill --kill-who=all --signal=SIGKILL "${unit}" || true
  systemctl reset-failed "${unit}" || true
}

read_camera_status() {
  (
    cd "${workspace}"
    timeout 2s /home/wuji-brain/miniconda3/envs/wuji/bin/python -c \
      'from camera_pipeline.client import CameraName, CameraPipelineClient; client = CameraPipelineClient("tcp://127.0.0.1:6200", timeout_ms=1000); status = client.get_camera_status(CameraName.LEFT_ARM, timeout_s=0.5); client.close(); print(status.service_version); raise SystemExit(0 if status.online else 1)'
  )
}

json_field() {
  local field="$1"
  /home/wuji-brain/miniconda3/envs/wuji/bin/python -c \
    'import json, sys; print(json.load(sys.stdin)[sys.argv[1]])' "${field}"
}

case "${stage_path}" in
  "${workspace}/.deploy_stage/"*) ;;
  *)
    echo "[deploy] invalid stage path: ${stage_path}"
    exit 1
    ;;
esac
cleanup() {
  rm -rf -- "${stage_path}"
  rm -f -- "/tmp/dingtai_sync_and_restart.sh"
}
trap cleanup EXIT

for exempt_dir in prior_data records; do
  if [[ -e "${stage_path}/record_replay/${exempt_dir}" ]]; then
    echo "[deploy] staged RecordReplay package must not contain exempt directory: ${exempt_dir}"
    exit 1
  fi
done

expected_count="$(wc -l < "${manifest_path}")"
actual_stage_count="$(
  find "${stage_path}/camera_pipeline" "${stage_path}/record_replay" "${stage_path}/calibration_service" "${stage_path}/robot_control" "${stage_path}/api_gateway" \
    "${stage_path}/scripts/restart_camera_pipeline_service.sh" \
    "${stage_path}/scripts/restart_record_replay_service.sh" \
    "${stage_path}/scripts/restart_robot_control_service.sh" \
    "${stage_path}/scripts/restart_api_gateway_service.sh" \
    "${stage_path}/scripts/restart_calibration_service.sh" \
    -type f ! -name '*.pyc' ! -name '*.log' ! -path '*/__pycache__/*' |
    wc -l
)"
if [[ "${actual_stage_count}" -ne "${expected_count}" ]]; then
  echo "[deploy] staged file count mismatch expected=${expected_count} actual=${actual_stage_count}"
  exit 1
fi
(
  cd "${stage_path}"
  sha256sum --check "${manifest_path}"
)
camera_import_output=""
if ! camera_import_output="$(
  cd "${stage_path}"
  export PYTHONPATH="${stage_path}"
  "/home/wuji-brain/miniconda3/envs/wuji/bin/python" -c 'import camera_pipeline.service.__main__'
)"; then
  echo "[deploy] refused: staged CameraPipeline import preflight failed"
  printf '%s\n' "${camera_import_output}"
  exit 1
fi
service_state="not_running"
if ss -ltn '( sport = :6300 )' | grep -q LISTEN; then
  status_payload="$(curl -fsS --max-time 5 http://127.0.0.1:6300/status)"
  service_state="$(
    printf '%s' "${status_payload}" |
      /home/wuji-brain/miniconda3/envs/wuji/bin/python \
        -c 'import json, sys; print(json.load(sys.stdin)["state"])'
  )"
  # RecordReplay 1.11.x 使用 waiting 表示空闲等待；1.12.x 统一为 idle。
  # 两者都表示可以安全进入部署停止阶段，busy 和未知状态仍必须拒绝。
  if [[ "${service_state}" != "idle" && "${service_state}" != "waiting" ]]; then
    echo "[deploy] refused: RecordReplay current state is ${service_state}"
    exit 1
  fi
fi

for tls_path in \
  "/etc/dingtai/api-gateway/tls/casiahand-root-ca.crt.pem" \
  "/etc/dingtai/api-gateway/tls/api-gateway.fullchain.pem" \
  "/etc/dingtai/api-gateway/tls/api-gateway.key.pem"; do
  [[ -r "${tls_path}" ]] || { echo "[deploy] missing or unreadable API Gateway TLS file: ${tls_path}"; exit 1; }
done

echo "[deploy] stopping RecordReplay before replacing service files"
stop_service "${record_unit}" 10
echo "[deploy] stopping RobotControl before replacing service files"
stop_service "${robot_unit}" 10
echo "[deploy] stopping Calibration Service before replacing service files"
stop_service "${calibration_unit}" 10
echo "[deploy] stopping API Gateway before replacing service files"
stop_service "${gateway_unit}" 10

echo "[deploy] disabling conflicting legacy CameraPipeline units"
for legacy_camera_unit in "${legacy_camera_units[@]}"; do
  if [[ -e "/etc/systemd/system/${legacy_camera_unit}" ]]; then
    systemctl disable --now "${legacy_camera_unit}" || true
  fi
done
stop_service "${camera_unit}" 15

for legacy_camera_unit in "${legacy_camera_units[@]}"; do
  legacy_unit_path="/etc/systemd/system/${legacy_camera_unit}"
  if [[ -e "${legacy_unit_path}" ]]; then
    rm -f -- "${legacy_unit_path}"
  fi
done
if [[ -e "${stage_path}/record_replay/.archive" ]]; then
  echo "[deploy] staged RecordReplay package must not contain runtime .archive"
  exit 1
fi
mkdir -p "${preserve_root}"
for preserved_name in prior_data records runtime_state.json; do
  if [[ -e "${workspace}/record_replay/${preserved_name}" ]]; then
    mv "${workspace}/record_replay/${preserved_name}" "${preserve_root}/${preserved_name}"
  fi
done
for package_name in camera_pipeline record_replay calibration_service robot_control api_gateway; do
  if [[ -d "${workspace}/${package_name}" ]]; then
    rm -rf -- "${workspace:?}/${package_name}"
  fi
done
for restart_script_name in \
  "restart_camera_pipeline_service.sh" \
  "restart_record_replay_service.sh" \
  "restart_robot_control_service.sh" \
  "restart_api_gateway_service.sh" \
  "restart_calibration_service.sh"; do
  if [[ -f "${workspace}/scripts/${restart_script_name}" ]]; then
    rm -f -- "${workspace}/scripts/${restart_script_name}"
  fi
done
mv "${stage_path}/camera_pipeline" "${workspace}/camera_pipeline"
mv "${stage_path}/record_replay" "${workspace}/record_replay"
for exempt_dir in prior_data records; do
  preserved_dir="${preserve_root}/${exempt_dir}"
  target_dir="${workspace}/record_replay/${exempt_dir}"
  if [[ -e "${preserved_dir}" ]]; then
    mv "${preserved_dir}" "${target_dir}"
    echo "[deploy] preserved remote RecordReplay directory: ${exempt_dir}"
  else
    mkdir -p "${target_dir}"
    echo "[deploy] created missing remote RecordReplay directory: ${exempt_dir}"
  fi
done
runtime_state_path="${workspace}/record_replay/runtime_state.json"
preserved_runtime_state_path="${preserve_root}/runtime_state.json"
case "${service_state}" in
  idle|waiting)
    printf '{\n  "state": "idle"\n}\n' > "${runtime_state_path}"
    echo "[deploy] restored pre-deploy RecordReplay state: idle"
    ;;
  not_running)
    if [[ -f "${preserved_runtime_state_path}" ]]; then
      mv "${preserved_runtime_state_path}" "${runtime_state_path}"
      echo "[deploy] preserved existing RecordReplay runtime_state.json"
    fi
    ;;
  *)
    echo "[deploy] refused unexpected captured RecordReplay state: ${service_state}"
    exit 1
    ;;
esac
mv "${stage_path}/calibration_service" "${workspace}/calibration_service"
mv "${stage_path}/robot_control" "${workspace}/robot_control"
mv "${stage_path}/api_gateway" "${workspace}/api_gateway"
mkdir -p "${workspace}/scripts"
mv "${stage_path}/scripts/restart_camera_pipeline_service.sh" \
  "${workspace}/scripts/restart_camera_pipeline_service.sh"
mv "${stage_path}/scripts/restart_record_replay_service.sh" \
  "${workspace}/scripts/restart_record_replay_service.sh"
mv "${stage_path}/scripts/restart_robot_control_service.sh" \
  "${workspace}/scripts/restart_robot_control_service.sh"
mv "${stage_path}/scripts/restart_api_gateway_service.sh" \
  "${workspace}/scripts/restart_api_gateway_service.sh"
mv "${stage_path}/scripts/restart_calibration_service.sh" \
  "${workspace}/scripts/restart_calibration_service.sh"
chmod 0755 \
  "${workspace}/scripts/restart_camera_pipeline_service.sh" \
  "${workspace}/scripts/restart_record_replay_service.sh" \
  "${workspace}/scripts/restart_robot_control_service.sh" \
  "${workspace}/scripts/restart_api_gateway_service.sh" \
  "${workspace}/scripts/restart_calibration_service.sh"

install -m 0644 \
  "${workspace}/camera_pipeline/service/camera-pipeline.service" \
  "/etc/systemd/system/${camera_unit}"
install -m 0644 \
  "${workspace}/record_replay/service/record-replay.service" \
  "/etc/systemd/system/${record_unit}"
install -m 0644 \
  "${workspace}/robot_control/service/robot-control.service" \
  "/etc/systemd/system/${robot_unit}"
install -m 0644 \
  "${workspace}/calibration_service/service/calibration.service" \
  "/etc/systemd/system/${calibration_unit}"
install -m 0644 \
  "${workspace}/api_gateway/service/api-gateway.service" \
  "/etc/systemd/system/${gateway_unit}"
if [[ -f "${workspace}/api_gateway/requirements.txt" ]]; then
  "/home/wuji-brain/miniconda3/envs/wuji/bin/python" -m pip install --disable-pip-version-check --no-input -r "${workspace}/api_gateway/requirements.txt"
else
  echo "[deploy] api_gateway/requirements.txt not present; skip API Gateway dependency installation"
fi
systemctl daemon-reload
systemctl enable "${camera_unit}" "${record_unit}" "${robot_unit}" "${calibration_unit}" "${gateway_unit}"

(
  cd "${workspace}"
  sha256sum --check "${manifest_path}"
)

echo "[deploy] starting CameraPipeline"
systemctl restart --no-block "${camera_unit}"
camera_ready=false
camera_deadline=$((SECONDS + 20))
while ((SECONDS < camera_deadline)); do
  if systemctl is-active --quiet "${camera_unit}" &&
     ss -ltn '( sport = :6200 )' | grep -q LISTEN; then
    if camera_status="$(read_camera_status 2>/dev/null)"; then
      camera_ready=true
      echo "[deploy] CameraPipeline ready: ${camera_status}"
      break
    fi
  fi
  sleep 1
done
if [[ "${camera_ready}" != "true" ]]; then
  echo "[deploy] CameraPipeline was not business-ready within 20s"
  systemctl status "${camera_unit}" --no-pager -l || true
  print_deploy_logs
  exit 1
fi
if [[ "${camera_status}" != "${expected_camera_version}" ]]; then
  echo "[deploy] CameraPipeline version mismatch expected=${expected_camera_version} actual=${camera_status}"
  exit 1
fi
echo "[deploy] CameraPipeline ready; version=${camera_status}"

echo "[deploy] starting RecordReplay"
systemctl restart --no-block "${record_unit}"
record_ready=false
record_deadline=$((SECONDS + 10))
while ((SECONDS < record_deadline)); do
  if systemctl is-active --quiet "${record_unit}" &&
     ss -ltn '( sport = :6300 )' | grep -q LISTEN; then
    record_ready=true
    break
  fi
  sleep 1
done
if [[ "${record_ready}" != "true" ]]; then
  echo "[deploy] RecordReplay was not HTTP-ready within 10s"
  systemctl status "${record_unit}" --no-pager -l || true
  print_deploy_logs
  exit 1
fi
curl -fsS --max-time 5 http://127.0.0.1:6300/status
echo
record_version="$(cd "${workspace}" && /home/wuji-brain/miniconda3/envs/wuji/bin/python -c 'from record_replay import RECORD_REPLAY_VERSION; print(RECORD_REPLAY_VERSION)')"
if [[ "${record_version}" != "${expected_record_version}" ]]; then
  echo "[deploy] RecordReplay version mismatch expected=${expected_record_version} actual=${record_version}"
  exit 1
fi
echo "[deploy] RecordReplay ready; version=${record_version}"
echo "[deploy] starting RobotControl"
systemctl restart --no-block "${robot_unit}"
robot_ready=false
robot_deadline=$((SECONDS + 20))
while ((SECONDS < robot_deadline)); do
  if systemctl is-active --quiet "${robot_unit}" &&
     ss -ltn '( sport = :6500 )' | grep -q LISTEN &&
     curl -fsS --max-time 5 http://127.0.0.1:6500/api/v1/health; then
    robot_ready=true
    echo
    break
  fi
  sleep 1
done
if [[ "${robot_ready}" != "true" ]]; then
  echo "[deploy] RobotControl was not HTTP-ready within 20s"
  systemctl status "${robot_unit}" --no-pager -l || true
  print_deploy_logs
  exit 1
fi
robot_payload="$(curl -fsS --max-time 5 http://127.0.0.1:6500/api/v1/health)"
robot_version="$(printf '%s' "${robot_payload}" | json_field service_version)"
if [[ "${robot_version}" != "${expected_robot_version}" ]]; then
  echo "[deploy] RobotControl version mismatch expected=${expected_robot_version} actual=${robot_version}"
  exit 1
fi
echo "[deploy] RobotControl ready; version=${robot_version}"
echo "[deploy] starting Calibration Service"
systemctl restart --no-block "${calibration_unit}"
calibration_ready=false
calibration_deadline=$((SECONDS + 10))
while ((SECONDS < calibration_deadline)); do
  if systemctl is-active --quiet "${calibration_unit}" &&
     ss -ltn '( sport = :6600 )' | grep -q LISTEN &&
     calibration_payload="$(curl -fsS --max-time 5 http://127.0.0.1:6600/api/v1/status)"; then
    calibration_ready=true
    printf '%s\n' "${calibration_payload}"
    break
  fi
  sleep 1
done
if [[ "${calibration_ready}" != "true" ]]; then
  echo "[deploy] Calibration Service was not HTTP-ready within 10s"
  systemctl status "${calibration_unit}" --no-pager -l || true
  print_deploy_logs
  exit 1
fi
calibration_version="$(cd "${workspace}" && /home/wuji-brain/miniconda3/envs/wuji/bin/python -c 'from calibration_service import CALIBRATION_SERVICE_VERSION; print(CALIBRATION_SERVICE_VERSION)')"
if [[ "${calibration_version}" != "${expected_calibration_version}" ]]; then
  echo "[deploy] Calibration Service version mismatch expected=${expected_calibration_version} actual=${calibration_version}"
  exit 1
fi
echo "[deploy] Calibration Service ready; version=${calibration_version}"
echo "[deploy] starting API Gateway"
systemctl restart --no-block "${gateway_unit}"
gateway_ready=false
gateway_deadline=$((SECONDS + 10))
gateway_hostname="$(hostname)"
while ((SECONDS < gateway_deadline)); do
  if systemctl is-active --quiet "${gateway_unit}" &&
     ss -ltn '( sport = :443 )' | grep -q LISTEN &&
     curl -fsS --max-time 5 --cacert /etc/dingtai/api-gateway/tls/casiahand-root-ca.crt.pem --resolve "${gateway_hostname}:443:127.0.0.1" "https://${gateway_hostname}/api/v1/gateway/health"; then
    gateway_ready=true
    echo
    break
  fi
  sleep 1
done
if [[ "${gateway_ready}" != "true" ]]; then
  echo "[deploy] API Gateway was not HTTPS-ready within 10s"
  systemctl status "${gateway_unit}" --no-pager -l || true
  print_deploy_logs
  exit 1
fi
gateway_payload="$(curl -fsS --max-time 5 --cacert /etc/dingtai/api-gateway/tls/casiahand-root-ca.crt.pem --resolve "${gateway_hostname}:443:127.0.0.1" "https://${gateway_hostname}/api/v1/gateway/health")"
gateway_version="$(printf '%s' "${gateway_payload}" | json_field gateway_version)"
if [[ "${gateway_version}" != "${expected_gateway_version}" ]]; then
  echo "[deploy] API Gateway version mismatch expected=${expected_gateway_version} actual=${gateway_version}"
  exit 1
fi
echo "[deploy] API Gateway ready; version=${gateway_version}"
print_deploy_logs
echo "[deploy] five services updated, restarted and version-verified: CameraPipeline=${camera_status} RecordReplay=${record_version} RobotControl=${robot_version} Calibration=${calibration_version} ApiGateway=${gateway_version}"
'@
    [System.IO.File]::WriteAllText(
        $remoteScriptPath,
        ($remoteScript.Replace([Environment]::NewLine, [string][char]10) + [char]10),
        [System.Text.UTF8Encoding]::new($false)
    )

    Write-Host "即将同步 $($deployFiles.Count) 个文件到 $SshTarget"
    Write-Host "远端旧版本由 Git 管理，部署时不生成 .archive 快照"
    Write-Warning "本次会重启 CameraPipeline、RecordReplay、RobotControl 和 API Gateway，但不会发送 RecordReplay /start 请求。缺少先验只会在人工调用 /start 时报告。"

    Invoke-CheckedCommand -FilePath "ssh.exe" -ArgumentList @(
        $SshOptions + @(
            $SshTarget,
            "mkdir -p '$remoteStagePath'"
        )
    )
    Invoke-CheckedCommand -FilePath "scp.exe" -ArgumentList @(
        $SshOptions + @(
            $archivePath,
            "${SshTarget}:$remoteArchivePath"
        )
    )
    Invoke-CheckedCommand -FilePath "scp.exe" -ArgumentList @(
        $SshOptions + @(
            $manifestPath,
            "${SshTarget}:$remoteManifestPath"
        )
    )
    Invoke-CheckedCommand -FilePath "scp.exe" -ArgumentList @(
        $SshOptions + @(
            $remoteScriptPath,
            "${SshTarget}:$RemoteDeployScript"
        )
    )
    Invoke-CheckedCommand -FilePath "ssh.exe" -ArgumentList @(
        $SshOptions + @(
            $SshTarget,
            "tar -xf '$remoteArchivePath' -C '$remoteStagePath'"
        )
    )

    $remoteCommand = "sudo -S -p '' bash '$RemoteDeployScript' " +
        "'$RemoteWorkspace' '$remoteStagePath' '$remoteManifestPath' " +
        "'$expectedCameraPipelineVersion' '$expectedRecordReplayVersion' " +
        "'$expectedRobotControlVersion' '$expectedCalibrationServiceVersion' '$expectedApiGatewayVersion'"
    $RemoteSudoPassword | & ssh.exe @SshOptions $SshTarget $remoteCommand
    if ($LASTEXITCODE -ne 0) {
        throw "远端同步或重启失败，请检查上方日志。"
    }
}
finally {
    if (Test-Path -LiteralPath $localTempRoot) {
        $resolvedTempRoot = [System.IO.Path]::GetFullPath($localTempRoot)
        if (
            -not $resolvedTempRoot.StartsWith(
                $localTempBase,
                [System.StringComparison]::OrdinalIgnoreCase
            ) -or
            [System.IO.Path]::GetFileName($resolvedTempRoot) -notlike "dingtai-deploy-*"
        ) {
            throw "拒绝清理非预期临时目录：$resolvedTempRoot"
        }
        Remove-Item -LiteralPath $localTempRoot -Recurse -Force
    }
}
