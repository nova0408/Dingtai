---
name: ssh-orin-remote-ops
description: Reliable workflow for working with the `orin` host from Windows PowerShell in the Dingtai project. Use when Codex needs to deploy files to `/home/wuji-brain/workspace`, synchronize or restart CameraPipeline, RecordReplay, RobotControl, Calibration Service or API Gateway through project scripts, run remote Python or bash commands over SSH, create temporary remote scripts, manage long-running Orin services, diagnose quoting or here-doc failures, handle CRLF/LF issues, or avoid PowerShell-to-SSH command mangling.
---

# SSH Orin Remote Ops

## Use This Workflow

Prefer this skill when the task includes any of these patterns:

- Running `ssh orin "..."` from Windows PowerShell.
- Starting or stopping remote Python services on Orin.
- Sending multi-line bash or Python snippets to Orin.
- Copying local files or model caches to `/home/wuji-brain/workspace`.
- Investigating failures that look like quoting, CRLF, `here-doc`, `pkill`, or detached-process problems.

## Core Rules

- Prefer short single-purpose SSH calls over one giant compound command.
- When changing Orin-side project code that also exists in the local workspace, always update the local source of truth first, then upload or sync the exact result to Orin. Do not edit only the remote copy when the same module is maintained locally.
- When running Python scripts, SDK examples, or Python-based diagnostics on `orin`, use the `wuji` Conda environment by default instead of the system Python.
- Prefer `conda run -n wuji ...` for one-shot remote Python commands so the environment choice stays explicit and does not depend on interactive shell initialization.
- Prefer uploading a temporary script to Orin for multi-line logic instead of nesting `bash`, `python`, quotes, and `here-doc` inside one Windows command.
- Prefer writing remote scripts with LF newlines from Linux or via remote Python, not via Windows-created CRLF shell files.
- When validating remote startup, test in the foreground first; only then detach to the background.
- Separate "service can start" from "service stays detached"; validate both.
- When the remote job needs long model loading or hardware warmup, use generous timeouts and stage the checks.
- Prefer the repository's Windows service-control entrypoint over hand-written `sudo`,
  `systemctl`, or nested SSH commands. It already supplies the configured sudo password and
  runs service-specific readiness checks.
- On Orin, service-to-service requests and local read-only diagnostics must use each service's
  `localhost`/`127.0.0.1` internal port directly. Do not route these requests through the API
  Gateway. The API Gateway is only the external client entry point; its HTTPS 443 endpoint and
  prefixed URLs are not the default path for Orin-local access.

## Project Service Control

Run these commands from the Dingtai repository root with PowerShell 7.

Restart only CameraPipeline without syncing files or touching RecordReplay:

```powershell
pwsh -NoProfile -File .\scripts\sync_and_restart_services.ps1 -RestartOnly -CameraPipelineOnly
```

Restart all deployed services without syncing files:

```powershell
pwsh -NoProfile -File .\scripts\sync_and_restart_services.ps1 -RestartOnly
```

Restart only RobotControl without syncing files:

```powershell
pwsh -NoProfile -File .\scripts\sync_and_restart_services.ps1 -RestartOnly -RobotControlOnly
```

Restart only the unified API Gateway without syncing files:

```powershell
pwsh -NoProfile -File .\scripts\sync_and_restart_services.ps1 -RestartOnly -ApiGatewayOnly
```

Restart only Calibration Service and then the dependent API Gateway without syncing files:

```powershell
pwsh -NoProfile -File .\scripts\sync_and_restart_services.ps1 -RestartOnly -CalibrationServiceOnly
```

Calibration Service uses the Orin-local `http://127.0.0.1:6600/api/v1/status` read-only check.
Restarting it does not capture a frame, calculate a prior, start calibration, control a device,
or start RecordReplay.

Use the all-service command only when the user explicitly authorizes restarting RecordReplay.
Restarting a service does not authorize sending RecordReplay `/start` or running a replay test.

Deploy and restart only RobotControl:

```powershell
pwsh -NoProfile -File .\scripts\sync_and_restart_services.ps1 -RobotControlOnly
```

The RobotControl-only deployment validates the staged SHA-256 manifest, backs up the previous
remote copy, installs `robot-control.service`, and checks only `GET /api/v1/health`; it does not
call `/api/v1/status` or any control POST.

The Orin-local RobotControl health check is intentionally direct, for example
`http://127.0.0.1:6500/api/v1/health`. The Gateway prefix
`/api/v1/robot-control/*` is reserved for external clients.

Deploy and restart only the unified API Gateway:

```powershell
pwsh -NoProfile -File .\scripts\sync_and_restart_services.ps1 -ApiGatewayOnly
```

This installs `api_gateway/requirements.txt` into the Orin `wuji` environment, validates the
staged SHA-256 manifest, and checks only `GET /api/v1/gateway/health`. Gateway unification changes
the client entry and URL prefixes; it does not merge processes or remove the CameraPipeline,
RecordReplay, or RobotControl internal ports.

Gateway deployment health is the exception because it checks Gateway itself. Backend checks from
Orin remain direct: CameraPipeline HTTP/WebSocket, RecordReplay, and RobotControl use their own
`localhost` ports and are not probed through Gateway.

Deploy and restart only CameraPipeline:

```powershell
pwsh -NoProfile -File .\scripts\sync_and_restart_services.ps1 -CameraPipelineOnly
```

This synchronizes only the `camera_pipeline/` package, verifies its SHA-256 manifest, backs up the
remote package, and performs a read-only camera status/version check. Public CameraPipeline
protocol changes still require the documented dependent RecordReplay deployment.

Deploy and restart only Calibration Service:

```powershell
pwsh -NoProfile -File .\scripts\sync_and_restart_services.ps1 -CalibrationServiceOnly
```

This synchronizes only the `calibration_service/` package, verifies its SHA-256 manifest, backs up
the remote package, and checks `GET /api/v1/status` is idle with the expected version. It does not
restart API Gateway, capture frames, calculate priors, or control devices.

Deploy CameraPipeline, RecordReplay, RobotControl, Calibration Service and the API Gateway as one version-aligned unit when the change spans services:

```powershell
pwsh -NoProfile -File .\scripts\sync_and_restart_services.ps1
```

## Five-service version synchronization rule

The five deployed services have independent version sources and must not be reported as remotely
updated based only on local edits:

| Service | Local version source | Version-change sync command |
| --- | --- | --- |
| CameraPipeline | `camera_pipeline/service/protocol.py: SERVICE_VERSION` | `-CameraPipelineOnly` |
| RecordReplay | `record_replay/__init__.py: RECORD_REPLAY_VERSION` | `-RecordReplayOnly` |
| RobotControl | `robot_control/__init__.py: ROBOT_CONTROL_VERSION` | `-RobotControlOnly` |
| Calibration Service | `calibration_service/__init__.py: CALIBRATION_SERVICE_VERSION` | `-CalibrationServiceOnly` |
| API Gateway | `api_gateway/__init__.py: API_GATEWAY_VERSION` | `-ApiGatewayOnly` |

默认部署规则：只要任务修改了上述任一已部署服务的版本源或部署包，完成相关本地静态检查和契约
检查后，必须执行对应的官方 `scripts/sync_and_restart_services.ps1` 同步并重启；除非用户明确要求
“仅本地修改/不部署”，否则不能把代码修改交付为仅本地状态。单服务变更使用对应的 `*Only` 参数，
跨服务协议或依赖变更使用全量命令。不要改用手写 `scp`、`ssh systemctl` 或 `RestartOnly` 替代默认
同步部署。

同步部署前必须检查所有将上传并由远端 Bash 执行的 `.sh` 文件为 LF 换行；若发现 CRLF，先修复
本地部署源文件并重新执行本地检查，再调用官方 PowerShell 脚本，不能重复提交已知会被 Bash 拒绝的
产物。部署结果必须记录本地期望版本、远端实际版本、文件清单和 SHA-256 结果、远端备份、服务
就绪状态及服务只读版本/健康响应。若 Orin 不可达、同步/就绪/版本校验失败，必须报告为未部署，
不得根据本地文件推断远端状态。
For changes spanning multiple services, use the full five-service deployment. Single-service
CameraPipeline or Calibration Service changes use their corresponding `*Only` deployment. These operations never
authorize RecordReplay `/start`, replay tests, device-control POSTs, or calibration capture.

## Mandatory deployment disposition

Before ending any task that changed a deployed package or service version, report one of these
explicit states:

- `已部署`：the official synchronization command succeeded, with local expected version, remote
  actual version, file count/manifest result, backup, restart/readiness, and direct read-only
  response recorded.
- `待授权部署`：deployment was not authorized. Give the exact scoped command and do not imply the
  remote service contains the local change.
- `部署失败`：the command was attempted but sync, readiness, or version verification failed;
  preserve the failure output and report the remote version actually observed.

When the user reports a manual restart or synchronization, verify every affected service by its
Orin-local read-only endpoint. A manual restart alone proves process lifecycle only; it does not
prove that local files were copied. For a Calibration Service plus CameraPipeline change, check
both `127.0.0.1:6600/api/v1/status` and `127.0.0.1:6400/api/v1/health`, and compare both versions
with the local source. Do not restart again unless the user authorizes it.

RecordReplay service startup no longer depends on `ball_debug_overlay.jpg` or any other prior file.
Missing or invalid runtime priors are reported only when现场人员 explicitly calls `POST /start`;
the synchronization script must not add an overlay-artifact precondition.

Before deployment, verify RecordReplay is waiting and that restarting it is in scope. Treat script
success and its business-readiness output as the primary result; use short read-only status checks
only when additional diagnosis is necessary.

Treat CameraPipeline public client, API, or wire-protocol changes as an atomic two-service
deployment. Never update and restart CameraPipeline while leaving a running RecordReplay process
loaded with the previous CameraPipeline client. A completed deployment must restore both services
to business-ready state; stopping RecordReplay is only a temporary replacement step.

## Recommended Patterns

### 1. Run a simple remote command

Use:

```powershell
ssh orin "ls -la /home/wuji-brain/workspace"
```

Keep the payload simple. If quoting starts to nest, switch to a remote script.

If the command runs Python on `orin`, prefer:

```powershell
ssh orin "/home/wuji-brain/miniconda3/bin/conda run -n wuji python script.py"
```

This avoids accidentally using `/usr/bin/python` or a shell session that did not initialize Conda.

### 2. Run multi-line remote logic

Preferred pattern:

1. Create a local temp script.
2. `scp` it to `/tmp/...` on Orin.
3. Execute it with `ssh orin "bash /tmp/that_script.sh"` or, for Python, `ssh orin "/home/wuji-brain/miniconda3/bin/conda run -n wuji python /tmp/that_script.py"`.

For examples and known pitfalls, read [references/pitfalls.md](references/pitfalls.md).

### 3. Start a long-running service

Use this sequence:

1. Foreground boot test with `timeout`.
2. Verify logs and port binding.
3. Detach with `setsid ... < /dev/null >log 2>&1 &`.
4. Confirm with `ps` or `pgrep`.
5. Smoke test the service from Orin or locally.

### 4. Deploy files

Use:

```powershell
scp .\local\file.py orin:/home/wuji-brain/workspace/orin/module/file.py
```

For directories or caches, prefer `tar | ssh tar` when there are many files.

## Failure Triage

- If `ssh` output shows PowerShell parsing errors, reduce quoting complexity or switch to uploaded scripts.
- If a Python script works in one session but fails over SSH with missing modules such as `qmlinker`, first confirm you are running it through the `wuji` environment instead of the system Python.
- If remote bash reports weird `$'...\r'` commands, suspect CRLF line endings.
- If `python - <<'PY'` leaves a trailing `NameError: name 'PY' is not defined`, the `here-doc` was mangled; use a remote script instead.
- If a service works in foreground but not in background, verify detachment method and stdio redirection.
- If `pkill` behaves oddly, inspect how quoting changed the pattern that actually reached Linux.
- If RPC times out, confirm the service process still exists before debugging application logic.

## References

- Read [references/pitfalls.md](references/pitfalls.md) for concrete symptoms, root causes, and command templates from the Orin tray-detection migration work.
