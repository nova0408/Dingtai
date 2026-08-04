---
name: ssh-orin-remote-ops
description: Reliable workflow for working with the `orin` host from Windows PowerShell in the Dingtai project. Use when Codex needs to deploy files to `/home/wuji-brain/workspace`, restart CameraPipeline or RecordReplay through project scripts, run remote Python or bash commands over SSH, create temporary remote scripts, manage long-running Orin services, diagnose quoting or here-doc failures, handle CRLF/LF issues, or avoid PowerShell-to-SSH command mangling.
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

Deploy CameraPipeline, RecordReplay, RobotControl and the API Gateway as one version-aligned unit:

```powershell
pwsh -NoProfile -File .\scripts\sync_and_restart_services.ps1
```

RecordReplay service startup no longer depends on `ball_debug_overlay.jpg` or any other prior file.
Missing or invalid runtime priors are reported only when现场人员 explicitly calls `POST /start`;
the synchronization script must not add an overlay-artifact precondition. `-CameraPipelineOnly` is
valid only together with `-RestartOnly` and is rejected for deployment.

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
