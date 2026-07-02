---
name: ssh-orin-remote-ops
description: Reliable workflow for working with the `orin` host from Windows PowerShell in the Dingtai project. Use when Codex needs to deploy files to `/home/wuji-brain/workspace`, run remote Python or bash commands over SSH, create temporary remote scripts, manage long-running Orin services, diagnose quoting or here-doc failures, handle CRLF/LF issues, or avoid PowerShell-to-SSH command mangling.
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
