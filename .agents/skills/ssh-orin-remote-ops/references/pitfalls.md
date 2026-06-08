# Orin SSH Pitfalls

## Windows PowerShell -> SSH quoting

Symptoms:

- Remote `python -c` turns into broken shell fragments.
- PowerShell says `import` is not recognized.
- Linux receives a different command than intended.

Root cause:

- Nested quoting across PowerShell, SSH, remote shell, and Python is fragile.

Preferred fix:

- Avoid deep nesting.
- Upload a temp script and run it remotely.

Safer pattern:

```powershell
$scriptPath = Join-Path $env:TEMP 'remote_task.sh'
@'
#!/usr/bin/env bash
set -e
source /home/wuji-brain/miniconda3/etc/profile.d/conda.sh
conda activate py38_tourch
cd /home/wuji-brain/workspace
python -m orin.tray_detection.smoke_test --service-addr tcp://127.0.0.1:6210
'@ | Set-Content -Path $scriptPath -Encoding utf8
scp $scriptPath orin:/tmp/remote_task.sh
ssh orin "bash /tmp/remote_task.sh"
```

## Here-doc mangling

Symptoms:

- Output partly succeeds, then ends with `NameError: name 'PY' is not defined`.
- Bash warns `here-document ... wanted 'PY'`.

Root cause:

- The `python - <<'PY' ... PY` block was corrupted before it reached the remote shell.

Preferred fix:

- Do not rely on inline `here-doc` from Windows PowerShell for important remote work.
- Use uploaded scripts or write the remote file with remote Python first.

## CRLF breaks remote shell scripts

Symptoms:

- Remote shell prints errors like `$'true\r': command not found`.
- A script copied from Windows behaves oddly on Linux.

Root cause:

- The remote file contains CRLF line endings.

Preferred fix:

- Write the file on Orin using remote Python with `newline='\n'`, or normalize after upload.

Reliable remote write pattern:

```powershell
ssh orin "python - <<'PY'
script = '''#!/usr/bin/env bash
set -e
echo ok
'''
with open('/tmp/example.sh', 'w', encoding='utf-8', newline='\n') as f:
    f.write(script)
print('written')
PY"
ssh orin "chmod +x /tmp/example.sh && bash /tmp/example.sh"
```

If `here-doc` itself is suspect, upload the file with `scp` and run `dos2unix` or `sed -i 's/\r$//'`.

## Background service does not stay alive

Symptoms:

- The service starts in foreground but disappears when started "in background".
- Log file is empty or the process dies after SSH exits.

Root cause:

- The process was not fully detached from the SSH session.
- Startup command mixed too many layers and failed before exec.

Preferred fix:

1. Prove the service starts in foreground.
2. Then detach with `setsid` and stdio redirection.

Pattern:

```bash
source /home/wuji-brain/miniconda3/etc/profile.d/conda.sh
conda activate py38_tourch
cd /home/wuji-brain/workspace
setsid python -u -m orin.tray_detection.service >/tmp/orin_tray_service.log 2>&1 < /dev/null &
```

Then verify:

```bash
ps -ef | grep orin.tray_detection.service | grep -v grep
tail -n 20 /tmp/orin_tray_service.log
```

## Foreground timeout closes ZMQ sockets noisily

Symptoms:

- `timeout 20s python -m ...` shows normal startup, then exits with `Socket operation on non-socket`.

Root cause:

- The timeout kills the process while the service loop is blocked in `recv`.

Interpretation:

- Treat this as acceptable for a startup smoke test if logs show the service reached the intended started state.
- Do not mistake it for the original startup failure.

## pkill pattern surprises

Symptoms:

- `pkill` reports invalid options or does not match the intended process.

Root cause:

- Shell quoting changed the pattern before Linux parsed it.

Preferred fix:

- Keep the pattern simple.
- Use `pgrep -af ...` first to inspect exact command lines.

Examples:

```bash
pgrep -af orin.tray_detection.service
pkill -f orin.tray_detection.service || true
```

## Remote network assumptions

Symptoms:

- Orin cannot download models directly.
- A configured proxy such as `127.0.0.1:4444` is unreachable.

Preferred fix:

1. Check direct network with `curl -I https://huggingface.co`.
2. Check proxy explicitly with `curl -I -x http://127.0.0.1:4444 https://huggingface.co`.
3. If both fail, download on the local machine and sync the cache to Orin.

## Large file sync

For many files, prefer tar streaming:

```powershell
tar -cf - -C .\.cache\hf_downloads\grounding-dino-base . | ssh orin "tar -xf - -C /home/wuji-brain/workspace/.cache/huggingface/project_store/grounding_dino_model/IDEA-Research__grounding-dino-base"
```

Use `scp` for single files, `tar | ssh tar` for directory trees and model caches.

## Debug order

When a remote workflow fails, debug in this order:

1. Can Orin import the module?
2. Can the remote script run in foreground?
3. Can the hardware or model dependency initialize?
4. Can the service bind the port?
5. Can the service stay detached?
6. Can a smoke test request succeed?
