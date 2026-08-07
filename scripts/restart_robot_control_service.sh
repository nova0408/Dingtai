#!/usr/bin/env bash
# Linux service restart helper; keep this file LF-encoded for remote execution.
set -euo pipefail

SERVICE_UNIT="robot-control.service"
SERVICE_PORT="6500"
SERVICE_URL="http://127.0.0.1:${SERVICE_PORT}/api/v1/health"
WAIT_TIMEOUT_SECONDS=20
restart_log_since="$(date '+%Y-%m-%d %H:%M:%S')"

print_restart_logs() {
  journalctl -u "${SERVICE_UNIT}" \
    --since "${restart_log_since}" --no-pager -o short-precise || true
}

echo "[restart] restarting ${SERVICE_UNIT} through systemd"
sudo systemctl restart "${SERVICE_UNIT}"

deadline=$((SECONDS + WAIT_TIMEOUT_SECONDS))
while ((SECONDS < deadline)); do
  if systemctl is-active --quiet "${SERVICE_UNIT}" &&
     ss -ltn "( sport = :${SERVICE_PORT} )" | grep -q LISTEN &&
     curl -fsS --max-time 5 "${SERVICE_URL}"; then
    echo
    echo "[restart] ${SERVICE_UNIT} is ready"
    systemctl show "${SERVICE_UNIT}" \
      -p ActiveState -p SubState -p MainPID \
      -p TimeoutStartUSec -p TimeoutStopUSec --no-pager
    print_restart_logs
    exit 0
  fi
  sleep 1
done

echo "[restart] ${SERVICE_UNIT} was not ready within ${WAIT_TIMEOUT_SECONDS} seconds"
systemctl status "${SERVICE_UNIT}" --no-pager -l || true
print_restart_logs
exit 1
