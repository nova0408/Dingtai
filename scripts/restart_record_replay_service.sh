#!/usr/bin/env bash
# Linux service restart helper; keep this file LF-encoded for remote execution.
set -euo pipefail

SERVICE_UNIT="record-replay.service"
SERVICE_PORT="6300"
SERVICE_URL="http://127.0.0.1:${SERVICE_PORT}"
SERVICE_PYTHON="/home/wuji-brain/miniconda3/envs/wuji/bin/python"
WORKSPACE_DIR="/home/wuji-brain/workspace"
LEFT_RECORD_DIR="${WORKSPACE_DIR}/record_replay/records/left"
RIGHT_RECORD_DIR="${WORKSPACE_DIR}/record_replay/records/right"
WAIT_TIMEOUT_SECONDS=10
restart_log_since="$(date '+%Y-%m-%d %H:%M:%S')"
non_interactive=false

if [[ "${1:-}" == "--non-interactive" ]]; then
  non_interactive=true
  shift
fi
if (($# != 0)); then
  echo "[restart] unsupported arguments: $*"
  exit 1
fi

print_restart_logs() {
  journalctl -u "${SERVICE_UNIT}" \
    --since "${restart_log_since}" --no-pager -o short-precise || true
}

if [[ "${non_interactive}" != "true" ]]; then
  if [[ ! -t 0 ]]; then
    echo "[restart] refused: interactive terminal required"
    exit 1
  fi
  read -r -p "我已确认现场安全并同意重启RecordReplay服务 [Y/n] " confirmation
  case "${confirmation:-Y}" in
    Y | y) ;;
    *)
      echo "[restart] cancelled"
      exit 1
      ;;
  esac
fi

if ! compgen -G "${LEFT_RECORD_DIR}/*.csv" >/dev/null; then
  echo "[restart] refused: no left-arm CSV in ${LEFT_RECORD_DIR}"
  exit 1
fi
if ! compgen -G "${RIGHT_RECORD_DIR}/*.csv" >/dev/null; then
  echo "[restart] refused: no right-arm CSV in ${RIGHT_RECORD_DIR}"
  exit 1
fi

if ss -ltn "( sport = :${SERVICE_PORT} )" | grep -q LISTEN; then
  if ! systemctl is-active --quiet "${SERVICE_UNIT}"; then
    echo "[restart] refused: port ${SERVICE_PORT} is not owned by active ${SERVICE_UNIT}"
    exit 1
  fi
  status_payload="$(curl -fsS --max-time 5 "${SERVICE_URL}/status")" || {
    echo "[restart] refused: running service status is unavailable"
    exit 1
  }
  service_state="$(
    printf '%s' "${status_payload}" |
      "${SERVICE_PYTHON}" -c 'import json, sys; print(json.load(sys.stdin)["state"])'
  )" || {
    echo "[restart] refused: running service returned invalid status"
    exit 1
  }
  if [[ "${service_state}" != "idle" ]]; then
    echo "[restart] refused: current service state is ${service_state}"
    exit 1
  fi
fi

echo "[restart] restarting ${SERVICE_UNIT} through systemd"
sudo systemctl restart "${SERVICE_UNIT}"

deadline=$((SECONDS + WAIT_TIMEOUT_SECONDS))
while ((SECONDS < deadline)); do
  if systemctl is-active --quiet "${SERVICE_UNIT}" &&
     ss -ltn "( sport = :${SERVICE_PORT} )" | grep -q LISTEN; then
    curl -fsS --max-time 5 "${SERVICE_URL}/status"
    echo
    echo "[restart] API service restarted; no replay start request was sent"
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
