#!/usr/bin/env bash
set -euo pipefail

SERVICE_PYTHON="/home/wuji-brain/miniconda3/envs/wuji/bin/python"
SERVICE_PATTERN="${SERVICE_PYTHON} -m record_replay.service --host 0.0.0.0 --port 6300"
SERVICE_CMD=("${SERVICE_PYTHON}" -m record_replay.service --host 0.0.0.0 --port 6300)
SERVICE_URL="http://127.0.0.1:6300"
LOG_PATH="/tmp/record_replay_service.log"
WORKSPACE_DIR="/home/wuji-brain/workspace"
PRIOR_PATH="${WORKSPACE_DIR}/record_replay/prior_data/ball_pose_prior.json"
HAND_EYE_PATH="${WORKSPACE_DIR}/record_replay/prior_data/hand_eye_result.txt"
LEFT_RECORD_DIR="${WORKSPACE_DIR}/record_replay/records/left"
RIGHT_RECORD_DIR="${WORKSPACE_DIR}/record_replay/records/right"

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

if [[ ! -f "${PRIOR_PATH}" ]]; then
  echo "[restart] refused: missing ${PRIOR_PATH}"
  exit 1
fi
if [[ ! -f "${HAND_EYE_PATH}" ]]; then
  echo "[restart] refused: missing ${HAND_EYE_PATH}"
  exit 1
fi
if ! compgen -G "${LEFT_RECORD_DIR}/*.csv" >/dev/null; then
  echo "[restart] refused: no left-arm CSV in ${LEFT_RECORD_DIR}"
  exit 1
fi
if ! compgen -G "${RIGHT_RECORD_DIR}/*.csv" >/dev/null; then
  echo "[restart] refused: no right-arm CSV in ${RIGHT_RECORD_DIR}"
  exit 1
fi

cd "${WORKSPACE_DIR}"
service_pids="$(pgrep -f "${SERVICE_PATTERN}" || true)"
port_is_listening=false
if ss -ltn '( sport = :6300 )' | grep -q LISTEN; then
  port_is_listening=true
fi

if [[ "${port_is_listening}" == true ]]; then
  if [[ -z "${service_pids}" ]]; then
    echo "[restart] refused: port 6300 is occupied by another process"
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
  if [[ "${service_state}" != "waiting" ]]; then
    echo "[restart] refused: current service state is ${service_state}"
    exit 1
  fi
fi

if [[ -n "${service_pids}" ]]; then
  echo "[restart] stopping pid list:"
  echo "${service_pids}"
  kill ${service_pids}
  for _ in {1..30}; do
    if ! pgrep -f "${SERVICE_PATTERN}" >/dev/null; then
      break
    fi
    sleep 1
  done
  if pgrep -f "${SERVICE_PATTERN}" >/dev/null; then
    echo "[restart] refused: service did not stop cleanly; no SIGKILL was sent"
    exit 1
  fi
fi

if ss -ltn '( sport = :6300 )' | grep -q LISTEN; then
  echo "[restart] refused: port 6300 is occupied by another process"
  exit 1
fi

echo "[restart] starting RecordReplay API service"
setsid "${SERVICE_CMD[@]}" >"${LOG_PATH}" 2>&1 < /dev/null &
sleep 3

running_pids="$(pgrep -f "${SERVICE_PATTERN}" || true)"
if [[ -z "${running_pids}" ]]; then
  echo "[restart] failed: service process is not running"
  tail -n 30 "${LOG_PATH}" || true
  exit 1
fi
if ! ss -ltn '( sport = :6300 )' | grep -q LISTEN; then
  echo "[restart] failed: service did not bind port 6300"
  tail -n 30 "${LOG_PATH}" || true
  exit 1
fi

echo "[restart] service pid(s): ${running_pids}"
echo "[restart] API: ${SERVICE_URL}"
echo "[restart] API service restarted; no replay start request was sent"
tail -n 20 "${LOG_PATH}" || true
