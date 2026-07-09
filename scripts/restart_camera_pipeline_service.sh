#!/usr/bin/env bash
set -euo pipefail

SERVICE_PATTERN="python -m camera_pipeline.service camera_pipeline_service --bind-addr tcp://0.0.0.0:6200 --control-port 5570 --stream-port 5562 --camera-id LEFT --camera-name left_hand_camera"
SERVICE_CMD="/home/wuji-brain/miniconda3/envs/py38_tourch/bin/python -m camera_pipeline.service camera_pipeline_service --bind-addr tcp://0.0.0.0:6200 --control-port 5570 --stream-port 5562 --camera-id LEFT --camera-name left_hand_camera"
LOG_PATH="/tmp/camera_pipeline_service.log"
WORKSPACE_DIR="/home/wuji-brain/workspace"

echo "[restart] workspace=${WORKSPACE_DIR}"
cd "${WORKSPACE_DIR}"

collect_pids() {
  local ports_output="$1"
  local pids=""
  while IFS= read -r line; do
    if [[ "${line}" =~ pid=([0-9]+) ]]; then
      pids+="${BASH_REMATCH[1]}"$'\n'
    fi
  done <<< "${ports_output}"
  printf '%s' "${pids}" | sed '/^$/d' | sort -u
}

echo "[restart] checking port usage"
port_usage=""
if command -v ss >/dev/null 2>&1; then
  port_usage="$(ss -ltnp '( sport = :6200 or sport = :6201 or sport = :6202 or sport = :6203 )' 2>/dev/null || true)"
  echo "${port_usage}"
fi

port_pids="$(collect_pids "${port_usage}")"
service_pids="$(pgrep -f "${SERVICE_PATTERN}" || true)"
all_pids="$(printf '%s\n%s\n' "${port_pids}" "${service_pids}" | sed '/^$/d' | sort -u)"

if [[ -n "${all_pids}" ]]; then
  echo "[restart] stopping pid list:"
  echo "${all_pids}"
  echo "[restart] sending SIGTERM"
  if command -v sudo >/dev/null 2>&1; then
    sudo kill ${all_pids} || true
  else
    kill ${all_pids} || true
  fi
  sleep 3
fi

port_usage_after_term=""
if command -v ss >/dev/null 2>&1; then
  port_usage_after_term="$(ss -ltnp '( sport = :6200 or sport = :6201 or sport = :6202 or sport = :6203 )' 2>/dev/null || true)"
fi
still_pids="$(collect_pids "${port_usage_after_term}")"
if [[ -n "${still_pids}" ]]; then
  echo "[restart] still busy after SIGTERM, sending SIGKILL"
  echo "${still_pids}"
  if command -v sudo >/dev/null 2>&1; then
    sudo kill -9 ${still_pids} || true
  else
    kill -9 ${still_pids} || true
  fi
  sleep 3
fi

port_usage_final=""
if command -v ss >/dev/null 2>&1; then
  port_usage_final="$(ss -ltnp '( sport = :6200 or sport = :6201 or sport = :6202 or sport = :6203 )' 2>/dev/null || true)"
fi
final_pids="$(collect_pids "${port_usage_final}")"
if [[ -n "${final_pids}" ]]; then
  echo "[restart] failed to stop port owners:"
  echo "${final_pids}"
  echo "${port_usage_final}"
  exit 1
fi

echo "[restart] starting service"
rm -f "${LOG_PATH}"
setsid ${SERVICE_CMD} >"${LOG_PATH}" 2>&1 < /dev/null &
sleep 3

echo "[restart] success summary"
running_pids="$(pgrep -f "${SERVICE_PATTERN}" || true)"
if [[ -n "${running_pids}" ]]; then
  echo "[restart] service pid(s): ${running_pids}"
else
  echo "[restart] service pid(s): not found"
fi
echo "[restart] bind ports: 6200(service), 6201(frame), 6202(color), 6203(depth)"
echo "[restart] last log lines:"
tail -n 20 "${LOG_PATH}" || true
