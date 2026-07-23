#!/usr/bin/env bash
set -euo pipefail

SERVICE_UNIT="camera-pipeline.service"
SERVICE_PORT="6200"
SERVICE_PYTHON="/home/wuji-brain/miniconda3/envs/wuji/bin/python"
WORKSPACE_DIR="/home/wuji-brain/workspace"
WAIT_TIMEOUT_SECONDS=20
restart_log_since="$(date '+%Y-%m-%d %H:%M:%S')"

print_restart_logs() {
  journalctl -u "${SERVICE_UNIT}" \
    --since "${restart_log_since}" --no-pager -o short-precise || true
}

read_camera_status() {
  (
    cd "${WORKSPACE_DIR}"
    timeout 2s "${SERVICE_PYTHON}" -c \
      'from camera_pipeline.client import CameraName, CameraPipelineClient; client = CameraPipelineClient("tcp://127.0.0.1:6200", timeout_ms=1000); status = client.get_camera_status(CameraName.LEFT_ARM, timeout_s=0.5); client.close(); print(f"camera_name={status.camera_name} online={status.online} service_version={status.service_version}"); raise SystemExit(0 if status.online else 1)'
  )
}

echo "[restart] restarting ${SERVICE_UNIT} through systemd"
sudo systemctl restart --no-block "${SERVICE_UNIT}"

deadline=$((SECONDS + WAIT_TIMEOUT_SECONDS))
while ((SECONDS < deadline)); do
  if systemctl is-active --quiet "${SERVICE_UNIT}" &&
     ss -ltn "( sport = :${SERVICE_PORT} )" | grep -q LISTEN; then
    if camera_status="$(read_camera_status 2>/dev/null)"; then
      echo "[restart] ${SERVICE_UNIT} is ready: ${camera_status}"
      systemctl show "${SERVICE_UNIT}" \
        -p ActiveState -p SubState -p MainPID \
        -p TimeoutStartUSec -p TimeoutStopUSec --no-pager
      print_restart_logs
      exit 0
    fi
  fi
  sleep 1
done

echo "[restart] ${SERVICE_UNIT} was not ready within ${WAIT_TIMEOUT_SECONDS} seconds"
systemctl status "${SERVICE_UNIT}" --no-pager -l || true
print_restart_logs
exit 1
