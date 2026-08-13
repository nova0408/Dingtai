#!/usr/bin/env bash
# Deploy one service package and verify its read-only readiness.
# Keep this file LF-encoded for remote execution.
set -euo pipefail

workspace="$1"
stage_path="$2"
manifest_path="$3"
package_name="$4"
service_unit="$5"
service_file_name="$6"
expected_version="$7"
readiness_mode="$8"
deploy_log_since="$(date '+%Y-%m-%d %H:%M:%S')"

cleanup() {
  rm -rf -- "${stage_path}"
  rm -f -- "/tmp/dingtai_single_service_deploy.sh"
}
trap cleanup EXIT

case "${package_name}:${service_unit}:${readiness_mode}" in
  camera_pipeline:camera-pipeline.service:camera) ;;
  calibration_service:calibration.service:calibration) ;;
  *) echo "[deploy] unsupported single-service deployment selection"; exit 1 ;;
esac
case "${stage_path}" in
  "${workspace}/.deploy_stage/"*) ;;
  *) echo "[deploy] invalid single-service stage path: ${stage_path}"; exit 1 ;;
esac

expected_count="$(wc -l < "${manifest_path}")"
actual_stage_count="$(find "${stage_path}/${package_name}" -type f ! -name '*.pyc' ! -name '*.log' ! -path '*/__pycache__/*' ! -path '*/.archive/*' | wc -l)"
if [[ "${actual_stage_count}" -ne "${expected_count}" ]]; then
  echo "[deploy] staged file count mismatch expected=${expected_count} actual=${actual_stage_count}"
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

record_replay_was_active=false
record_replay_state="not_running"
if [[ "${readiness_mode}" == "camera" ]] &&
   systemctl is-active --quiet record-replay.service &&
   ss -ltn '( sport = :6300 )' | grep -q LISTEN; then
  record_replay_payload="$(curl -fsS --max-time 5 http://127.0.0.1:6300/status)"
  record_replay_state="$(printf '%s' "${record_replay_payload}" | json_field state)"
  case "${record_replay_state}" in
    idle|waiting|rapid_stop) ;;
    *)
      echo "[deploy] refused: RecordReplay current state is ${record_replay_state}"
      exit 1
      ;;
  esac
  record_replay_was_active=true
  echo "[deploy] captured RecordReplay pre-deploy state=${record_replay_state}"
fi

echo "[deploy] stopping ${service_unit}"
systemctl stop --no-block "${service_unit}" || true
for attempt in $(seq 1 15); do
  active_state="$(systemctl show "${service_unit}" -p ActiveState --value 2>/dev/null || true)"
  if [[ -z "${active_state}" || "${active_state}" == "inactive" || "${active_state}" == "failed" ]]; then
    systemctl reset-failed "${service_unit}" || true
    break
  fi
  sleep 1
done

if [[ -d "${workspace}/${package_name}" ]]; then
  rm -rf -- "${workspace:?}/${package_name}"
fi
mv "${stage_path}/${package_name}" "${workspace}/${package_name}"
install -m 0644 \
  "${workspace}/${package_name}/service/${service_file_name}" \
  "/etc/systemd/system/${service_unit}"
systemctl daemon-reload
systemctl enable "${service_unit}"
systemctl restart --no-block "${service_unit}"

if [[ "${readiness_mode}" == "camera" ]]; then
  ready=false
  camera_version=""
  for attempt in $(seq 1 20); do
    if systemctl is-active --quiet "${service_unit}" &&
       ss -ltn '( sport = :6200 )' | grep -q LISTEN; then
      if camera_version="$(cd "${workspace}" && timeout 5s /home/wuji-brain/miniconda3/envs/wuji/bin/python -c 'from camera_pipeline.client import CameraName, CameraPipelineClient; client = CameraPipelineClient("tcp://127.0.0.1:6200", timeout_ms=1000); status = client.get_camera_status(CameraName.LEFT_ARM, timeout_s=0.5); client.close(); print(status.service_version); raise SystemExit(0 if status.online else 1)' 2>/dev/null)"; then
        ready=true
        break
      fi
    fi
    sleep 1
  done
  if [[ "${ready}" != "true" ]]; then
    echo "[deploy] CameraPipeline was not business-ready within 20s"
    systemctl status "${service_unit}" --no-pager -l || true
    journalctl -u "${service_unit}" --since "${deploy_log_since}" --no-pager -o short-precise || true
    exit 1
  fi
  if [[ "${camera_version}" != "${expected_version}" ]]; then
    echo "[deploy] CameraPipeline version mismatch expected=${expected_version} actual=${camera_version}"
    exit 1
  fi
  echo "[deploy] CameraPipeline updated and restarted; version=${camera_version}; read-only camera status passed"

  if [[ "${record_replay_was_active}" == "true" ]]; then
    echo "[deploy] restoring record-replay.service after CameraPipeline deployment"
    for attempt in $(seq 1 15); do
      record_active_state="$(systemctl show record-replay.service -p ActiveState --value 2>/dev/null || true)"
      if [[ -z "${record_active_state}" || "${record_active_state}" == "inactive" || "${record_active_state}" == "failed" ]]; then
        break
      fi
      sleep 1
    done
    if [[ "${record_replay_state}" == "idle" || "${record_replay_state}" == "waiting" ]]; then
      printf '{\n  "state": "idle"\n}\n' > "${workspace}/record_replay/runtime_state.json"
    fi
    systemctl reset-failed record-replay.service || true
    systemctl start --no-block record-replay.service
    record_replay_ready=false
    for attempt in $(seq 1 20); do
      if systemctl is-active --quiet record-replay.service &&
         ss -ltn '( sport = :6300 )' | grep -q LISTEN &&
         record_replay_payload="$(curl -fsS --max-time 5 http://127.0.0.1:6300/status)"; then
        restored_record_replay_state="$(printf '%s' "${record_replay_payload}" | json_field state)"
        if [[ "${record_replay_state}" == "rapid_stop" && "${restored_record_replay_state}" == "rapid_stop" ]] ||
           [[ "${record_replay_state}" != "rapid_stop" && "${restored_record_replay_state}" == "idle" ]]; then
          record_replay_ready=true
          break
        fi
      fi
      sleep 1
    done
    if [[ "${record_replay_ready}" != "true" ]]; then
      echo "[deploy] RecordReplay was not restored after CameraPipeline deployment"
      systemctl status record-replay.service --no-pager -l || true
      journalctl -u record-replay.service --since "${deploy_log_since}" --no-pager -o short-precise || true
      exit 1
    fi
    echo "[deploy] RecordReplay restored; GET /status state=${restored_record_replay_state}"
  fi

  echo "[deploy] restoring calibration.service after CameraPipeline deployment"
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
    echo "[deploy] Calibration Service was not restored after CameraPipeline deployment"
    systemctl status calibration.service --no-pager -l || true
    journalctl -u calibration.service --since "${deploy_log_since}" --no-pager -o short-precise || true
    exit 1
  fi
  echo "[deploy] Calibration Service restored; GET /api/v1/status state=idle"
else
  ready=false
  calibration_version=""
  for attempt in $(seq 1 10); do
    if systemctl is-active --quiet "${service_unit}" &&
       ss -ltn '( sport = :6600 )' | grep -q LISTEN &&
       calibration_payload="$(curl -fsS --max-time 5 http://127.0.0.1:6600/api/v1/status)"; then
      calibration_state="$(printf '%s' "${calibration_payload}" | json_field state)"
      calibration_version="$(printf '%s' "${calibration_payload}" | json_field service_version)"
      if [[ "${calibration_state}" == "idle" ]]; then
        ready=true
        break
      fi
    fi
    sleep 1
  done
  if [[ "${ready}" != "true" ]]; then
    echo "[deploy] Calibration Service was not idle and HTTP-ready within 10s"
    systemctl status "${service_unit}" --no-pager -l || true
    journalctl -u "${service_unit}" --since "${deploy_log_since}" --no-pager -o short-precise || true
    exit 1
  fi
  if [[ "${calibration_version}" != "${expected_version}" ]]; then
    echo "[deploy] Calibration Service version mismatch expected=${expected_version} actual=${calibration_version}"
    exit 1
  fi
  echo "[deploy] Calibration Service updated and restarted; version=${calibration_version}; GET /api/v1/status passed"
fi
