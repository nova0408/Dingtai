#!/usr/bin/env bash
# Linux service restart helper; keep this file LF-encoded for remote execution.
set -euo pipefail

SERVICE_UNIT="api-gateway.service"
SERVICE_PORT="443"
WAIT_TIMEOUT_SECONDS=15
CA_PATH="/etc/dingtai/api-gateway/tls/casiahand-root-ca.crt.pem"
CERT_PATH="/etc/dingtai/api-gateway/tls/api-gateway.fullchain.pem"
KEY_PATH="/etc/dingtai/api-gateway/tls/api-gateway.key.pem"
restart_log_since="$(date '+%Y-%m-%d %H:%M:%S')"
gateway_hostname="$(hostname)"

print_restart_logs() {
  journalctl -u "${SERVICE_UNIT}" \
    --since "${restart_log_since}" --no-pager -o short-precise || true
}

for tls_path in "${CA_PATH}" "${CERT_PATH}" "${KEY_PATH}"; do
  if [[ ! -r "${tls_path}" ]]; then
    echo "[restart] refused: missing or unreadable TLS file ${tls_path}"
    exit 1
  fi
done

echo "[restart] restarting ${SERVICE_UNIT} through systemd"
sudo systemctl restart "${SERVICE_UNIT}"

deadline=$((SECONDS + WAIT_TIMEOUT_SECONDS))
while ((SECONDS < deadline)); do
  if systemctl is-active --quiet "${SERVICE_UNIT}" &&
     ss -ltn "( sport = :${SERVICE_PORT} )" | grep -q LISTEN &&
     curl -fsS --max-time 5 \
       --cacert "${CA_PATH}" \
       --resolve "${gateway_hostname}:${SERVICE_PORT}:127.0.0.1" \
       "https://${gateway_hostname}/api/v1/gateway/health"; then
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
