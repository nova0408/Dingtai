#!/usr/bin/env bash
set -euo pipefail

if (( $# != 0 )); then
  echo "此脚本不需要参数，直接运行即可。" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${script_dir}/api_gateway_certificate_sans.sh"

if (( EUID != 0 )); then
  echo "run this script with sudo" >&2
  exit 1
fi

source_dir="/home/wuji-brain/casiahand-pki/orin"
install_dir="/etc/dingtai/api-gateway/tls"
archive_dir="/etc/dingtai/api-gateway/.archive/tls-$(date '+%Y%m%d-%H%M%S')"
ca_cert="${source_dir}/casiahand-root-ca.crt.pem"
server_cert="${source_dir}/api-gateway.crt.pem"
fullchain="${source_dir}/api-gateway.fullchain.pem"
server_key="${source_dir}/api-gateway.key.pem"

echo "[1/4] 检查待安装的 CasiaHand 证书文件"
for path in "${ca_cert}" "${server_cert}" "${fullchain}" "${server_key}"; do
  [[ -f "${path}" ]] || { echo "missing certificate file: ${path}" >&2; exit 1; }
done
echo "[2/4] 验证服务器证书链和私钥匹配"
openssl verify -CAfile "${ca_cert}" "${server_cert}"
cert_public_key="$(openssl x509 -in "${server_cert}" -pubkey -noout | openssl sha256)"
key_public_key="$(openssl pkey -in "${server_key}" -pubout | openssl sha256)"
[[ "${cert_public_key}" == "${key_public_key}" ]] || {
  echo "server certificate and private key do not match" >&2
  exit 1
}

if [[ -d "${install_dir}" ]] && find "${install_dir}" -mindepth 1 -print -quit | grep -q .; then
  install -d -m 0700 -o root -g root "${archive_dir}"
  cp -a "${install_dir}/." "${archive_dir}/"
  echo "已备份旧 TLS 文件：${archive_dir}"
fi
echo "[3/4] 安装 Gateway TLS 文件"
install -d -m 0750 -o root -g wuji-brain "${install_dir}"
install -m 0644 -o root -g wuji-brain "${ca_cert}" "${install_dir}/casiahand-root-ca.crt.pem"
install -m 0644 -o root -g wuji-brain "${server_cert}" "${install_dir}/api-gateway.crt.pem"
install -m 0644 -o root -g wuji-brain "${fullchain}" "${install_dir}/api-gateway.fullchain.pem"
install -m 0640 -o root -g wuji-brain "${server_key}" "${install_dir}/api-gateway.key.pem"

echo "[4/4] 验证已安装文件"
installed_ca="${install_dir}/casiahand-root-ca.crt.pem"
installed_cert="${install_dir}/api-gateway.crt.pem"
installed_fullchain="${install_dir}/api-gateway.fullchain.pem"
installed_key="${install_dir}/api-gateway.key.pem"
gateway_hostname="$(hostname)"
openssl verify -CAfile "${installed_ca}" -verify_hostname "${gateway_hostname}" "${installed_cert}"
api_gateway_verify_certificate_ip_sans "${installed_ca}" "${installed_cert}"
openssl x509 -in "${installed_fullchain}" -noout -subject -issuer -dates >/dev/null
installed_cert_public_key="$(openssl x509 -in "${installed_cert}" -pubkey -noout | openssl sha256)"
installed_key_public_key="$(openssl pkey -in "${installed_key}" -pubout | openssl sha256)"
[[ "${installed_cert_public_key}" == "${installed_key_public_key}" ]] || {
  echo "错误：已安装的服务器证书和私钥不匹配。" >&2
  exit 1
}
[[ "$(stat -c '%a' "${installed_key}")" == "640" ]] || {
  echo "错误：服务器私钥权限不是 0640。" >&2
  exit 1
}

echo "证书文件已安装并通过链验证：${install_dir}"
if systemctl is-active --quiet api-gateway.service &&
  ss -ltn '( sport = :443 )' | grep -q LISTEN; then
  echo "正在执行 Gateway 只读 HTTPS 健康检查"
  health_payload="$(curl -fsS --max-time 5 --cacert "${installed_ca}" \
    --resolve "${gateway_hostname}:443:127.0.0.1" \
    "https://${gateway_hostname}/api/v1/gateway/health")"
  echo "Gateway HTTPS 健康检查通过：${health_payload}"
else
  echo "提示：api-gateway.service 尚未在 443 监听，已跳过在线健康检查。"
  echo "证书安装已完成；更新服务文件后请运行 sync_and_restart_services.ps1 -ApiGatewayOnly。"
fi
echo "CasiaHand CA 与 API Gateway TLS 安装完成。"
