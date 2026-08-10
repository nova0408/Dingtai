#!/usr/bin/env bash
set -euo pipefail

if (( $# != 0 )); then
  echo "此脚本不需要参数，直接运行即可。" >&2
  exit 2
fi

pki_root="/home/wuji-brain/casiahand-pki"
ca_dir="${pki_root}/ca"
server_dir="${pki_root}/orin"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
umask 077
common_name="$(hostname)"

if [[ -e "${server_dir}" ]]; then
  echo "refused: existing API Gateway PKI output found at ${server_dir}" >&2
  echo "Archive it manually before registering again." >&2
  exit 1
fi

mkdir -p "${pki_root}"
if [[ -e "${ca_dir}" ]]; then
  [[ -f "${ca_dir}/casiahand-root-ca.key.pem" &&
    -f "${ca_dir}/casiahand-root-ca.crt.pem" ]] || {
    echo "CasiaHand CA directory exists but is incomplete: ${ca_dir}" >&2
    exit 1
  }
  echo "Reusing the existing CasiaHand Root CA in ${ca_dir}"
else
  echo "Creating CasiaHand Root CA in ${ca_dir}"
  bash "${script_dir}/create_casiahand_ca.sh"
fi

echo "为 ${common_name} 签发 API Gateway 证书"
bash "${script_dir}/issue_api_gateway_certificate.sh"

echo "安装 API Gateway 服务器证书"
sudo bash "${script_dir}/install_api_gateway_certificate.sh"

echo
echo "Registration complete."
echo "CA public certificate: ${server_dir}/casiahand-root-ca.cer"
echo "Server certificate identities: ${common_name}, 192.168.100.70, and 192.168.1.1-192.168.1.254"
echo "Existing Root CA clients do not need to reinstall a certificate."
echo "Install the public certificate on clients only when this run created a new Root CA."
echo "The CA private key remains at ${ca_dir}/casiahand-root-ca.key.pem; keep it offline."
