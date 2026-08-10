#!/usr/bin/env bash
set -euo pipefail

if (( $# != 0 )); then
  echo "此脚本不需要参数，直接运行即可。" >&2
  exit 2
fi

ca_dir="/home/wuji-brain/casiahand-pki/ca"
output_dir="/home/wuji-brain/casiahand-pki/orin"
common_name="$(hostname)"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${script_dir}/api_gateway_certificate_sans.sh"

ca_key="${ca_dir}/casiahand-root-ca.key.pem"
ca_cert="${ca_dir}/casiahand-root-ca.crt.pem"
key_path="${output_dir}/api-gateway.key.pem"
csr_path="${output_dir}/api-gateway.csr.pem"
cert_path="${output_dir}/api-gateway.crt.pem"
fullchain_path="${output_dir}/api-gateway.fullchain.pem"
output_ca_cert="${output_dir}/casiahand-root-ca.crt.pem"
output_ca_der="${output_dir}/casiahand-root-ca.cer"

[[ -f "${ca_key}" && -f "${ca_cert}" ]] || {
  echo "CasiaHand CA key or certificate is missing in ${ca_dir}" >&2
  exit 1
}
mkdir -p "${output_dir}"
chmod 0700 "${output_dir}"
for path in "${key_path}" "${csr_path}" "${cert_path}" "${fullchain_path}" \
  "${output_ca_cert}" "${output_ca_der}"; do
  [[ ! -e "${path}" ]] || { echo "refused: output exists: ${path}" >&2; exit 1; }
done

san_csv="DNS:${common_name},$(api_gateway_certificate_san_csv)"

extension_file="$(mktemp "${output_dir}/api-gateway.extensions.XXXXXX")"
trap 'rm -f -- "${extension_file}"' EXIT
cat > "${extension_file}" <<EOF
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
subjectAltName=${san_csv}
authorityKeyIdentifier=keyid,issuer
subjectKeyIdentifier=hash
EOF

openssl genrsa -out "${key_path}" 3072
chmod 0600 "${key_path}"
openssl req -new -sha256 -key "${key_path}" \
  -subj "/C=CN/O=CasiaHand/CN=${common_name}" -out "${csr_path}"
openssl x509 -req -sha256 -days 825 \
  -in "${csr_path}" -CA "${ca_cert}" -CAkey "${ca_key}" -CAcreateserial \
  -extfile "${extension_file}" -out "${cert_path}"
cat "${cert_path}" "${ca_cert}" > "${fullchain_path}"
cp "${ca_cert}" "${output_ca_cert}"
openssl x509 -in "${ca_cert}" -outform DER -out "${output_ca_der}"
chmod 0644 "${csr_path}" "${cert_path}" "${fullchain_path}" \
  "${output_ca_cert}" "${output_ca_der}"

openssl verify -CAfile "${ca_cert}" -verify_hostname "${common_name}" "${cert_path}"
api_gateway_verify_certificate_ip_sans "${ca_cert}" "${cert_path}"
openssl x509 -in "${cert_path}" -noout -subject -issuer -dates -ext subjectAltName
echo "API Gateway certificate created: ${fullchain_path}"
