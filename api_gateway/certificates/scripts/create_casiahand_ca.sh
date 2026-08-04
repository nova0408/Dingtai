#!/usr/bin/env bash
set -euo pipefail

if (( $# != 0 )); then
  echo "此脚本不需要参数，直接运行即可。" >&2
  exit 2
fi

output_dir="/home/wuji-brain/casiahand-pki/ca"
mkdir -p "${output_dir}"
chmod 0700 "${output_dir}"

key_path="${output_dir}/casiahand-root-ca.key.pem"
cert_path="${output_dir}/casiahand-root-ca.crt.pem"
der_path="${output_dir}/casiahand-root-ca.cer"
config_file="$(mktemp "${output_dir}/casiahand-ca.XXXXXX.cnf")"
trap 'rm -f -- "${config_file}"' EXIT

if [[ -e "${key_path}" || -e "${cert_path}" || -e "${der_path}" ]]; then
  echo "refused: CasiaHand CA output already exists in ${output_dir}" >&2
  exit 1
fi

openssl genrsa -aes256 -out "${key_path}" 4096
chmod 0600 "${key_path}"
cat > "${config_file}" <<'EOF'
[req]
prompt = no
distinguished_name = req_distinguished_name
x509_extensions = v3_ca

[req_distinguished_name]
C = CN
O = CasiaHand
CN = CasiaHand Root CA

[v3_ca]
basicConstraints = critical,CA:TRUE,pathlen:0
keyUsage = critical,keyCertSign,cRLSign
subjectKeyIdentifier = hash
EOF
openssl req -x509 -new -sha256 -days 3650 \
  -key "${key_path}" \
  -config "${config_file}" \
  -extensions v3_ca \
  -out "${cert_path}"
openssl x509 -in "${cert_path}" -outform DER -out "${der_path}"
chmod 0644 "${cert_path}" "${der_path}"

echo "CasiaHand CA created: ${cert_path}"
echo "Keep the encrypted private key offline: ${key_path}"
