#!/usr/bin/env bash
set -euo pipefail

if (( $# != 0 )); then
  echo "此脚本不需要参数，直接运行即可。" >&2
  exit 2
fi
if (( EUID != 0 )); then
  echo "run this script with sudo" >&2
  exit 1
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
certificate="${script_dir}/../client/casiahand-root-ca.crt.pem"
[[ -f "${certificate}" ]] || { echo "certificate not found: ${certificate}" >&2; exit 1; }

if command -v update-ca-certificates >/dev/null 2>&1; then
  install -m 0644 "${certificate}" /usr/local/share/ca-certificates/casiahand-root-ca.crt
  update-ca-certificates
elif command -v update-ca-trust >/dev/null 2>&1; then
  install -m 0644 "${certificate}" /etc/pki/ca-trust/source/anchors/casiahand-root-ca.crt
  update-ca-trust extract
else
  echo "unsupported Linux trust-store tool; install the CA manually" >&2
  exit 1
fi

echo "CasiaHand CA installed in the system trust store."
