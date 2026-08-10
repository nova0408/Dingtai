#!/usr/bin/env bash

# These are server-certificate identities, not Root CA identities.

api_gateway_certificate_ip_sans() {
  printf '%s\n' '192.168.100.70'
  for last_octet in {1..254}; do
    printf '192.168.1.%s\n' "${last_octet}"
  done
}

api_gateway_certificate_san_csv() {
  local san_csv=''
  local ip
  local separator=''
  while IFS= read -r ip; do
    san_csv+="${separator}IP:${ip}"
    separator=','
  done < <(api_gateway_certificate_ip_sans)
  printf '%s\n' "${san_csv}"
}

api_gateway_verify_certificate_ip_sans() {
  local ca_cert="$1"
  local server_cert="$2"
  local ip

  openssl verify -CAfile "${ca_cert}" "${server_cert}" >/dev/null
  while IFS= read -r ip; do
    openssl verify -CAfile "${ca_cert}" -verify_ip "${ip}" \
      "${server_cert}" >/dev/null
  done < <(api_gateway_certificate_ip_sans)
}
