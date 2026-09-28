#!/usr/bin/env bash
set -euo pipefail

board_address="ubuntu@139.19.64.227"
ssh_key="/home/sadutta/.ssh/kria_codex_ed25519"
local_port="18765"
remote_port="8765"
runtime_dir="${XDG_RUNTIME_DIR:-/tmp}"
control_socket="${runtime_dir}/kv260-mnist-ui-${UID}.sock"
url="http://127.0.0.1:${local_port}/"

if ! ssh -S "${control_socket}" -O check "${board_address}" >/dev/null 2>&1; then
  rm -f "${control_socket}"
  ssh -fNT -M -S "${control_socket}" \
    -i "${ssh_key}" \
    -o BatchMode=yes \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=3 \
    -L "127.0.0.1:${local_port}:127.0.0.1:${remote_port}" \
    "${board_address}"
fi

for _attempt in $(seq 1 30); do
  if curl --silent --fail --max-time 1 "http://127.0.0.1:${local_port}/health" >/dev/null; then
    break
  fi
  sleep 0.2
done

curl --silent --fail --max-time 2 "http://127.0.0.1:${local_port}/health" >/dev/null
echo "Opening ${url}"
xdg-open "${url}" >/dev/null 2>&1 &
