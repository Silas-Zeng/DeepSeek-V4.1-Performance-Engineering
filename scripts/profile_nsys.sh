#!/usr/bin/env bash
set -euo pipefail

output="profiles/v41"
if [[ "${1:-}" == "--output" ]]; then
  output="$2"
  shift 2
fi
if [[ "${1:-}" != "--" ]]; then
  echo "usage: $0 [--output PATH] -- COMMAND [ARGS...]" >&2
  exit 2
fi
shift
command -v nsys >/dev/null || {
  echo "nsys is required on the Linux GPU host" >&2
  exit 1
}
mkdir -p "$(dirname "$output")"
exec nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true \
  --force-overwrite=true -o "$output" "$@"
