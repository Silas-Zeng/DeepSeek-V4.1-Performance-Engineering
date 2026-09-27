#!/usr/bin/env bash
set -euo pipefail

# Install one framework in an isolated uv environment on a Linux GPU host.
framework="${1:-}"
ref="${2:-}"
if [[ "$framework" != "vllm" && "$framework" != "sglang" ]] || [[ -z "$ref" ]]; then
  echo "usage: $0 vllm|sglang <git-commit-or-tag>" >&2
  exit 2
fi
command -v uv >/dev/null || { echo "uv is required on the Linux GPU host" >&2; exit 1; }
command -v git >/dev/null || { echo "git is required" >&2; exit 1; }

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
base="$repo_root/.framework-env"
src="$base/src/$framework"
venv="$base/venvs/$framework"
if [[ "$framework" == "vllm" ]]; then
  repository="https://github.com/vllm-project/vllm.git"
else
  repository="https://github.com/sgl-project/sglang.git"
fi

mkdir -p "$base/src" "$base/venvs"
if [[ ! -d "$src/.git" ]]; then
  git clone "$repository" "$src"
fi
git -C "$src" fetch --tags origin
git -C "$src" checkout --detach "$ref"

uv venv --python 3.12 "$venv"
if [[ "$framework" == "vllm" ]]; then
  uv pip install --python "$venv/bin/python" -e "$src" --torch-backend=auto
else
  uv pip install --python "$venv/bin/python" -e "$src"
fi

"$venv/bin/python" "$repo_root/scripts/collect_environment.py" \
  --output "$repo_root/reports/environment-$framework.json"
echo "Installed $framework at $ref"
echo "Environment: $venv"
