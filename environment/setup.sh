#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 -m venv "$repo_root/.venv"
"$repo_root/.venv/bin/python" -m pip install -r "$repo_root/requirements-analysis.txt"
"$repo_root/.venv/bin/python" "$repo_root/scripts/collect_environment.py" \
  --output "$repo_root/reports/environment.json"
