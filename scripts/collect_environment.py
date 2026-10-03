"""Collect reproducibility metadata for a performance run."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def command_output(command: list[str], cwd: Path | None = None) -> str | None:
    executable = shutil.which(command[0])
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, *command[1:]],
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=10,
            cwd=cwd,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    output = result.stdout.strip()
    return output or None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_environment() -> dict:
    """Describe this repository and host, independently of the caller's cwd."""
    status = command_output(["git", "status", "--porcelain"], cwd=REPO_ROOT)
    return {
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git": command_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT),
        "git_worktree_dirty": bool(status),
        "gpu": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "packages": {
            name: package_version(name)
            for name in (
                "torch",
                "vllm",
                "sglang",
                "flashinfer-python",
                "deep-gemm",
                "PyYAML",
                "psutil",
            )
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metadata = collect_environment()

    rendered = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
