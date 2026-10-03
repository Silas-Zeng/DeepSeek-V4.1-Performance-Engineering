"""Small CPU-only profiler exercise for the DeepSeek-V4.1 study.

This is a learning workload, not a performance claim. It mimics three indexer
stages so that a trace can be read before a CUDA host is available.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "profiles/cpu-tutorial/indexer_cpu_trace.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the DeepSeek-V4.1 CPU profiler tutorial.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT.relative_to(REPO_ROOT)),
        help="Relative or absolute path for the Chrome trace output.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="Number of profile iterations to execute for the synthetic indexer workload.",
    )
    return parser


def resolve_output_path(output: str | Path | None = None) -> Path:
    if output is None:
        return DEFAULT_OUTPUT

    output_path = Path(output)
    if output_path.is_absolute():
        return output_path
    return REPO_ROOT / output_path


def run(output: str | Path | None = None, repeat: int = 5) -> None:
    output_path = resolve_output_path(output)
    torch.manual_seed(0)
    torch.set_num_threads(1)

    query_rows = 32
    history = 8192
    head_dim = 64
    top_k = 64

    query = torch.randn(query_rows, head_dim)
    keys = torch.randn(history, head_dim)
    value_cache = torch.randn(history, 128)

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(max(1, repeat)):
            with record_function("indexer_score_matmul"):
                scores = query @ keys.transpose(0, 1)

            with record_function("indexer_topk"):
                _, indices = scores.topk(top_k, dim=-1)

            with record_function("indexer_value_gather"):
                _ = value_cache[indices]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(output_path))
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
    print(f"trace={output_path.resolve()}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(output=args.output, repeat=args.repeat)


if __name__ == "__main__":
    main()

