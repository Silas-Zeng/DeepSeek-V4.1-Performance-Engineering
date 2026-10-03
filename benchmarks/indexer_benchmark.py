"""Run a small, reproducible dense-vs-chunked indexer benchmark.

The benchmark is intentionally framework-independent. It measures the score and
Top-K part of the indexer with model-shaped tensors, checks that a chunked path
selects the same positions as the dense reference, and writes a JSON record that
can be compared with later GPU runs. It is a component baseline, not an
end-to-end DeepSeek-V4.1 performance claim.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "results/indexer-benchmark.json"


@dataclass(frozen=True)
class BenchmarkConfig:
    """Inputs that define one benchmark invocation."""

    query_rows: int = 32
    history: int = 8192
    head_dim: int = 64
    top_k: int = 64
    chunk_size: int = 1024
    warmup: int = 2
    repeat: int = 10
    device: str = "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark dense and chunked synthetic indexer paths."
    )
    parser.add_argument("--query-rows", type=int, default=BenchmarkConfig.query_rows)
    parser.add_argument("--history", type=int, default=BenchmarkConfig.history)
    parser.add_argument("--head-dim", type=int, default=BenchmarkConfig.head_dim)
    parser.add_argument("--top-k", type=int, default=BenchmarkConfig.top_k)
    parser.add_argument("--chunk-size", type=int, default=BenchmarkConfig.chunk_size)
    parser.add_argument("--warmup", type=int, default=BenchmarkConfig.warmup)
    parser.add_argument("--repeat", type=int, default=BenchmarkConfig.repeat)
    parser.add_argument(
        "--device",
        default=BenchmarkConfig.device,
        help="Torch device, for example cpu or cuda. CUDA must be available.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON output path; relative paths are resolved from the repository root.",
    )
    return parser


def validate_config(config: BenchmarkConfig) -> None:
    """Reject invalid shapes before allocating tensors."""

    positive = {
        "query_rows": config.query_rows,
        "history": config.history,
        "head_dim": config.head_dim,
        "top_k": config.top_k,
        "chunk_size": config.chunk_size,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"values must be positive: {', '.join(invalid)}")
    if config.top_k > config.history:
        raise ValueError("top_k cannot exceed history")
    if config.warmup < 0 or config.repeat <= 0:
        raise ValueError("warmup must be non-negative and repeat must be positive")
    if not config.device:
        raise ValueError("device cannot be empty")


def dense_indexer(query: Any, keys: Any, top_k: int) -> tuple[Any, Any]:
    """Return Top-K scores and positions after materializing the full score matrix."""

    scores = query @ keys.transpose(0, 1)
    return scores.topk(top_k, dim=-1)


def chunked_indexer(
    query: Any, keys: Any, top_k: int, chunk_size: int
) -> tuple[Any, Any]:
    """Return the exact dense Top-K result while bounding score workspace."""

    import torch

    best_scores = None
    best_indices = None
    for start in range(0, keys.shape[0], chunk_size):
        stop = min(start + chunk_size, keys.shape[0])
        chunk_scores = query @ keys[start:stop].transpose(0, 1)
        local_k = min(top_k, chunk_scores.shape[-1])
        local_scores, local_indices = chunk_scores.topk(local_k, dim=-1)
        local_indices = local_indices + start

        if best_scores is None:
            best_scores, best_indices = local_scores, local_indices
            continue

        merged_scores = torch.cat((best_scores, local_scores), dim=-1)
        merged_indices = torch.cat((best_indices, local_indices), dim=-1)
        keep = min(top_k, merged_scores.shape[-1])
        best_scores, order = merged_scores.topk(keep, dim=-1)
        best_indices = merged_indices.gather(-1, order)

    return best_scores, best_indices


def _synchronize(device: Any, torch_module: Any) -> None:
    if device.type == "cuda":
        torch_module.cuda.synchronize(device)


def _timed(
    operation: Callable[[], tuple[Any, Any]],
    warmup: int,
    repeat: int,
    device: Any,
    torch_module: Any,
) -> tuple[list[float], int | None]:
    with torch_module.no_grad():
        for _ in range(warmup):
            operation()
        _synchronize(device, torch_module)

        if device.type == "cuda":
            torch_module.cuda.reset_peak_memory_stats(device)

        samples: list[float] = []
        for _ in range(repeat):
            _synchronize(device, torch_module)
            started = time.perf_counter()
            operation()
            _synchronize(device, torch_module)
            samples.append((time.perf_counter() - started) * 1000.0)

        peak = (
            int(torch_module.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        )
    return samples, peak


def _summary(samples: list[float], peak_memory_bytes: int | None) -> dict[str, Any]:
    ordered = sorted(samples)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "samples_ms": [round(value, 6) for value in samples],
        "median_ms": round(statistics.median(samples), 6),
        "p95_ms": round(ordered[p95_index], 6),
        "peak_memory_bytes": peak_memory_bytes,
    }


def resolve_output_path(output: str | Path) -> Path:
    output_path = Path(output)
    return output_path if output_path.is_absolute() else REPO_ROOT / output_path


def run_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    """Run the benchmark and return a JSON-serializable result."""

    validate_config(config)
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the host environment
        raise RuntimeError("PyTorch is required; install the GPU or analysis environment") from exc

    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    torch.manual_seed(0)
    query = torch.randn(config.query_rows, config.head_dim, device=device)
    keys = torch.randn(config.history, config.head_dim, device=device)

    dense_scores, dense_indices = dense_indexer(query, keys, config.top_k)
    chunked_scores, chunked_indices = chunked_indexer(
        query, keys, config.top_k, config.chunk_size
    )
    same_positions = torch.equal(
        torch.sort(dense_indices, dim=-1).values,
        torch.sort(chunked_indices, dim=-1).values,
    )
    max_score_error = float(
        (torch.sort(dense_scores, dim=-1).values - torch.sort(chunked_scores, dim=-1).values)
        .abs()
        .max()
        .item()
    )
    if not same_positions or max_score_error > 1e-5:
        raise RuntimeError(
            "chunked path disagrees with dense reference: "
            f"same_positions={same_positions}, max_score_error={max_score_error}"
        )

    dense_samples, dense_peak = _timed(
        lambda: dense_indexer(query, keys, config.top_k),
        config.warmup,
        config.repeat,
        device,
        torch,
    )
    chunked_samples, chunked_peak = _timed(
        lambda: chunked_indexer(query, keys, config.top_k, config.chunk_size),
        config.warmup,
        config.repeat,
        device,
        torch,
    )
    return {
        "benchmark": "synthetic_sparse_indexer",
        "torch_version": torch.__version__,
        "config": asdict(config),
        "correctness": {
            "topk_positions_match": same_positions,
            "max_score_abs_error": max_score_error,
        },
        "paths": {
            "dense": _summary(dense_samples, dense_peak),
            "chunked": _summary(chunked_samples, chunked_peak),
        },
        "notes": [
            "Component-level synthetic measurement; not an end-to-end model result.",
            "CPU peak memory is not reported; CUDA peak memory is allocator-reported.",
        ],
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = BenchmarkConfig(
        query_rows=args.query_rows,
        history=args.history,
        head_dim=args.head_dim,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
        warmup=args.warmup,
        repeat=args.repeat,
        device=args.device,
    )
    try:
        result = run_benchmark(config)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    output_path = resolve_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"result={output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
