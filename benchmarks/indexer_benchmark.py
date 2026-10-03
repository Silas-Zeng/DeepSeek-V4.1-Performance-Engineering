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
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "results/indexer-benchmark.json"
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_environment import collect_environment


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
    seed: int = 0
    threads: int = 1


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
    parser.add_argument("--seed", type=int, default=BenchmarkConfig.seed)
    parser.add_argument("--threads", type=int, default=BenchmarkConfig.threads)
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
        "threads": config.threads,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"values must be positive: {', '.join(invalid)}")
    if config.top_k > config.history:
        raise ValueError("top_k cannot exceed history")
    if config.warmup < 0 or config.repeat <= 0:
        raise ValueError("warmup must be non-negative and repeat must be positive")
    if config.threads <= 0:
        raise ValueError("threads must be positive")
    if not config.device:
        raise ValueError("device cannot be empty")
    if not 0 <= config.seed < 2**63:
        raise ValueError("seed must be in [0, 2**63)")


def dense_indexer(query: Any, keys: Any, top_k: int) -> tuple[Any, Any]:
    """Return Top-K scores and positions after materializing the full score matrix."""

    scores = query @ keys.transpose(0, 1)
    return scores.topk(top_k, dim=-1)


def chunked_indexer(
    query: Any, keys: Any, top_k: int, chunk_size: int
) -> tuple[Any, Any]:
    """Stream Top-K while bounding scores; equal scores may choose other indices."""

    import torch

    best_scores = None
    best_indices = None
    for start in range(0, keys.shape[0], chunk_size):
        stop = min(start + chunk_size, keys.shape[0])
        chunk_scores = query @ keys[start:stop].transpose(0, 1)
        local_k = min(top_k, chunk_scores.shape[-1])
        local_scores, local_indices = chunk_scores.topk(local_k, dim=-1)
        local_indices = local_indices + start
        del chunk_scores

        if best_scores is None:
            best_scores, best_indices = local_scores, local_indices
            continue

        merged_scores = torch.cat((best_scores, local_scores), dim=-1)
        merged_indices = torch.cat((best_indices, local_indices), dim=-1)
        keep = min(top_k, merged_scores.shape[-1])
        best_scores, order = merged_scores.topk(keep, dim=-1)
        best_indices = merged_indices.gather(-1, order)
        del merged_scores, merged_indices, local_scores, local_indices, order

    return best_scores, best_indices


def _synchronize(device: Any, torch_module: Any) -> None:
    if device.type == "cuda":
        torch_module.cuda.synchronize(device)


def check_selection(query: Any, keys: Any, top_k: int, selected: tuple[Any, Any]) -> dict:
    """Check indices against dense scores, accepting only exact-score tie swaps.

    PyTorch does not promise stable indices for tied Top-K elements. Every index
    must still be unique, in range, and select the reference Top-K score multiset.
    Chunked GEMM scores may differ within the documented FP32 tolerance.
    """
    import torch

    values, indices = selected
    expected_shape = (query.shape[0], top_k)
    if tuple(values.shape) != expected_shape or tuple(indices.shape) != expected_shape:
        raise RuntimeError("selection has an invalid shape")
    if indices.dtype != torch.int64:
        raise RuntimeError("selection indices must use int64")
    if bool(((indices < 0) | (indices >= keys.shape[0])).any()):
        raise RuntimeError("selection contains out-of-range indices")
    ordered_indices = indices.sort(dim=-1).values
    if bool((ordered_indices[:, 1:] == ordered_indices[:, :-1]).any()):
        raise RuntimeError("selection contains duplicate indices")

    reference = query @ keys.transpose(0, 1)
    reference_values, reference_indices = reference.topk(top_k, dim=-1)
    selected_reference_values = reference.gather(-1, indices)
    if not bool(torch.isfinite(values).all()):
        raise RuntimeError("selection contains non-finite scores")
    if not torch.equal(
        selected_reference_values.sort(dim=-1).values,
        reference_values.sort(dim=-1).values,
    ):
        raise RuntimeError("selection does not contain the dense Top-K scores")
    if not torch.allclose(values, selected_reference_values, rtol=1e-5, atol=1e-5):
        raise RuntimeError("selection scores disagree with the dense reference")
    return {
        "passed": True,
        "topk_positions_match": torch.equal(
            ordered_indices, reference_indices.sort(dim=-1).values
        ),
        "max_score_abs_error": float((values - selected_reference_values).abs().max().item()),
        "tie_policy": "accept alternate indices only for equal dense reference scores",
        "score_rtol": 1e-5,
        "score_atol": 1e-5,
    }


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
            baseline_memory = torch_module.cuda.memory_allocated(device)
            torch_module.cuda.reset_peak_memory_stats(device)

        samples: list[float] = []
        for _ in range(repeat):
            _synchronize(device, torch_module)
            started = time.perf_counter()
            operation()
            _synchronize(device, torch_module)
            samples.append((time.perf_counter() - started) * 1000.0)

        peak = (
            int(torch_module.cuda.max_memory_allocated(device) - baseline_memory)
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
        "peak_temporary_memory_bytes": peak_memory_bytes,
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

    torch.manual_seed(config.seed)
    torch.set_num_threads(config.threads)
    query = torch.randn(config.query_rows, config.head_dim, device=device)
    keys = torch.randn(config.history, config.head_dim, device=device)

    dense_scores, dense_indices = dense_indexer(query, keys, config.top_k)
    chunked_scores, chunked_indices = chunked_indexer(
        query, keys, config.top_k, config.chunk_size
    )
    dense_check = check_selection(query, keys, config.top_k, (dense_scores, dense_indices))
    chunked_check = check_selection(query, keys, config.top_k, (chunked_scores, chunked_indices))

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
        "environment": collect_environment(),
        "config": asdict(config),
        "correctness": {
            "dense": dense_check,
            "chunked": chunked_check,
            "topk_positions_match": chunked_check["topk_positions_match"],
            "max_score_abs_error": chunked_check["max_score_abs_error"],
        },
        "paths": {
            "dense": _summary(dense_samples, dense_peak),
            "chunked": _summary(chunked_samples, chunked_peak),
        },
        "notes": [
            "Component-level synthetic measurement; not an end-to-end model result.",
            "Peak memory is temporary allocator memory observed during each timed path.",
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
        seed=args.seed,
        threads=args.threads,
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
