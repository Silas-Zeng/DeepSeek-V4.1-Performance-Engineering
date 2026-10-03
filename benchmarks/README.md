# Benchmark contract

## Reproducible component baseline

`indexer_benchmark.py` provides the first runnable baseline in this project. It
compares a dense score matrix with a chunked score path, verifies that both
return the same Top-K positions, and records median and p95 latency in JSON.
The default shapes mirror the initial experiment matrix while remaining small
enough to run on a CPU host.

```bash
python benchmarks/indexer_benchmark.py \
  --history 8192 --query-rows 32 --head-dim 64 --top-k 64 \
  --chunk-size 1024 --warmup 2 --repeat 10 \
  --output results/indexer-benchmark.json
```

Use `--device cuda` on a CUDA host. The output includes the exact configuration,
Torch version, correctness result, per-path samples, median, p95, and CUDA peak
allocator memory when available. `results/` is ignored so large or host-specific
outputs do not enter Git; commit a reduced summary and a research note when a
run is worth preserving.

This is component-level evidence only. It does not measure vLLM/SGLang, paged
address mapping, or end-to-end TTFT/TPOT.

Run the focused tests with:

```bash
pytest -q tests/test_indexer_benchmark.py tests/test_tutorial_cpu_profiler.py
```

## Environment and full experiment contract

The repository provides the environment metadata collector and an Nsight
Systems wrapper. Install the lightweight analysis requirements first, then run
the collector on every framework/GPU environment:

```bash
python scripts/collect_environment.py --output reports/environment.json
```

Use `configs/workload-matrix.yaml` as the starting workload contract. Do not
compare frameworks until their model, precision, parallelism and workload
shapes are recorded and equivalent.

The framework benchmark runner will be extended after the vLLM source path is
pinned. The synthetic baseline above remains useful for checking correctness
and measurement plumbing before a GPU host is available.

Every result should record:

```text
framework commit
CUDA / driver / PyTorch versions
GPU model and clocks
input shape and history length
candidate count and Top-K
execution mode
warmup and measured iterations
latency, workspace, HBM traffic and kernel count
correctness result
```

Use separate commands for normal timing and Nsight profiling. Profiling output
belongs outside the repository unless it has been reduced to a small summary.
