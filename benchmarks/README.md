# Benchmark contract

The repository provides the environment metadata collector and an Nsight
Systems wrapper. Install the lightweight analysis requirements first, then run
the collector on every framework/GPU environment:

```bash
python scripts/collect_environment.py --output reports/environment.json
```

Use `configs/workload-matrix.yaml` as the starting workload contract. Do not
compare frameworks until their model, precision, parallelism and workload
shapes are recorded and equivalent.

The benchmark runner will be added after the vLLM source path is pinned.

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
