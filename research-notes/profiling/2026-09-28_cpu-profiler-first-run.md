# First PyTorch Profiler run: CPU indexer-shaped workload

## Command

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python scripts/tutorial_cpu_profiler.py
```

The environment variable is only a temporary workaround for the current
Windows Anaconda process, which reports duplicate Intel OpenMP runtimes. This
run is a learning trace, not a benchmark.

## Workload

- query rows: 32
- history length: 8,192
- head dimension: 64
- Top-K: 64
- repeated iterations: 5
- stages: score matrix multiplication, Top-K, value-cache gather

## First observation

On this CPU run, `aten::topk` used the most self CPU time (56.75%), followed by
the matrix multiplication (22.22%) and indexed gather (13.43%). The result is
specific to this CPU, tensor shape, and implementation. It does not establish
the bottleneck on H100/H200; it demonstrates how to read a profiler table.

## Trace

The Chrome trace is written to the ignored path:

```text
profiles/cpu-tutorial/indexer_cpu_trace.json
```

Open it with [Perfetto](https://ui.perfetto.dev/) when you want to inspect the
timeline. The next exercise will replace the synthetic operations with a small
vLLM workload on a GPU host.

