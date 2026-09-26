# Benchmark contract

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
