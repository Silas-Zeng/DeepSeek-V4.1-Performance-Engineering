# DeepSeek-V4.1 Inference Performance Engineering

*Bottleneck Localization and Optimization Across vLLM and SGLang*

Measurement-driven performance engineering for DeepSeek-V4.1 inference. The
sparse indexer is the initial case study, with vLLM and SGLang evaluated under
matched workloads and hardware configurations.

## Research focus

The project follows a complete performance loop:

```text
baseline -> bottleneck localization -> targeted optimization -> validation
```

## Research question

At long context lengths, can a memory-bounded indexer execution policy reduce
temporary score materialization and data movement without changing the selected
positions or the model's sparse-attention semantics?

The initial scope is deliberately narrow:

```text
indexer query/key preparation
  -> score computation
  -> Top-K selection
  -> candidate/page address mapping
```

The project will measure the complete indexer component, including workspace
allocation, candidate gathering and index remapping. A faster isolated kernel
is not treated as an end-to-end improvement until the full path is measured.

## Initial scope: sparse indexer across frameworks

- lock framework commits and document the actual DeepSeek-V4.1 execution paths;
- align model configuration, precision, parallelism, workload shapes and hardware;
- measure indexer, sparse MQA, Top-K, metadata and downstream attention stages;
- localize compute, memory, launch, synchronization and communication bottlenecks;
- apply one targeted optimization at a time and validate component and
  end-to-end results.

Framework-specific capability differences will be recorded instead of being
treated as performance regressions.

## Repository map

- [`environment/README.md`](environment/README.md): local and GPU-host setup
- [`environment/frameworks.md`](environment/frameworks.md): isolated vLLM and SGLang source environments
- [`configs/framework-environments.yaml`](configs/framework-environments.yaml): framework commit lock
- [`configs/workload-matrix.yaml`](configs/workload-matrix.yaml): matched workload contract
- [`docs/vllm-first-plan.md`](docs/vllm-first-plan.md): first-phase plan and boundaries
- [`configs/vllm-baseline.yaml`](configs/vllm-baseline.yaml): version and experiment lock
- [`benchmarks/README.md`](benchmarks/README.md): benchmark entry-point contract
- [`docs/existing-vllm-work.md`](docs/existing-vllm-work.md): existing vLLM work
  and the remaining research gap

## Quick CPU profiler

A small CPU-only trace is included to inspect the indexer pipeline before a CUDA host is available:

```bash
python scripts/tutorial_cpu_profiler.py
python scripts/tutorial_cpu_profiler.py --output profiles/custom/indexer_cpu_trace.json --repeat 3
```

The script writes a Chrome trace to `profiles/cpu-tutorial/indexer_cpu_trace.json` under the repository root by default. It accepts a custom relative or absolute output path and supports tuning the number of profile iterations.

## Status

Repository scaffold and scope correction created. No performance claim has been made yet.
