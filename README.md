# V4.1-Sparse-Indexer

Performance research for DeepSeek-V4.1 sparse indexer execution on Hopper GPUs.

The first phase focuses on the vLLM execution path. SGLang is kept as a later
comparison point after the vLLM baseline and correctness harness are stable.

## Working title

**Memory-Bounded Sparse Indexer Execution for DeepSeek-V4.1 on Hopper GPUs: A Comparative Study of SGLang and vLLM**

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

## Current scope: vLLM first

- lock a vLLM commit and document the actual DeepSeek-V4.1/Hopper path;
- build a small reference implementation for index and Top-K correctness;
- reproduce dense, chunked and candidate-aware indexer baselines;
- measure peak temporary memory, HBM traffic, kernel launches and latency;
- only then evaluate fused scoring/Top-K or paged-KV address mapping changes.

SGLang comparison will be added after the vLLM path has a reproducible baseline.

## Repository map

- [`docs/vllm-first-plan.md`](docs/vllm-first-plan.md): first-phase plan and boundaries
- [`configs/vllm-baseline.yaml`](configs/vllm-baseline.yaml): version and experiment lock
- [`benchmarks/README.md`](benchmarks/README.md): benchmark entry-point contract
- [`src/sparse_indexer/reference.py`](src/sparse_indexer/reference.py): framework-independent reference semantics
- [`tests/test_reference.py`](tests/test_reference.py): Top-K and paged-KV address correctness tests

Run the current reference tests with:

```powershell
python -m unittest discover -s tests -v
```

## Status

Repository scaffold created. No performance claim has been made yet.
