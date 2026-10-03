# Contributing

The project accepts reproducible measurements, source-reading notes, and small
changes to the benchmark tooling. Keep a change narrow enough that its evidence
can be reviewed in one commit.

## Local validation

Install the lightweight analysis dependencies and a CPU PyTorch build, then run:

```bash
python -m pytest -q
python benchmarks/indexer_benchmark.py --output results/indexer-benchmark.json
```

The benchmark must report matching dense and chunked Top-K positions before its
timings are useful. Use `--device cuda` only on a host with a compatible CUDA
installation. Do not commit files under `results/`, `profiles/`, or raw Nsight
output.

## Research notes

Add dated notes under `research-notes/`. Record the source URL or commit,
hardware and software assumptions, and label statements as observations,
inferences, or hypotheses. A synthetic or CPU result must not be presented as
an end-to-end DeepSeek-V4.1 claim.
