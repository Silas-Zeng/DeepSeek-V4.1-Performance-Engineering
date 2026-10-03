# Synthetic dense versus chunked baseline

## Environment

- Date: 2026-10-03
- Host: Windows x86_64, CPU-only PyTorch
- PyTorch: `2.8.0+cpu`
- Device: `cpu`
- Model: synthetic tensors shaped like the initial indexer exercise

## Command

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python benchmarks/indexer_benchmark.py --warmup 1 --repeat 3 --output results/indexer-benchmark.json
```

## Configuration

The run used 32 query rows, history length 8,192, head dimension 64, Top-K 64,
and a chunk size of 1,024. The script compared the dense full-score path with
the streaming chunked path and checked selected positions before timing.

## Observation

Both paths selected the same positions and had a maximum score error of `0.0`.
The dense median was `0.485 ms`; the chunked median was `7.1487 ms` on this
CPU-only run. The result is a correctness and measurement-plumbing baseline,
not evidence that either path wins on a GPU or in a full model.

## Next experiment

Repeat the same JSON-producing run on a pinned CUDA environment, add peak
allocator memory to the comparison, and then connect the measured path to the
framework-specific vLLM indexer before discussing end-to-end latency.
