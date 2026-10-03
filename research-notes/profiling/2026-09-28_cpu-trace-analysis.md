# Analysis of the first CPU trace

## What the JSON contains

The trace contains 115 events, including 71 duration events. It records CPU
operator names, input shapes, memory annotations, and the custom ranges for the
three synthetic indexer stages. It has no CUDA activity because this run used a
CPU-only PyTorch build.

## Observed time breakdown

Using the operator self-time table:

| Operator | Self time | Share |
| --- | ---: | ---: |
| `aten::topk` | 7.838 ms | 56.75% |
| `aten::mm` | 3.073 ms | 22.22% |
| `aten::index` | 1.855 ms | 13.43% |

These three operators account for about 92% of the measured self CPU time.
The wrapper ranges are useful for reading the timeline, but the `aten::` rows
are the better rows for avoiding double counting nested operations.

## Shape-level interpretation

- score: `[32, 64] @ [64, 8192] -> [32, 8192]`;
- Top-K: select 64 entries from each of 32 rows;
- gather: index a value cache shaped `[8192, 128]`.

The score matrix contains 262,144 float values, about 1 MiB for one tensor. The
trace therefore shows the cost of materializing the full-history score matrix
before Top-K. A chunked or candidate-aware implementation would be expected to
change this workspace and the Top-K input shape, but that must be measured and
checked for exactness.

## What this does not prove

This is not a DeepSeek-V4.1, vLLM, Hopper, or CUDA result. The CPU backend,
small tensor shape, five measured iterations, and lack of warmup all limit the
claim. The trace is a tool-learning baseline and a workload-shape sanity check.

## Next experiment

Add a chunked score path with the same query, history, and Top-K values. Warm up
three iterations, measure at least twenty, compare median latency and peak
temporary memory, and verify that the selected indices match the dense path.

