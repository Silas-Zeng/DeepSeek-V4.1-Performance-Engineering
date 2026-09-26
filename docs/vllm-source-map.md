# vLLM DeepSeek-V4.1 Indexer source map

This map records the first vLLM path to inspect. Links point at `main` for
orientation only; an experiment must replace them with a pinned commit hash.

## Entry points

| Stage | vLLM location | What to record |
| --- | --- | --- |
| Model attention integration | [`vllm/models/deepseek_v41/attention.py`](https://github.com/vllm-project/vllm/blob/main/vllm/models/deepseek_v41/attention.py) | indexer construction, source-layer sharing, candidate buffers, eager-break boundary |
| Indexer implementation | [`vllm/model_executor/layers/sparse_attn_indexer.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/sparse_attn_indexer.py) | score/selection orchestration and output index semantics |
| MQA/indexer kernels | [`vllm/model_executor/layers/sparse_mqa_indexer.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/sparse_mqa_indexer.py) | candidate/full-history score path and kernel dispatch |
| MLA backend metadata | [`vllm/v1/attention/backends/mla/indexer.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/indexer.py) | prefill/decode metadata, chunking and cache layout |
| V4.1 model-specific attention | [`vllm/models/deepseek_v41/attention.py`](https://github.com/vllm-project/vllm/blob/main/vllm/models/deepseek_v41/attention.py) | `DeepseekV4Indexer`, index-K cache ownership and sparse-attention call order |

## Observed V4.1 call order

The model-specific attention layer prepares the query and cache writes, invokes
the indexer, then enters sparse attention. The first implementation task is to
confirm this sequence at a pinned commit and instrument its boundaries:

```text
input hidden states
  -> query / KV projections
  -> indexer query and indexer weights
  -> index-K cache publication
  -> indexer score + Top-K
  -> logical index / page mapping
  -> sparse MLA attention
```

The indexer and the downstream MLA path must be timed separately and together.
The latter is the only number that can support an end-to-end claim.

## First hypotheses to test

1. Long-context prefill can make score materialization and temporary workspace
   larger than the final Top-K output by a large factor.
2. Chunking bounds peak memory but may increase cumulative HBM traffic and
   launch overhead.
3. Candidate-aware scoring can reduce arithmetic while making gather and paged
   address mapping more expensive.
4. Eager and CUDA Graph executions may prefer different workspace sizes and
   dispatch paths.

## Existing evidence and open boundaries

- [vLLM #45663](https://github.com/vllm-project/vllm/issues/45663) reports a
  long-context sparse-indexer temporary-buffer OOM.
- [vLLM #53563](https://github.com/vllm-project/vllm/issues/53563) proposes an
  SM90 exact fused score + Top-K prefill path that avoids materializing the full
  score tensor. Treat it as related work and a comparison point, not as a
  claimed contribution of this repository.
- Candidate filtering, paged KV and CUDA Graph metadata must be measured
  together. A standalone `topk` speedup is insufficient evidence.

## Instrumentation boundary

The first benchmark should expose timestamps around:

```text
indexer input preparation
score kernel(s)
score chunk materialization / release
Top-K
candidate or page address mapping
output publication
```

Use CUDA events for normal timing. Use Nsight Systems/Compute only for selected
representative cases, because profiler instrumentation changes the execution
cost.
