# Existing vLLM work to reuse

This project does not claim to introduce a new Top-K or paged-address
algorithm. vLLM already has the relevant model path and tests.

## Existing pieces

- `SparseAttnIndexer` provides the full-history score/mask path.
- `SparseMQAIndexer` consumes candidate blocks and runs sparse MQA-logits
  kernels on supported hardware.
- The candidate-consuming path gathers paged prefill K into a bounded workspace,
  then selects Top-K inside the candidate blocks.
- DeepSeek-V4.1 attention owns index-K cache setup, candidate buffers and the
  eager-break boundary around sparse indexer plus MLA attention.
- vLLM has kernel and gating tests for the DeepSeek sparse-indexer paths.

References:

- [SparseMQAIndexer API](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/sparse_mqa_indexer/)
- [V4.1 attention integration](https://github.com/vllm-project/vllm/blob/main/vllm/models/deepseek_v41/attention.py)
- [MLA indexer metadata](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/indexer.py)
- [SM90 fused prefill RFC](https://github.com/vllm-project/vllm/issues/53563)

## What this repository should investigate instead

1. **Path selection:** when vLLM uses dense, chunked or candidate-consuming
   execution as context length, batch shape and hardware change.
2. **Remaining memory costs:** whether score workspaces, gathered K buffers or
   graph metadata still grow with the configured maximum rather than live work.
3. **Ragged and graph coverage:** whether dynamic request lengths and CUDA Graph
   replay preserve the same efficient path as uniform synthetic cases.
4. **Complete cost:** whether candidate-only score savings are lost to gather,
   page mapping, Top-K and launch overhead.

Any proposed change must be compared against the existing vLLM implementation
and its own correctness tests. A standalone reimplementation in this
repository is not a contribution by itself.
