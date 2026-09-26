# vLLM-first research plan

## Why start with vLLM

The first implementation target is vLLM's DeepSeek sparse-indexer path. vLLM
already contains the indexer, candidate-consuming MQA path, paged-KV mapping,
and kernel-level correctness tests. This project must measure and extend those
paths rather than reimplement them as a generic Top-K library. SGLang becomes a
controlled comparison once the vLLM baseline can be reproduced.

## Scope boundary

The first milestone covers the indexer component only:

1. prepare the indexer query and compressed key inputs;
2. compute indexer scores over a complete or candidate-restricted history;
3. select Top-K positions;
4. map logical positions to paged KV-cache addresses.

The milestone does not change model weights, the Top-K value, quantization
precision, candidate semantics or the downstream sparse-attention equation.

## Baselines

The benchmark should make three existing or experimentally composed paths explicit:

- **dense**: score the full history and select Top-K;
- **chunked**: bound the score workspace by processing history chunks;
- **candidate-aware**: score only an existing candidate set and resolve its
  paged KV addresses.

The candidate-aware path is already implemented in vLLM for supported hardware
through `SparseMQAIndexer`; the research task is to measure its complete cost
and identify cases that still fall back to dense or chunked execution. It must
not be presented as a new algorithmic selection method.

## First experiment matrix

| Dimension | Values |
| --- | --- |
| GPU | H100/H200, SM90 |
| History length | 8K, 32K, 128K, 1M where supported |
| Query rows | 1, 32, 512 |
| Candidate count | 256, 1K, 4K, 16K |
| Top-K | 32, 64, 128, 512 |
| Batch/concurrency | 1, 2, 8 |
| Execution | eager, CUDA Graph where supported |

The first run can use synthetic tensors with model-shaped dimensions. A real
model run is required before making an end-to-end TTFT or TPOT claim.

## Required measurements

Measure the whole indexer call, not only a GEMM or `topk` kernel:

- score and Top-K latency;
- candidate gather and logical-to-paged address mapping latency;
- peak temporary workspace;
- cumulative HBM read/write bytes;
- kernel launch count and synchronization points;
- Top-K exactness and index mapping correctness;
- prefill TTFT and decode TPOT when the full model is available.

Keep profiler runs separate from normal benchmark runs because instrumentation
changes launch and memory behavior.

## Correctness contract

Reuse vLLM's existing DeepSeek-V4.1 indexer tests and compare any new path at
each boundary:

- selected indices and valid-length handling;
- score error and tie behavior;
- page/block address mapping;
- sparse-attention input positions;
- final logits on a small real-model case.

Add only missing cases discovered during profiling, such as ragged batches,
candidate/page-table layout changes, and repeated CUDA Graph replays with
changed metadata.

## Decision gate

Do not write a new CUDA kernel until profiling shows that a stable share of the
complete indexer time is spent in a kernel or data movement that the proposed
change can affect. If gather, remapping or graph metadata dominates, optimize
that path instead.

The first milestone is a reproducible vLLM baseline and a cost breakdown. It is
successful even if the candidate-aware path does not win for every context
length; an optimization is only justified when vLLM's existing paths leave a
measured gap.
