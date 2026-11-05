<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# QSA sparse attention

QSA is the sparse attention path made of two `com.microsoft` contrib ops:

- `SparseAttentionIndexer` with `policy_mode = qsa` scores pooled key blocks
  for every query row and writes the selected token indices.
- `DynamicSparseAttention` in `selected_only` mode attends over those tokens
  of the main KV cache.

Both ops run on custom HIP kernels and are wave32 (RDNA) only. The supported
geometries are listed in
[supported-operations.md](../supported-operations.md).

## Components

| Layer | Indexer | DSA |
|---|---|---|
| ONNX to HIP | [SparseAttentionIndexerConversion.cpp](../../lib/Conversion/OnnxToHip/SparseAttentionIndexerConversion.cpp), [IndexerRotaryFold.cpp](../../lib/Conversion/OnnxToHip/IndexerRotaryFold.cpp) | [DynamicSparseAttentionConversion.cpp](../../lib/Conversion/OnnxToHip/DynamicSparseAttentionConversion.cpp) |
| HIP op | `hip.sparse_attention_indexer` in [HipOps.td](../../include/hip/Dialect/IR/HipOps.td) | `hip.dynamic_sparse_attention` |
| HIP to LLVM | [SparseAttentionIndexerLowering.cpp](../../lib/Conversion/HipToLLVM/SparseAttentionIndexerLowering.cpp) | [DynamicSparseAttentionLowering.cpp](../../lib/Conversion/HipToLLVM/DynamicSparseAttentionLowering.cpp) |
| Runtime wrapper | [sparse_attention_indexer.cpp](../../lib/Runtime/real/sparse_attention_indexer.cpp) | [dynamic_sparse_attention.cpp](../../lib/Runtime/real/dynamic_sparse_attention.cpp) |
| Kernels | [sparse_attention_indexer_kernel.hip](../../lib/Runtime/Kernels/hip/sparse_attention_indexer_kernel.hip) | [dynamic_sparse_attention_kernel.hip](../../lib/Runtime/Kernels/hip/dynamic_sparse_attention_kernel.hip) |

The indexer follows the ORT CUDA kernel's numerics: pooled keys, RMSNorm, RoPE
and scores are fp32, and blocks are ranked by score descending, then by block
index ascending.

## Indexer rotary-table fold

### Problem

An exporter with MRoPE positions builds the indexer's
`cos_cache` and `sin_cache` in the graph from `rotary_position_ids`
`[M, B, X]` and two base tables `[Y, R/2]`:

```text
pos_k = Gather(rotary_position_ids, k, axis 0)          [B, X]
g_k   = Gather(base, pos_k, axis 0)                     [B, X, R/2]
w     = Where(mask_2, g_2, Where(mask_1, g_1, g_0))     [B, X, R/2]
cache = Concat(w, w, axis -1)                           [B, X, R]
```

The masks are constants and pick the section of each frequency.
onnxruntime-genai binds the full `max_length` history as `X`, so this subgraph
rewrites `B * X * R` elements of both tables on every call, although the
indexer only rotates rows at present positions. At decode the rebuild costs
far more than the indexer itself and does not shrink with the context.

### Fold

`IndexerRotaryFold` is a pre-lowering pattern that runs next to the other
`populate*FoldPatterns` in [OnnxToHip.cpp](../../lib/Conversion/OnnxToHip/OnnxToHip.cpp).
It matches a qsa `SparseAttentionIndexer` when:

- `cos_cache` and `sin_cache` are each `Concat(w, w)` on the last axis of a
  `Where` tree whose leaves are `Gather(base, Gather(ids, k))`.
- Every `Where` condition is an integer constant of `R/2` elements, so each
  frequency's section is known at compile time.
- Both tables use the same `ids`, the same section per frequency, rank-2 bases
  of the query's element type, and section indices below `min(M, 256)`.

The rewritten `onnx.Custom` passes the base tables in the cache slots, appends
`rotary_position_ids` at operand index 14, and records the section map as
`hipdnn.rotary_sections`. The converter then emits:

```mlir
hip.sparse_attention_indexer(%ctx) ins(..., %cos_base, %sin_base, ...)
    past_sequence_length(%psl : tensor<1xi32>)
    rotary_position_ids(%ids : tensor<3x?x?xi64>)
    outs(...) {rotary_sections = array<i32: 0, 1, 2, ...>, ...}
```

The fold erases the dead `Concat`, `Where` and `Gather` ops. It leaves the
constants to constant lowering. The fold doesn't apply when any part of the
pattern differs, such as a runtime mask, mismatched cos and sin sections, or
another table producer. In those cases the materialized tables reach the
indexer unchanged.

### Semantics

With `rotary_position_ids`, channel `d` of row `t` in batch `b` uses base row
`ids[rotary_sections[d mod R/2], b, t]`. A negative position wraps once, as
ONNX `Gather` indices do, and the row is clamped to `[0, Y - 1]`. Row `t` is
clamped to `[0, X - 1]` in the same way as the unfused `[B, X, R]` table.
Every node in the replaced subgraph is a copy, so the indexer's outputs are
bit-identical with and without the fold.

### Lowering and runtime

- The lowering emits the section map as an internal byte global
  (`__hip_sai_rotary_sections_<hex>`), shared by every indexer with the same
  map, and passes the ids pointer and the `M`, `B` and `X` sizes.
- `wrap_sparse_attention_indexer` and `hip_sparse_attention_indexer` take
  `rotary_position_ids`, `rotary_sections` and the id sizes after `scale`. Null
  ids select the table path.
- The kernels take a `Rope<T>` argument holding either the table strides or
  the ids, the strides and the section map. `ropeAngle` reads one base entry per
  channel instead of a materialized table row.

## Remaining graph glue

These exported subgraphs around QSA still lower op by op:

- `GreaterOrEqual`, `Where` and `ReduceSum` turn `selected_indices` into the
  per-row counts that DSA consumes.
- The q/k `SimplifiedLayerNormalization` and `MRotaryEmbedding` before DSA,
  and the sigmoid gate after it.

## Validation

- LIT:
  - [test_indexer_rotary_fold.mlir](../../test/lit/Conversion/onnx-to-hip/test_indexer_rotary_fold.mlir)
    checks the full converted IR of the fold and of each case it rejects.
  - [test_sparse_attention_indexer_rotary_ids.mlir](../../test/lit/Conversion/hip-to-llvm/test_sparse_attention_indexer_rotary_ids.mlir)
    checks the runtime call and the shared section-map global.
  - [hip-sparse-attention-indexer-verifier.mlir](../../test/lit/Dialect/hip-sparse-attention-indexer-verifier.mlir)
    checks the verifier.
- Numeric: `test_rotary_fold_matches_tables` in
  [test_sparse_attention_indexer.py](../../test/numeric/tests/test_sparse_attention_indexer.py)
  requires the folded graph and the materialized tables to agree bit for bit.
- Kernel resources: run
  [check_kernel_resources.py](../../lib/Runtime/Kernels/tools/check_kernel_resources.py)
  after changing either kernel.

Run numeric tests with `HIPDNN_EP_STRICT=1`. Otherwise a failed compile can
pass by falling back to the CPU.
