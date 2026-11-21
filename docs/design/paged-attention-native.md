<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Native FP16 PagedAttention

The native entry performs causal self-attention over a shared physical KV page
pool. It accepts separate Q, new K and new V, applies optional RoPE, appends K/V
in place, and computes attention without gathering the full history into a dense
cache. Prefill and decode are separate calls selected by `phase`; requests within
a call may have different lengths.

## Public interface and storage

The public C interface is declared in
[`hip_custom_kernels.h`](../../lib/Runtime/Kernels/include/hip_custom_kernels.h):

```c
typedef struct hip_paged_attention_f16_desc {
  const void *q, *k_new, *v_new;
  void *k_pool, *v_pool, *o;
  const int32_t *block_table, *cum, *past_seqlens;
  const void *cos_cache, *sin_cache;
  int32_t batch, tokens, num_heads, kv_num_heads, head_dim;
  int32_t num_pages, page_size, table_columns, max_position;
  hip_paged_attention_phase_t phase;
  int32_t do_rotary, rotary_interleaved, rotary_dim;
  int32_t local_window_size;
  float scale, softcap;
} hip_paged_attention_f16_desc;

int hip_paged_attention_f16_workspace_size(
    const hip_paged_attention_f16_desc *desc, size_t *bytes);
int hip_paged_attention_f16(
    void *stream, const hip_paged_attention_f16_desc *desc,
    void *workspace, size_t workspace_bytes);

typedef struct hip_paged_attention_f16_config_v2 {
  int splits, bkv, hb, qw, mt, nd;
} hip_paged_attention_f16_config_v2;

int hip_paged_attention_f16_configured_v2(
    void *stream, const hip_paged_attention_f16_desc *desc,
    void *workspace, size_t workspace_bytes,
    const hip_paged_attention_f16_config_v2 *config, size_t config_size);
int hip_paged_attention_f16_prepare_v2(
    void *stream, const hip_paged_attention_f16_desc *desc,
    void *workspace, size_t workspace_bytes,
    hip_paged_attention_f16_config_v2 *config, size_t config_size);
```

For `B=batch`, `T=tokens`, `Hq=num_heads`, `Hkv=kv_num_heads`, `D=head_dim`,
`P=num_pages`, `S=page_size` and `M=table_columns`:

- Q/O are contiguous FP16 `[T,Hq,D]`; new K/V are `[T,Hkv,D]`.
- Each pool is contiguous FP16 `[P,S,Hkv,D]` (PSHD). The int32
  `block_table[B,M]` gives each request its own logical-to-physical page mapping.
  Logical KV position `p` uses physical page `block_table[b,p/S]`, row `p%S`.
- Int32 `cum[B+1]` partitions packed query rows: request `b` owns
  `[cum[b],cum[b+1])`. It starts at zero and ends at T.
  `past_seqlens[B]` contains lengths **before this call**, not cache values.
  The resulting length is `past_seqlens[b]+cum[b+1]-cum[b]`; input lengths are
  not modified. Decode permits zero or one new token per request.

The caller allocates pages and performs copy-on-write. Read-only prefix pages
may be shared; appended physical slots must have a single writer and must not
overlap another request's historical read region. There is no allocator or
reference-count management in the entry.

## Computation and GQA reuse

For local query index `j`, absolute position is `past_seqlens[b]+j`. Causal
attention includes KV positions through that position. A positive
`local_window_size=W` retains at most W positions including the current token;
nonpositive values use the full causal history. `scale=0` selects `1/sqrt(D)`.

With `do_rotary=1`, FP16 cos/sin tables have shape `[max_position,D/2]` and use
the absolute position above. Only this call's Q and new K are rotated, with FP16
storage rounding; historical K and V are unchanged. New K/V are then written to
the mapped pool slots.

The production call chain is:

```text
hip_paged_attention_f16[_configured_v2]
  -> metadata validation -> RoPE / KV append
  -> gqa_paged_attention_f16_launch
       prefill: shared GQA v6x/v8x dispatch and body, PAGED=true
       decode:  paged split/reduce launch adapters -> shared GQA bodies
```

This reuses internal templates in `gqa_kernel.hip`, rather than calling the
dense public `hip_gqa_flash_*` C APIs or copying a separate attention core.
Paged specializations supply ragged row mapping and page-aware KV loads;
the Flash attention math and online softmax accumulation remain shared.
Physical page size is independent of the attention tile size.

## Workspace and asynchronous errors

The descriptor is host-resident; tensor, metadata and workspace pointers are
device-accessible on the supplied stream. The workspace query is CPU-only and
does not inspect device metadata. Workspace must be 256-byte aligned; it holds
device status, derived lengths, optional rotated Q, and decode partial results.
Q/new K/new V/pools/O require 16-byte alignment, int32 metadata 4-byte alignment,
and cos/sin 2-byte alignment.

A zero host return from a nonempty submission means launches were submitted.
After stream completion, copy the first workspace int32 to the host and require
`HIP_PAGED_ATTENTION_DEVICE_SUCCESS` before consuming O. Device validation covers
lengths, page bounds, positions and write conflicts; a metadata error suppresses
O/pool writes. Also check normal HIP stream execution errors. Host argument,
unsupported-option and workspace errors are returned directly.

Keep storage and metadata stable until completion. O, workspace and both pools
must be disjoint and must not overlap read-only inputs. Workspace reuse requires
completion of its preceding call. `T=0` is a validated no-launch no-op with zero
workspace requirement and no device-status write.

## Configuration and LUTs

The no-config entry uses a geometry heuristic; decode adapts split count to
device-side effective lengths/window, up to 64 splits. It does not implicitly
tune. `configured_v2` validates and executes a supplied final geometry, without
metadata readback or implicit tuning. `config_size` must match the v2 struct.
Decode `splits=0` selects adaptive dispatch; positive values select explicit
power-of-two splits. The legacy two-int configured ABI remains available.

`prepare_v2` runs a validated append, synchronizes, and tunes shared GQA launches
on the actual buffers. It can modify valid pools/O and belongs outside hot
submissions and graph capture. With invalid metadata it returns a safe config;
the workspace status remains authoritative. Callers own configuration reuse.

The C++ `preparePagedAttention` resolver adds exact LUT lookup and explicit
online preparation on a miss. Page LUTs use schema2, ABI `paged_fp16_pshd_v2`,
PSHD/FP16 identity and final v2 geometry. Dense GQA tables are incompatible.
Native UTs support external FB input and the shared ranking/refresh command
family; the supplied empty JSON is a new-table seed, not a measured table.
Loading, compilation and selection stay outside timed configured calls.

## Scope and validation

The first version supports FP16, D=64/96/128/256, and integer Hq/Hkv. Decode
supports ratios 1/2/3/4/5/6/8/16, plus 9 for D=128. RoPE supports full-D,
non-interleaved split-half rotation only. Nonzero softcap and D=256 prefill with
a positive window are rejected. The operator is causal; no bidirectional mode
or ONNX lowering/EP integration is included.

Native prefill/decode leaves provide `COVERAGE=1|2|3`, custom shapes/options,
and `TIMING=screen|standard|confirm`. Checks include independent FP32 references,
FP16 cache writes, untouched storage/guards, ragged and empty requests, shared
prefixes, page boundaries and expected errors. The original-Flash comparison
adapter separately compiles a caller-specified original source with isolated
symbols; it is a test baseline, not a production Page implementation.

CMake integration registers the Page source, schema, resolver and embedded LUT
registry. This build wiring is separate from native UT validation: a full
EP/package CMake build has not yet been compiled for this change. See the
[prefill UT README](../../lib/Runtime/Kernels/unit_test/paged_attention/prefill/Xfp16_Wfp16_Yfp16/README.md)
and [decode UT README](../../lib/Runtime/Kernels/unit_test/paged_attention/decode/Xfp16_Wfp16_Yfp16/README.md)
for reproducible native commands and measurement details.
