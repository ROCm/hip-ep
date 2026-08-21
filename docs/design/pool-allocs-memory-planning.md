<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# GPU memory planning and PoolAllocs

**Date:** 2026-07-24
**Document Type:** Design
**Status:** Implemented
**Related:** [pipeline_pass_menu.md](../pipeline_pass_menu.md), [compiler-runtime-contract.md](compiler-runtime-contract.md), [output-allocator-design.md](output-allocator-design.md), [constant-handling-design.md](constant-handling-design.md)

## Purpose

The compiler uses a session-owned GPU memory plan for transient intermediate tensors. It replaces function-local `memref.alloc` operations with views into one or more grow-on-demand byte pools, avoiding per-inference `hipMalloc`/`hipFree` traffic.

Three constraints shape the design:

1. **Steady-state inference performs no transient allocator calls.** Pools are reused across invocations.
2. **Memory grows to a high-water mark and does not shrink during the session.** A larger input shape may grow a pool after synchronizing the stream.
3. **Pool size and offset SSA must obey dominance.** In the single-block functions PoolAllocs accepts, dominance is textual order: a pool acquisition must follow every in-block dynamic-size definition it consumes and precede every allocation it replaces.

When all dynamic sizes can be computed before the earliest pooled allocation, the function uses one pool. When a size depends on a non-speculatable value produced later in the block, PoolAllocs creates another **dominance domain** with its own acquisition point and runtime buffer.

## Ownership model

- Pool memory belongs to `RuntimeState`, not to an inference call.
- `hip.get_pool(%ctx, %size) {site_id = S, domain_id = N}` obtains the backing
  buffer for a function-local domain.
- Pool buffers grow on demand and are freed during session cleanup.
- Pooled `memref.alloc` operations become `memref.view` operations; matching per-allocation deallocations are removed.
- Graph outputs are not pooled. `hip-use-output-allocator` rewrites them to `hip.alloc_output`.
- Tiny host-written shape buffers are not pooled. `hip-materialize-host-scalars` redirects them to session-owned host-mapped scratch.

`site_id` is the function's ordinal in top-level module symbol order. This
deterministic module-level assignment is collision-free and independent of
symbol spelling; no string hash is used. `domain_id` remains local to the
function, preserving existing per-domain packing.

The runtime keys storage by `(site_id, domain_id)`. A caller's local domain zero
and an outlined helper's local domain zero therefore have distinct backing
allocations while both are live; growing the helper cannot invalidate the
caller's pool.

## Pipeline relationship

[pipeline_pass_menu.md](../pipeline_pass_menu.md) documents pass names and extension anchors. The order source of truth is `lib/Dialect/Transforms/Pipelines.cpp`; the passes immediately surrounding memory planning have distinct roles:

```text
... hip-infer-shapes
→ canonicalize → CSE
→ hip-split-duplicate-dps-inits
→ hip-resolve-tensor-dims
→ one-shot-bufferize
→ hip-loop-body-to-out-params
→ hip-use-output-allocator
→ hip-fix-loop-accumulator-offset
→ CSE → canonicalize
→ convert-linalg-to-loops
→ hip-optimize-memrefs
→ hip-promote-strided-operands
→ hip-materialize-host-scalars
→ hip-resolve-memref-dims
→ CSE
→ hip-hoist-alloc-size-arith
→ hip-pool-allocs
→ convert-bufferization-to-memref
→ CSE → canonicalize
→ hip-lower-allocs
```

### Correctness and ABI preparation

`hip-promote-strided-operands` materializes identity-layout temporaries for non-identity-layout DPS-input memrefs. Selecting by interface covers HIP and plugin runtime consumers without coupling the pass to a dialect or operation-name prefix. The pass must run before pooling so its temporaries become pool views.

`hip-use-output-allocator` must run before pooling so returned buffers are rewritten to runtime-owned `hip.alloc_output` operations rather than absorbed into the transient pool. The production pipeline intentionally omits ownership-based buffer deallocation because every transient is pooled and outputs are runtime-owned. See [output-allocator-design.md](output-allocator-design.md).

`convert-linalg-to-loops` must run before host-scalar materialization: a rank-zero `linalg.fill` must become `memref.store` before the host-I/O candidate scan.

`hip-materialize-host-scalars` redirects small integer/index buffers with host accesses to host-mapped scratch. It runs after allocation-producing rewrites and before PoolAllocs so host-written memory never becomes a view into device-only pool storage.

### Pool-quality preconditions

`hip-resolve-memref-dims` folds post-bufferization `memref.dim` queries through view chains to root-buffer dimensions. These views may be created by bufferization or operand promotion, so this pass runs after both.

The following CSE removes repeated size queries exposed by late allocation and
view rewrites.

`hip-hoist-alloc-size-arith` moves pure dynamic-size arithmetic above
the earliest used allocation when the complete operand cone can move safely.

These preconditions preserve correctness when omitted, but omission may create
more dominance domains and increase peak memory.

Pre-bufferization `hip-resolve-tensor-dims` serves a related purpose for tensor reshape chains. It lets upstream reification and canonicalization reduce `tensor.dim` queries before they become memref-level size arithmetic. Coverage of standard tensor reshape operations requires `tensor::registerInferTypeOpInterfaceExternalModels` on the dialect registry.

## PoolAllocs algorithm

PoolAllocs operates on a single-block `func.func` whose first argument is `!hip.context`.

Input contract:

- only used `memref.alloc` operations that are direct children of the entry block are candidates;
- unused allocations are ignored;
- allocations nested inside region-bearing operations are not pooled by the enclosing function's pass;
- a single used allocation remains a valid candidate.

### Phase 1: liveness

For each used `memref.alloc`, compute:

```text
[definition index, last transitive aliased use]
```

`BufferViewFlowAnalysis` follows view-like aliases, including `arith.select` through an externally registered view-flow model. Deallocation operations are not treated as data uses. Nested-region users are mapped to their enclosing entry-block operation; users that cannot be mapped are conservatively treated as live to block end.

Two allocations may share an address range only when their intervals do not overlap.

### Phase 1.5: dominance domains

Allocations are processed in textual order. An allocation joins the most recent domain for which one insertion point can:

- follow every in-block dynamic-size definition used by the domain; and
- precede every allocation in the domain.

If no existing domain is feasible, a new domain is created. Domain count is unbounded; an unusually high count emits a performance advisory rather than failing compilation.

Minimal two-domain example:

```mlir
%n = memref.dim %arg, %c0
%a = memref.alloc(%n) : memref<?xf16>       // domain 0
// ...
%late = memref.load %host_shape[%c0] : memref<1xindex>
%b = memref.alloc(%late) : memref<?xf16>    // domain 1
```

`%late` cannot move above `%a` because it reads mutable memory. No one insertion point can follow `%late` while preceding both allocations, so `%b` opens a second domain.

### Phase 2: static and dynamic split

Each domain is partitioned into:

- fully static allocations, whose byte sizes are compile-time constants;
- dynamic allocations, whose byte sizes depend on runtime SSA values.

### Phase 3: static packing

Static allocations use greedy best-fit gap packing in byte space. Allocations with disjoint lifetimes may reuse the same offset.

### Phase 4: dynamic packing

Dynamic allocation size is:

```text
staticFactor × product(dynamic dimension SSA values)
```

where `staticFactor` is the element byte width multiplied by all static dimensions.

With `lifetime-only=true`, each allocation forms a group. The pass assigns a
group to the first slab whose existing groups have no overlapping member
lifetimes, or creates a new slab. Groups in a slab share its base. The slab
width is the runtime maximum of their footprints, rounded to pool alignment
when required.

With `lifetime-only=false`, allocations with SSA-identical ordered dynamic-size
operands and alignment-multiple `staticFactor`s are best-fit-packed into groups
before the same slab assignment. Non-alignment-multiple allocations use
first-fit bins keyed by `{staticFactor, dynamic operands}`.

The modes differ in packing granularity. For a common factor `F`, one
allocation of size `512*F` that is disjoint from two mutually overlapping
allocations of size `256*F` requires `768*F` bytes in lifetime-only mode and
`512*F` bytes in grouped mode.

### Phase 5: IR emission

For each domain, the pass emits:

- dynamic size arithmetic;
- one `hip.get_pool`;
- offset arithmetic;
- one `memref.view` per original allocation.

The domain pool size is:

```text
static prefix
+ sum(dynamic slab runtime widths)
+ sum(small-bucket aligned size × bin count)
```

Each original allocation is replaced at its original position, preserving its dynamic dimensions and memref type.

A domain with no dynamic groups and a zero-byte static prefix emits no `hip.get_pool` because there is nothing to reserve. Multi-domain metadata still records a zero static prefix for that domain when present.

`hipdnn.buffer_offsets` records `-1` when a view offset is runtime SSA; non-negative values are compile-time byte offsets within the allocation's domain.

## Empty and single-allocation functions

- A function with no body blocks (`funcOp.empty()`) is left unchanged and emits no pool module attributes.
- If the entry block has no used `memref.alloc` candidates—because it has no allocations or every allocation is unused—PoolAllocs emits zeroed legacy pool attributes and makes no IR replacement.
- A single used allocation is still pooled. Leaving it to lower as an unsupported per-inference device allocation is not a valid fallback.

## Compiler/runtime contract

PoolAllocs writes MLIR module attributes:

| Attribute | Meaning |
|---|---|
| `hipdnn.pool_size` | Domain 0 static prefix; dynamic contributions live only in SSA feeding `hip.get_pool` |
| `hipdnn.buffer_count` | Total pooled allocation count |
| `hipdnn.buffer_offsets` | Constant offset, or `-1` when offset is runtime SSA |
| `hipdnn.domain_count` | Number of domains; emitted only for multi-domain functions |
| `hipdnn.pool_sizes` | Per-domain static prefixes |
| `hipdnn.buffer_domains` | Domain ID for each pooled allocation |
| `hipdnn.pool_site_id` | Function site owning legacy eager domain-zero metadata |

The pool attributes are MLIR code-generation inputs, not fields in `schemas/model_metadata.fbs`. `generate-interface` emits `hipdnn_ep_pool_init` only when all three legacy attributes are present and `hipdnn.pool_size > 0`. Dynamic contributions are handled by the size operands on `hip.get_pool` during `inference_compute`.

Pool behavior is therefore carried by generated LLVM IR plus `RuntimeState`,
not by the FlatBuffers metadata blob. Runtime selection comes from each lowered
`hip.get_pool` call's `(site_id, domain_id)` pair.

Each `hip.get_pool` lowers to:

```c
void *hipdnn_ep_get_pool_base(RuntimeState *state,
                              int site_id,
                              int domain_id,
                              size_t needed_size);
```

The runtime maintains a grow-on-demand function-site table whose entries own
grow-on-demand arrays of domain bases and capacities. A zero-byte request is
promoted to one byte before calling HIP.

When one site/domain grows, the runtime synchronizes the session stream, frees
that pool's old buffer, allocates the new capacity, and leaves every other pool
untouched. Pools never shrink.

This generated-call ABI is internal but not backwards compatible. Delete and
rebuild stale cached model bitcode/native artifacts after upgrading; artifacts
compiled against the former three-argument pool helper cannot be mixed with
the new runtime.

`hipdnn_ep_get_buffer_from_pool(state, index)` remains the legacy domain-zero accessor for compile-time offsets. Multi-domain dynamic paths use `hipdnn_ep_get_pool_base`.

`hip.get_pool` carries an Allocate memory effect for the returned memref.

## Host scalar scratch

`hip-materialize-host-scalars` selects static-shape integer/index allocations of at most 16 elements with host reads or writes, including accesses reachable through supported view/alias operations. Accepted descriptor-only and terminal users include memref dimension/deallocation operations, supported views, memref copies, reshape shape operands, and HIP operations.

All selected allocations in a function share one `hip.get_host_scratch` buffer. Their offsets are aligned to 64 bytes, independently of PoolAllocs' default 256-byte alignment.

Functions without `!hip.context` as argument zero are skipped.

## Diagnostics and tests

Hard failures include:

- a multi-block function;
- missing `!hip.context` argument zero;
- invalid alignment;
- inability to place dynamic size arithmetic or pool acquisition despite domain partitioning.

`hip-hoist-alloc-size-arith` silently skips multi-block functions, while PoolAllocs fails on them. Both passes consider only entry-block allocations.

More than eight domains emits a non-fatal performance remark. Recommended triage, broader than the text of that remark:

1. `hip-infer-shapes` and the following canonicalize/CSE;
2. pre-bufferization `hip-resolve-tensor-dims` and tensor InferType external-model registration;
3. post-bufferization `hip-resolve-memref-dims`;
4. `hip-hoist-alloc-size-arith`;
5. remaining non-speculatable runtime dependencies.

LLVM statistics report allocation, domain, dynamic slab, small-bucket, and
small-bin counts; static excess bytes; comparable dynamic coefficient excess;
and small-bucket excess bins.
`--hip-pool-allocs='emit-fragmentation-report=true'` emits per-domain remarks
without changing IR. The dynamic coefficient comparison is numeric only when
all groups share one runtime factor and every effective slab-width coefficient
is alignment-multiple. Otherwise the remark states why the coefficient is
unavailable.

Primary regression coverage:

- `test/lit/Dialect/hip-pool-allocs.mlir`
- `test/lit/Dialect/hip-pool-allocs-multi-domain-metadata.mlir`
- `test/lit/Dialect/hip-pool-allocs-many-domains.mlir`
- `test/lit/Dialect/hip-pool-allocs-dynamic-binning.mlir`
- `test/lit/Dialect/hip-pool-allocs-lifetime-only.mlir`
- `test/lit/Dialect/hip-pool-allocs-fragmentation-report.mlir`
- `test/lit/Dialect/hip-pool-preconditions.mlir`
- `test/lit/Dialect/hip-resolve-tensor-dims.mlir`
- `test/lit/Dialect/hip-resolve-memref-dims.mlir`
- `test/lit/Dialect/hip-hoist-alloc-size-arith.mlir`
- `test/lit/Dialect/hip-materialize-host-scalars.mlir`
- `test/lit/Pipeline/pipeline-pool-lower.mlir`
- `test/lit/e2e/test_mlp_model.mlir`

`test/lit/Pipeline/output-allocator-dealloc.mlir` separately characterizes the
historical interaction if ownership-based deallocation is reintroduced; it is
not a production-pipeline test.

## Current limitations

- PoolAllocs requires single-block functions.
- Pools in separate domains cannot share storage.
- Static-prefix and dynamic regions do not share offsets.
- Dynamic reuse is limited to whole allocations in lifetime-only mode, whole
  groups in grouped mode, and allocations with identical
  `{staticFactor, dynamic operands}` keys in grouped-mode bins.
- Pool alignment defaults to 256 bytes.
