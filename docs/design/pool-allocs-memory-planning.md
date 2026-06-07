<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Pool-Allocs Memory Planning

**Date:** 2026-06-06
**Document Type:** Design
**Status:** Implemented
**Related:** [compiler-runtime-contract.md](compiler-runtime-contract.md), [constant-handling-design.md](constant-handling-design.md)

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Position](#pipeline-position)
- [Cooperating Passes](#cooperating-passes)
- [The hip-pool-allocs Algorithm](#the-hip-pool-allocs-algorithm)
- [Module-Attribute Contract](#module-attribute-contract)
- [Compiler-Runtime ABI](#compiler-runtime-abi)
- [Design Decisions](#design-decisions)
- [Worked Example: Two-Domain Partition](#worked-example-two-domain-partition)
- [Failure Modes & Diagnostics](#failure-modes--diagnostics)
- [Future Work](#future-work)

---

## Overview

The MLIR compiler converts ONNX models into a single-block `func.func @main_graph` whose body contains hundreds of `memref.alloc` ops — one per intermediate tensor materialised on the GPU. Three constraints shape the memory plan:

1. **No per-inference allocator traffic.** Any `memref.alloc` left intact at lowering becomes a runtime allocate + free pair issued on *every* inference. Allocator calls are expensive and sit on the critical path; a function that materializes many transient buffers pays that cost multiplied by the buffer count on every invocation. The plan must keep allocation off the per-inference path.

2. **Bounded, monotonic memory growth.** Buffers are reused across inferences, not re-acquired. A stable workload performs no allocator calls after warm-up; a workload whose required sizes grow pays one resize per new high-water size and never shrinks. The footprint grows only when the working set genuinely grows, never per inference.

3. **SSA dominance.** Every value consumed inside the function body must be dominated by its definition. That includes the size operand of `hip.get_pool` and the offset operand of every `memref.view` derived from it. With dynamic shapes — where alloc sizes are runtime arithmetic over input dims — emitting `hip.get_pool` at function entry forces every dim-arith chain to also live at function entry, which does not always hold.

Pool-allocs replaces every `memref.alloc` in `@main_graph` with a `memref.view` into one or more grow-on-demand byte buffers (`memref<?xi8>`) acquired via `hip.get_pool(%ctx, %size) {domain_id = N : i64}`. The buffers live on `RuntimeState` for the session lifetime; they grow on shape change and are freed at session cleanup. No per-inference allocator activity in the steady state.

The dominance constraint is what makes the design non-trivial: a single function-wide pool requires all alloc sizes to be computable above the earliest pooled alloc, which does not always hold. The design partitions allocs into **dominance domains**, each with its own pool acquisition point and its own runtime backing buffer.

---

## Pipeline Position

Several passes cooperate to produce the final pooled IR. The relevant stretch of `lib/Dialect/Transforms/Pipelines.cpp::buildOnnxToHipPipelineTail` is:

```
... (onnx-to-hip conversion)
    → hip-infer-shapes                       (1b: refine `?` dims via ReifyRankedShapedTypeOpInterface)
    → hip-resolve-tensor-dims                (1c: fold `tensor.dim` via upstream reify patterns)
    → one-shot-bufferize                     (tensor → memref)
    → buffer-results-to-out-params           (function results → out-param memrefs)
    → buildBufferDeallocationPipeline        (insert ownership-based deallocs)
    → CSE → canonicalize
    → hip-optimize-memrefs                   (HIP-specific buffer cleanup)
    → hip-promote-strided-hip-operands       (6a: contiguous temporaries for HIP-op operands)
    → hip-materialize-host-scalars           (6b: redirect tiny host-fed scalars to host scratch)
    → hip-hoist-alloc-size-arith             (6c: pull speculatable size arith above earliest dyn-alloc)
    → hip-pool-allocs                        (this doc's main subject)
    → ConvertBufferizationToMemRef → CSE → canonicalize
    → hip-lower-allocs                       (memref.alloc → hip.alloc/hip.free for what's left)
    → ...
```

The 1c step (`--hip-resolve-tensor-dims`) is **pre-bufferize and load-bearing**. It operates on `tensor.dim` queries against `tensor.expand_shape` / `tensor.collapse_shape` — operations that bufferize lowers to `memref.expand_shape` / `memref.collapse_shape` whose shape SSA (`output_shape` operands and reassociation maps) is opaque to the post-bufferize `memref.dim` patterns. Pre-bufferize is the last useful position. Without it, downstream `hip-pool-allocs` sees one scattered `memref.dim` query per reshape site and partitions them into separate dominance domains — graphs with per-layer same-rank dynamic `onnx.Reshape` (typical: norm / projection chains in transformer decoders) fragment into one pool per reshape site, wasting pool-prefix overhead and defeating the point of pooling (the partition is unbounded, so this degrades efficiency rather than failing compilation). After 1c, the chain bottoms out at `tensor.dim %arg, %const` on a function argument and pool-allocs sees a single domain. See the dedicated subsection in Cooperating Passes below.

The 6a/6b/6c ordering is load-bearing and **load-bearing in the order shown** — the Pipelines.cpp comments around each `addNestedPass` call spell out why:

- **`hip-promote-strided-hip-operands` runs FIRST (6a).** Some HIP-op operands arrive as `memref.subview` / strided memrefs; the runtime ABI for `hip.*` ops only forwards a bare `alignedPtr` per memref operand (no offset / per-dim strides channel), so this pass materialises a contiguous temporary `memref.alloc` + copy for any strided operand. Those new transient allocs then need to flow through the rest of the buffer pipeline.

- **`hip-materialize-host-scalars` runs SECOND (6b), AFTER promote-strided.** Tiny static-shape integer-or-index allocs (≤16 elements) with at least one host I/O user (`memref.store` or `memref.load`) must live in host-mapped memory acquired via `hip.get_host_scratch` — not in the GPU pool, where on some targets a host store SEGVs (real device memory) while on others it silently works (UMA-mapped). The "AFTER promote-strided" placement is what the file header at `MaterializeHostScalars.cpp` specifically calls out: any contiguous-temporary `memref.alloc` introduced by 6a must also be visible to the host-scalar candidate scan, and 6b must run BEFORE pool-allocs so candidates are removed from its input set.

- **`hip-hoist-alloc-size-arith` runs THIRD (6c), immediately before pool-allocs.** It moves speculatable size-arith ops above the earliest dynamic alloc, reducing the number of dominance domains pool-allocs sees.

Each pass has a single, named purpose — no internal phases that quietly do work belonging to a sibling pass. When something breaks, bisect by reading `--debug-only=<pass>` traces independently instead of one tangled snapshot.

---

## Cooperating Passes

### `hip-resolve-tensor-dims`

A single `func.func`-nested pass, run between `hip-infer-shapes` and `one-shot-bufferize`. Resolves `tensor.dim` queries that arise from same-rank dynamic `onnx.Reshape` decompositions (e.g. transformer-decoder `q_norm`/`k_norm`/projection chains lowered as `tensor.expand_shape + tensor.collapse_shape` pairs). Without it, the queries cross the bufferize boundary as scattered `memref.dim` ops on the resulting reshape ops; pool-allocs then partitions per reshape site, fragmenting the function into many single-alloc domains and degrading pooling efficiency (the partition is unbounded, so this is a perf cost, not a compile failure).

#### Implementation

A thin wrapper over upstream MLIR's `populateResolveRankedShapedTypeResultDimsPatterns` (and the `InferShapedTypeOpInterface` companion populator) plus the tensor-dialect canonicalisers (`tensor::DimOp::getCanonicalizationPatterns`, `tensor::ExpandShapeOp::getCanonicalizationPatterns`, `tensor::CollapseShapeOp::getCanonicalizationPatterns`), run as a single greedy fixed point.

The wrapper carries **no new logic** — every fold is performed by an upstream pattern. The pass exists only because the same fixed point composes patterns that live in different upstream populators (`memref::populate*` on one side, `tensor::*::getCanonicalizationPatterns` on the other) and we want them to converge against each other in one pass invocation, so chains like `dim(collapse(expand(arg)))` bottom out at `tensor.dim %arg, %const` in a single greedy run.

The mental model: any op implementing `ReifyRankedShapedTypeOpInterface` exposes its result shape as either static type info or a chain of SSA values reachable from its operands; the upstream `DimOfReifyRankedShapedTypeOpInterface` pattern dispatches `tensor.dim` of such an op through `reifyDimOfResult` → `reifyShapeOfResult` → `reifyResultShapes`, returning the appropriate `OpFoldResult` (a static `IntegerAttr`, an existing SSA value, or a freshly-built `affine.apply` over upstream-extracted source dims).

#### Required external-model registration

Coverage of `tensor.{expand,collapse}_shape` and `tensor.pad` requires `mlir::tensor::registerInferTypeOpInterfaceExternalModels(registry)` to have been called on the dialect registry. Upstream attaches the `Reify{Expand,Collapse,Pad}ShapeOp` external models (in `mlir/lib/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.cpp`) only when this registration runs; without it, the upstream patterns silently no-op on these ops and the bufferize boundary leaks every reshape-dim query as a `memref.dim` of a reshape op.

The registration is wired into `hip::compiler::registerAllDialects` in `include/hip/InitAllPasses.h` (which `CompilerDriver.cpp` calls via `loadAllDialects`) and into the inline registry built by `tools/hip-mlir-opt/hip-mlir-opt.cpp::main` (which builds its own registry rather than going through `registerAllDialects`). Both sites are load-bearing: omitting either loses coverage in that tool and lets a regression slip past LIT or end-to-end compile.

If a future MLIR rebase moves the registration into a different translation unit, ensure that translation unit links the `MLIRTensorInferTypeOpInterfaceImpl` library (the fold logic is in a separate static library from `MLIRTensorDialect` and is not pulled in transitively).

#### Worked example

Same-rank dynamic Reshape pair (the canonical transformer-decoder norm / projection trigger):

Before:

```mlir
%d0 = tensor.dim %arg0, %c0 : tensor<?x?x4096xf16>
%d1 = tensor.dim %arg0, %c1 : tensor<?x?x4096xf16>
%expand   = tensor.expand_shape %arg0 [[0], [1], [2, 3]]
              output_shape [%d0, %d1, 32, 128]
              : tensor<?x?x4096xf16> into tensor<?x?x32x128xf16>
%collapse = tensor.collapse_shape %expand [[0], [1, 2], [3]]
              : tensor<?x?x32x128xf16> into tensor<?x?x128xf16>
%use = tensor.dim %collapse, %c1 : tensor<?x?x128xf16>      // queries seq*32
```

After:

```mlir
#map = affine_map<()[s0] -> (s0 * 32)>
%d1  = tensor.dim %arg0, %c1 : tensor<?x?x4096xf16>
%use = affine.apply #map()[%d1]                             // (reshapes are
                                                            //  DCE'd because
                                                            //  the dim was
                                                            //  their only use)
```

The static factor is folded INTO the affine map by upstream reify rather than emitted as a separate `arith.muli %dyn, %c32`. Downstream `--lower-affine` (already in `buildHipToLLVMPipeline`) lowers `affine.apply` to arith ops in time for `--convert-hip-to-llvm`.

#### Pattern correctness

Inherited from upstream — `DimOfReifyRankedShapedTypeOpInterface` carries `setHasBoundedRewriteRecursion()` and the `ReifyShapeOfResult` -> `reifyResultShapes` chain returns a finite SSA construction per call. Greedy non-convergence calls `signalPassFailure()`, mirroring upstream `ResolveShapedTypeResultDimsPass`.

#### Idempotence

The pass emits no new `tensor.dim` ops on reshape operands once it has converged. A second invocation has no work; the LIT fixture `test/lit/Dialect/hip-resolve-tensor-dims.mlir` pins this down across the canonical cases (collapse with two/three dyn sources, mixed static/dyn, expand at static and dynamic slots, chained collapse-of-expand, fully-static tensors, no-op on functions without reshape-dim queries).

#### What this pass cannot fix

Queries reaching values produced by `memref.load` of a host-scratch slot or other memory-mutating ops survive into bufferize. Those are exactly the chains `hip-pool-allocs`'s multi-domain partition is designed to handle (the `%alloc_11` case in the worked example below). This pass removes the structural noise from reshape chains; the partition handles the residue.

#### Retirement

This pass retires entirely (pipeline call + LIT fixture removed) when any of:

1. Upstream MLIR's `tensor::DimOp::fold` learns `tensor.expand_shape` / `tensor.collapse_shape` chains directly (the LIT fixture becomes a regression catcher for the upstream fold and the pipeline entry is removed; the external-model registration stays as cheap insurance).
2. The compiler migrates to an IREE-style `ShapeAwareOpInterface` that carries explicit shape SSA on HIP-dialect ops, eliminating `tensor.dim` queries from dynamic-shape IR entirely.
3. The same-rank dynamic Reshape decomposition in `ReshapeConversion.cpp` is replaced by a single op that emits no `expand_shape` / `collapse_shape` pair (the canonical trigger disappears; the pass keeps doing useful work for any other reshape-pair-emitting path).

#### History

The pass originally shipped as this upstream-pattern wrapper plus a separate custom-pattern pass `--hip-resolve-reshape-dims` that explicitly rewrote `tensor.dim` of `collapse_shape` and the SSA-valued slot of `tensor.dim` of `expand_shape`. The custom patterns were necessary because the upstream `Reify{Expand,Collapse}ShapeOp` external models had never been registered in this project's dialect registry — without registration, the upstream `DimOfReifyRankedShapedTypeOpInterface` pattern silently failed on every reshape-dim query.

Once the registration was added, every case the custom pass handled folds via upstream as cleanly or cleaner (e.g. mixed static/dyn collapse groups become `affine.apply [s0 * 32]` instead of `arith.muli %dyn, %c32`), so the custom pass was retired and the wrapper now stands alone. End-to-end compile of dynamic-shape models also improved on this transition: the cleaner reify-driven IR (static factors folded into `affine.apply` rather than scattered `arith.muli`) composes better with bufferize's allocation sizing.

### `hip-hoist-alloc-size-arith`

Before:

```mlir
%alloc_a = memref.alloc(%dim) : memref<?xf16>     // earliest dyn alloc
... unrelated ops ...
%size_b = arith.muli %dim, %dim_0 : index         // could have lived above %alloc_a
%alloc_b = memref.alloc(%size_b) : memref<?xf16>
```

After:

```mlir
%size_b = arith.muli %dim, %dim_0 : index         // hoisted
%alloc_a = memref.alloc(%dim) : memref<?xf16>
... unrelated ops ...
%alloc_b = memref.alloc(%size_b) : memref<?xf16>
```

#### Hoist-eligibility predicate

There is **no fixed allow-list of op kinds.** The pass uses MLIR's standard `mlir::isSpeculatable` predicate (the same one upstream LICM uses) — any speculatable op is potentially hoistable. The recursive descent in `isReachableHoistable` (`HoistAllocSizeArith.cpp`) accepts an op iff:

1. It lives in the entry block (out-of-block defs already dominate any in-block insertion point).
2. It is below `earliestAlloc` (in-block but already-above-earliest defs already dominate).
3. It is **not** a `memref.alloc` / `memref.alloca` (those are the things we're trying to make feasible — moving one would alter buffer lifetimes).
4. It satisfies `mlir::isSpeculatable(op)` (e.g. `memref.load`, `func.call` without `NoSideEffect`, `hip.host_sync`, `arith.divsi` — anything that could trap or read mutable memory — fail this).
5. **Every transitive operand** also satisfies the same recursion (or already dominates `earliestAlloc`).

Condition 5 is the non-obvious one. A chain like `%loaded = memref.load …; %doubled = arith.muli %loaded, %c2; %alloc(%doubled)` looks half-hoistable — `%doubled` IS speculatable, but its operand `%loaded` is NOT. Moving `%doubled` above `%alloc` while `%loaded` stays below would break SSA dominance at the new `%doubled` site. The recursive check ensures we only hoist `%doubled` when its entire transitive cone is also hoistable.

#### Move algorithm (and why the "displacement property" matters)

The recursive descent inserts each accepted op into a `SetVector` AFTER recursing into its operands — so the SetVector ends up in operand-before-use order. The pass then **forward-iterates** the SetVector and calls `op->moveBefore(earliestAlloc)` on each op:

```cpp
for (Operation *op : toMove)
  op->moveBefore(earliestAlloc);
```

The non-obvious correctness mechanic is that each `moveBefore(earliestAlloc)` displaces every previously-moved op up by one slot — so the deepest operands (inserted first into the SetVector, moved first by the loop) end up at the top of the hoist region, and the closest-to-the-alloc ops (inserted last, moved last) sit just above `earliestAlloc`. Final layout: a topologically-ordered slice of the speculatable predecessor cone, operands at the top, uses at the bottom.

A naive "move each op directly to the slot above its consumer" would require recomputing slot positions after every move; relying on the displacement property keeps the loop one line.

#### Why a separate pass

Pool-allocs's downstream invariant — every alloc's dyn-operand defs precede the earliest pooled alloc — becomes feasible for many more inputs without entangling pool-allocs's placement logic with backward-slice analysis. Tests for the two passes stay independent. When something breaks at the boundary, bisect by reading `--debug-only=hip-hoist-alloc-size-arith` and `--debug-only=hip-pool-allocs` separately instead of one tangled trace.

#### Idempotence

A second invocation finds nothing below `earliestAlloc` that still needs moving (everything in the cone now dominates), so the pass is a no-op. A LIT fixture pins this down.

#### What hoisting cannot fix

Chains that reach a `memref.load` of a value computed mid-block (the canonical case is a host-scratch slot used to pass a runtime-dependent scalar from CPU to GPU). Those chains stay where they are; pool-allocs handles them by partitioning into domains.

### `hip-pool-allocs`

The subject of this doc. It partitions allocations into dominance domains and replaces each `memref.alloc` with a `memref.view` into one of N independent, grow-on-demand runtime pools, acquired per domain via `hip.get_pool`.

### Briefly: `hip-materialize-host-scalars`

Some allocs need to be host-writable AND device-readable: the canonical case is a single `int64` carrying `total_seq_len` written by CPU code from a `memref.dim` and read by the GPU as the `seqlens_k` argument of a `hip.gqa` op. Putting these in the GPU pool is incorrect on architectures where `hipMalloc` returns true device memory (the host store SEGVs); on others it silently works because `hipMalloc` returns UMA-mapped host memory there, masking the bug.

#### Candidate filter

`isHostScalarCandidate` (in `MaterializeHostScalars.cpp`) accepts a `memref.alloc` iff:

- **Static shape**, **≤ 16 elements**.
- **Integer or index** element type. Float allocs are excluded because they are almost always GPU-consumed in flight, where the GPU pool is the right home.
- **At least one host I/O user**: `memref.store` OR `memref.load` directly on the alloc. Either flavour is enough — both are the SEGV trigger on real-device-memory targets.
- **All other users are either `memref.dim` / `memref.dealloc` OR any `hip.*` op.** This is the load-bearing inclusion: the canonical regression is `memref.alloc<i64>` → `memref.store<HOST>` → `hip.cast<GPU>`, and `hipHostMalloc(hipHostMallocMapped)` memory is GPU-readable on UMA so the bare-ptr ABI used by `--convert-hip-to-llvm` consumes the same buffer regardless of whether it was `hipMalloc`'d or `hipHostMalloc`'d. An earlier "no-hip-users" filter rejected this exact pattern → pass was a no-op → SEGV persisted.

#### Rewriting

All candidates in a function share a single `hip.get_host_scratch(%ctx, %total) : memref<?xi8>` op emitted at the entry block. Each candidate is rewritten to `memref.view %scratch[%offset]` at its original alloc site. Offsets within the shared scratch are **64-byte aligned** (matches the GPU pool's alignment scale and gives every candidate its own cache line); element sizes derived from `getIntOrFloatBitWidth()` (index = 64 bits). Original `memref.dealloc`s are erased — the scratch buffer is runtime-owned.

Pool-allocs and host-scalars are mutually exclusive: an alloc is either pool-eligible or host-scalar-eligible, never both.

---

## The hip-pool-allocs Algorithm

Input contract (assumed by the pass; violations either signal failure or get skipped):

- Function has exactly one block (`funcOp.getBody().hasOneBlock()`).
- Argument 0 is `!hip.context` (run `hip-add-context-arg` before).
- Every alloc whose result has at least one use is a candidate; allocs with `result.use_empty()` are ignored.

Output: every input `memref.alloc` is replaced by a `memref.view %pool[%offset]` into one of N grow-on-demand `memref<?xi8>` pools. Per-domain `hip.get_pool(%ctx, %pool_size) {domain_id = N : i64}` calls are emitted at points that dominate every pooled alloc in their domain.

Algorithm (executed in `runOnOperation` of `lib/Dialect/Transforms/PoolAllocs.cpp`):

```
Phase 1   - Liveness.
            Index every op in the block (sequential `defIndex`).
            For each alloc, follow view-flow aliasing transitively to find
            `lastUseIndex` (highest index of any aliased use).

Phase 1.5 - Domain partition.
            Greedy textual-order clustering by dominance feasibility.
            Output: SmallVector<Domain>, each carrying a vector of allocs.

Phases 2..5 (per domain):
  Phase 2 - Static / dynamic split for this domain.
  Phase 3 - Static packing: greedy best-fit gap-finding offset assignment.
  Phase 4 - Dynamic packing: bucket by structural byte-size key, bin-pack
            within bucket by lifetime.
  Phase 5 - IR emission: per-bucket size arith, hip.get_pool, per-alloc
            offset SSA, view replacements.

Module metadata:
  Always: legacy single-domain attrs (hipdnn.pool_size, hipdnn.buffer_count,
          hipdnn.buffer_offsets).
  When domain count > 1: also emit hipdnn.domain_count, hipdnn.pool_sizes,
          hipdnn.buffer_domains (consumed by the runtime).
```

### Phase 1: Liveness Analysis

For each alloc, compute `[defIndex, lastUseIndex]` using the helper `findLastAliasedUseIndex` in `BufferUtils.cpp`:

- `defIndex` = sequential position of the alloc's `Operation*` in the block.
- `lastUseIndex` = maximum `defIndex` over all uses of the alloc's result, **transitively through view-flow ops** (`memref.view`, `memref.subview`, `memref.cast`, `memref.expand_shape`, `memref.collapse_shape`, etc.) computed via MLIR's `BufferViewFlowAnalysis`.

Two non-obvious behaviours of `findLastAliasedUseIndex`:

- **Users in nested regions** (e.g. an `scf.for` body that consumes a view of the alloc) are mapped to their enclosing op's index in the entry block via `block.findAncestorOpInBlock`. Without this, a use inside a region would not be findable in `opIndex` and would be treated as "no users".
- **Users unreachable from the entry block** conservatively return `blockSize - 1` (one past the last in-block op), making the alloc live to function exit. Strictly conservative — this branch is a defensive fallback, not a hot path.

`PoolAllocs::getDependentDialects` registers `arith::registerBufferViewFlowOpInterfaceExternalModels` so view-flow tracking correctly follows `arith.select` of two memrefs (a value-flow case the upstream `BufferViewFlowAnalysis` would otherwise miss).

Two allocs `A` and `B` have **overlapping lifetimes** iff their `[defIndex, lastUseIndex]` intervals overlap. Non-overlapping allocs may share offset slots in subsequent phases.

`staticByteSize` is computed from the alloc's MemRefType (`getStaticByteSize` in `BufferUtils.cpp`): if the type is fully static, it is `elementBytes * product(shape)` rounded to a byte boundary. For dynamic shapes it is 0 (signalling the dynamic path).

The single-block constraint is what makes sequential indices a valid liveness model. Multi-block / region-bearing functions would need MLIR's full `Liveness` analysis; today the upstream pipeline never produces multi-block `@main_graph` (control flow is collapsed into ops like `hip.if`, `hip.loop` whose regions are treated opaquely).

### Phase 1.5: Dominance-Domain Partition

The hardest and newest part of the design.

A **domain** is a subset of pooled allocs whose `hip.get_pool` can share a single insertion point that

- comes **strictly after** every dynamic-size operand definition reachable from any alloc in the domain, AND
- comes **strictly before** every alloc in the domain.

Such a position lets one `hip.get_pool` dominate every alloc in the domain, while every dyn-operand SSA value dominates the pool acquisition.

The helper `findLatestLegalInsertionPoint(block, requiredAfter, requiredBefore)` returns the latest in-block iterator satisfying both constraints, or `std::nullopt` when no such point exists. (Implementation: take the latest op in `requiredAfter` as `lo` and the earliest in `requiredBefore` as `hi`; the answer is `std::next(lo)` iff `lo` is strictly before `hi`.)

A domain is **feasible** iff `findLatestLegalInsertionPoint` over the union of (dyn-operand defs) and (allocs themselves) returns a value.

#### Greedy clustering rule

```
domains = []
for alloc in allocs sorted by defIndex:
  for D in reversed(domains):              // most-recent first
    speculatively add alloc to D
    if D ∪ {alloc} is feasible: keep, break
    else:                                   roll back
  else:                                    // no D accepted
    domains.append(Domain([alloc]))         // unbounded — one more pool
return domains
```

Why most-recent-first: domains created later have later first-allocs, so their dominance window is the strictest. If alloc `A`'s dyn-operand chain lives above the latest-created domain `D_recent`, the merge into `D_recent` succeeds; if `A`'s dyn-defs are below `D_recent`'s earliest alloc, they are structurally below every earlier domain's earliest alloc too, and merging anywhere will fail. So the first feasible domain is always the most recent one. We still walk reverse order to remain robust against future reorderings.

Why greedy textual-order: the ordering of allocs in the block IS the dataflow order. Processing in `defIndex` order means each alloc only ever joins a domain whose earlier members are already settled — no backtracking needed.

#### Single-domain output

In the common case (after `hip-hoist-alloc-size-arith` has run), every alloc's dyn-operand defs dominate every pooled alloc, so the first probe of `D[0]` always succeeds and the partition returns a single domain containing every alloc. Phases 2..5 then run on that single domain exactly as they would with no partitioning step at all — single-domain IR and metadata are identical whether or not a second domain is ever needed.

#### Multi-domain output

A second domain opens iff some alloc's dyn-operand chain comes from values defined **below** the earliest alloc of every existing domain. The canonical trigger is a `memref.load` from a host-scratch slot whose `memref.store` lives between two pooled allocs — the load cannot be hoisted (memory side effect) and the alloc that consumes it cannot share a domain with allocs above the load.

**The domain count is unbounded — there is no compile-time cap.** The partition produces exactly as many domains as the dataflow requires (real graphs produce 1, canonical, or 2, one host-load chain), and the runtime backs them with per-domain pool arrays that grow on demand — `num_pool_domains` starts at 1 (domain 0) and the arrays realloc, zero-filling new slots, the first time a higher `domain_id` is seen (on the cold first inference). A fixed cap is not acceptable in production: it would turn a legitimate-but-unusual graph (e.g. several independent host-scratch chains) into a hard compile failure with no correct fallback. Soft-merging unrelated domains is never done — it would silently mis-place allocations — so the only correct behavior is to open another pool, which the dynamic backing makes free.

A surprisingly high domain count still usually means upstream canonicalisation (`hip-resolve-tensor-dims`, `hip-hoist-alloc-size-arith`) left scattered size queries in place, which hurts pooling efficiency. That condition is surfaced as a non-fatal advisory (see Failure Modes), never as a compile error.

### Phase 2: Static / Dynamic Split

For each domain, partition allocs by `staticByteSize > 0`. Statics go to Phase 3 (constant-offset slot assignment); dynamics go to Phase 4 (runtime size + bin-shared offsets).

Statics are sorted largest-first to give the gap-finder more room to work with.

### Phase 3: Static Packing

Greedy best-fit gap-finding in a 1-D byte address space.

For each static alloc (largest first):

1. Walk existing reservations sorted by offset.
2. For each reservation whose lifetime overlaps the new alloc, check the gap before it.
3. Place the alloc in the smallest gap that fits (best-fit).
4. If no gap fits, append after the latest overlapping reservation.

Two allocs whose lifetimes do not overlap can share the same address range — captured by ignoring non-overlapping reservations during the walk.

Worst case grows as `O(n²)` in the number of static allocs per domain, but `n` is small (typically <16). Production graphs have ≤4 static allocs after upstream canonicalisation, so the constant factor dominates and this is effectively O(1).

Why best-fit (not first-fit): `n` is small enough that the difference is one extra iteration of the inner loop, and best-fit produces tighter packings on the long tail of unbalanced gap distributions. The static prefix size feeds directly into `hip.get_pool`'s size operand, so tighter packings reduce pool footprint.

### Phase 4: Dynamic Packing

Two-level grouping: bucket by structural byte-size key, then bin-pack by lifetime within each bucket.

#### Level 1: Bucket by `DynSizeKey`

```
DynSizeKey = { staticFactor: int64, dynOperands: SmallVector<Value> }
staticFactor = elementBytes * product(static dims)
dynOperands  = SSA values for the dynamic dims, in declaration order
```

Two allocs with the same key have identical runtime byte sizes:

```
byte_size = staticFactor * dynDim0 * dynDim1 * ...
```

This is a structural equivalence on the alloc's MemRefType + dynamic operand values, not a value-numbering or CSE pass. Two allocs of `memref<?x?x4096xf16>` whose `?` dims come from the same SSA values share a bucket. Two allocs of `memref<?x?x4096xf16>` whose `?` dims come from different SSA values do **not** share a bucket — even if those SSA values would be equal at runtime — because the type-level grouping is what guarantees correctness.

Bucket creation order is insertion order (linear scan over a SmallVector), giving deterministic IR output across runs.

#### Level 2: First-fit bin packing within bucket

Within a bucket, allocs whose lifetimes do not overlap can share a single offset slot at runtime. First-fit packs each alloc into the first existing bin without conflicts; if none accepts, a new bin opens.

Number of bins per bucket = number of simultaneously-live allocs of that size class. Each bin maps 1:1 to a runtime offset slot.

Why first-fit (not best-fit): bins are unordered (no notion of "size left over"), so best-fit reduces to first-fit. The algorithmic distinction matters only for static packing where offsets are linear.

### Phase 5: IR Emission

Per-domain emission of size arithmetic, pool acquisition, offset SSA values, and view replacements. Each domain is processed independently; the emitted IR for domain N consumes only its own allocs' constraints, not the global function's.

#### Per-bucket size arithmetic

For each `DynBucket`, find the latest legal insertion point given:

- `requiredAfter` = the bucket's `dynOperands` defs in this block,
- `requiredBefore` = the bucket's allocs.

Set the builder there and emit:

```mlir
%static  = arith.constant <staticFactor> : index
%byte    = arith.muli %static, %dyn0 : index
%byte    = arith.muli %byte,   %dyn1 : index    // for each dyn operand
%aligned = align_up(%byte, alignment)           // skipped if staticFactor % align == 0
```

The skip condition is a small but real optimisation: when `staticFactor` is already a multiple of the alignment (very common for tensor types where `elementBytes * product(static dims)` is a cache-line multiple), the `align_up` is a semantic no-op and we save the `arith.divui + muli + addi` triple it would otherwise produce.

#### Pool acquisition

Find the latest legal insertion point given:

- `requiredAfter` = every bucket's `alignedSize` def (forces pool acquisition after all bucket sizes are known),
- `requiredBefore` = every alloc in this domain.

Set the builder there and emit:

```mlir
%pool_size = arith.constant <static_prefix> : index
%pool_size = arith.addi %pool_size, %bucket_0_aligned * %bucket_0_bin_count : index
%pool_size = arith.addi %pool_size, %bucket_1_aligned * %bucket_1_bin_count : index
...
%pool = hip.get_pool(%ctx, %pool_size) {domain_id = N : i64} : memref<?xi8>
```

Domain `N == 0` omits the `domain_id` attribute — the operation definition declares it as `DefaultValuedAttr<I64Attr, "0">` so the printer elides it. A single-domain function therefore prints no `domain_id` anywhere, so its textual IR is unchanged by the existence of multi-domain support.

#### Per-alloc offset assignment + view replacement

Static allocs: each gets an `arith.constant` for its offset, emitted right below the pool acquisition.

Dynamic allocs: offset = static prefix + cumulative bucket size + bin index × bucket aligned size. The `currentBase` cursor advances past each bucket's full footprint (`alignedSize × numBins`) to ensure distinct buckets never alias within this pool.

For each alloc: `setInsertionPoint` to the alloc's original position, emit `memref.view %pool[%offset][%dyn_dims...]` of the alloc's original `MemRefType`, replace all uses of the alloc's result with the view, erase the original alloc.

After all domains: walk the function once to find any `memref.dealloc` whose operand is now a `memref.view` and erase those — pools are runtime-owned, not function-scoped, so no per-alloc dealloc is appropriate.

---

## Module-Attribute Contract

Pool-allocs writes module-level attributes that downstream `GenerateInterface` reads to produce the FlatBuffers metadata blob baked into `model.dll`. The contract is **additive**: legacy single-domain attributes are emitted on every run; multi-domain attributes are emitted only when `domain_count > 1`.

### Always emitted (legacy contract)

| Attribute | Type | Meaning |
|---|---|---|
| `hipdnn.pool_size` | `i64` | Domain 0's static prefix in bytes (the constant baked into `hip.get_pool(domain=0)`'s size). |
| `hipdnn.buffer_count` | `i64` | Total pooled allocs across all domains. |
| `hipdnn.buffer_offsets` | `array<i64>` | Per-buffer offset within its domain's pool. `-1` indicates a runtime-computed offset (dynamic allocs inside non-trivial buckets). Length = `buffer_count`. |

A single-domain model produces only these three. A pre-multi-domain runtime that does not understand multi-domain attrs reads only these and behaves correctly.

### Multi-domain attributes (emitted when `domain_count > 1`)

| Attribute | Type | Meaning |
|---|---|---|
| `hipdnn.domain_count` | `i64` | Number of distinct pools (= number of `hip.get_pool` ops in the function). |
| `hipdnn.pool_sizes` | `array<i64>` | Per-domain static prefix in bytes. Length = `domain_count`. |
| `hipdnn.buffer_domains` | `array<i64>` | Per-buffer domain id (which pool a buffer lives in). Length = `buffer_count`. |

A pre-multi-domain runtime would silently ignore these extra attrs and alias all domains onto its single pool slot, which is **incorrect** (pools are independent). The runtime ABI bump described next is what makes multi-domain output safe to ship.

### Trivial-input edge case

When a function has fewer than two pooled allocs, the pass emits zeroed legacy attrs (`pool_size = 0`, `buffer_count = 0`, `buffer_offsets = []`) and returns without touching the IR. A missing attribute would crash the metadata reader; a zeroed one is a well-defined "no pool".

---

## Compiler-Runtime ABI

### `hip.get_pool` op

`include/hip/Dialect/IR/HipOps.td`:

```td
def Hip_GetPoolOp : Hip_Op<"get_pool",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins Hip_ContextType:$ctx, Index:$pool_size,
                       DefaultValuedAttr<I64Attr, "0">:$domain_id);
  let results = (outs AnyMemRef:$pool);
  let assemblyFormat = "`(` $ctx `,` $pool_size `)` attr-dict `:` type($pool)";
}
```

Textual IR:

```mlir
%pool0 = hip.get_pool(%ctx, %size0) : memref<?xi8>                             // domain_id elided
%pool1 = hip.get_pool(%ctx, %size1) {domain_id = 1 : i64} : memref<?xi8>       // explicit
```

The default-valued attribute is what makes single-domain textual IR bit-identical to pre-multi-domain output: the printer omits attrs equal to their default, and the parser fills in `0` when the attr is absent. Existing LIT goldens that do not touch multi-domain cases continue to pass without edits. The `MemoryEffectsOpInterface` declaration lets canonicalize / CSE reason about the op (it has no read/write effects on its operands — the side-effect is the implicit runtime malloc, which we model out-of-band).

### LLVM lowering

`ConvertHipToLLVM` (`lib/Conversion/HipToLLVM/MemoryLowering.cpp::GetPoolOpLowering`) lowers `hip.get_pool` to a 3-arg C call:

```mlir
%domain_const = llvm.mlir.constant(N : i32) : i32
%base = llvm.call @hipdnn_ep_get_pool_base(%state, %domain_const, %size)
        : (!llvm.ptr, i32, i64) -> !llvm.ptr
```

The `domain_id` attribute is materialised as a small `i32` constant immediately above the call, then passed as the second argument. The runtime returns a generic AS-0 pointer; if the result memref is in a non-zero address space (e.g. AS-1 = AMDGPU global), an `llvm.addrspacecast` follows. The result is wrapped into a `memref<?xi8>` descriptor with stride 1 and the dynamic length equal to `%pool_size`.

### Runtime state layout

`lib/Runtime/runtime_state_internal.h`:

```cpp
struct RuntimeState {
  // ... other fields (constants blob, stream, library handles, ...) ...
  int     num_pool_domains;             // slots currently allocated; grows on demand
  void  **pool_base;                    // [num_pool_domains] per-domain GPU pool base pointers
  size_t *pool_size;                    // [num_pool_domains] per-domain pool size in bytes
  size_t *buffer_offsets;               // offsets for static buffers in domain 0
  size_t  num_buffers;                  // static buffer count in domain 0
  // ... host scratch, qmoe scratch, descriptor caches, ... ...
};
```

There is no compile-time domain cap. `pool_base` / `pool_size` are heap arrays of `num_pool_domains` entries that **grow on demand**: domain 0's slot is ensured at `inference_init`, and the arrays are reallocated (zero-filling the new slots) the first time a higher `domain_id` is observed — by `hipdnn_ep_get_pool_base`. Every `domain_id` is first seen on the cold first inference, so `num_pool_domains` stabilises after that and no further array realloc happens at steady state. `realloc`-move is safe because nothing caches `&pool_base[i]` across calls; `pool_base[domain_id]` is re-derived from `state` on every access. Freed in `hipdnn_ep_state_cleanup`.

### Runtime entry points

Three entry points consume this state. All declared in `lib/Runtime/hipdnn_ep_runtime.h`, all `extern "C"` so generated bitcode can call them.

#### `hipdnn_ep_pool_init` — eager domain-0 allocation at session start

```cpp
int hipdnn_ep_pool_init(RuntimeState *state,
                        size_t        pool_size,
                        const size_t *buffer_offsets,
                        size_t        num_buffers);
```

Called once from `inference_init` with values pulled from the FlatBuffers metadata blob. Maps to the legacy attribute set:

- `pool_size` ← `hipdnn.pool_size` (= domain 0's static prefix in bytes)
- `buffer_offsets` / `num_buffers` ← the constant entries of `hipdnn.buffer_offsets` (= the static-offset slots in domain 0)

It ensures the per-domain arrays have a slot for domain 0 (`ensure_pool_domains(1)`) before touching `pool_base[0]`. The runtime does **not** read `hipdnn.domain_count` to pre-size the arrays — higher domains are grown lazily (below), so the compiled `domain_count` is informational metadata only. Then: if `pool_size > 0`, allocates `state->pool_base[0]` via `hipMalloc(pool_size)` — eagerly, so the first inference does not pay the malloc latency. If `pool_size == 0`, leaves `pool_base[0]` NULL and lets the dynamic grow path own the first allocation. `state->pool_size[0]` is set to `pool_size`. **Domains 1..N have no array slot yet; the first `hip.get_pool(domain=N)` call grows the arrays and lazily allocates that domain's pool.**

The `buffer_offsets` array is `memcpy`'d into a `malloc`'d slot on `state` for use by `hipdnn_ep_get_buffer_from_pool` (next).

#### `hipdnn_ep_get_buffer_from_pool` — legacy static-offset accessor

```cpp
void *hipdnn_ep_get_buffer_from_pool(RuntimeState *state, size_t index);
```

Returns `state->pool_base[0] + state->buffer_offsets[index]`. Used by code paths that consume the static-offset side of the pre-multi-domain ABI (the buffer-index entries in `hipdnn.buffer_offsets` that came out non-negative). All static buffers live in domain 0 by construction; multi-domain functions only use the dynamic `hip.get_pool` path.

#### `hipdnn_ep_get_pool_base` — grow-on-demand pool acquisition

```cpp
void *hipdnn_ep_get_pool_base(RuntimeState *state,
                              int           domain_id,
                              size_t        needed_size);
```

Called by every `hip.get_pool` lowering. Behaviour:

- Rejects `domain_id < 0` (returns NULL + stderr diagnostic flagging a compiler bug — `hip-pool-allocs` assigns domain ids starting at 0). There is no upper bound: `ensure_pool_domains(domain_id + 1)` grows the per-domain arrays (zero-filling new slots) the first time a higher `domain_id` is seen, a cold-path event on the first inference.
- If `needed_size > state->pool_size[domain_id]`, grows: `hipStreamSynchronize(state->stream)` (so any in-flight kernel reading the old buffer finishes), `hipFree(old)`, `hipMalloc(needed_size)`, store new base + size. **Stream sync covers all domains** because they share a single compute stream, but the grow itself only touches the one domain — `pool_base[M]` for M ≠ domain_id are untouched.
- If `needed_size <= state->pool_size[domain_id]`, just returns the existing base.
- Pools never shrink. A session that hits a peak shape keeps that footprint until cleanup.
- Cleanup (`hipdnn_ep_state_cleanup`): `hipFree` every non-NULL `pool_base[i]` for `i ∈ [0, num_pool_domains)`, then `free` the `pool_base` / `pool_size` arrays themselves.

---

## Design Decisions

### Single-block-only constraint

`PoolAllocs::runOnOperation` requires `funcOp.getBody().hasOneBlock()` and signals failure otherwise. The diagnostic mentions that liveness analysis uses sequential op indices that do not generalise to control flow.

Generalising to multi-block / region-bearing functions would require MLIR's full `Liveness` analysis (forward+backward dataflow) and dominator-tree-aware insertion points (instead of `Operation::isBeforeInBlock`). The codebase does not currently produce multi-block `@main_graph` — control flow is collapsed into single-result ops with regions (`hip.if`, `hip.loop`) whose bodies are opaque to pool-allocs.

This is conservative-correct: the pass refuses to touch IR shapes it cannot reason about. Future work if multi-block bodies show up.

### Greedy textual-order partition

Alternatives considered:

- **Tight (one domain per alloc):** trivially correct, defeats pooling.
- **Loose (cluster by transitive equivalence):** more expressive, requires solving a graph-partition problem.

The greedy approach picked here is correct for single-block functions because textual order = SSA dominance order, and the greedy rule is exactly "join the first domain whose existing constraints still admit a common dominator with this alloc's operands". Most-recent-first iteration is robust to future reorderings of the domain list without changing semantics.

The partition is unbounded in the number of domains it can produce. An earlier design capped it at 8 and failed compilation past that, on the theory that a high count only ever indicated pathological IR. That is the wrong trade for production: a graph that *legitimately* needs many domains (several independent host-scratch chains) would fail to compile with no correct fallback, and the only alternative — soft-merging unrelated domains — is silently incorrect. Since the runtime backs domains with grow-on-demand arrays (reallocated the first time a higher `domain_id` is seen), opening another domain costs one more array slot on the cold first inference, so there is no reason to cap. A high count is now a perf advisory (it points at missing upstream canonicalisation), not an error.

### Best-fit static packing, first-fit bin packing

Static allocs have a 1-D byte-offset address space, so best-fit's "smallest gap that fits" produces tighter packings than first-fit. The cost is one extra inner iteration; on `n ≤ 16` allocs per domain the difference is invisible.

Bin packing has no notion of "smallest gap" — bins are bags of non-overlapping lifetimes. First-fit is the natural choice and matches best-fit asymptotically.

Cross-bucket lifetime sharing (e.g. a static slot freed by an alloc dying early reused by a dynamic alloc later) is **not** attempted: it would require resolving offsets at runtime for static allocs too, defeating the constant-offset advantage. The same restriction applies to cross-domain sharing — pools are independent buffers.

### Default-elided `domain_id` attribute

The `Hip_GetPoolOp` declaration uses `DefaultValuedAttr<I64Attr, "0">`. This produces three properties simultaneously:

1. **Backwards-compat IR.** Pre-multi-domain textual IR (no `domain_id` attr) parses as `domain_id = 0`.
2. **Bit-identical single-domain output.** The printer elides default-valued attrs, so single-domain output looks textually unchanged.
3. **Explicit multi-domain output.** Domains 1..N print with `{domain_id = N : i64}`, which is grep-able from IR dumps and machine-checkable from LIT goldens.

The alternative (a separate op kind like `hip.get_pool_n`) would have doubled the surface area for an additive feature. The default-attr approach kept the LIT golden churn tiny.

### Per-domain lazy growth (domain 0 eager)

Domain 0 is allocated eagerly at `pool_init` because:

- The static prefix size is known at compile time and baked into the metadata.
- For canonical single-domain models, domain 0 is the only domain and the eager allocation amortises the `hipMalloc` cost into session startup instead of the first inference.

Domains 1..N start NULL and grow lazily because:

- Their static prefix is often zero (the canonical case is a pure-dynamic late-bound alloc).
- The shape that triggers them might not be hit on every shape combination a session sees. Lazy avoids unnecessary `hipMalloc`s.

The grow path is identical across all domains: stream-sync-then-free-then-malloc. There is no shrink path; the pool keeps its peak footprint for the session.

### 256-byte alignment

The `--hip-pool-allocs` `alignment` option defaults to 256. Two reasons:

1. **GPU coalesced-access requirement.** Most current AMD GPUs require 128-byte alignment for coalesced loads; 256 is the next power of two and matches the largest cache-line size that future GPUs in this family are likely to use. Going lower risks split-transaction penalties on memory-bound kernels.
2. **No reserved-prefix mechanic.** When a domain has zero static allocs, `staticPoolSize == 0` and the first dynamic bin starts at offset 0 — there is no implicit 256-byte slot. The "static prefix" the IR shows in the worked example above is **emergent**: it is `staticPoolSize = max(end of each static reservation)` after the per-alloc end has been rounded up to alignment, and it happens to equal 256 in that example only because the only static there is one 4-byte alloc rounded up.

Hardcoded today; a `ResourceConfig`-style configurable mechanism would be a much larger architectural change and the project is currently single-target.

### Pool ownership: runtime-owned, no per-alloc dealloc

After view replacement, the pass walks the function once and erases any orphaned `memref.dealloc` whose operand is now a `memref.view` of a pool. The pool itself has no `memref.dealloc` — its lifetime is the session, owned by `RuntimeState::pool_base[]`, freed in `hipdnn_ep_state_cleanup`.

Alternative considered: a function-scoped "dealloc on exit" would let the pass ship without runtime-side state. Rejected because (a) ONNX models that get called repeatedly would re-allocate per-call, defeating reuse, and (b) the EP already owns runtime state for other reasons (constants, host scratch, KV cache aliasing) and adding pool slots is uniformly cheap.

---

## Worked Example: Two-Domain Partition

Hand-traced from a real production case — the `embedding.onnx` sub-model of an encoder-style multimodal pipeline whose `main_graph` has eight pooled allocs and triggers a two-domain split.

### Input IR — alloc inventory

Eight allocs in the entry block, classified by dyn-operand chain:

| Alloc | Type | Element bytes | Dyn-operands | Origin of dyn-operands |
|---|---|---|---|---|
| `%alloc` | `memref<?x?xui8>` | 1 | `%dim, %dim_0` | `memref.dim %arg1, %c{0,1}` — function args, already at top |
| `%alloc_1` | `memref<?x?x4096xf16>` | 2 | `%dim, %dim_0` | same |
| `%alloc_3` | `memref<?x?x4096xui8>` | 1 | `%dim, %dim_0` | same |
| `%alloc_4` | `memref<?x?x4096xui8>` | 1 | `%dim, %dim_0` | same |
| `%alloc_6` | `memref<3x?xi64>` | 8 | `%7` | `%7 = arith.muli %6, %c4096`; `%6 = arith.muli %dim, %dim_0` — speculatable arith **interleaved with allocs** |
| `%alloc_7` | `memref<1xi32>` | 4 | (static — none) | n/a (4-byte static) |
| `%alloc_8` | `memref<?x3xi64>` | 8 | `%7` | same as `%alloc_6` |
| `%alloc_11` | `memref<?xf16>` | 2 | `%26` | `%26 = arith.select %25, %dim_10, %24`; chain reaches `%11 = memref.load %expand_shape_9[%c0]` — **non-speculatable load through host-scratch** |

All deallocs land at function tail; every alloc's lifetime overlaps every other's, so there are no intra-domain bin-sharing opportunities in this graph beyond same-key buckets.

### Hoist analysis (runs before pool-allocs)

`hip-hoist-alloc-size-arith` walks SSA backward from each pooled alloc's dyn-operands:

| Chain root | Speculatable predecessors | Hoist action |
|---|---|---|
| `%dim, %dim_0` | already at top | no-op |
| `%6 = arith.muli %dim, %dim_0` | speculatable; operands at top | **hoist to before earliest alloc** |
| `%7 = arith.muli %6, %c4096` | speculatable; operand `%6` will be hoisted | **hoist to before earliest alloc** (after `%6`) |
| `%26` chain | reaches `%11 = memref.load %expand_shape_9[%c0]` — NOT speculatable | hoist stops at the load; the chain stays where it was |

Net hoist effect: moves exactly two ops (`%6`, `%7`) above the earliest pooled alloc. Allocs depending on `%7` (`%alloc_6`, `%alloc_8`) become feasible for a single-domain placement; `%alloc_11`'s chain is unchanged.

### Phase 1.5 partition

Probe each alloc against the most-recent domain in reverse order:

| Step | Alloc | Probe | Feasible? | Action |
|---|---|---|---|---|
| 1 | `%alloc` | (none) | n/a | open `D0` |
| 2..5 | `%alloc_1, %alloc_3, %alloc_4, %alloc_7` | `D0 ∪ {…}` | yes — all depend on `%dim, %dim_0` (already at top) or are static | merge into `D0` |
| 6 | `%alloc_6` | `D0 ∪ {%alloc_6}` | yes — `%7` was hoisted above the earliest alloc | merge into `D0` |
| 7 | `%alloc_8` | `D0 ∪ {%alloc_8}` | yes — same `%7` | merge into `D0` |
| 8 | `%alloc_11` | `D0 ∪ {%alloc_11}` | NO — `%26`'s def is below `%alloc` (the earliest), no insertion point dominates `%alloc` while following `%26`'s def | roll back |
| 8' | `%alloc_11` | (no other domain) | n/a | open `D1` |

Result: `D0 = {%alloc, %alloc_1, %alloc_3, %alloc_4, %alloc_6, %alloc_7, %alloc_8}` (7 allocs, including the rank-0 i32 static), `D1 = {%alloc_11}` (1 dynamic alloc).

### Phases 2..5 — Domain 0

- **Phase 2 split:** 1 static (`%alloc_7`, 4 bytes), 6 dynamic.
- **Phase 3 (statics):** `%alloc_7` placed at offset 0; `staticPoolSize = alignTo(4, 256) = 256` (one 4-byte alloc rounded to alignment 256). The "256-byte static prefix" the IR shows is **not** a reserved slot — it is exactly what falls out of `staticPoolSize = max(end of each static reservation)` when the only static is one tiny i32.
- **Phase 4 buckets** (per `DynSizeKey = {staticFactor, dynOperands}`):

| Bucket | Allocs | `staticFactor` | `dynOperands` | byte size SSA |
|---|---|---|---|---|
| B0 | `%alloc` | 1 | `{%dim, %dim_0}` | `dim * dim_0` |
| B1 | `%alloc_1` | 8192 (= 4096 × 2) | `{%dim, %dim_0}` | `dim * dim_0 * 8192` |
| B2 | `%alloc_3, %alloc_4` | 4096 | `{%dim, %dim_0}` | `dim * dim_0 * 4096`; two bins (lifetimes overlap) |
| B3 | `%alloc_6, %alloc_8` | 24 (= 3 × 8) | `{%7}` | `%7 * 24`; two bins (lifetimes overlap) |

`%alloc_3` and `%alloc_4` share key — same `(staticFactor, dynOperands)` — so they end up in one bucket with two bins. Same story for `%alloc_6` / `%alloc_8` (different memref types `<3x?xi64>` vs `<?x3xi64>` but identical key).

- **Phase 5 anchor (D0):** just after `%7`'s hoisted def, just before `%alloc`. Pool acquisition + bucket size emission:

```mlir
%6 = arith.muli %dim, %dim_0 : index                              // hoisted
%7 = arith.muli %6, %c4096 : index                                // hoisted

%b0_size    = %6                                                  // dim * dim_0
%b0_aligned = align_up(%b0_size, 256)
%b1_size    = arith.muli ..., %c8192                              // staticFactor 8192 % 256 == 0 → align skipped
%b1_aligned = %b1_size
%b2_size    = arith.muli ..., %c4096                              // 4096 % 256 == 0 → align skipped
%b2_aligned = %b2_size
%b3_size    = arith.muli %7, %c24                                 // (3 cols × 8 bytes)
%b3_aligned = align_up(%b3_size, 256)

%dom0_total = arith.constant 256 : index                          // static prefix
%dom0_total = arith.addi %dom0_total, %b0_aligned                 // 1 bin
%dom0_total = arith.addi %dom0_total, %b1_aligned                 // 1 bin
%dom0_total = arith.addi %dom0_total, arith.muli %b2_aligned, %c2 // 2 bins
%dom0_total = arith.addi %dom0_total, arith.muli %b3_aligned, %c2 // 2 bins

%pool_0 = hip.get_pool(%ctx, %dom0_total) : memref<?xi8>          // domain_id = 0 (elided)
```

Per-alloc views (offsets):

| Alloc | Offset | Constant? | `bufferOffsets[i]` |
|---|---|---|---|
| `%alloc` (B0, bin 0) | 256 | yes (static prefix is constant) | `256` |
| `%alloc_1` (B1) | 256 + b0_aligned | SSA | `-1` |
| `%alloc_3` (B2 bin 0) | … + b1_aligned | SSA | `-1` |
| `%alloc_4` (B2 bin 1) | … + b2_aligned | SSA | `-1` |
| `%alloc_6` (B3 bin 0) | … | SSA | `-1` |
| `%alloc_7` (static) | 0 | yes | `0` |
| `%alloc_8` (B3 bin 1) | … | SSA | `-1` |

Static (`%alloc_7`) and the first bin of the first bucket (`%alloc`) end up with constant offsets that survive `arith` folding — `bufferOffsets` records both. Every other alloc's offset depends on SSA chains threaded through the bucket arithmetic, so `bufferOffsets` records `-1`.

### Phases 2..5 — Domain 1

- **Phase 2 split:** 1 dynamic (`%alloc_11`), 0 static.
- **Phase 4:** one bucket B5 = `{staticFactor=2, dynOperands={%26}}`, byte size `%26 * 2`.
- **Phase 5 anchor (D1):** just after `%26`'s def, just before `%alloc_11`. The host-scratch path between Domain 0 and Domain 1 (`memref.store`s, `memref.load`, the `%26` chain) is left untouched.

```mlir
// ... %26 = arith.select %25, %dim_10, %24 : index ...

%b5_size    = arith.muli %26, %c2 : index                         // f16 = 2 bytes
%b5_aligned = align_up(%b5_size, 256)
%dom1_total = %b5_aligned

%pool_1 = hip.get_pool(%ctx, %dom1_total) {domain_id = 1 : i64} : memref<?xi8>

%off_d1 = arith.constant 0 : index
%view_alloc_11 = memref.view %pool_1[%off_d1][%26] : memref<?xi8> to memref<?xf16>
```

`%alloc_11` lives in `D1` at offset 0 (no static prefix in this domain → first dynamic bin starts at 0). Constant offset → `bufferOffsets` records `0`.

### Module attributes (this example)

```
hipdnn.pool_size      = 256                                  : i64
hipdnn.buffer_count   = 8                                    : i64
hipdnn.buffer_offsets = [256, -1, -1, -1, -1, 0, -1, 0]      : array<i64>
                        ↑    ↑ ↑ ↑ ↑    ↑   ↑ ↑
                        |    | | | |    |   | └ %alloc_11 (D1, offset 0)
                        |    | | | |    |   └── %alloc_8 (D0, dyn)
                        |    | | | |    └────── %alloc_7 (D0, static, offset 0)
                        |    | | | └─────────── %alloc_6 (D0, dyn)
                        |    | | └───────────── %alloc_4 (D0, dyn)
                        |    | └─────────────── %alloc_3 (D0, dyn)
                        |    └───────────────── %alloc_1 (D0, dyn)
                        └────────────────────── %alloc    (D0, offset 256)

hipdnn.domain_count   = 2                                    : i64
hipdnn.pool_sizes     = [256, 0]                             : array<i64>
hipdnn.buffer_domains = [0, 0, 0, 0, 0, 0, 0, 1]             : array<i64>
```

`hipdnn.pool_size` (legacy attr) carries domain 0's static prefix = 256. `hipdnn.pool_sizes[0]` repeats this; `hipdnn.pool_sizes[1] = 0` because D1 has no static allocs.

### Lowering output (LLVM dialect, sketched)

```mlir
%c0_i32 = llvm.mlir.constant(0 : i32) : i32
%base0  = llvm.call @hipdnn_ep_get_pool_base(%state, %c0_i32, %dom0_total)
          : (!llvm.ptr, i32, i64) -> !llvm.ptr

%c1_i32 = llvm.mlir.constant(1 : i32) : i32
%base1  = llvm.call @hipdnn_ep_get_pool_base(%state, %c1_i32, %dom1_total)
          : (!llvm.ptr, i32, i64) -> !llvm.ptr
```

At runtime, `pool_base[0]` was eagerly malloc'd at session start to 256 bytes (the static prefix); the first inference's larger `%dom0_total` triggers the grow path on D0 only. the domain-1 array slot did not exist at session start; the first call to `hip.get_pool(%dom1_total) {domain_id = 1}` grows the per-domain arrays and allocates `pool_base[1]` lazily. Subsequent inferences with stable `%dim, %dim_0, %26` are zero-allocation; a shape that pushes `needed_size` beyond the current capacity triggers grow on the affected domain only — the other pool is untouched.

---

## Failure Modes & Diagnostics

### Hard failures (`signalPassFailure`)

| Trigger | Diagnostic | Likely fix |
|---|---|---|
| `funcOp.getBody().hasOneBlock() == false` | "hip-pool-allocs requires single-block functions" | Run earlier passes that collapse control flow into region-bearing ops. |
| `funcOp.getNumArguments() == 0 || arg0 not !hip.context` | "function missing !hip.context as arg 0 — run hip-add-context-arg before hip-pool-allocs" | Add `--hip-add-context-arg` to the pipeline. |
| `alignment` not a positive power of 2 | "alignment must be a positive power of 2 (got N)" | Fix the pass option. |
| `findLatestLegalInsertionPoint` returns `nullopt` for a bucket | "cannot place bucket size arithmetic at a legal point in the block (a dyn-operand def is at-or-after the earliest pooled alloc in its domain)" | Either the hoist pass is missing, or Phase 1.5 partition logic regressed (a dyn-operand below the earliest alloc was admitted into a domain whose anchor is above it). Verify the pipeline; if hoist is in place, this is a partition bug. |
| `findLatestLegalInsertionPoint` returns `nullopt` for the pool itself | "cannot place hip.get_pool at a legal point dominating its domain's allocs" | Same root cause as the bucket-placement variant; surfaces in the second insertion-point query (per-domain pool acquisition rather than per-bucket size arith). |

### Soft no-ops

| Trigger | Behaviour |
|---|---|
| Function is empty | Return without modification. |
| Fewer than 2 pooled allocs (after dropping `result.use_empty()` allocs) | Emit zeroed legacy attrs and return. |
| All allocs are static and lifetime-disjoint | Phase 4 produces no buckets; Phase 3 packs everything into the static prefix; pool acquisition uses a constant `pool_size`. |
| Per-domain: domain has no dynamic buckets AND `staticPoolSize == 0` | Phase 5 skips emission of `hip.get_pool` and the offset arithmetic for that domain entirely (no work to do, no SSA values to thread). The domain still appears in `hipdnn.pool_sizes` as `0` so downstream metadata indexing stays consistent across domains. |
| Domain count unexpectedly high (advisory threshold) | Non-fatal remark suggesting missing upstream canonicalisation; compilation continues with one pool per domain. Triage in this order: (1) `hip-resolve-tensor-dims` is in the pipeline AND `mlir::tensor::registerInferTypeOpInterfaceExternalModels` has been called on the dialect registry (without either, per-layer same-rank dynamic Reshape chains scatter `memref.dim` ops onto bufferized reshape ops and force one domain per reshape site); (2) `hip-hoist-alloc-size-arith` is in the pipeline; (3) any speculatable arithmetic that could be hoisted but is not (e.g. an `arith.divui` whose operands could not all be proven safe). The compiled model is still correct — each extra domain is a valid independent pool — but pooling efficiency suffers. |

### Statistics (collected via LLVM `STATISTIC` macros)

```
hip-pool-allocs.NumAllocsPooled  = total allocs replaced by view
hip-pool-allocs.NumStaticPacked  = static allocs that went through Phase 3
hip-pool-allocs.NumDynBuckets    = number of dynamic buckets created
```

Visible via `hip-mlir-opt --stats` or `--debug-only=hip-pool-allocs`.

---

## Future Work

### Lifetime slot packing within bucket

Currently each bucket's bins map 1:1 to runtime offset slots. A bucket with 4 simultaneously-live allocs of `memref<?x?x4096xf16>` produces 4 distinct slots, each sized `dim*dim_0*8192` bytes. Lifetime-based slot packing could overlap two same-shape allocs whose lifetimes do not actually overlap into one slot — bounded by `max(simultaneously-live sizes)` rather than `sum(sizes)`.

Profiling on real models is needed first: if bin-sharing is rare in production, the algorithmic complexity is not worth it.

### Multi-block functions

If the upstream pipeline ever produces multi-block `@main_graph` (e.g. for early-exit inference or speculative-decoding sub-graphs that share a function), the sequential-index liveness model breaks. Replacement: MLIR's full `Liveness` analysis + dominator-tree-aware insertion points.

### Cross-domain bin sharing

Allocs in different domains can never share offset slots today — pools are independent buffers. A future design where domains share an underlying allocation (with per-domain offset windows) could reclaim that footprint. Probably not worth it: the dominant pool footprint comes from one or two large dynamic buckets, not from prefix overhead.

### Alignment configurability

The 256-byte alignment is hard-coded. Multi-target HIP support (or a future non-HIP backend) would benefit from a `ResourceConfig`-style mechanism. Out of scope until a second target lands.

---

## Related Documents

- [compiler-runtime-contract.md](compiler-runtime-contract.md) — `model_metadata.fbs` schema; how pool attributes are baked into `__metadata_blob`.
- [constant-handling-design.md](constant-handling-design.md) — sibling runtime allocation: model constants (read-only, session-lifetime) vs pool buffers (read-write, grow-on-demand).
- [morphizen-ep-integration.md](morphizen-ep-integration.md) — `inference_init` / `inference_compute` / `inference_cleanup` lifecycle that owns `RuntimeState::pool_base[]`.
