/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===-- HipPasses.td - HIP pass definitions ----------------*- tablegen -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef HIP_PASSES
#define HIP_PASSES

include "mlir/Pass/PassBase.td"

def HipAddContextArgPass : Pass<"hip-add-context-arg", "mlir::ModuleOp"> {
  let summary = "Insert !hip.context as arg 0 into every func.func";
  let description = [{
    Inserts a !hip.context argument at position 0 into every func.func in
    the module. The OnnxToHip conversion patterns assume that the context
    value is always available as the first function argument.

    Skips functions that already have a !hip.context as argument 0.
  }];
  let dependentDialects = [
    "mlir::hip::HipDialect",
    "mlir::func::FuncDialect"
  ];
}

def SimplifyOnnxPass : Pass<"simplify-onnx", "mlir::ModuleOp"> {
  let summary = "Pre-lowering simplification of ONNX-dialect graphs";
  let description = [{
    Runs a small fixed set of ONNX-dialect rewrites that must complete
    BEFORE convert-onnx-to-hip captures the function signatures as input
    metadata. After the per-op rewrites, drops any function argument
    whose only remaining uses were eliminated by them.

    The pass is pure ONNX-dialect: it does not depend on the HIP dialect
    and is intentionally positioned at the head of the pipeline, BEFORE
    hip-add-context-arg, so it operates in the original ONNX function
    index space and could be reused unchanged by any other frontend.

    Pipeline position:
      simplify-onnx -> hip-add-context-arg -> convert-onnx-to-hip

    Per-op rewrites currently implemented:
      * onnx.CastLike -> onnx.Cast
          CastLike's second operand is a *type donor* whose data is never
          read -- the target dtype is statically known from the result
          type. Rewriting to a plain onnx.Cast lets the shared
          CastConversion pattern in convert-onnx-to-hip do the rest, AND
          frees up any function argument that was wired up only to feed
          that type-donor slot.

    Future-friendly: additional simplifiers (e.g. onnx.EyeLike /
    onnx.RandomNormalLike, which are similar shape/type-donor patterns)
    can be added by writing one `static void simplify<Op>(func::FuncOp)`
    helper in SimplifyOnnx.cpp and calling it from the pass dispatcher,
    between the live-before-rewrite snapshot and the shared
    dead-function-argument sweep.

    Invariants:
      * Arguments that were already dead in the input IR are preserved,
        so the existing "metadata captures the original signature"
        contract asserted by test/lit/Pipeline/module-metadata.mlir
        stays intact.
  }];
  let dependentDialects = [
    "mlir::func::FuncDialect"
  ];
}

def ConvertOnnxToHipPass : Pass<"convert-onnx-to-hip", "mlir::ModuleOp"> {
  let summary = "Convert ONNX dialect to HIP dialect (tensor DPS)";
  let description = [{
    Converts ONNX dialect IR (from onnx-mlir) into HIP dialect IR using
    destination-passing style (DPS) with tensor types.

    Transformations:
      1. Constants: onnx.Constant -> hip.constant. The two existing constant
         sweeps remain around compute-op conversion, but neither applies
         storage policy or performs file I/O.
      2. Op mapping: onnx.{MatMul,Transpose,Mul,Softmax} -> hip.* (tensor DPS)
      3. Cleanup: erase onnx.EntryPoint

    Requires hip-add-context-arg to have run first so that !hip.context is
    available as function argument 0.
  }];
  let options = [
    Option<"kvShareBuffer", "kv-share-buffer", "bool", /*default=*/"false",
           "present_key/value name the same allocation as past_key/value "
           "rather than a freshly grown past+current, so the present buffer's "
           "capacity is the past extent alone. Must agree with "
           "past_present_share_buffer in the model's genai_config.json: the "
           "operand shapes cannot tell a right-sized shared buffer from a "
           "growing separate one, since both have past < total.">
  ];
  let dependentDialects = [
    "mlir::hip::HipDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::tensor::TensorDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::memref::MemRefDialect",
    "mlir::bufferization::BufferizationDialect"
  ];
}

def AssignOpStateSlotsPass
    : Pass<"assign-op-state-slots", "mlir::ModuleOp"> {
  let summary = "Assign one op-state slot per stateful op instance";
  let description = [{
    Identity half of the op-state-slots design
    (docs/design/op-state-slots-design.md). Walks the module in program order
    and gives EACH op implementing `OpStateOpInterface` its own dense slot
    `0..N-1`, stamping `hip.op_state_slot` on the op and recording the total as
    the module attribute `hipdnn.num_op_state_slots`.

    Per-instance is the default: two ops of the same kind get two slots and
    two independent state objects. No per-class deduplication is done here.

    Runs before --convert-hip-to-llvm so the slot is available to the lowering
    (threaded into the op's wrap_* call) and to --generate-op-state-init. A
    no-op (no module attr) when no op implements the interface.
  }];
  let dependentDialects = [
    "mlir::hip::HipDialect"
  ];
}

def GenerateOpStateInitPass
    : Pass<"generate-op-state-init", "mlir::ModuleOp"> {
  let summary = "Emit @hipdnn_ep_op_states_init_fn from per-op generateOpStateInit";
  let description = [{
    Construction half of the op-state-slots design
    (docs/design/op-state-slots-design.md). Runs AFTER --assign-op-state-slots
    and BEFORE --convert-hip-to-llvm, while the HIP ops still exist. Emits a
    standalone `llvm.func @hipdnn_ep_op_states_init_fn(!llvm.ptr) -> i32` that:
      1. calls @hipdnn_ep_op_states_alloc(state, N) to size the slot array,
      2. for each stateful op (in slot order) calls the op's
         `generateOpStateInit` (which emits its construct_* call from the op's
         own attributes) and stores the result via @hipdnn_ep_op_state_set.

    --generate-interface later calls this function from inference_init. A no-op
    when `hipdnn.num_op_state_slots` is absent or zero.
  }];
  let dependentDialects = [
    "mlir::hip::HipDialect",
    "mlir::LLVM::LLVMDialect"
  ];
}

def ConvertHipToLLVMPass : Pass<"convert-hip-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert HIP dialect to LLVM dialect";
  let description = [{
    Lowers HIP dialect operations to LLVM dialect by converting each HIP op
    into calls to the HIP runtime C API (hipCreateHandle, hipblasltMatmul,
    etc.). MemRef operands are unpacked into bare pointers for the C calls.
  }];
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def OptimizeMemRefsPass : Pass<"hip-optimize-memrefs", "mlir::func::FuncOp"> {
  let summary = "Optimize memref buffers to reduce allocation count";
  let description = [{
    Performs liveness-based best-fit buffer reuse on straight-line bufferized
    HIP IR.  Replaces memref.alloc ops with earlier compatible buffers whose
    live ranges have ended, reducing peak memory usage.

    Compatibility rules:
      - Static shapes: any slot whose byte-size is >= the needed allocation
        (same element type and memory space).  A memref.reinterpret_cast is
        inserted when shapes differ.
      - Dynamic shapes: the MemRefType must be identical and every dynamic-size
        operand must be the same SSA value.

    Only single-block functions are optimized; multi-block functions are left
    unchanged.
  }];
  let dependentDialects = [
    "mlir::memref::MemRefDialect"
  ];
}

def PromoteStridedHipOperandsPass
    : Pass<"hip-promote-strided-operands", "mlir::func::FuncOp"> {
  let summary = "Materialize identity-layout temporaries for DPS-input memrefs "
                "with non-identity layouts";
  let description = [{
    Walks every DestinationStyleOpInterface op. For each DPS-input memref with
    a non-identity layout (a non-zero offset or non-contiguous strides), inserts:

    ```mlir
    %tmp = memref.alloc() : memref<...identity-layout...>
    memref.copy %strided, %tmp : ..., ...
    // consumer is rewritten to read from %tmp instead of %strided
    memref.dealloc %tmp : memref<...identity-layout...>
    ```

    DPS-init operands are intentionally left untouched because promoting a
    write target requires copy-back and must preserve read-before-write
    semantics. The production pipeline normally supplies identity-layout inits
    through fresh allocations, `hip.alloc_output`, or `IdentityLayoutMap` at
    function boundaries.

    The pass exists to bridge the contract gap between the IR (which can
    represent non-identity layouts faithfully) and runtime ABIs that receive a
    bare pointer with no offset or strides. Matching the DPS interface rather
    than a dialect or operation name covers plugin ops without dialect
    coupling. In the ONNX-to-HIP pipeline, this pass runs after linalg cleanup;
    the remaining DPS operations are HIP or plugin operations.

    Recommended pipeline position:
      hip-optimize-memrefs -> hip-promote-strided-operands -> hip-pool-allocs

    This placement ensures the new transient allocations are routed through
    the HIP memory pool instead of producing extra hipMalloc calls.
  }];
  let dependentDialects = [
    "mlir::memref::MemRefDialect"
  ];
}

def PoolAllocsPass : Pass<"hip-pool-allocs", "mlir::func::FuncOp"> {
  let summary =
      "Pool memref.alloc ops into per-domain grow-on-demand i8 byte buffers";
  let description = [{
    Packs used memref.alloc ops in a single-block function into one byte pool
    per dominance domain. Each pool is obtained with hip.get_pool, and each
    original allocation is replaced with a memref.view at a computed byte
    offset.

    The pool size is computed as an SSA value and passed to hip.get_pool,
    which uses grow-on-demand semantics: the runtime grows the pool when the
    requested size exceeds current capacity.

    Static allocations use greedy best-fit offset assignment. By default, each
    dynamic allocation forms a separate lifetime group. Groups with pairwise
    disjoint member lifetimes may share a slab across different dynamic-size
    operands; the slab width is the runtime maximum of the group footprints.
    Grouped mode first best-fit-packs alignment-multiple allocations with
    SSA-identical ordered dynamic-size operands.

    Sub-buffer offsets are aligned to 256 bytes by default (configurable).

    Recommended pipeline position:
      hip-optimize-memrefs (optional) -> hip-pool-allocs -> hip-lower-allocs

    Dynamic size arithmetic and pool acquisition are emitted after their SSA
    dependencies and before every allocation in the corresponding domain.
  }];
  let options = [
    Option<"alignment", "alignment", "int64_t", /*default=*/"256",
           "Sub-buffer byte offset alignment (must be power of 2)">,
    Option<"emitFragmentationReport", "emit-fragmentation-report", "bool",
           /*default=*/"false",
           "Debug-only: emit per-domain packing remarks with lower-bound "
           "comparisons where representable. No effect on generated code.">,
    Option<"lifetimeOnly", "lifetime-only", "bool", /*default=*/"true",
           "Default: treat each dynamic alloc as a separate lifetime group. "
           "This permits slab reuse across different dynamic-size operands but "
           "does not pack gaps among allocs with identical operands. Set false "
           "to best-fit-pack alignment-multiple allocs with SSA-identical "
           "ordered dynamic-size operands before slab assignment; other allocs "
           "use per-size bins.">
  ];
  let dependentDialects = [
    "mlir::hip::HipDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def HoistAllocSizeArithPass
    : Pass<"hip-hoist-alloc-size-arith", "mlir::func::FuncOp"> {
  let summary = "Hoist pure size arithmetic feeding memref.alloc dynamic "
                "operands above the earliest used memref.alloc in the entry "
                "block";
  let description = [{
    Hoists pure producers of `memref.alloc` dynamic operands to a
    point above the earliest used `memref.alloc` in the function's single entry
    block. After this pass, every hoistable dynamic-size value dominates every
    allocation that `--hip-pool-allocs` may absorb, allowing pool-size
    arithmetic and `hip.get_pool` to share one insertion point.

    Algorithm
    ---------
    Single-entry-block functions only (PoolAllocs already enforces this);
    multi-block functions are skipped.  Only entry-block allocs are
    considered — allocs nested inside region-carrying ops are untouched.

    1. Collect every used entry-block `memref.alloc`, and identify those with
       dynamic operands.
    2. Pick `earliestAlloc` = the earliest used allocation in the block (per
       `Operation::isBeforeInBlock`).
    3. For each dynamic alloc, walk SSA backward from each of its dynamic
       operands.  An op is hoistable when:
         - it is in the entry block,
         - it is below `earliestAlloc` (otherwise it already dominates),
         - it is not a `memref.alloc` / `memref.alloca`,
         - it has no regions and satisfies `mlir::isPure(op)`,
         - and every transitive operand is itself hoistable (or already
           dominates `earliestAlloc`).
    4. Topologically sorted (operands before uses), each hoistable op is
       moved to immediately before `earliestAlloc`.  Forward iteration of
       insertion order is operands-first, so each `moveBefore` pushes
       prior moves up by one slot — final layout has the deepest operands
       at the top of the hoist region.

    Why "pure" is the right predicate
    ---------------------------------
    `mlir::isPure` requires both speculatability and freedom from memory
    effects, matching upstream MLIR's LICM. Conditionally-speculatable ops
    (`arith.divsi`, `arith.divui`) move only when their interface proves
    speculation safe. Region-bearing operations are rejected because their
    regions may capture values not listed as parent-op operands.

    Pipeline placement
    ------------------
    Wired into `buildOnnxToHipPipelineTail` between
    `--hip-materialize-host-scalars` and `--hip-pool-allocs`.  Running
    after host-scalar materialisation means rewrites that turn an alloc
    into a `memref.view` over host scratch are visible (we will not try
    to hoist their producers).  Running before pool-allocs means
    pool-allocs sees IR where every dynamic alloc's operand chain has
    already been pulled above the earliest used alloc when safe.

    Why a separate pass (vs. an internal phase of pool-allocs)
    ----------------------------------------------------------
    `--hip-pool-allocs` deliberately does not move ops or fold shape
    queries internally; it only places `hip.get_pool` + pool-size
    arithmetic at a single dominator point.  Keeping the hoist as a
    standalone, separately-named pass — composed via the pipeline rather
    than embedded in pool-allocs — keeps each pass single-purpose and is
    idiomatic with MLIR convention (`--loop-invariant-code-motion`,
    `--canonicalize`, IREE's `--iree-stream-emplace-transients`, etc.
    are all standalone composable passes).

    Caveat — does NOT pre-canonicalize
    ----------------------------------
    The pass moves existing ops; it does not duplicate, fold, or
    simplify.  If an input IR has e.g. `arith.muli %x, %c0` (which would
    fold to zero), this pass leaves it alone.  Canonicalisation runs
    separately and earlier in the pipeline.

    Idempotent: a second invocation finds no ops below `earliestAlloc`
    that need moving, so it is a no-op.

    Example IR (canonical case after bufferisation)
    ------------------------------------------------------------

    Before — `%6` and `%7` are pure but live below `%alloc`,
    blocking single-block dominator-emit:

    ```mlir
    %dim   = memref.dim %arg, %c0 : memref<?x?xi64>
    %dim_0 = memref.dim %arg, %c1 : memref<?x?xi64>
    %alloc = memref.alloc(%dim, %dim_0) : memref<?x?xui8>
    // ... unrelated ops ...
    %6 = arith.muli %dim, %dim_0      : index
    %7 = arith.muli %6,   %c4096      : index
    %alloc_6 = memref.alloc(%7) : memref<3x?xi64>
    ```

    After — `%6` and `%7` hoisted to just before `%alloc`; original locations
    now live above the earliest used alloc:

    ```mlir
    %dim   = memref.dim %arg, %c0 : memref<?x?xi64>
    %dim_0 = memref.dim %arg, %c1 : memref<?x?xi64>
    %6 = arith.muli %dim, %dim_0      : index
    %7 = arith.muli %6,   %c4096      : index
    %alloc = memref.alloc(%dim, %dim_0) : memref<?x?xui8>
    // ... unrelated ops ...
    %alloc_6 = memref.alloc(%7) : memref<3x?xi64>
    ```

    Non-goals
    ---------
      - Multi-block functions: `--hip-pool-allocs` does not pool them
        either, so they are silently skipped.  Allocs nested in
        region-carrying ops within a single entry block are also left
        untouched (only entry-block allocs are considered).
      - Op duplication / cloning: an op with multiple uses is moved
        once; hoisting follows existing SSA edges.  Cross-region cloning
        à la IREE's `computeSizeSliceInRegion` is unnecessary for
        single-block IR.
      - Static-only allocs: no dynamic operands → nothing to hoist.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def MaterializeHostScalarsPass
    : Pass<"hip-materialize-host-scalars", "mlir::func::FuncOp"> {
  let summary = "Redirect tiny host-fed memref.alloc ops to a host-mapped "
                "scratch buffer";
  let description = [{
    Walks `memref.alloc` ops in the function and redirects ones that look
    like host-fed scalar staging buffers — small static-shape, integer-or-
    index memrefs (up to 16 elements) that have at least one host I/O user
    (memref.store / load) — away from the GPU pool.  They are replaced with
    `memref.view` ops over a single per-function
    `hip.get_host_scratch(%ctx, %total_size)` buffer
    (hipHostMalloc(hipHostMallocMapped), host-writable AND GPU-readable at
    the same VA on UMA targets).

    Hip-dialect users are explicitly accepted (the implementation classifies
    `hip.*` ops as legal users alongside memref.store/load/dim/dealloc).
    The canonical
    `tensor.from_elements -> reduce_sum + sub + cast -> seqlens_k` GQA
    pattern produces an alloc with a host store followed by `hip.cast`
    reading it; rejecting it left the host store crashing inside the GPU
    pool, so the pass intentionally allows it.

    Why: on some targets the GPU pool is real device memory; the bufferized
    `tensor.from_elements` lowering emits `memref.alloc + memref.store`
    which, once absorbed by `hip-pool-allocs`, becomes a host store into
    device memory and SEGVs.  (Other targets silently worked because hipMalloc
    returned UMA-mapped host memory there, masking the bug.)  This pass runs
    BEFORE `hip-pool-allocs` so candidates never enter the GPU pool.

    Pipeline placement is load-bearing — see Pipelines.cpp for the exact
    ordering and the static-shape lockdown LIT under `test/lit/Pipeline/`.
  }];
  let dependentDialects = [
    "mlir::hip::HipDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def LowerAllocsPass : Pass<"hip-lower-allocs", "mlir::func::FuncOp"> {
  let summary = "Lower memref.alloc to hip.alloc and insert hip.free";
  let description = [{
    Replaces memref.alloc operations with hip.alloc (device allocation via
    hipMalloc) and inserts hip.free for buffers that are not returned from
    the function.  Requires !hip.context as function argument 0.

    This pass bridges one-shot-bufferize output (generic memref allocations)
    to the HIP-specific lowering that produces hipMalloc/hipFree calls.
  }];
  let dependentDialects = [
    "mlir::hip::HipDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def RelaxMultiDynExpandShapePass
    : Pass<"hip-relax-multi-dyn-expand-shape", "mlir::func::FuncOp"> {
  let summary = "Rewrite memref.expand_shape ops with multiple dynamic output "
                "dims per reassociation group into memref.reinterpret_cast";
  let description = [{
    Workaround for upstream `expand-strided-metadata` (LLVM as of our
    prebuilt) asserting "There must be at most one dynamic size per group"
    when a `memref.expand_shape` carries a reassociation group containing
    more than one dynamic output dim — even though the op carries explicit
    `output_shape` operands that fully determine every dim's size.

    This pass matches exactly that case and rewrites the op into a
    `memref.reinterpret_cast` with explicit sizes/strides computed from the
    `output_shape` operands and the source's static layout.  The resulting
    `memref.reinterpret_cast` is already lowerable by the existing
    downstream LLVM-conversion patterns.

    Single-dyn-per-group `memref.expand_shape` and all `memref.collapse_shape`
    are left untouched — upstream handles those correctly.

    Canonical IR pattern that triggers a match (ONNX `Range -> Reshape`
    materialising 2-D position_ids):

        %expand = memref.expand_shape %src [[0, 1]]
                  output_shape [%bs, %ss]
                  : memref<?xi64> into memref<?x?xi64>

    After this pass:

        %cast = memref.reinterpret_cast %src to
                  offset: [0], sizes: [%bs, %ss], strides: [%ss, 1]
                  : memref<?xi64> to memref<?x?xi64>

    Pipeline placement: very first sub-pass of `buildHipToLLVMPipeline`,
    immediately before `memref::createExpandStridedMetadataPass()`.

    Retirement: when upstream `expand-strided-metadata` is upgraded to
    consult `output_shape`, this pass becomes a no-op and can be deleted.
  }];
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def ResolveTensorDimsPass
    : Pass<"hip-resolve-tensor-dims", "mlir::func::FuncOp"> {
  let summary = "Fold `tensor.dim` queries via upstream reify patterns "
                "(pre-bufferize)";
  let description = [{
    Per-`func.func` greedy rewrite that wraps upstream MLIR's
    `populateResolveRankedShapedTypeResultDimsPatterns` (plus the
    `InferShapedTypeOpInterface` companion populator and the tensor-
    dialect canonicalisers) in a single fixed point. Folds `tensor.dim`
    of any op carrying `ReifyRankedShapedTypeOpInterface` --
    `tensor.expand_shape`, `tensor.collapse_shape`, `tensor.pad`, and
    every HIP DPS op (which carries the interface via the dialect's
    own external models registered in `InitAllPasses.h`).

    Coverage assumes
    `mlir::tensor::registerInferTypeOpInterfaceExternalModels` has
    been wired into the dialect registry; without that registration
    the upstream patterns are inert and reshape-dim queries silently
    fall through to bufferize, where they get materialised as
    `memref.dim` of a reshape op and explode the dominance domain
    count of `--hip-pool-allocs`.

    Pipeline placement: between `--hip-infer-shapes` and
    `--one-shot-bufferize`, so refined dim values propagate into
    bufferize's allocation sizing. Idempotent.
  }];
  // affine + arith are materialised by the upstream reify folds
  // (`affine.apply` for collapse products, `arith.constant` for static
  // slots); tensor is already loaded as the input dialect, so it is not
  // listed here.
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect"
  ];
}

def ResolveMemRefDimsPass
    : Pass<"hip-resolve-memref-dims", "mlir::func::FuncOp"> {
  let summary = "Fold `memref.dim` of view ops to root-buffer dims "
                "(post-bufferize, before pooling)";
  let description = [{
    Post-bufferize twin of `--hip-resolve-tensor-dims`: re-applies the
    reify-based `memref` dim folds plus the `memref.dim` canonicalisers so
    `memref.dim` of a `memref.subview` / `expand_shape` / `collapse_shape` /
    `view` folds to a dim of the root buffer (ultimately `memref.dim` of a
    function argument).

    The strided views that pin these queries are created during bufferization
    and by `--hip-promote-strided-operands`, i.e. after resolve-tensor-dims
    ran, so the patterns had nothing to fold then. Re-rooting the size to a
    block-top arg dim lets `--hip-hoist-alloc-size-arith` lift it, so
    `--hip-pool-allocs` shares one pool across many allocs instead of splitting
    a dominance domain per alloc. Placement: after promote-strided-operands /
    materialize-host-scalars, before the final CSE / hoist-alloc-size-arith /
    pool-allocs sequence. Idempotent. See ResolveMemRefDims.cpp for the
    before/after IR.
  }];
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def LoopBodyToOutParamsPass
    : Pass<"hip-loop-body-to-out-params", "mlir::ModuleOp"> {
  let summary = "Promote outlined hip.loop body returns to out-param ABI";
  let description = [{
    Runs AFTER `one-shot-bufferize` and BEFORE the pool/lowering passes that
    consume the out-param ABI.

    `onnx-loop-outline` emits each `hip.loop` body as a private
    `func.func` named `*_loop_body_*`. After bufferization those helpers
    still return loop-carried memrefs, while `convert-hip-to-llvm`
    LoopLowering expects the out-param ABI (`v_in` + `v_out` per
    loop-carried value, with `{bufferize.result}` on each out-param).

    This is the loop-body out-param ABI, which is INTERNAL to the DLL and
    unrelated to the graph-entry output allocator ABI. `@main_graph`
    outputs are handled by `hip-use-output-allocator` (`hip.alloc_output`
    + EP callback); this pass only promotes the private `*_loop_body_*`
    helpers by calling MLIR's `promoteBufferResultsToOutParams` with
    `modifyPublicFunctions = false` and a filter on `*_loop_body_*` only.

    Example (one loop-carried memref):
      Before:  private @body(..., %v_in) -> memref<16xf32> { return %out }
      After:   private @body(..., %v_in, %v_out {bufferize.result}) { return }
  }];
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def UseOutputAllocatorPass
    : Pass<"hip-use-output-allocator", "mlir::func::FuncOp"> {
  let summary = "Rewrite returned memref.alloc outputs to hip.alloc_output";
  let description = [{
    Replaces each graph-output memref.alloc with hip.alloc_output, reusing the
    alloc's dynamic-size operands and setting out_idx to its position in
    func.return.  Output buffers then come from the EP output allocator (the EP
    owns and frees them) instead of being deallocated or pooled.

    An alloc counts as a graph output when func.return hands it back -- either
    directly, or after it passes through view ops (memref.cast,
    memref.collapse_shape, memref.expand_shape, memref.subview, any number of
    them).  The pass uses BufferViewFlowAnalysis to check this: an alloc is an
    output if any value made from it via those view ops is returned.  This
    handles reshaped / cast / subview'd outputs without special-casing each op.
    The view ops between the alloc and the return are left in place.

    Only public (graph-entry) functions are rewritten.  Private helpers (e.g.
    outlined onnx.Loop bodies) also take a !hip.context arg and return allocs,
    but those buffers stay inside the DLL and are not EP outputs, so they are
    left unchanged.  Functions whose arg 0 is not !hip.context are skipped too.

    Leaves the function signature and func.return untouched.
  }];
  let dependentDialects = [
    "mlir::hip::HipDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def InferShapesPass : Pass<"hip-infer-shapes", "mlir::ModuleOp"> {
  let summary = "Refine dynamic dims in HIP DPS op result types via "
                "`ReifyRankedShapedTypeOpInterface`";
  let description = [{
    Walks every op that implements `ReifyRankedShapedTypeOpInterface` and
    has at least one `RankedTensorType` result (memref-mode DPS ops are
    skipped). For each such op, calls `reifyResultShapes` to discover
    static dim values that the op contract knows but the IR types still
    spell as `?`, and refines:

      1. The DPS `outs` operand's producer result type, when the producer
         is a `tensor.empty` -- the only zero-cost refinement we can do
         today. The tensor.empty is rebuilt with the refined shape; any
         dynamic dim operands that became static are dropped.
      2. The op's own result type, in place.
      3. Each consumer whose operand type no longer matches: a
         `tensor.cast` is inserted on the use-edge to bridge the new
         (more-static) type back to the original type. DPS consumers
         that take the refined value as their `outs` operand are
         deliberately NOT cast back -- they will be refined in turn on
         the same module walk (topo-ordered by IR position).

    Why DPS-aware
    -------------
    HIP DPS ops require `result_type == outs_operand_type`, so refining
    the op's result without matching the outs operand's producer would
    produce a verifier error. This pass handles the DPS contract by
    rewriting the producer at the same time as the result type. See
    [docs/design/hip-shape-inference.md](docs/design/hip-shape-inference.md)
    for the full design discussion.

    Pipeline placement
    ------------------
    Wired into `buildOnnxToHipPipelineTail` (the shared post-conversion
    segment of `buildOnnxToHipPipeline`) immediately after
    `convert-onnx-to-hip` and before `one-shot-bufferize`. Refining
    tensor result types BEFORE bufferize propagates static dims through
    the alloc / pool sizing computations and removes spurious
    `memref.dim` ops that the legalisation chain would otherwise leave
    at the top of the function. The pass is also a no-op when no op in
    the function carries the interface (or when no further refinement
    is possible), so it is safe to run unconditionally on every model.

    Idempotent: a second invocation that finds no further refinements is
    a no-op.

    Example:
    ```mlir
    // before
    %empty = tensor.empty(%dynM, %dynN) : tensor<?x?xf16>
    %y = hip.matmul(%ctx) ins(%a, %b : tensor<2x4xf16>, tensor<4x8xf16>)
                          outs(%empty : tensor<?x?xf16>) -> tensor<?x?xf16>

    // after
    %empty = tensor.empty() : tensor<2x8xf16>
    %y = hip.matmul(%ctx) ins(%a, %b : tensor<2x4xf16>, tensor<4x8xf16>)
                          outs(%empty : tensor<2x8xf16>) -> tensor<2x8xf16>
    ```
  }];
  let dependentDialects = [
    "mlir::hip::HipDialect",
    "mlir::tensor::TensorDialect"
  ];
}

def InferLoopBodyShapesPass
    : Pass<"hip-infer-loop-body-shapes", "mlir::ModuleOp"> {
  let summary = "Rank-establish unranked tensors inside outlined hip.loop "
                "body functions before `convert-onnx-to-hip`";
  let description = [{
    Runs AFTER `onnx-loop-outline` and BEFORE `convert-onnx-to-hip`. The
    importer (MorphiZen) emits `tensor<*xT>` (unranked) for values whose
    shape it cannot infer -- most commonly an outlined `hip.loop` body's
    loop-carried output (e.g. an `onnx.Concat` that accumulates along the
    sequence axis). `convert-onnx-to-hip`'s op converters require ranked
    result types (`ConcatConversion` bails on unranked), so an unranked
    body output blocks conversion AND leaves the body `func.return`
    operand type-incompatible with the func signature.

    This pass closes that gap by, for every `hip.loop` body func:
      1. Seeding the body's loop-carried block args from the loop op's
         `$v_init` operand types (already ranked post-outline; done for
         robustness / parity with onnx-mlir `ONNXLoopOp::inferShapes`).
      2. Forward-inferring unranked `onnx.*` result types from operand
         types (currently: `onnx.Concat`), iterated to a fixed point.
      3. Loop-contract backstop: any loop-carried body output still
         unranked is set to the matching `$v_init` type -- the ONNX Loop
         spec guarantees body-output type == loop-carried-input type.
      4. Reconciling the body func's declared result types from the (now
         ranked) terminator operand types.

    Post-conversion `--hip-infer-shapes` still handles all `?`-dim
    narrowing on HIP-dialect ops; this pass only establishes rank so
    conversion can proceed.

    Example (one body func):
    ```mlir
    // before
    func.func private @loop_body(%ctx, %i, %c, %acc: tensor<1x?x1152xf16>, ...)
        -> tensor<*xf16> {
      %a = hip.multi_head_attention ... : tensor<?x?x?xf16>
      %r = "onnx.Concat"(%acc, %a) {axis = 1} :
             (tensor<1x?x1152xf16>, tensor<?x?x?xf16>) -> tensor<*xf16>
      return %r : tensor<*xf16>
    }

    // after
    func.func private @loop_body(%ctx, %i, %c, %acc: tensor<1x?x1152xf16>, ...)
        -> tensor<1x?x1152xf16> {
      %a = hip.multi_head_attention ... : tensor<?x?x?xf16>
      %r = "onnx.Concat"(%acc, %a) {axis = 1} :
             (tensor<1x?x1152xf16>, tensor<?x?x?xf16>) -> tensor<1x?x1152xf16>
      return %r : tensor<1x?x1152xf16>
    }
    ```
  }];
  let dependentDialects = [
    "mlir::hip::HipDialect",
    "mlir::func::FuncDialect"
  ];
}

def FixLoopAccumulatorOffsetPass
    : Pass<"hip-fix-loop-accumulator-offset", "mlir::func::FuncOp"> {
  let summary = "Rewrite frozen Concat-accumulator offsets in outlined "
                "hip.loop body functions to iter-driven offsets";
  let description = [{
    Out-param promotion + the LoopLowering trampoline alias each loop-carried
    `v_in` arg and its `{bufferize.result}` `v_out` out-param onto one
    immutable memref descriptor, so `memref.dim %v_in, %cN` is frozen at the
    v_init dim every iteration. A growing-Concat accumulator then appends each
    chunk at that frozen offset and only the last chunk survives.

    This pass rewrites each chunk-append `memref.subview` OFFSET from the frozen
    `memref.dim` to the real per-iter chunk start: `seqlens_k[iter]` (read via a
    synchronized `hip.readback_scalar` of the body's first
    `hip.gather -> memref<1xi32>`), or `iter * chunk_size` when there is no
    seqlens gather. Only OFFSETs are touched. Runs after out-param promotion
    and before the pool/hoist passes. See FixLoopAccumulatorOffset.cpp for the
    matcher and a Before/After snippet.
  }];
  let dependentDialects = [
    "mlir::hip::HipDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def ResolveExternConstantsPass
    : Pass<"hip-resolve-extern-constants", "mlir::ModuleOp"> {
  let summary = "Replace extern memref.global constants with "
                "hip.get_constant ops";
  let description = [{
    Resolves externalized constants (memref.global ops with hip.external_data
    attributes produced by --convert-onnx-to-hip) by replacing each
    memref.get_global with a hip.get_constant(%ctx, %index) operation.

    The index is read from the hip.external_data attribute on each extern
    memref.global. The !hip.context is taken from function argument 0
    (inserted by --hip-add-context-arg).

    The runtime is responsible for loading the constants.bin file
    via morphizen::FileSystem, uploading constants to GPU memory, and
    returning GPU pointers via hipdnn_ep_constant_get(state, index).

    This pass should run after all buffer optimizations (hip-pool-allocs,
    hip-lower-allocs) and before --convert-hip-to-llvm.
  }];
  let dependentDialects = [
    "mlir::hip::HipDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def ExternalizeConstantsPass
    : Pass<"hip-externalize-constants", "mlir::ModuleOp"> {
  let summary = "Apply storage policy to hip.constant carriers";
  let description = [{
    Validates every `hip.constant` and the unique compiler-owned absolute order,
    appends unmarked plugin carriers in stable module walk order, plans one
    64-byte-aligned layout, serializes the current constants.bin/json artifacts,
    and only then commits IR rewrites and the existing parallel module metadata
    arrays.

    Inline carriers below the threshold become `arith.constant`. Carriers at or
    above it, plus every file-backed or memory-backed carrier when
    externalization is enabled, become the existing private extern
    `memref.global` +
    `bufferization.to_tensor` bridge consumed by
    `hip-resolve-extern-constants`.

    This pass preserves full-sidecar, streaming, and hybrid skip-data behavior.
    It must run after all constant producers and shape inference, and before
    canonicalization and bufferization.
  }];
  let options = [
    Option<"externalizeMinNumElements", "externalize-min-num-elements",
           "int64_t", /*default=*/"0",
           "Minimum number of tensor elements to externalize (0 = disabled)">,
    Option<"externalizeOutputDir", "externalize-output-dir", "std::string",
           /*default=*/"\"\"",
           "Directory for constants-file .constants.bin/.json files (empty = cwd)">,
    Option<"skipConstantData", "skip-constant-data", "bool",
           /*default=*/"false",
           "Use existing streaming/hybrid source metadata instead of writing "
           "all constant bytes to constants.bin">
  ];
  let dependentDialects = [
    "mlir::hip::HipDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect",
    "mlir::bufferization::BufferizationDialect"
  ];
}

def SplitDuplicateDpsInitsPass
    : Pass<"hip-split-duplicate-dps-inits", "mlir::func::FuncOp"> {
  let summary = "Give repeated tensor.empty DPS init uses distinct allocations";
  let description = [{
    CSE may merge pure, same-typed `tensor.empty` operations that seed unrelated
    destination-style operations. The sharing carries no tensor-level data
    dependence, but preserving it through bufferization can alias outputs that
    must be distinct and can form large One-Shot Bufferize alias classes.

    The pass preserves the first DPS init use of each `tensor.empty` and
    replaces later init uses with fresh `bufferization.alloc_tensor` operations.
    This is value-preserving because both operations produce undefined tensor
    contents. Non-empty init values remain unchanged. Downstream lifetime
    analysis may still reuse the resulting buffers.

    Run this pass after CSE and before One-Shot Bufferize.

    Before (post-CSE; one %e feeds two DPS ops):

        %e = tensor.empty() : tensor<AxBxf16>
        %a = hip.sigmoid ... outs(%e)
        %b = hip.tanh    ... outs(%e)

    After:

        %e  = tensor.empty()               : tensor<AxBxf16>
        %e2 = bufferization.alloc_tensor() : tensor<AxBxf16>
        %a = hip.sigmoid ... outs(%e)
        %b = hip.tanh    ... outs(%e2)
  }];
  let dependentDialects = [
    "mlir::bufferization::BufferizationDialect",
    "mlir::tensor::TensorDialect"
  ];
}

#endif // HIP_PASSES
