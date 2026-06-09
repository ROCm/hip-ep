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
      1. Constants: onnx.Constant -> arith.constant (small/inline) or
         memref.global extern + memref.get_global + bufferization.to_tensor
         (large/externalized to sidecar .constants.bin + .constants.json)
      2. Op mapping: onnx.{MatMul,Transpose,Mul,Softmax} -> hip.* (tensor DPS)
      3. Cleanup: erase onnx.EntryPoint

    Requires hip-add-context-arg to have run first so that !hip.context is
    available as function argument 0.

    Constant externalization (opt-in via --externalize-min-num-elements):
      Large constants above the threshold are written to a binary sidecar file
      and replaced with extern memref.global ops carrying a hip.external_data
      attribute (offset + size).  The filename is recorded as a
      hip.constants_file attribute on the module.  This follows the onnx-mlir
      pattern (krnl.global + omMMapBinaryFile) adapted to standard MLIR ops.
      Splat constants are always kept inline regardless of size.
  }];
  let options = [
    Option<"externalizeOutputDir", "externalize-output-dir", "std::string",
           /*default=*/"\"\"",
           "Directory for sidecar .constants.bin/.json files (empty = cwd)">,
    Option<"externalizeMinNumElements", "externalize-min-num-elements",
           "int64_t", /*default=*/"0",
           "Minimum number of tensor elements to externalize (0 = disabled)">
  ];
  let dependentDialects = [
    "mlir::hip::HipDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::tensor::TensorDialect",
    "mlir::memref::MemRefDialect",
    "mlir::bufferization::BufferizationDialect"
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
  let summary = "Materialize contiguous temporaries for non-canonical memref "
                "operands of hip.* ops";
  let description = [{
    Walks every DestinationStyleOpInterface op in the `hip` dialect.  For
    each input (DPS-input) operand whose memref type has a non-identity
    layout (non-zero offset or non-contiguous strides), inserts:

    ```mlir
    %tmp = memref.alloc() : memref<...identity-layout...>
    memref.copy %strided, %tmp : ..., ...
    // consumer is rewritten to read from %tmp instead of %strided
    memref.dealloc %tmp : memref<...identity-layout...>
    ```

    DPS-init (output) operands are left untouched: in this pipeline they are
    always either fresh `memref.alloc` results or function arguments under
    `IdentityLayoutMap`, so they are guaranteed contiguous by construction.

    The pass exists to bridge the contract gap between the IR (which can
    represent strided memrefs faithfully via memref.subview) and the HIP
    runtime call ABI (which receives a single bare pointer with no offset /
    strides channel).  Without this promotion, hip.* runtime calls silently
    read the base of the parent buffer instead of the intended slice.

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
  let summary = "Pool memref.alloc ops into a grow-on-demand i8 byte buffer";
  let description = [{
    Packs all memref.alloc ops in a single-block function into one contiguous
    byte pool (memref<?xi8>) obtained via hip.get_pool.  Each original
    allocation is replaced with a memref.view at a computed byte offset.

    The pool size is computed as an SSA value and passed to hip.get_pool,
    which uses grow-on-demand semantics: the runtime grows the pool when the
    requested size exceeds current capacity.

    Static allocations use greedy best-fit offset assignment.  Dynamic
    allocations are bucketed by their byte-size SSA value: non-overlapping
    lifetimes within a bucket share an offset.

    Sub-buffer offsets are aligned to 256 bytes by default (configurable).

    Recommended pipeline position:
      hip-optimize-memrefs (optional) -> hip-pool-allocs -> hip-lower-allocs

    Emission strategy:
      Per-bucket size arithmetic and the pool acquisition are anchored at
      the latest legal point in the block — after every dyn-operand def
      and before the earliest pooled alloc. No code motion, no hoist
      whitelist, no reshape pre-folds. See `findLatestLegalInsertionPoint`
      in PoolAllocs.cpp for the helper that picks the point.
  }];
  let options = [
    Option<"alignment", "alignment", "int64_t", /*default=*/"256",
           "Sub-buffer byte offset alignment (must be power of 2)">
  ];
  let dependentDialects = [
    "mlir::hip::HipDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def HoistAllocSizeArithPass
    : Pass<"hip-hoist-alloc-size-arith", "mlir::func::FuncOp"> {
  let summary = "Hoist speculatable size arithmetic feeding memref.alloc dynamic "
                "operands above the earliest dynamic memref.alloc in the entry "
                "block";
  let description = [{
    Hoists speculatable producers of `memref.alloc` dynamic operands to a
    point above the earliest dynamic `memref.alloc` in the function's
    single entry block.  After this pass, every dynamic alloc's
    dyn-operand SSA values dominate the earliest dynamic alloc — which is
    exactly the precondition `--hip-pool-allocs` needs to anchor its
    bucket-size arithmetic and `hip.get_pool` at a single-block-dominant
    point.

    Algorithm
    ---------
    Single-entry-block functions only (PoolAllocs already enforces this);
    multi-block functions are skipped.  Only entry-block allocs are
    considered — allocs nested inside region-carrying ops are untouched.

    1. Collect every entry-block `memref.alloc` with at least one dynamic
       operand.
    2. Pick `earliestAlloc` = the one earliest in the block (per
       `Operation::isBeforeInBlock`).
    3. For each dynamic alloc, walk SSA backward from each of its dynamic
       operands.  An op is hoistable when:
         - it is in the entry block,
         - it is below `earliestAlloc` (otherwise it already dominates),
         - it is not a `memref.alloc` / `memref.alloca`,
         - it satisfies `mlir::isSpeculatable(op)`,
         - and every transitive operand is itself hoistable (or already
           dominates `earliestAlloc`).
    4. Topologically sorted (operands before uses), each hoistable op is
       moved to immediately before `earliestAlloc`.  Forward iteration of
       insertion order is operands-first, so each `moveBefore` pushes
       prior moves up by one slot — final layout has the deepest operands
       at the top of the hoist region.

    Why "speculatable" is the right predicate
    -----------------------------------------
    `mlir::isSpeculatable` checks for both side effects AND undefined
    behaviour.  Conditionally-speculatable ops (`arith.divsi`, `arith.divui`)
    return false unless their divisor is a known-non-zero constant — so
    hoisting them across a runtime guard cannot expose UB earlier.  This
    is the same predicate upstream MLIR's LICM uses, and it admits new
    arith ops automatically as the dialect evolves (vs. an explicit
    allow-list that has to be maintained by hand).

    Pipeline placement
    ------------------
    Wired into `buildOnnxToHipPipelineTail` between
    `--hip-materialize-host-scalars` and `--hip-pool-allocs`.  Running
    after host-scalar materialisation means rewrites that turn an alloc
    into a `memref.view` over host scratch are visible (we will not try
    to hoist their producers).  Running before pool-allocs means
    pool-allocs sees IR where every dynamic alloc's operand chain has
    already been pulled above the earliest dynamic alloc.

    Why a separate pass (vs. an internal phase of pool-allocs)
    ----------------------------------------------------------
    `--hip-pool-allocs` deliberately does not move ops or fold shape
    queries internally; it only places `hip.get_pool` + bucket-size
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

    Before — `%6` and `%7` are speculatable but live below `%alloc`,
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

    After — `%6` and `%7` hoisted to just before `%alloc`; original
    locations now live above the earliest dynamic alloc:

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

def UseOutputAllocatorPass
    : Pass<"hip-use-output-allocator", "mlir::func::FuncOp"> {
  let summary = "Rewrite returned memref.alloc outputs to hip.alloc_output";
  let description = [{
    Replaces each graph-output memref.alloc (a value returned by func.return)
    with hip.alloc_output, reusing the alloc's dynamic-size operands and setting
    out_idx to the return operand position.  Output buffers are then obtained
    from the EP output allocator (EP/runtime-owned) instead of being deallocated
    or pooled.

    Scope: only public (graph-entry) functions are rewritten.  Private helpers
    (e.g. outlined onnx.Loop bodies) carry a !hip.context arg 0 and return
    memref.allocs too, but their results are DLL-internal, never EP outputs, so
    they are left unchanged.  Functions lacking !hip.context as argument 0 are
    likewise skipped.

    The pass also unconditionally stamps the `hipdnn.use_output_allocator` unit
    attribute on the parent module -- the allocator-mode marker that downstream
    convert-hip-to-llvm and generate-interface read to select the allocator ABI.
    It is stamped even when no alloc is rewritten, because the mode is decided by
    the pass being run, not by the IR contents (a zero-output graph must still be
    marked).

    Standalone for now -- registered for hip-mlir-opt and LIT but NOT wired into
    any pipeline (the allocator pipeline placement is a later phase; see
    docs/design/output-allocator-design.md).  Leaves the function signature and
    func.return untouched.
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

    The runtime is responsible for loading the sidecar constants.bin file
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

#endif // HIP_PASSES
