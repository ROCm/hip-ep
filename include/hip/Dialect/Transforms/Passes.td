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

    Static allocations use greedy best-fit offset assignment (inspired by
    greedy best-fit strip packing).  Dynamic allocations
    are bucketed by their byte-size SSA value: non-overlapping lifetimes within
    a bucket share an offset.

    Sub-buffer offsets are aligned to 256 bytes by default (configurable).

    Recommended pipeline position:
      hip-optimize-memrefs (optional) -> hip-pool-allocs -> hip-lower-allocs
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

def UseOutputAllocatorPass
    : Pass<"hip-use-output-allocator", "mlir::func::FuncOp"> {
  let summary = "Rewrite returned memref.alloc outputs to hip.alloc_output";
  let description = [{
    Replaces each graph-output memref.alloc (a value returned by func.return)
    with hip.alloc_output, reusing the alloc's dynamic-size operands and setting
    out_idx to the return operand position.  Output buffers are then obtained
    from the EP output allocator (EP/runtime-owned) instead of being deallocated
    or pooled.  Requires !hip.context as function argument 0; functions without
    it are left unchanged.

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
