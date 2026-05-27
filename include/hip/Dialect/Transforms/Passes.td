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

def InsertTensorDumpPass
    : Pass<"hip-insert-tensor-dump", "mlir::ModuleOp"> {
  let summary = "Insert hip.dump_tensor after every HIP compute op";
  let description = [{
    Walks every hip.* compute operation and inserts a hip.dump_tensor op
    after it for each memref-typed result.  Skipped for infrastructure ops
    (alloc, free, get_constant, get_pool, yield, dump_tensor).

    The op name is derived from the MLIR Location metadata when available,
    falling back to {hip_op_type}_{index}.

    This pass is conditionally added by the compiler driver when the
    dump_tensors compilation option is enabled.
  }];
  let options = [
    Option<"dumpTensorsDir", "dump-tensors-dir", "std::string", /*default=*/"\"\"",
           "Directory where .npy files will be written at inference time">
  ];
  let dependentDialects = [
    "mlir::hip::HipDialect"
  ];
}

#endif // HIP_PASSES
