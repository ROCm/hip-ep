//===-- Passes.td - HIP dialect transform pass definitions -*- tablegen -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.  All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
//
// TableGen definitions for HIP-dialect *transform* passes (passes that
// rewrite HIP IR in place rather than convert across dialects).  Conversion
// passes (ONNX -> HIP, HIP -> LLVM, ONNX -> HipDNN graph outliner) live in
// the sibling umbrella `include/hip/Conversion/Passes.td`, mirroring the
// upstream `mlir/Conversion/Passes.td` / `mlir/Dialect/<X>/Transforms/
// Passes.td` split.
//
// `GenerateInterfacePass` is intentionally hand-rolled in
// `Dialect/Transforms/Passes.h` because it takes a non-TableGen-expressible
// `CompilationOptionsT` constructor argument.
//
//===----------------------------------------------------------------------===//

#ifndef HIP_DIALECT_TRANSFORMS_PASSES
#define HIP_DIALECT_TRANSFORMS_PASSES

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

#endif // HIP_DIALECT_TRANSFORMS_PASSES
