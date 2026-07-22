//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// Aggregate definitions for TableGen-based HIP conversion passes, mirroring
// upstream mlir/Conversion/Passes.td. New conversions add their `def` here; the
// per-conversion header under Conversion/<XToY>/ pulls just its declaration via
// GEN_PASS_DECL_<NAME>, and Conversion/Passes.h registers them all.
//

#ifndef HIP_CONVERSION_PASSES
#define HIP_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertOnnxToHipsrPass : Pass<"convert-onnx-to-hipsr", "mlir::ModuleOp"> {
  let summary = "Convert ONNX dialect to the hipsr dialect (tensor DPS)";
  let description = [{
    Converts ONNX dialect IR into hipsr dialect IR using destination-passing
    style with tensor types. ONNX ops are matched by name via the generic MLIR
    Operation API, so no onnx-mlir headers are required.

    Each shaped op's output-shape computation is emitted by the op's own
    ShapeRegionInterface::populateShapeRegion(), the single source of truth,
    so this pass exercises that code path (unlike round-trip tests).
  }];
  let dependentDialects = [
    "::mlir::hipsr::HipsrDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::shape::ShapeDialect",
    "::mlir::tensor::TensorDialect"
  ];
}

def ConvertHipsrToLLVMPass : Pass<"convert-hipsr-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower the hipsr dialect to LLVM";
  let description = [{
    Lowers hipsr ops to the LLVM dialect. Currently handles `hipsr.constant`:
    an externalized constant (offset/size set) becomes a
    `@wrap_get_global(ctx, offset, size)` runtime call wrapped in a memref
    descriptor, mirroring the `hip.get_constant` lowering. `!hip.context`
    function arguments convert to `!llvm.ptr`, and `#hipsr.mem<device>` memrefs
    map to their AMDGPU integer address space.
  }];
  let dependentDialects = [
    "::mlir::LLVM::LLVMDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::hipsr::HipsrDialect"
  ];
}

#endif // HIP_CONVERSION_PASSES
