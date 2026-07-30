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

    Converted DPS operations form a data graph, while their placeholders form
    a parallel shape-dependency graph. Block arguments, `arith.constant`, and
    `hipsr.constant` results are graph roots.

    Shape regions remain empty for the later population pass.
  }];
  let dependentDialects = [
    "::mlir::hipsr::HipsrDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::shape::ShapeDialect",
    "::mlir::tensor::TensorDialect"
  ];
}

#endif // HIP_CONVERSION_PASSES
