//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef HIPSR_TRANSFORMS_PASSES
#define HIPSR_TRANSFORMS_PASSES

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

#endif // HIPSR_TRANSFORMS_PASSES
