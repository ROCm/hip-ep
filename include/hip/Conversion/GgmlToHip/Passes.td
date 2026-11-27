//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef HIP_CONVERSION_GGMLTOHIP_PASSES_TD
#define HIP_CONVERSION_GGMLTOHIP_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertGgmlToHipPass
    : Pass<"convert-ggml-to-hip", "mlir::ModuleOp"> {
  let summary = "Convert GGML operations to HIP target dialects";
  let description = [{
    Converts registered `ggml.*` operations directly to tensor-form `hip.*`
    and standard MLIR operations without routing through ONNX. The conversion is all-or-nothing: any GGML operation
    without a semantics-preserving lowering causes the pass to fail instead of
    leaving a mixed source/target module.

    The pass expects `hip-add-context-arg` to have inserted `!hip.context` as
    argument zero of each function containing a lowered compute operation.
    A packed K-quant weight input stays packed: the native kernels read its
    bytes where the caller put them.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::bufferization::BufferizationDialect",
    "::mlir::func::FuncDialect",
    "::mlir::ggml::GgmlDialect",
    "::mlir::hip::HipDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tensor::TensorDialect"
  ];
}

#endif // HIP_CONVERSION_GGMLTOHIP_PASSES_TD
