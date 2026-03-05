/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===-- Passes.td - OnnxToHip conversion pass definitions -*- tablegen -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef ONNXTOHIP_PASSES
#define ONNXTOHIP_PASSES

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

#endif // ONNXTOHIP_PASSES
