// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// onnx.Return forms that do not survive the conversion.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --onnx-dialect=modeled --convert-onnx-to-hipsr --split-input-file --verify-diagnostics %s

// onnx.Return carries HasParent<FuncOp>, so that trait rejects an onnx.Return
// nested in a region before any conversion pattern runs.
func.func @return_outside_func(%ctx: !hipsr.context) {
  hipsr.compute(%ctx) ins() outs() {
  ^bb0(%body_ctx: !hipsr.context):
    // expected-error @+1 {{'onnx.Return' op expects parent op 'func.func'}}
    "onnx.Return"() : () -> ()
    hipsr.compute_yield
  }
  return
}
