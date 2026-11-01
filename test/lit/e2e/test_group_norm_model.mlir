// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// com.microsoft GroupNorm through the full pipeline:
//   onnx.Custom -> hip.group_norm -> llvm.call @wrap_group_norm

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 3
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_group_norm
// CHECK: llvm.func @inference_compute
// CHECK-NOT: onnx.Custom
module {
  func.func @main_graph(%arg0: tensor<1x8x4x4xf16> {onnx.name = "X"},
                        %arg1: tensor<8xf16> {onnx.name = "gamma"},
                        %arg2: tensor<8xf16> {onnx.name = "beta"})
      -> (tensor<1x8x4x4xf16> {onnx.name = "Y"}) {
    %0 = "onnx.Custom"(%arg0, %arg1, %arg2)
        <{function_name = "GroupNorm"}>
        {activation = 1 : si64,
         channels_last = 0 : si64,
         domain_name = "com.microsoft",
         epsilon = 9.99999974E-6 : f32,
         groups = 4 : si64,
         onnx_node_name = "GroupNorm_0"}
        : (tensor<1x8x4x4xf16>, tensor<8xf16>, tensor<8xf16>)
        -> tensor<1x8x4x4xf16>
    "onnx.Return"(%0) : (tensor<1x8x4x4xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
