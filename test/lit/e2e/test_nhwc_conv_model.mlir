// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// com.microsoft.NhwcConv through the full pipeline. Layout transposes reuse
// wrap_transpose. The convolution is 1x1 with stride 1 and no padding, so
// hip-fusion-transform runs it as a hipBLASLt matmul plus a bias add
// (wrap_elementwise) rather than wrap_conv.

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 3
// CHECK-SAME: hipdnn.output_count = 1
// CHECK-DAG: llvm.func @wrap_transpose
// CHECK-DAG: llvm.func @wrap_hipblasLtMatmul
// CHECK-DAG: llvm.func @wrap_elementwise
// CHECK-NOT: llvm.func @wrap_conv
// CHECK: llvm.func @inference_compute
// CHECK-NOT: NhwcConv
module {
  func.func @main_graph(%arg0: tensor<1x8x8x4xf16> {onnx.name = "X"},
                        %arg1: tensor<4x1x1x4xf16> {onnx.name = "W"},
                        %arg2: tensor<4xf16> {onnx.name = "B"})
      -> (tensor<1x8x8x4xf16> {onnx.name = "Y"}) {
    %0 = "onnx.Custom"(%arg0, %arg1, %arg2)
        <{function_name = "NhwcConv"}>
        {auto_pad = "NOTSET",
         dilations = [1, 1],
         domain_name = "com.microsoft",
         group = 1 : si64,
         kernel_shape = [1, 1],
         pads = [0, 0, 0, 0],
         strides = [1, 1],
         onnx_node_name = "NhwcConv_0"}
        : (tensor<1x8x8x4xf16>, tensor<4x1x1x4xf16>, tensor<4xf16>)
        -> tensor<1x8x8x4xf16>
    "onnx.Return"(%0) : (tensor<1x8x8x4xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
