// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Test Pow E2E full pipeline (constant scalar exponent decompose path).
// onnx.Pow(x, 2) is decomposed to hip.mul (x*x), which lowers to the
// wrap_elementwise runtime call.
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.Pow -> hip.mul (decompose)
// 2. canonicalize: Simplify redundant operations
// 3. memory-pooling: Pool output buffer into single allocation
// 4. convert-hip-to-llvm: HIP ops -> LLVM runtime calls
// 5. generate-interface: Create inference_init/compute/cleanup/metadata

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_elementwise
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Pow
module {
  func.func @main_graph(%arg0: tensor<1x128x512xf16> {onnx.name = "input"}) -> (tensor<1x128x512xf16> {onnx.name = "output"}) {
    %exp = "onnx.Constant"() {value = dense<2.0> : tensor<f16>} : () -> tensor<f16>
    %0 = "onnx.Pow"(%arg0, %exp) {onnx_node_name = "pow_node"} : (tensor<1x128x512xf16>, tensor<f16>) -> tensor<1x128x512xf16>
    "onnx.Return"(%0) : (tensor<1x128x512xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
