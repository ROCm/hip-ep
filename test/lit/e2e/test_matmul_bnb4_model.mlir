// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Static MatMulBnb4. The dequant kernel and hipBLASLt are both in the
// artifact. A fully static graph may also fold the dequant into rocMLIR when
// that pass runs; this pipeline check is the hip-cc fallback.

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 3
// CHECK-SAME: hipdnn.output_count = 1
// CHECK-NOT: onnx.Custom
// CHECK-DAG: llvm.func @wrap_dequant_bnb4
// CHECK-DAG: llvm.func @wrap_hipblasLtMatmulTyped
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
module {
  func.func @main_graph(
      %arg0: tensor<1x4x8xf32> {onnx.name = "input"},
      %arg1: tensor<16xui8> {onnx.name = "packed"},
      %arg2: tensor<2xf32> {onnx.name = "absmax"})
      -> (tensor<1x4x4xf32> {onnx.name = "output"}) {
    %y = "onnx.Custom"(%arg0, %arg1, %arg2) {
      function_name = "MatMulBnb4",
      domain_name = "com.microsoft",
      K = 8 : si64, N = 4 : si64, block_size = 16 : si64,
      quant_type = 1 : si64, transB = 1 : si64,
      onnx_node_name = "MatMul_Bnb4"
    } : (tensor<1x4x8xf32>, tensor<16xui8>, tensor<2xf32>) -> tensor<1x4x4xf32>
    "onnx.Return"(%y) : (tensor<1x4x4xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
