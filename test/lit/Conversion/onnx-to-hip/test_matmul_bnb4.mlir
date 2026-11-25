// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

// K=8, N=4, block_size=16 -> 32 weights, 16 packed bytes, 2 absmax values.
// The dequant writes [N, K] and the matmul transposes that buffer.

// CHECK-LABEL: func.func @main_graph
// CHECK: tensor.empty() : tensor<4x8xf32>
// CHECK: hip.dequant_bnb4
// CHECK-SAME: K = 8
// CHECK-SAME: N = 4
// CHECK-SAME: block_size = 16
// CHECK-SAME: quant_type = 1
// CHECK: tensor.dim
// CHECK: hip.matmul
// CHECK-SAME: transB = 1
// CHECK-NOT: onnx.Custom
func.func @main_graph(%a: tensor<?x4x8xf32>, %packed: tensor<16xui8>,
                       %absmax: tensor<2xf32>) -> tensor<?x4x4xf32> {
  %y = "onnx.Custom"(%a, %packed, %absmax) {
    function_name = "MatMulBnb4",
    domain_name = "com.microsoft",
    K = 8 : si64, N = 4 : si64, block_size = 16 : si64,
    quant_type = 1 : si64, training_mode = 0 : si64, transB = 1 : si64,
    onnx_node_name = "MatMul_Bnb4"
  } : (tensor<?x4x8xf32>, tensor<16xui8>, tensor<2xf32>) -> tensor<?x4x4xf32>
  return %y : tensor<?x4x4xf32>
}

// CHECK-LABEL: func.func @matmul_bnb4_training
// CHECK: onnx.Custom
// CHECK-SAME: function_name = "MatMulBnb4"
func.func @matmul_bnb4_training(%a: tensor<1x4x8xf32>, %packed: tensor<16xui8>,
                                %absmax: tensor<2xf32>) -> tensor<1x4x4xf32> {
  %y = "onnx.Custom"(%a, %packed, %absmax) {
    function_name = "MatMulBnb4",
    domain_name = "com.microsoft",
    K = 8 : si64, N = 4 : si64, block_size = 16 : si64,
    quant_type = 1 : si64, training_mode = 1 : si64, transB = 1 : si64
  } : (tensor<1x4x8xf32>, tensor<16xui8>, tensor<2xf32>) -> tensor<1x4x4xf32>
  return %y : tensor<1x4x4xf32>
}
