// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: rm -rf %t && mkdir -p %t && cd %t && echo 0123456789abcdef > w.bin && hip-mlir-opt --convert-onnx-to-hipsr --split-input-file --mlir-elide-resource-strings-if-larger=0 %s | FileCheck %s

// CHECK-LABEL: func.func @scalar_const
func.func @scalar_const() -> tensor<f32> {
  // CHECK: arith.constant dense<3.000000e+00> : tensor<f32>
  // CHECK-NOT: hipsr.constant
  %0 = "onnx.Constant"() {value = dense<3.0> : tensor<f32>} : () -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func.func @inline_const
func.func @inline_const() -> tensor<4xf32> {
  // CHECK: hipsr.constant {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>} : tensor<4xf32>
  %0 = "onnx.Constant"() {value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>} : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func.func @mem_resource_const
func.func @mem_resource_const() -> tensor<2x4xf32> {
  // CHECK: hipsr.constant {value = dense_resource<"mem|0x7ff620910000"> : tensor<2x4xf32>} : tensor<2x4xf32>
  %0 = "onnx.Constant"() {location = "*/_ORT_MEM_ADDR_/*", offset = 140695085056000 : i64, size = 32 : i64} : () -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

// CHECK-LABEL: func.func @file_resource_const
func.func @file_resource_const() -> tensor<4xi8> {
  // CHECK: hipsr.constant {value = dense_resource<"file|w.bin|4"> : tensor<4xi8>} : tensor<4xi8>
  %0 = "onnx.Constant"() {location = "w.bin", offset = 4 : i64, size = 4 : i64} : () -> tensor<4xi8>
  return %0 : tensor<4xi8>
}
