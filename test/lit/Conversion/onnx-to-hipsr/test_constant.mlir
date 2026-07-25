// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// convert-onnx-to-hipsr: onnx.Constant -> hipsr.constant / arith.constant.
// Four branches keyed on storage form (no size threshold here):
//   rank-0 scalar          -> arith.constant (compile-time)
//   rank>=1 inline value   -> hipsr.constant {value}
//   "*/_ORT_MEM_ADDR_/*"   -> hipsr.constant {mem_source}
//   other location path    -> hipsr.constant {file_source}
// onnx.Constant is written in generic form so no onnx dialect is required.

// RUN: hip-mlir-opt --convert-onnx-to-hipsr %s | FileCheck %s

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

// CHECK-LABEL: func.func @mem_source_const
func.func @mem_source_const() -> tensor<2x4xf32> {
  // CHECK: hipsr.constant {source = #hipsr.mem_source<140695085056000, 32>} : tensor<2x4xf32>
  %0 = "onnx.Constant"() {location = "*/_ORT_MEM_ADDR_/*", offset = 140695085056000 : i64, size = 32 : i64} : () -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

// CHECK-LABEL: func.func @file_source_const
func.func @file_source_const() -> tensor<100xf32> {
  // CHECK: hipsr.constant {source = #hipsr.file_source<"weights.bin", 1048576, 400>} : tensor<100xf32>
  %0 = "onnx.Constant"() {location = "weights.bin", offset = 1048576 : i64, size = 400 : i64} : () -> tensor<100xf32>
  return %0 : tensor<100xf32>
}
