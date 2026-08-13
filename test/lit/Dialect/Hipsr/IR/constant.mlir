// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @inline_const(
// CHECK-NEXT: hipsr.constant {value = dense<{{.*}}> : tensor<4xf16>} : memref<4xf16, #hipsr.mem<device>>
func.func @inline_const() -> memref<4xf16, #hipsr.mem<device>> {
  %c = hipsr.constant {value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf16>}
     : memref<4xf16, #hipsr.mem<device>>
  return %c : memref<4xf16, #hipsr.mem<device>>
}

// -----

// CHECK-LABEL: func.func @inline_const_tensor(
// CHECK-NEXT: hipsr.constant {value = dense<{{.*}}> : tensor<4xf16>} : tensor<4xf16>
func.func @inline_const_tensor() -> tensor<4xf16> {
  %c = hipsr.constant {value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf16>}
     : tensor<4xf16>
  return %c : tensor<4xf16>
}

// -----

// CHECK-LABEL: func.func @file_resource_const(
// CHECK-NEXT: hipsr.constant {value = dense_resource<"file|w.bin|1024"> : tensor<100xf32>} : memref<100xf32, #hipsr.mem<device>>
func.func @file_resource_const() -> memref<100xf32, #hipsr.mem<device>> {
  %c = hipsr.constant {value = dense_resource<"file|w.bin|1024"> : tensor<100xf32>}
     : memref<100xf32, #hipsr.mem<device>>
  return %c : memref<100xf32, #hipsr.mem<device>>
}

// -----

// CHECK-LABEL: func.func @mem_resource_const(
// CHECK-NEXT: hipsr.constant {value = dense_resource<"mem|0x7ff620910000"> : tensor<2x4xf32>} : memref<2x4xf32, #hipsr.mem<device>>
func.func @mem_resource_const() -> memref<2x4xf32, #hipsr.mem<device>> {
  %c = hipsr.constant {value = dense_resource<"mem|0x7ff620910000"> : tensor<2x4xf32>}
     : memref<2x4xf32, #hipsr.mem<device>>
  return %c : memref<2x4xf32, #hipsr.mem<device>>
}

// -----

// CHECK-LABEL: func.func @externalized_const(
// CHECK-NEXT: hipsr.constant {index = 0 : i64, offset = 0 : i64, size = 524288 : i64, value = dense_resource<"mem|0x7ff620910000"> : tensor<512x512xf16>} : memref<512x512xf16, #hipsr.mem<device>>
func.func @externalized_const() -> memref<512x512xf16, #hipsr.mem<device>> {
  %c = hipsr.constant {value = dense_resource<"mem|0x7ff620910000"> : tensor<512x512xf16>, offset = 0 : i64, size = 524288 : i64, index = 0 : i64}
     : memref<512x512xf16, #hipsr.mem<device>>
  return %c : memref<512x512xf16, #hipsr.mem<device>>
}

// -----

func.func @missing_value_rejected() -> memref<4xf16, #hipsr.mem<device>> {
  // expected-error @+1 {{requires attribute 'value'}}
  %c = hipsr.constant {} : memref<4xf16, #hipsr.mem<device>>
  return %c : memref<4xf16, #hipsr.mem<device>>
}

// -----

func.func @sparse_value_rejected() -> memref<3x4xi32, #hipsr.mem<device>> {
  // expected-error @+1 {{value must be dense<...> or dense_resource<...>}}
  %c = hipsr.constant {value = sparse<[[0, 0], [1, 2]], [1, 5]> : tensor<3x4xi32>}
     : memref<3x4xi32, #hipsr.mem<device>>
  return %c : memref<3x4xi32, #hipsr.mem<device>>
}

// -----

func.func @splat_value_rejected() -> memref<4xf16, #hipsr.mem<device>> {
  // expected-error @+1 {{splat value is not supported.}}
  %c = hipsr.constant {value = dense<1.0> : tensor<4xf16>}
     : memref<4xf16, #hipsr.mem<device>>
  return %c : memref<4xf16, #hipsr.mem<device>>
}

// -----

// A single-element tensor is stored splat but its raw data is already complete.
// CHECK-LABEL: func.func @single_element_const(
// CHECK-NEXT: hipsr.constant {value = dense<{{.*}}> : tensor<1xf16>} : memref<1xf16, #hipsr.mem<device>>
func.func @single_element_const() -> memref<1xf16, #hipsr.mem<device>> {
  %c = hipsr.constant {value = dense<1.0> : tensor<1xf16>}
     : memref<1xf16, #hipsr.mem<device>>
  return %c : memref<1xf16, #hipsr.mem<device>>
}

// -----

// A memref with no memory space is rejected by the ODS type constraint
// (Hipsr_TensorOrDeviceMemRef), not by the verifier.
func.func @non_device_space_rejected() -> memref<4xf16> {
  // expected-error @+1 {{result #0 must be ranked tensor or device memref}}
  %c = hipsr.constant {value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf16>}
     : memref<4xf16>
  return %c : memref<4xf16>
}
