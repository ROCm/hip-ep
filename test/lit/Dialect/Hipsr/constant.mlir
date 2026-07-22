// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// Round-trips the hipsr.constant forms and checks the verifier: exactly one of
// {value, source} is always present, offset/size/index are a grouped
// externalization marker layered on top, and the result must be device memory.

// RUN: hip-mlir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @inline_const
func.func @inline_const() -> memref<4xf16, #hipsr.mem<device>> {
  // CHECK: hipsr.constant {value = dense<{{.*}}> : tensor<4xf16>} : memref<4xf16, #hipsr.mem<device>>
  %c = hipsr.constant {value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf16>}
     : memref<4xf16, #hipsr.mem<device>>
  return %c : memref<4xf16, #hipsr.mem<device>>
}

// -----

// A ranked tensor result is allowed (pre-bufferization form, as produced by
// the onnx->hipsr conversion before bufferization).
// CHECK-LABEL: func.func @inline_const_tensor
func.func @inline_const_tensor() -> tensor<4xf16> {
  // CHECK: hipsr.constant {value = dense<{{.*}}> : tensor<4xf16>} : tensor<4xf16>
  %c = hipsr.constant {value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf16>}
     : tensor<4xf16>
  return %c : tensor<4xf16>
}

// -----

// CHECK-LABEL: func.func @file_source_const
func.func @file_source_const() -> memref<100xf32, #hipsr.mem<device>> {
  // CHECK: hipsr.constant {source = #hipsr.file_source<"w.bin", 0, 1000>} : memref<100xf32, #hipsr.mem<device>>
  %c = hipsr.constant {source = #hipsr.file_source<"w.bin", 0, 1000>}
     : memref<100xf32, #hipsr.mem<device>>
  return %c : memref<100xf32, #hipsr.mem<device>>
}

// -----

// CHECK-LABEL: func.func @mem_source_const
func.func @mem_source_const() -> memref<2x4xf32, #hipsr.mem<device>> {
  // CHECK: hipsr.constant {source = #hipsr.mem_source<140695085056000, 32>} : memref<2x4xf32, #hipsr.mem<device>>
  %c = hipsr.constant {source = #hipsr.mem_source<140695085056000, 32>}
     : memref<2x4xf32, #hipsr.mem<device>>
  return %c : memref<2x4xf32, #hipsr.mem<device>>
}

// -----

// Externalized: data source (value here) is kept; offset/size/index are added.
// CHECK-LABEL: func.func @externalized_const
func.func @externalized_const() -> memref<512x512xf16, #hipsr.mem<device>> {
  // CHECK: hipsr.constant {index = 0 : i64, offset = 0 : i64, size = 524288 : i64, value = dense<{{.*}}> : tensor<512x512xf16>} : memref<512x512xf16, #hipsr.mem<device>>
  %c = hipsr.constant {value = dense<1.0> : tensor<512x512xf16>, offset = 0 : i64, size = 524288 : i64, index = 0 : i64}
     : memref<512x512xf16, #hipsr.mem<device>>
  return %c : memref<512x512xf16, #hipsr.mem<device>>
}

// -----

func.func @both_value_and_source_rejected() -> memref<4xf16, #hipsr.mem<device>> {
  // expected-error @+1 {{expected exactly one of {value} or {source}}}
  %c = hipsr.constant {value = dense<1.0> : tensor<4xf16>, source = #hipsr.mem_source<140695085056000, 8>}
     : memref<4xf16, #hipsr.mem<device>>
  return %c : memref<4xf16, #hipsr.mem<device>>
}

// -----

func.func @offset_without_size_rejected() -> memref<4xf16, #hipsr.mem<device>> {
  // expected-error @+1 {{`offset`, `size` and `index` must be set together}}
  %c = hipsr.constant {value = dense<1.0> : tensor<4xf16>, offset = 0 : i64}
     : memref<4xf16, #hipsr.mem<device>>
  return %c : memref<4xf16, #hipsr.mem<device>>
}

// -----

// A memref with no memory space is rejected by the ODS type constraint
// (Hipsr_TensorOrDeviceMemRef), not by the verifier.
func.func @non_device_space_rejected() -> memref<4xf16> {
  // expected-error @+1 {{result #0 must be ranked tensor or device memref}}
  %c = hipsr.constant {value = dense<1.0> : tensor<4xf16>} : memref<4xf16>
  return %c : memref<4xf16>
}
