// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hipsr-externalize-constants %s -split-input-file | FileCheck %s

// An inline dense value is slot 0 at offset 0.
// CHECK-LABEL: module attributes {hip.constants_file = "constants.bin", hipdnn.constant_offsets = array<i64: 0>, hipdnn.constant_sizes = array<i64: 16>} {
// CHECK-NEXT: func.func @inline_value() -> tensor<4xf32, #hipsr.mem<device>> {
// CHECK-NEXT: %[[CONSTANT_0:.*]] = hipsr.constant {index = 0 : i64, offset = 0 : i64, size = 16 : i64, value = dense<{{\[}}1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00{{\]}}> : tensor<4xf32>} : tensor<4xf32, #hipsr.mem<device>>
// CHECK-NEXT: return %[[CONSTANT_0]] : tensor<4xf32, #hipsr.mem<device>>
// CHECK-NEXT: }
// CHECK-NEXT: }
func.func @inline_value() -> tensor<4xf32, #hipsr.mem<device>> {
  %0 = hipsr.constant {value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>} : tensor<4xf32, #hipsr.mem<device>>
  return %0 : tensor<4xf32, #hipsr.mem<device>>
}

// -----

// A file resource uses the same sidecar layout as an inline value.
// CHECK-LABEL: module attributes {hip.constants_file = "constants.bin", hipdnn.constant_offsets = array<i64: 0>, hipdnn.constant_sizes = array<i64: 16>} {
// CHECK-NEXT: func.func @file_resource() -> tensor<2xi64, #hipsr.mem<device>> {
// CHECK-NEXT: %[[CONSTANT_0:.*]] = hipsr.constant {index = 0 : i64, offset = 0 : i64, size = 16 : i64, value = dense_resource<"file|w.bin|100"> : tensor<2xi64>} : tensor<2xi64, #hipsr.mem<device>>
// CHECK-NEXT: return %[[CONSTANT_0]] : tensor<2xi64, #hipsr.mem<device>>
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK-NEXT: {-#
// CHECK-NEXT: dialect_resources: {
// CHECK-NEXT: builtin: {
// CHECK-NEXT: "file|w.bin|100": "0x0100000001000000000000000200000000000000"
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: #-}
func.func @file_resource() -> tensor<2xi64, #hipsr.mem<device>> {
  %0 = hipsr.constant {value = dense_resource<"file|w.bin|100"> : tensor<2xi64>} : tensor<2xi64, #hipsr.mem<device>>
  return %0 : tensor<2xi64, #hipsr.mem<device>>
}

{-#
  dialect_resources: {
    builtin: {
      "file|w.bin|100": "0x0100000001000000000000000200000000000000"
    }
  }
#-}

// -----

// A file resource and an inline value share one 64-byte-aligned layout.
// CHECK-LABEL: module attributes {hip.constants_file = "constants.bin", hipdnn.constant_offsets = array<i64: 0, 64>, hipdnn.constant_sizes = array<i64: 16, 8>} {
// CHECK-NEXT: func.func @cumulative_alignment() -> (tensor<2xi64, #hipsr.mem<device>>, tensor<2xf32, #hipsr.mem<device>>) {
// CHECK-NEXT: %[[CONSTANT_0:.*]] = hipsr.constant {index = 0 : i64, offset = 0 : i64, size = 16 : i64, value = dense_resource<"file|w.bin|0"> : tensor<2xi64>} : tensor<2xi64, #hipsr.mem<device>>
// CHECK-NEXT: %[[CONSTANT_1:.*]] = hipsr.constant {index = 1 : i64, offset = 64 : i64, size = 8 : i64, value = dense<{{\[}}5.000000e+00, 6.000000e+00{{\]}}> : tensor<2xf32>} : tensor<2xf32, #hipsr.mem<device>>
// CHECK-NEXT: return %[[CONSTANT_0]], %[[CONSTANT_1]] : tensor<2xi64, #hipsr.mem<device>>, tensor<2xf32, #hipsr.mem<device>>
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK-NEXT: {-#
// CHECK-NEXT: dialect_resources: {
// CHECK-NEXT: builtin: {
// CHECK-NEXT: "file|w.bin|0": "0x0100000001000000000000000200000000000000"
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: #-}
func.func @cumulative_alignment() -> (tensor<2xi64, #hipsr.mem<device>>, tensor<2xf32, #hipsr.mem<device>>) {
  %0 = hipsr.constant {value = dense_resource<"file|w.bin|0"> : tensor<2xi64>} : tensor<2xi64, #hipsr.mem<device>>
  %1 = hipsr.constant {value = dense<[5.0, 6.0]> : tensor<2xf32>} : tensor<2xf32, #hipsr.mem<device>>
  return %0, %1 : tensor<2xi64, #hipsr.mem<device>>, tensor<2xf32, #hipsr.mem<device>>
}

{-#
  dialect_resources: {
    builtin: {
      "file|w.bin|0": "0x0100000001000000000000000200000000000000"
    }
  }
#-}

// -----

// Offsets accumulate across functions because the pass runs on the module.
// CHECK-LABEL: module attributes {hip.constants_file = "constants.bin", hipdnn.constant_offsets = array<i64: 0, 64>, hipdnn.constant_sizes = array<i64: 16, 8>} {
// CHECK-NEXT: func.func @first() -> tensor<4xf32, #hipsr.mem<device>> {
// CHECK-NEXT: %[[CONSTANT_0:.*]] = hipsr.constant {index = 0 : i64, offset = 0 : i64, size = 16 : i64, value = dense<{{\[}}1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00{{\]}}> : tensor<4xf32>} : tensor<4xf32, #hipsr.mem<device>>
// CHECK-NEXT: return %[[CONSTANT_0]] : tensor<4xf32, #hipsr.mem<device>>
// CHECK-NEXT: }
// CHECK-NEXT: func.func @second() -> tensor<2xf32, #hipsr.mem<device>> {
// CHECK-NEXT: %[[CONSTANT_1:.*]] = hipsr.constant {index = 1 : i64, offset = 64 : i64, size = 8 : i64, value = dense<{{\[}}7.000000e+00, 8.000000e+00{{\]}}> : tensor<2xf32>} : tensor<2xf32, #hipsr.mem<device>>
// CHECK-NEXT: return %[[CONSTANT_1]] : tensor<2xf32, #hipsr.mem<device>>
// CHECK-NEXT: }
// CHECK-NEXT: }
func.func @first() -> tensor<4xf32, #hipsr.mem<device>> {
  %0 = hipsr.constant {value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>} : tensor<4xf32, #hipsr.mem<device>>
  return %0 : tensor<4xf32, #hipsr.mem<device>>
}
func.func @second() -> tensor<2xf32, #hipsr.mem<device>> {
  %0 = hipsr.constant {value = dense<[7.0, 8.0]> : tensor<2xf32>} : tensor<2xf32, #hipsr.mem<device>>
  return %0 : tensor<2xf32, #hipsr.mem<device>>
}

// -----

// A second run replaces stale per-op stamps and module layout attributes.
// CHECK-LABEL: module attributes {hip.constants_file = "constants.bin", hipdnn.constant_offsets = array<i64: 0>, hipdnn.constant_sizes = array<i64: 16>} {
// CHECK-NEXT: func.func @restamps_existing() -> tensor<4xf32, #hipsr.mem<device>> {
// CHECK-NEXT: %[[CONSTANT_0:.*]] = hipsr.constant {index = 0 : i64, offset = 0 : i64, size = 16 : i64, value = dense<{{\[}}1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00{{\]}}> : tensor<4xf32>} : tensor<4xf32, #hipsr.mem<device>>
// CHECK-NEXT: return %[[CONSTANT_0]] : tensor<4xf32, #hipsr.mem<device>>
// CHECK-NEXT: }
// CHECK-NEXT: }
module attributes {
  hip.constants_file = "old.bin",
  hipdnn.constant_offsets = array<i64: 0, 64>,
  hipdnn.constant_sizes = array<i64: 16, 8>
} {
  func.func @restamps_existing() -> tensor<4xf32, #hipsr.mem<device>> {
    %0 = hipsr.constant {value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>, offset = 999 : i64, size = 16 : i64, index = 3 : i64} : tensor<4xf32, #hipsr.mem<device>>
    return %0 : tensor<4xf32, #hipsr.mem<device>>
  }
}
