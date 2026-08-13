// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hipsr-externalize-constants %s -split-input-file | FileCheck %s

// -----

// CHECK-LABEL: func.func @inline_value
// CHECK: hipsr.constant {index = 0 : i64, offset = 0 : i64, size = 16 : i64, value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>} : tensor<4xf32>
func.func @inline_value() -> tensor<4xf32> {
  %0 = hipsr.constant {value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>} : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func.func @file_resource
// CHECK: hipsr.constant {index = 0 : i64, offset = 0 : i64, size = 16 : i64, value = dense_resource<"file|w.bin|100"> : tensor<2xi64>} : tensor<2xi64>
func.func @file_resource() -> tensor<2xi64> {
  %0 = hipsr.constant {value = dense_resource<"file|w.bin|100"> : tensor<2xi64>} : tensor<2xi64>
  return %0 : tensor<2xi64>
}

{-#
  dialect_resources: {
    builtin: {
      "file|w.bin|100": "0x0100000001000000000000000200000000000000"
    }
  }
#-}

// -----

// CHECK-LABEL: func.func @cumulative_alignment
// CHECK: hipsr.constant {index = 0 : i64, offset = 0 : i64, size = 16 : i64, value = dense_resource<"file|w.bin|0"> : tensor<2xi64>} : tensor<2xi64>
// CHECK: hipsr.constant {index = 1 : i64, offset = 64 : i64, size = 8 : i64, value = dense<[5.000000e+00, 6.000000e+00]> : tensor<2xf32>} : tensor<2xf32>
func.func @cumulative_alignment() -> (tensor<2xi64>, tensor<2xf32>) {
  %0 = hipsr.constant {value = dense_resource<"file|w.bin|0"> : tensor<2xi64>} : tensor<2xi64>
  %1 = hipsr.constant {value = dense<[5.0, 6.0]> : tensor<2xf32>} : tensor<2xf32>
  return %0, %1 : tensor<2xi64>, tensor<2xf32>
}

{-#
  dialect_resources: {
    builtin: {
      "file|w.bin|0": "0x0100000001000000000000000200000000000000"
    }
  }
#-}

// -----
// Module-scoped: the pass runs on the whole module, so offsets accumulate
// across functions (the second func's constant lands after the first, aligned)
// rather than restarting at 0 per function.

// CHECK-LABEL: func.func @first
// CHECK: hipsr.constant {index = 0 : i64, offset = 0 : i64, size = 16 : i64, value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>} : tensor<4xf32>
// CHECK-LABEL: func.func @second
// CHECK: hipsr.constant {index = 1 : i64, offset = 64 : i64, size = 8 : i64, value = dense<[7.000000e+00, 8.000000e+00]> : tensor<2xf32>} : tensor<2xf32>
func.func @first() -> tensor<4xf32> {
  %0 = hipsr.constant {value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>} : tensor<4xf32>
  return %0 : tensor<4xf32>
}
func.func @second() -> tensor<2xf32> {
  %0 = hipsr.constant {value = dense<[7.0, 8.0]> : tensor<2xf32>} : tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

// A pre-stamped constant is re-stamped from scratch (offset/index recomputed,
// not preserved).
// CHECK-LABEL: func.func @restamps_existing
// CHECK: hipsr.constant {index = 0 : i64, offset = 0 : i64, size = 16 : i64, value = dense<{{.*}}> : tensor<4xf32>} : tensor<4xf32>
func.func @restamps_existing() -> tensor<4xf32> {
  %0 = hipsr.constant {value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>, offset = 999 : i64, size = 16 : i64, index = 3 : i64} : tensor<4xf32>
  return %0 : tensor<4xf32>
}
