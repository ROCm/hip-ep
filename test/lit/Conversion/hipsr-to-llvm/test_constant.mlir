// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// REQUIRES: hipsr

// RUN: hip-mlir-opt %s --convert-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @single_constant
// CHECK-SAME:  (%[[CTX:.*]]: !llvm.ptr)
func.func @single_constant(%ctx: !llvm.ptr)
    -> memref<3x4xf32, #hipsr.mem<device>> {
  // CHECK:   %[[IDX:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK:   %[[PTR:.*]] = llvm.call @hipdnn_ep_constant_get(%[[CTX]], %[[IDX]]) : (!llvm.ptr, i64) -> !llvm.ptr<1>
  %c = hipsr.constant {value = dense<1.0> : tensor<3x4xf32>, offset = 0 : i64, size = 48 : i64, index = 0 : i64}
     : memref<3x4xf32, #hipsr.mem<device>>
  return %c : memref<3x4xf32, #hipsr.mem<device>>
}

// CHECK-LABEL: llvm.func @two_constants
// CHECK-SAME:  (%[[CTX2:.*]]: !llvm.ptr, %[[N:.*]]: i64)
func.func @two_constants(%ctx: !llvm.ptr, %n: i64)
    -> (memref<64xf32, #hipsr.mem<device>>, memref<8xf32, #hipsr.mem<device>>) {
  // CHECK:   %[[IDX0:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK:   %[[P0:.*]] = llvm.call @hipdnn_ep_constant_get(%[[CTX2]], %[[IDX0]]) : (!llvm.ptr, i64) -> !llvm.ptr<1>
  %w = hipsr.constant {value = dense<1.0> : tensor<64xf32>, offset = 0 : i64, size = 256 : i64, index = 0 : i64}
     : memref<64xf32, #hipsr.mem<device>>
  // CHECK:   llvm.insertvalue {{.*}}[4, 0]
  // CHECK:   %[[IDX1:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:   %[[P1:.*]] = llvm.call @hipdnn_ep_constant_get(%[[CTX2]], %[[IDX1]]) : (!llvm.ptr, i64) -> !llvm.ptr<1>
  %b = hipsr.constant {value = dense<2.0> : tensor<8xf32>, offset = 256 : i64, size = 32 : i64, index = 1 : i64}
     : memref<8xf32, #hipsr.mem<device>>
  return %w, %b : memref<64xf32, #hipsr.mem<device>>, memref<8xf32, #hipsr.mem<device>>
}
