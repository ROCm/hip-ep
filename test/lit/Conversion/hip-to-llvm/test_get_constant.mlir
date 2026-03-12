// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Test: hip.get_constant lowering to LLVM.
//
// Verifies:
// - hip.get_constant(%ctx, %index) lowers to llvm.call @hipdnn_ep_constant_get
// - Returned GPU pointer is wrapped in a memref descriptor
// - Address space cast from AS 0 to AS 1 is inserted
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // CHECK-LABEL: llvm.func @test_get_constant
  // CHECK-SAME:  (%[[CTX:.*]]: !llvm.ptr)
  func.func @test_get_constant(%ctx: !hip.context) -> memref<3x4xf32, 1> {
    %idx = arith.constant 0 : i64
    // CHECK:   %[[IDX:.*]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK:   %[[PTR:.*]] = llvm.call @hipdnn_ep_constant_get(%[[CTX]], %[[IDX]]) : (!llvm.ptr, i64) -> !llvm.ptr
    // CHECK:   llvm.addrspacecast %[[PTR]] : !llvm.ptr to !llvm.ptr<1>
    %c = hip.get_constant(%ctx, %idx) : memref<3x4xf32, 1>
    return %c : memref<3x4xf32, 1>
  }

  // CHECK-LABEL: llvm.func @test_get_constant_multi
  // CHECK-SAME:  (%[[CTX2:.*]]: !llvm.ptr)
  func.func @test_get_constant_multi(%ctx: !hip.context) -> (memref<64x3x3x3xf32, 1>, memref<64xf32, 1>) {
    %idx0 = arith.constant 0 : i64
    %idx1 = arith.constant 1 : i64
    // CHECK:   %[[I0:.*]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK:   %[[I1:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK:   %[[P0:.*]] = llvm.call @hipdnn_ep_constant_get(%[[CTX2]], %[[I0]]) : (!llvm.ptr, i64) -> !llvm.ptr
    // CHECK:   llvm.addrspacecast %[[P0]] : !llvm.ptr to !llvm.ptr<1>
    // CHECK:   %[[P1:.*]] = llvm.call @hipdnn_ep_constant_get(%[[CTX2]], %[[I1]]) : (!llvm.ptr, i64) -> !llvm.ptr
    // CHECK:   llvm.addrspacecast %[[P1]] : !llvm.ptr to !llvm.ptr<1>
    %w = hip.get_constant(%ctx, %idx0) : memref<64x3x3x3xf32, 1>
    %b = hip.get_constant(%ctx, %idx1) : memref<64xf32, 1>
    return %w, %b : memref<64x3x3x3xf32, 1>, memref<64xf32, 1>
  }
}
