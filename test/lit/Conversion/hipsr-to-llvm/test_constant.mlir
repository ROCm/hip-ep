// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Test: hipsr.constant lowering to LLVM.
//
// Verifies:
// - An externalized hipsr.constant (offset/size set) lowers to
//   llvm.call @hipdnn_ep_constant_get(%ctx, %index).
// - The !hip.context function argument is used as %ctx (converted to !llvm.ptr,
//   not hardcoded to arg 0).
// - The returned AS 0 pointer is address-space-cast to the device space (AS 1).
// - Multiple externalized constants get module-walk-order indices 0, 1, ...
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --convert-hipsr-to-llvm | FileCheck %s

module {
  // CHECK-LABEL: llvm.func @single_constant
  // CHECK-SAME:  (%[[CTX:.*]]: !llvm.ptr)
  func.func @single_constant(%ctx: !hip.context)
      -> memref<3x4xf32, #hipsr.mem<device>> {
    // CHECK:   %[[IDX:.*]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK:   %[[PTR:.*]] = llvm.call @hipdnn_ep_constant_get(%[[CTX]], %[[IDX]]) : (!llvm.ptr, i64) -> !llvm.ptr
    // CHECK:   llvm.addrspacecast %[[PTR]] : !llvm.ptr to !llvm.ptr<1>
    %c = hipsr.constant {value = dense<1.0> : tensor<3x4xf32>, offset = 0 : i64, size = 48 : i64}
       : memref<3x4xf32, #hipsr.mem<device>>
    return %c : memref<3x4xf32, #hipsr.mem<device>>
  }

  // Two externalized constants. Indices are the module-global walk order, so
  // they continue after @single_constant's constant (which is index 0): %w is
  // 1, %b is 2. The context argument is not arg 0 here, exercising the
  // per-function ctx lookup.
  // CHECK-LABEL: llvm.func @two_constants
  // CHECK-SAME:  (%[[N:.*]]: i64, %[[CTX2:.*]]: !llvm.ptr)
  func.func @two_constants(%n: i64, %ctx: !hip.context)
      -> (memref<64xf32, #hipsr.mem<device>>, memref<8xf32, #hipsr.mem<device>>) {
    // CHECK:   %[[I1:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK:   %[[P0:.*]] = llvm.call @hipdnn_ep_constant_get(%[[CTX2]], %[[I1]]) : (!llvm.ptr, i64) -> !llvm.ptr
    // CHECK:   llvm.addrspacecast %[[P0]] : !llvm.ptr to !llvm.ptr<1>
    %w = hipsr.constant {value = dense<1.0> : tensor<64xf32>, offset = 0 : i64, size = 256 : i64}
       : memref<64xf32, #hipsr.mem<device>>
    // CHECK:   %[[I2:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK:   %[[P1:.*]] = llvm.call @hipdnn_ep_constant_get(%[[CTX2]], %[[I2]]) : (!llvm.ptr, i64) -> !llvm.ptr
    // CHECK:   llvm.addrspacecast %[[P1]] : !llvm.ptr to !llvm.ptr<1>
    %b = hipsr.constant {value = dense<2.0> : tensor<8xf32>, offset = 256 : i64, size = 32 : i64}
       : memref<8xf32, #hipsr.mem<device>>
    return %w, %b : memref<64xf32, #hipsr.mem<device>>, memref<8xf32, #hipsr.mem<device>>
  }
}
