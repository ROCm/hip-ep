// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Test: hipsr.constant lowering to LLVM.
//
// HipsrDialect implements ConvertToLLVMPatternInterface, so the standard LLVM
// conversion driver (--convert-hip-to-llvm) discovers and applies it; there is
// no dedicated hipsr->LLVM pass.
//
// Verifies:
// - An externalized hipsr.constant (offset/size/index set) lowers to
//   llvm.call @hipdnn_ep_constant_get(%ctx, %index), index read off the op.
// - %ctx is the enclosing function's arg 0 (runtime state), converted to
//   !llvm.ptr.
// - The returned AS 0 pointer is address-space-cast to the device space (AS 1).
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @single_constant
// CHECK-SAME:  (%[[CTX:.*]]: !llvm.ptr)
func.func @single_constant(%ctx: !hip.context)
    -> memref<3x4xf32, #hipsr.mem<device>> {
  // CHECK:   %[[IDX:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK:   %[[PTR:.*]] = llvm.call @hipdnn_ep_constant_get(%[[CTX]], %[[IDX]]) : (!llvm.ptr, i64) -> !llvm.ptr
  // CHECK:   llvm.addrspacecast %[[PTR]] : !llvm.ptr to !llvm.ptr<1>
  %c = hipsr.constant {value = dense<1.0> : tensor<3x4xf32>, offset = 0 : i64, size = 48 : i64, index = 0 : i64}
     : memref<3x4xf32, #hipsr.mem<device>>
  return %c : memref<3x4xf32, #hipsr.mem<device>>
}

// Two externalized constants; each call carries its own index. ctx is arg 0
// (runtime state) per the EP ABI.
// CHECK-LABEL: llvm.func @two_constants
// CHECK-SAME:  (%[[CTX2:.*]]: !llvm.ptr, %[[N:.*]]: i64)
func.func @two_constants(%ctx: !hip.context, %n: i64)
    -> (memref<64xf32, #hipsr.mem<device>>, memref<8xf32, #hipsr.mem<device>>) {
  // CHECK:   %[[IDX0:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK:   %[[P0:.*]] = llvm.call @hipdnn_ep_constant_get(%[[CTX2]], %[[IDX0]]) : (!llvm.ptr, i64) -> !llvm.ptr
  // CHECK:   llvm.addrspacecast %[[P0]] : !llvm.ptr to !llvm.ptr<1>
  %w = hipsr.constant {value = dense<1.0> : tensor<64xf32>, offset = 0 : i64, size = 256 : i64, index = 0 : i64}
     : memref<64xf32, #hipsr.mem<device>>
  // Advance past the first descriptor's stride insert (a constant(1)) so IDX1
  // binds to the second constant's index, not that stride constant.
  // CHECK:   llvm.insertvalue {{.*}}[4, 0]
  // CHECK:   %[[IDX1:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:   %[[P1:.*]] = llvm.call @hipdnn_ep_constant_get(%[[CTX2]], %[[IDX1]]) : (!llvm.ptr, i64) -> !llvm.ptr
  // CHECK:   llvm.addrspacecast %[[P1]] : !llvm.ptr to !llvm.ptr<1>
  %b = hipsr.constant {value = dense<2.0> : tensor<8xf32>, offset = 256 : i64, size = 32 : i64, index = 1 : i64}
     : memref<8xf32, #hipsr.mem<device>>
  return %w, %b : memref<64xf32, #hipsr.mem<device>>, memref<8xf32, #hipsr.mem<device>>
}
