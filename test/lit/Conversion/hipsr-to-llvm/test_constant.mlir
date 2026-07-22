// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Test: hipsr.constant lowering to LLVM.
//
// Verifies:
// - An externalized hipsr.constant (offset/size set) lowers to
//   llvm.call @wrap_get_global(%ctx, %offset, %size), with offset/size read
//   off the op.
// - The !hip.context function argument is used as %ctx (converted to !llvm.ptr,
//   not hardcoded to arg 0).
// - The returned AS 0 pointer is address-space-cast to the device space (AS 1).
// - The pass stamps hipdnn.constant_sizes / hipdnn.constant_offsets on the
//   module (source=NONE metadata, one entry per externalized constant in
//   module walk order), and stamps nothing when there is no externalized
//   constant.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s -split-input-file --convert-hipsr-to-llvm | FileCheck %s

// Module-level metadata: one array entry per externalized constant, in module
// walk order (single_constant's %c, then two_constants' %w, %b). Attributes
// print alphabetically, so offsets precede sizes.
// CHECK: module attributes {hipdnn.constant_offsets = array<i64: 0, 0, 256>, hipdnn.constant_sizes = array<i64: 48, 256, 32>}
module {
  // CHECK-LABEL: llvm.func @single_constant
  // CHECK-SAME:  (%[[CTX:.*]]: !llvm.ptr)
  func.func @single_constant(%ctx: !hip.context)
      -> memref<3x4xf32, #hipsr.mem<device>> {
    // CHECK:   %[[OFF:.*]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK:   %[[SZ:.*]] = llvm.mlir.constant(48 : i64) : i64
    // CHECK:   %[[PTR:.*]] = llvm.call @wrap_get_global(%[[CTX]], %[[OFF]], %[[SZ]]) : (!llvm.ptr, i64, i64) -> !llvm.ptr
    // CHECK:   llvm.addrspacecast %[[PTR]] : !llvm.ptr to !llvm.ptr<1>
    %c = hipsr.constant {value = dense<1.0> : tensor<3x4xf32>, offset = 0 : i64, size = 48 : i64}
       : memref<3x4xf32, #hipsr.mem<device>>
    return %c : memref<3x4xf32, #hipsr.mem<device>>
  }

  // Two externalized constants; each call carries its own offset/size. The
  // context argument is not arg 0 here, exercising the per-function ctx lookup.
  // CHECK-LABEL: llvm.func @two_constants
  // CHECK-SAME:  (%[[N:.*]]: i64, %[[CTX2:.*]]: !llvm.ptr)
  func.func @two_constants(%n: i64, %ctx: !hip.context)
      -> (memref<64xf32, #hipsr.mem<device>>, memref<8xf32, #hipsr.mem<device>>) {
    // CHECK:   %[[OFF0:.*]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK:   %[[SZ0:.*]] = llvm.mlir.constant(256 : i64) : i64
    // CHECK:   %[[P0:.*]] = llvm.call @wrap_get_global(%[[CTX2]], %[[OFF0]], %[[SZ0]]) : (!llvm.ptr, i64, i64) -> !llvm.ptr
    // CHECK:   llvm.addrspacecast %[[P0]] : !llvm.ptr to !llvm.ptr<1>
    %w = hipsr.constant {value = dense<1.0> : tensor<64xf32>, offset = 0 : i64, size = 256 : i64}
       : memref<64xf32, #hipsr.mem<device>>
    // CHECK:   %[[OFF1:.*]] = llvm.mlir.constant(256 : i64) : i64
    // CHECK:   %[[SZ1:.*]] = llvm.mlir.constant(32 : i64) : i64
    // CHECK:   %[[P1:.*]] = llvm.call @wrap_get_global(%[[CTX2]], %[[OFF1]], %[[SZ1]]) : (!llvm.ptr, i64, i64) -> !llvm.ptr
    // CHECK:   llvm.addrspacecast %[[P1]] : !llvm.ptr to !llvm.ptr<1>
    %b = hipsr.constant {value = dense<2.0> : tensor<8xf32>, offset = 256 : i64, size = 32 : i64}
       : memref<8xf32, #hipsr.mem<device>>
    return %w, %b : memref<64xf32, #hipsr.mem<device>>, memref<8xf32, #hipsr.mem<device>>
  }
}

// -----

// No externalized constant -> the pass stamps no constant metadata attrs.
// CHECK-LABEL: llvm.func @no_constant
// CHECK-NOT: hipdnn.constant_sizes
// CHECK-NOT: hipdnn.constant_offsets
func.func @no_constant(%ctx: !hip.context) {
  return
}
