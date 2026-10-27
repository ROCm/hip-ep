// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

// The compiled kernel takes the data pointers of the inputs, then one output
// per kernel result in result order, so the kernargs array holds the `ins`
// and then the `outs` operands.

module {
  // CHECK-LABEL: llvm.func @rocmlir_two_results
  // CHECK-SAME: (%[[CTX:arg0]]: !llvm.ptr,
  func.func @rocmlir_two_results(
      %ctx: !hip.context,
      %x: memref<1x8x4x4xf16, 1>,
      %w: memref<16x8x3x3xf16, 1>,
      %y: memref<1x16x4x4xf16, 1>,
      %z: memref<1x16x4x4xf16, 1>) {
    // CHECK: %[[P0:.*]] = llvm.addrspacecast
    // CHECK: %[[P1:.*]] = llvm.addrspacecast
    // CHECK: %[[P2:.*]] = llvm.addrspacecast
    // CHECK: %[[P3:.*]] = llvm.addrspacecast
    // CHECK: %[[ARGS:.*]] = llvm.alloca %{{.*}} x !llvm.array<4 x ptr>
    // CHECK: llvm.store %[[P0]], %{{.*}} : !llvm.ptr, !llvm.ptr
    // CHECK: llvm.store %[[P1]], %{{.*}} : !llvm.ptr, !llvm.ptr
    // CHECK: llvm.store %[[P2]], %{{.*}} : !llvm.ptr, !llvm.ptr
    // CHECK: llvm.store %[[P3]], %{{.*}} : !llvm.ptr, !llvm.ptr
    // CHECK-NOT: llvm.store
    // CHECK: %[[SIZE:.*]] = llvm.mlir.constant(32 : i64) : i64
    // CHECK: llvm.call @wrap_rocmlir(%[[CTX]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[ARGS]], %[[SIZE]])
    hip.rocmlir(%ctx) @rocMlir0 ins(%x, %w : memref<1x8x4x4xf16, 1>,
                                             memref<16x8x3x3xf16, 1>)
        outs(%y, %z : memref<1x16x4x4xf16, 1>, memref<1x16x4x4xf16, 1>)
        {block_size = 128 : i64, grid_size = 4 : i64, kernel_binary = "ELF"}
    return
  }
}
