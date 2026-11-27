// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --split-input-file --convert-hip-to-llvm -verify-diagnostics | FileCheck %s

// hip.winograd_conv calls wrap_winograd_conv with its op-state slot, the
// embedded code object and symbol, the launch geometry, the input, filter,
// bias (null when absent) and output pointers, K, C and the filter store kind.

module {
  // The code object and the NUL-terminated symbol are module globals.
  // CHECK-DAG: llvm.mlir.global internal constant @hip_winograd_conv_leaky_relu_bias_binary("ELF")
  // CHECK-DAG: llvm.mlir.global internal constant @hip_winograd_conv_leaky_relu_bias_name("hip_winograd_conv_leaky_relu_bias\00")
  // CHECK-LABEL: llvm.func @winograd_bias
  // CHECK-SAME: (%[[CTX:arg0]]: !llvm.ptr,
  func.func @winograd_bias(%ctx: !hip.context,
                           %x: memref<1x16x8x8xf16, 1>,
                           %w: memref<32x16x3x3xf16, 1>,
                           %b: memref<32xf16, 1>,
                           %y: memref<1x32x8x8xf16, 1>) {
    // CHECK-DAG: %[[SLOT:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK-DAG: %[[BLOCK:.*]] = llvm.mlir.constant(128 : i64) : i64
    // CHECK-DAG: %[[GRID:.*]] = llvm.mlir.constant(9 : i64) : i64
    // CHECK-DAG: %[[K:.*]] = llvm.mlir.constant(32 : i64) : i64
    // CHECK-DAG: %[[C:.*]] = llvm.mlir.constant(16 : i64) : i64
    // CHECK-DAG: %[[FT:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: llvm.call @wrap_winograd_conv(%[[CTX]], %[[SLOT]], %{{.*}}, %{{.*}}, %[[BLOCK]], %[[GRID]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[K]], %[[C]], %[[FT]])
    hip.winograd_conv(%ctx) ins(%x, %w, %b : memref<1x16x8x8xf16, 1>,
                                memref<32x16x3x3xf16, 1>, memref<32xf16, 1>)
        outs(%y : memref<1x32x8x8xf16, 1>)
        {activation = "leaky_relu", alpha = 0.1 : f64, block_size = 128 : i64,
         grid_size = 9 : i64, kernel_binary = "ELF",
         kernel_name = "hip_winograd_conv_leaky_relu_bias",
         full_transform = true, hip.op_state_slot = 7 : i32}
    return
  }
}

// -----

module {
  // No bias: the bias argument is a null pointer.
  // CHECK-LABEL: llvm.func @winograd_no_bias
  func.func @winograd_no_bias(%ctx: !hip.context,
                              %x: memref<1x16x8x8xf16, 1>,
                              %w: memref<32x16x3x3xf16, 1>,
                              %y: memref<1x32x8x8xf16, 1>) {
    // CHECK: %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
    // CHECK: llvm.call @wrap_winograd_conv(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[NULL]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})
    hip.winograd_conv(%ctx) ins(%x, %w : memref<1x16x8x8xf16, 1>,
                                memref<32x16x3x3xf16, 1>)
        outs(%y : memref<1x32x8x8xf16, 1>)
        {block_size = 128 : i64, grid_size = 9 : i64, kernel_binary = "ELF",
         kernel_name = "hip_winograd_conv_none"}
    return
  }
}

// -----

module {
  func.func @winograd_bad_filter(%ctx: !hip.context,
                                 %x: memref<1x16x8x8xf16, 1>,
                                 %w: memref<32x16x5x5xf16, 1>,
                                 %y: memref<1x32x8x8xf16, 1>) {
    // expected-error @+1 {{requires a 3x3 filter}}
    hip.winograd_conv(%ctx) ins(%x, %w : memref<1x16x8x8xf16, 1>,
                                memref<32x16x5x5xf16, 1>)
        outs(%y : memref<1x32x8x8xf16, 1>)
    return
  }
}
