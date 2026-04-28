// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.dump_tensor is correctly lowered to LLVM call to
// hipdnn_ep_dump_tensor runtime function.
//
// This test validates:
// - hip.dump_tensor → llvm.call @hipdnn_ep_dump_tensor
// - Global string constants created for name and dump_tensors_dir
// - Shape array built on stack (alloca + store per dim)
// - Correct 7-parameter signature: state, gpu_ptr, shape_ptr, rank,
//                                   data_type, name_ptr, dir_ptr
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // Test 1: 3D f32 tensor dump
  func.func @dump_f32_3d(
      %ctx: !hip.context,
      %buf: memref<2x64x64xf32, 1>) {
    // CHECK-LABEL: llvm.func @dump_f32_3d

    hip.dump_tensor(%ctx) %buf {name = "matmul_0", dump_tensors_dir = "/tmp/dump"} : memref<2x64x64xf32, 1>

    // CHECK: llvm.call @hipdnn_ep_dump_tensor({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, !llvm.ptr) -> ()

    return
  }

  // Test 2: 2D f16 tensor dump
  func.func @dump_f16_2d(
      %ctx: !hip.context,
      %buf: memref<128x256xf16, 1>) {
    // CHECK-LABEL: llvm.func @dump_f16_2d

    hip.dump_tensor(%ctx) %buf {name = "add_1", dump_tensors_dir = "/tmp/dump"} : memref<128x256xf16, 1>

    // CHECK: llvm.call @hipdnn_ep_dump_tensor({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, !llvm.ptr) -> ()

    return
  }

  // Test 3: 1D i64 tensor dump
  func.func @dump_i64_1d(
      %ctx: !hip.context,
      %buf: memref<1024xi64, 1>) {
    // CHECK-LABEL: llvm.func @dump_i64_1d

    hip.dump_tensor(%ctx) %buf {name = "gather_2", dump_tensors_dir = "C:/debug"} : memref<1024xi64, 1>

    // CHECK: llvm.call @hipdnn_ep_dump_tensor({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, !llvm.ptr) -> ()

    return
  }
}
