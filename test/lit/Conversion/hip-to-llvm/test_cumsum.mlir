// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.cumsum lowers to llvm.call @wrap_cumsum with the full
// 12-parameter signature:
//   (state, x, axis, y, x_shape_ptr, x_rank,
//    num_elements, data_type, axis_dtype, exclusive, reverse, host_axis) -> i32.
// The x_shape buffer is an LLVM alloca that the lowering populates via
// GEP + store of each dim size. host_axis is `axis_value` when present and
// INT64_MIN ("read `axis` back") otherwise.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // Static f32 input, i64 axis, default attrs.
  func.func @cumsum_static_default(
      %ctx: !hip.context,
      %x: memref<3x4xf32, 1>,
      %axis: memref<i64, 1>,
      %y: memref<3x4xf32, 1>) {
    // CHECK-LABEL: llvm.func @cumsum_static_default

    hip.cumsum(%ctx) ins(%x, %axis : memref<3x4xf32, 1>, memref<i64, 1>)
                     outs(%y : memref<3x4xf32, 1>)

    // The lowering stack-allocates an array for the input shape:
    // CHECK: llvm.alloca {{.*}} x !llvm.array<2 x i64>
    // CHECK: %[[NOAXIS:.*]] = llvm.mlir.constant(-9223372036854775808 : i64) : i64
    // CHECK: llvm.call @wrap_cumsum({{.*}}, %[[NOAXIS]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32
    return
  }

  // Compile-time axis: forwarded as host_axis so the runtime skips its readback.
  func.func @cumsum_host_axis(
      %ctx: !hip.context,
      %x: memref<3x4xf32, 1>,
      %axis: memref<i64, 1>,
      %y: memref<3x4xf32, 1>) {
    // CHECK-LABEL: llvm.func @cumsum_host_axis

    hip.cumsum(%ctx) ins(%x, %axis : memref<3x4xf32, 1>, memref<i64, 1>)
                     outs(%y : memref<3x4xf32, 1>) {axis_value = -1 : i64}

    // CHECK: %[[HOSTAXIS:.*]] = llvm.mlir.constant(-1 : i64) : i64
    // CHECK: llvm.call @wrap_cumsum({{.*}}, %[[HOSTAXIS]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32
    return
  }

  // With exclusive + reverse attrs, i32 axis dtype.
  func.func @cumsum_attrs(
      %ctx: !hip.context,
      %x: memref<3x4xf32, 1>,
      %axis: memref<i32, 1>,
      %y: memref<3x4xf32, 1>) {
    // CHECK-LABEL: llvm.func @cumsum_attrs

    hip.cumsum(%ctx) ins(%x, %axis : memref<3x4xf32, 1>, memref<i32, 1>)
                     outs(%y : memref<3x4xf32, 1>)
                     {exclusive = 1 : i64, reverse = 1 : i64}

    // CHECK: llvm.call @wrap_cumsum({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32
    return
  }

  // Dynamic shape.
  func.func @cumsum_dynamic(
      %ctx: !hip.context,
      %x: memref<?x?xf32, 1>,
      %axis: memref<i64, 1>,
      %y: memref<?x?xf32, 1>) {
    // CHECK-LABEL: llvm.func @cumsum_dynamic

    hip.cumsum(%ctx) ins(%x, %axis : memref<?x?xf32, 1>, memref<i64, 1>)
                     outs(%y : memref<?x?xf32, 1>)

    // CHECK: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK: llvm.extractvalue %{{.*}}[3, 1]
    // CHECK: llvm.call @wrap_cumsum({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32
    return
  }
}
