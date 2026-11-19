// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

module {
  // A reverse reads its strided input through the descriptor: the source
  // pointer is advanced to the last element of the reversed axis and that
  // axis's stride is negated.
  func.func @reverse_strided_view(
      %ctx: !hip.context,
      %input: memref<4x49xf32, strided<[51, 1], offset: 1>, 1>,
      %output: memref<4x49xf32, 1>) {
    hip.reverse(%ctx) ins(%input : memref<4x49xf32, strided<[51, 1], offset: 1>, 1>)
                      outs(%output : memref<4x49xf32, 1>) {axes = [1]}
    return
  }

  func.func @reverse_two_axes_f16(
      %ctx: !hip.context,
      %input: memref<3x5xf16, 1>,
      %output: memref<3x5xf16, 1>) {
    hip.reverse(%ctx) ins(%input : memref<3x5xf16, 1>)
                      outs(%output : memref<3x5xf16, 1>) {axes = [0, 1]}
    return
  }

  // A bool buffer holds one byte per element, so elem_bytes is 1.
  func.func @reverse_bool(
      %ctx: !hip.context,
      %input: memref<2x6xi1, 1>,
      %output: memref<2x6xi1, 1>) {
    hip.reverse(%ctx) ins(%input : memref<2x6xi1, 1>)
                      outs(%output : memref<2x6xi1, 1>) {axes = [1]}
    return
  }

  // outer = 8 * 2, src_widths = [3, 5].
  func.func @concat_last_axis(
      %ctx: !hip.context,
      %a: memref<8x2x3xf32, 1>,
      %b: memref<8x2x5xf32, 1>,
      %output: memref<8x2x8xf32, 1>) {
    hip.concat(%ctx) ins(%a, %b : memref<8x2x3xf32, 1>, memref<8x2x5xf32, 1>)
                     outs(%output : memref<8x2x8xf32, 1>) {axis = 2 : i64}
    return
  }
}

// CHECK-LABEL: llvm.func @reverse_strided_view
// CHECK: %[[OFF:.*]] = llvm.extractvalue %{{.*}}[2]
// CHECK: %[[LAST:.*]] = llvm.sub %{{.*}}, %{{.*}} : i64
// CHECK: %[[STEP:.*]] = llvm.mul %[[LAST]], %{{.*}} : i64
// CHECK: llvm.add %[[OFF]], %[[STEP]] : i64
// CHECK: llvm.sub %{{.*}}, %{{.*}} : i64
// CHECK: llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
// CHECK: llvm.call @wrap_strided_copy_nd({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32

// CHECK-LABEL: llvm.func @reverse_two_axes_f16
// CHECK: llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
// CHECK: llvm.call @wrap_strided_copy_nd({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32

// CHECK-LABEL: llvm.func @reverse_bool
// CHECK: llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i1
// CHECK: llvm.call @wrap_strided_copy_nd({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32

// CHECK-LABEL: llvm.func @concat_last_axis
// CHECK: llvm.alloca {{.*}} x !llvm.array<2 x ptr>
// CHECK: llvm.alloca {{.*}} x !llvm.array<2 x i64>
// CHECK: llvm.call @wrap_concat({{.*}}) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64) -> i32
