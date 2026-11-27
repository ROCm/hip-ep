// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// A constant flip of an interior slice, joined back onto the source, and an
// interleave of two [..., 1] tensors, through the full pipeline.
//
// The flip is a strided-copy launch that reads the [1:-1] view in place: no
// hip.slice parameter readback and no identity-layout copy of the view. Each
// Concat is one wrap_concat launch rather than one strided copy per input.

// CHECK: module attributes {
// CHECK-NOT: onnx.Slice
// CHECK-NOT: onnx.Concat
// CHECK-NOT: llvm.call @wrap_slice_typed
// CHECK-NOT: llvm.call @wrap_strided_copy(
// CHECK-DAG: llvm.func @wrap_strided_copy_nd(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @wrap_concat(!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64) -> i32
// CHECK: llvm.func private @main_graph
// CHECK-COUNT-1: llvm.call @wrap_strided_copy_nd
// CHECK-COUNT-2: llvm.call @wrap_concat
// CHECK-NOT: llvm.call @wrap_strided_copy(
// CHECK: llvm.func @inference_compute
module {
  func.func @main_graph(%arg0: tensor<4x1x64x51xf32> {onnx.name = "x"},
                        %arg1: tensor<4x64x51x1xf32> {onnx.name = "re"},
                        %arg2: tensor<4x64x51x1xf32> {onnx.name = "im"})
      -> (tensor<4x1x64x100xf32> {onnx.name = "joined"},
          tensor<4x64x51x2xf32> {onnx.name = "interleaved"}) {
    %s1 = arith.constant dense<[1]> : tensor<1xi64>
    %e1 = arith.constant dense<[-1]> : tensor<1xi64>
    %ax = arith.constant dense<[3]> : tensor<1xi64>
    %st1 = arith.constant dense<[1]> : tensor<1xi64>
    %inner = "onnx.Slice"(%arg0, %s1, %e1, %ax, %st1)
        : (tensor<4x1x64x51xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>,
           tensor<1xi64>) -> tensor<4x1x64x49xf32>
    %s2 = arith.constant dense<[-1]> : tensor<1xi64>
    %e2 = arith.constant dense<[-9223372036854775807]> : tensor<1xi64>
    %st2 = arith.constant dense<[-1]> : tensor<1xi64>
    %flip = "onnx.Slice"(%inner, %s2, %e2, %ax, %st2)
        : (tensor<4x1x64x49xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>,
           tensor<1xi64>) -> tensor<4x1x64x49xf32>
    %joined = "onnx.Concat"(%arg0, %flip) {axis = 3 : si64}
        : (tensor<4x1x64x51xf32>, tensor<4x1x64x49xf32>)
        -> tensor<4x1x64x100xf32>
    %inter = "onnx.Concat"(%arg1, %arg2) {axis = -1 : si64}
        : (tensor<4x64x51x1xf32>, tensor<4x64x51x1xf32>)
        -> tensor<4x64x51x2xf32>
    "onnx.Return"(%joined, %inter)
        : (tensor<4x1x64x100xf32>, tensor<4x64x51x2xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
