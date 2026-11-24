// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// A Gather of one constant in-range index on static floating-point data, with
// a result of at least 4096 elements, becomes a size-1 tensor.extract_slice of
// the gathered axis, plus a tensor.collapse_shape that drops the axis for a
// scalar index. Small results, integer data, runtime or dynamically shaped
// indices, several indices and out-of-range indices keep hip.gather.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // Scalar index into the innermost axis.
  func.func @scalar_innermost(%x: tensor<2x32x64x2xf32>) -> tensor<2x32x64xf32> {
    %i = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %y = "onnx.Gather"(%x, %i) {axis = 3 : si64} : (tensor<2x32x64x2xf32>, tensor<i64>) -> tensor<2x32x64xf32>
    return %y : tensor<2x32x64xf32>
  }
  // CHECK-LABEL: func.func @scalar_innermost
  // CHECK-SAME: (%{{.*}}: !hip.context, %[[X:.*]]: tensor<2x32x64x2xf32>)
  // CHECK-NEXT: %[[S:.*]] = tensor.extract_slice %[[X]][0, 0, 0, 1] [2, 32, 64, 1] [1, 1, 1, 1] : tensor<2x32x64x2xf32> to tensor<2x32x64x1xf32>
  // CHECK-NEXT: %[[Y:.*]] = tensor.collapse_shape %[[S]] {{\[}}[0], [1], [2, 3]] : tensor<2x32x64x1xf32> into tensor<2x32x64xf32>
  // CHECK-NEXT: return %[[Y]] : tensor<2x32x64xf32>
  // CHECK-NEXT: }

  // Negative scalar index on a middle axis; the unit dim folds into the next.
  func.func @scalar_negative_middle(%x: tensor<4x6x1024xf16>) -> tensor<4x1024xf16> {
    %i = "onnx.Constant"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
    %y = "onnx.Gather"(%x, %i) {axis = 1 : si64} : (tensor<4x6x1024xf16>, tensor<i32>) -> tensor<4x1024xf16>
    return %y : tensor<4x1024xf16>
  }
  // CHECK-LABEL: func.func @scalar_negative_middle
  // CHECK-SAME: (%{{.*}}: !hip.context, %[[X:.*]]: tensor<4x6x1024xf16>)
  // CHECK-NEXT: %[[S:.*]] = tensor.extract_slice %[[X]][0, 5, 0] [4, 1, 1024] [1, 1, 1] : tensor<4x6x1024xf16> to tensor<4x1x1024xf16>
  // CHECK-NEXT: %[[Y:.*]] = tensor.collapse_shape %[[S]] {{\[}}[0], [1, 2]] : tensor<4x1x1024xf16> into tensor<4x1024xf16>
  // CHECK-NEXT: return %[[Y]] : tensor<4x1024xf16>
  // CHECK-NEXT: }

  // A one-element 1-D index keeps the unit dim: only the slice.
  func.func @one_element_index(%x: tensor<5x4096xf32>) -> tensor<1x4096xf32> {
    %i = "onnx.Constant"() {value = dense<[2]> : tensor<1xi64>} : () -> tensor<1xi64>
    %y = "onnx.Gather"(%x, %i) {axis = 0 : si64} : (tensor<5x4096xf32>, tensor<1xi64>) -> tensor<1x4096xf32>
    return %y : tensor<1x4096xf32>
  }
  // CHECK-LABEL: func.func @one_element_index
  // CHECK-SAME: (%{{.*}}: !hip.context, %[[X:.*]]: tensor<5x4096xf32>)
  // CHECK-NEXT: %[[S:.*]] = tensor.extract_slice %[[X]][2, 0] [1, 4096] [1, 1] : tensor<5x4096xf32> to tensor<1x4096xf32>
  // CHECK-NEXT: return %[[S]] : tensor<1x4096xf32>
  // CHECK-NEXT: }

  // A result under 4096 elements keeps hip.gather.
  func.func @small_result(%x: tensor<2x3x5x2xf32>) -> tensor<2x3x5xf32> {
    %i = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %y = "onnx.Gather"(%x, %i) {axis = 3 : si64} : (tensor<2x3x5x2xf32>, tensor<i64>) -> tensor<2x3x5xf32>
    return %y : tensor<2x3x5xf32>
  }
  // CHECK-LABEL: func.func @small_result
  // CHECK-NOT: tensor.extract_slice
  // CHECK: hip.gather

  // Integer data keeps hip.gather.
  func.func @integer_data(%x: tensor<4x8192xi64>) -> tensor<8192xi64> {
    %i = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %y = "onnx.Gather"(%x, %i) {axis = 0 : si64} : (tensor<4x8192xi64>, tensor<i64>) -> tensor<8192xi64>
    return %y : tensor<8192xi64>
  }
  // CHECK-LABEL: func.func @integer_data
  // CHECK-NOT: tensor.extract_slice
  // CHECK: hip.gather

  // A runtime index keeps hip.gather.
  func.func @runtime_index(%x: tensor<4x8192xf32>, %i: tensor<i64>) -> tensor<8192xf32> {
    %y = "onnx.Gather"(%x, %i) {axis = 0 : si64} : (tensor<4x8192xf32>, tensor<i64>) -> tensor<8192xf32>
    return %y : tensor<8192xf32>
  }
  // CHECK-LABEL: func.func @runtime_index
  // CHECK-NOT: tensor.extract_slice
  // CHECK: hip.gather

  // Dynamically shaped indices keep hip.gather even with a static result.
  func.func @dynamic_indices(%x: tensor<16x4096xf32>, %i: tensor<?xi64>) -> tensor<1x4096xf32> {
    %y = "onnx.Gather"(%x, %i) {axis = 0 : si64} : (tensor<16x4096xf32>, tensor<?xi64>) -> tensor<1x4096xf32>
    return %y : tensor<1x4096xf32>
  }
  // CHECK-LABEL: func.func @dynamic_indices
  // CHECK-NOT: tensor.extract_slice
  // CHECK: hip.gather

  // An out-of-range index keeps hip.gather, which defines that case.
  func.func @out_of_range(%x: tensor<4x8192xf32>) -> tensor<8192xf32> {
    %i = "onnx.Constant"() {value = dense<7> : tensor<i64>} : () -> tensor<i64>
    %y = "onnx.Gather"(%x, %i) {axis = 0 : si64} : (tensor<4x8192xf32>, tensor<i64>) -> tensor<8192xf32>
    return %y : tensor<8192xf32>
  }
  // CHECK-LABEL: func.func @out_of_range
  // CHECK-NOT: tensor.extract_slice
  // CHECK: hip.gather

  // Several indices keep hip.gather.
  func.func @several_indices(%x: tensor<4x8192xf32>) -> tensor<2x8192xf32> {
    %i = "onnx.Constant"() {value = dense<[0, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
    %y = "onnx.Gather"(%x, %i) {axis = 0 : si64} : (tensor<4x8192xf32>, tensor<2xi64>) -> tensor<2x8192xf32>
    return %y : tensor<2x8192xf32>
  }
  // CHECK-LABEL: func.func @several_indices
  // CHECK-NOT: tensor.extract_slice
  // CHECK: hip.gather
}
