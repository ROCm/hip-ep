// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// An export whose shape inference gave up reaches conversion carrying
// `tensor<*xT>`. The known producer is onnx.Slice with non-ascending `axes`
// (ORT declines to infer through it, so every downstream tensor loses its
// shape), but any op ONNX cannot infer through does the same.
//
// Every converter here must REFUSE such an operand, leaving the onnx.* op live
// for the unconverted-op report. Two older failure modes are what this guards
// against: building a hip.* op the verifier then rejects far from the converter
// responsible, and reading a rank off the unranked type through an unchecked
// `mlir::cast<RankedTensorType>`, which under NDEBUG reinterprets memory
// instead of asserting. onnx.Gather showed the sharpest form of the latter --
// its dim-copy loops ran to a garbage rank and the pass span forever.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

// The pass generates module metadata from @main_graph, so one must exist.
func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  return %arg0 : tensor<4xf32>
}

func.func @unranked_gather(%data: tensor<*xf32>, %idx: tensor<2xi64>)
    -> tensor<2x4xf32> {
  // CHECK-LABEL: @unranked_gather
  // CHECK-NOT: hip.gather
  // CHECK: "onnx.Gather"
  %r = "onnx.Gather"(%data, %idx) {axis = 0 : si64}
      : (tensor<*xf32>, tensor<2xi64>) -> tensor<2x4xf32>
  return %r : tensor<2x4xf32>
}

func.func @unranked_matmul(%a: tensor<*xf32>, %b: tensor<8x16xf32>)
    -> tensor<4x16xf32> {
  // CHECK-LABEL: @unranked_matmul
  // CHECK-NOT: hip.matmul
  // CHECK: "onnx.MatMul"
  %r = "onnx.MatMul"(%a, %b)
      : (tensor<*xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  return %r : tensor<4x16xf32>
}

func.func @unranked_gemm(%a: tensor<*xf32>, %b: tensor<8x16xf32>)
    -> tensor<4x16xf32> {
  // CHECK-LABEL: @unranked_gemm
  // CHECK-NOT: hip.gemm
  // CHECK: "onnx.Gemm"
  %r = "onnx.Gemm"(%a, %b)
      : (tensor<*xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  return %r : tensor<4x16xf32>
}

func.func @unranked_conv_transpose(%x: tensor<*xf32>, %w: tensor<8x3x3x3xf32>)
    -> tensor<1x8x64x88xf32> {
  // CHECK-LABEL: @unranked_conv_transpose
  // CHECK-NOT: hip.conv_transpose
  // CHECK: "onnx.ConvTranspose"
  %r = "onnx.ConvTranspose"(%x, %w) {auto_pad = "NOTSET", dilations = [1, 1],
      group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1],
      strides = [1, 1]}
      : (tensor<*xf32>, tensor<8x3x3x3xf32>) -> tensor<1x8x64x88xf32>
  return %r : tensor<1x8x64x88xf32>
}

func.func @unranked_softmax(%x: tensor<*xf32>) -> tensor<4x8xf32> {
  // CHECK-LABEL: @unranked_softmax
  // CHECK-NOT: hip.miopen.softmax
  // CHECK: "onnx.Softmax"
  %r = "onnx.Softmax"(%x) {axis = -1 : si64}
      : (tensor<*xf32>) -> tensor<4x8xf32>
  return %r : tensor<4x8xf32>
}

func.func @unranked_sigmoid(%x: tensor<*xf32>) -> tensor<4x8xf32> {
  // CHECK-LABEL: @unranked_sigmoid
  // CHECK-NOT: hip.sigmoid
  // CHECK: "onnx.Sigmoid"
  %r = "onnx.Sigmoid"(%x) : (tensor<*xf32>) -> tensor<4x8xf32>
  return %r : tensor<4x8xf32>
}

func.func @unranked_cast(%x: tensor<*xf32>) -> tensor<4x8xf16> {
  // CHECK-LABEL: @unranked_cast
  // CHECK-NOT: hip.cast
  // CHECK: "onnx.Cast"
  %r = "onnx.Cast"(%x) {to = f16} : (tensor<*xf32>) -> tensor<4x8xf16>
  return %r : tensor<4x8xf16>
}

func.func @unranked_mul(%a: tensor<*xf32>, %b: tensor<4x8xf32>)
    -> tensor<4x8xf32> {
  // CHECK-LABEL: @unranked_mul
  // CHECK-NOT: hip.mul
  // CHECK: "onnx.Mul"
  %r = "onnx.Mul"(%a, %b) : (tensor<*xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %r : tensor<4x8xf32>
}

func.func @unranked_sub(%a: tensor<*xf32>, %b: tensor<4x8xf32>)
    -> tensor<4x8xf32> {
  // CHECK-LABEL: @unranked_sub
  // CHECK-NOT: hip.sub
  // CHECK: "onnx.Sub"
  %r = "onnx.Sub"(%a, %b) : (tensor<*xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %r : tensor<4x8xf32>
}

func.func @unranked_tile(%x: tensor<*xf32>, %reps: tensor<4xi64>)
    -> tensor<1x3x64x88xf32> {
  // CHECK-LABEL: @unranked_tile
  // CHECK-NOT: hip.tile
  // CHECK: "onnx.Tile"
  %r = "onnx.Tile"(%x, %reps)
      : (tensor<*xf32>, tensor<4xi64>) -> tensor<1x3x64x88xf32>
  return %r : tensor<1x3x64x88xf32>
}

func.func @unranked_expand(%x: tensor<*xf32>, %shape: tensor<4xi64>)
    -> tensor<1x3x64x88xf32> {
  // CHECK-LABEL: @unranked_expand
  // CHECK-NOT: hip.expand
  // CHECK: "onnx.Expand"
  %r = "onnx.Expand"(%x, %shape)
      : (tensor<*xf32>, tensor<4xi64>) -> tensor<1x3x64x88xf32>
  return %r : tensor<1x3x64x88xf32>
}

// ReduceMean takes the operand guard only: an unranked RESULT is expected here
// and recovered by inferReduceResultType, which needs the ranked data to do it.
func.func @unranked_reduce_mean(%x: tensor<*xf32>) -> tensor<4x1xf32> {
  // CHECK-LABEL: @unranked_reduce_mean
  // CHECK-NOT: hip.reduce_mean
  // CHECK: "onnx.ReduceMean"
  %r = "onnx.ReduceMean"(%x) {axes = [1], keepdims = 1 : si64}
      : (tensor<*xf32>) -> tensor<4x1xf32>
  return %r : tensor<4x1xf32>
}

// The result-side guards: a ranked operand but a result shape inference never
// resolved. These are the cases that previously went through an unchecked
// `mlir::cast<RankedTensorType>` on the result type.
func.func @unranked_result_sigmoid(%x: tensor<4x8xf32>) -> tensor<*xf32> {
  // CHECK-LABEL: @unranked_result_sigmoid
  // CHECK-NOT: hip.sigmoid
  // CHECK: "onnx.Sigmoid"
  %r = "onnx.Sigmoid"(%x) : (tensor<4x8xf32>) -> tensor<*xf32>
  return %r : tensor<*xf32>
}

func.func @unranked_result_matmul(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>)
    -> tensor<*xf32> {
  // CHECK-LABEL: @unranked_result_matmul
  // CHECK-NOT: hip.matmul
  // CHECK: "onnx.MatMul"
  %r = "onnx.MatMul"(%a, %b)
      : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<*xf32>
  return %r : tensor<*xf32>
}
