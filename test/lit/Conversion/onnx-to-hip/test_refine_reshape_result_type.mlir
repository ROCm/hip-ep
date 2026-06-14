// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Exercises the pre-lowering result-type refinement patterns in
// lib/Conversion/OnnxToHip/RefineReshapeResultType.cpp. HF dynamic-shape
// exports drop the static dims on shape-deterministic ops, leaving
// `tensor<?x?x?>` results that block the downstream static-dim lowerings
// (collapse-reassociation choice, `tensor.empty` sizing). The patterns thread
// the recoverable statics forward; here we drive the full ONNX->HIP conversion
// and check that the lowered ops carry the refined extents.
//
// Coverage: Reshape-from-shape-operand (Slice(Shape(x)) prefix + inferred -1,
// with the dynamic batch dim cancelled out of the element-count arithmetic),
// Transpose forwarding through `perm`, and elementwise numpy-broadcast.
//
// CHECK style mirrors upstream monotone-refinement tests (e.g. StableHLO's
// stablehlo_refine_shapes.mlir): inline, SSA-captured checks asserting the
// refined result extent right at the op it lands on.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x128x4096xf16>) -> tensor<1x128x4096xf16> {
    return %arg0 : tensor<1x128x4096xf16>
  }

  // Reshape target = Concat(Slice(Shape(x), [0:2]), -1): the [B, 1152] prefix
  // comes from the data's own shape (batch left dynamic, 1152 pinned) and the
  // trailing -1 resolves to 16*16 = 256 by element count. A second Reshape to a
  // distinct static shape keeps the module self-consistent and proves the
  // ?x1152x256 extent came from refinement (it is absent from the signature).
  //
  // CHECK-LABEL: func.func @refine_reshape_neg1
  // CHECK-NOT:   onnx.Reshape
  // CHECK:       %[[COLL:.*]] = tensor.collapse_shape %{{.*}} {{\[\[}}0], [1], [2, 3]] : tensor<?x1152x16x16xf16> into tensor<?x1152x256xf16>
  // CHECK:       tensor.collapse_shape %[[COLL]] {{\[\[}}0], [1, 2]] : tensor<?x1152x256xf16> into tensor<?x294912xf16>
  func.func @refine_reshape_neg1(%x: tensor<?x1152x16x16xf16>) -> tensor<?x294912xf16> {
    %sh = "onnx.Shape"(%x) : (tensor<?x1152x16x16xf16>) -> tensor<4xi64>
    %s0 = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
    %e2 = "onnx.Constant"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>
    %a0 = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
    %sl = "onnx.Slice"(%sh, %s0, %e2, %a0) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
    %m1 = "onnx.Constant"() {value = dense<-1> : tensor<1xi64>} : () -> tensor<1xi64>
    %shp = "onnx.Concat"(%sl, %m1) {axis = 0 : si64} : (tensor<2xi64>, tensor<1xi64>) -> tensor<3xi64>
    %r = "onnx.Reshape"(%x, %shp) : (tensor<?x1152x16x16xf16>, tensor<3xi64>) -> tensor<?x?x?xf16>
    %shp2 = "onnx.Constant"() {value = dense<[-1, 294912]> : tensor<2xi64>} : () -> tensor<2xi64>
    %out = "onnx.Reshape"(%r, %shp2) : (tensor<?x?x?xf16>, tensor<2xi64>) -> tensor<?x294912xf16>
    return %out : tensor<?x294912xf16>
  }

  // Refined extents propagate down a Reshape -> Transpose -> broadcast-Mul
  // chain: the transpose permutes [256, 1152] under perm=[0,2,1], and the Mul
  // broadcasts a tensor<?x256x1> operand against the transposed tensor.
  //
  // CHECK-LABEL: func.func @refine_chain
  // CHECK-NOT:   onnx.Reshape
  // CHECK:       tensor.collapse_shape %{{.*}} {{\[\[}}0], [1], [2, 3]] : tensor<?x1152x16x16xf16> into tensor<?x1152x256xf16>
  // CHECK:       hip.transpose
  // CHECK-SAME:    ins(%{{.*}} : tensor<?x1152x256xf16>)
  // CHECK-SAME:    outs(%{{.*}} : tensor<?x256x1152xf16>)
  // CHECK-SAME:    {perm = [0, 2, 1]}
  // CHECK:       hip.mul
  // CHECK-SAME:    ins(%{{.*}}, %{{.*}} : tensor<?x256x1152xf16>, tensor<?x256x1xf16>)
  // CHECK-SAME:    -> tensor<?x256x1152xf16>
  func.func @refine_chain(%x: tensor<?x1152x16x16xf16>, %b: tensor<?x256x1xf16>) -> tensor<?x294912xf16> {
    %sh = "onnx.Shape"(%x) : (tensor<?x1152x16x16xf16>) -> tensor<4xi64>
    %s0 = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
    %e2 = "onnx.Constant"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>
    %a0 = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
    %sl = "onnx.Slice"(%sh, %s0, %e2, %a0) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
    %m1 = "onnx.Constant"() {value = dense<-1> : tensor<1xi64>} : () -> tensor<1xi64>
    %shp = "onnx.Concat"(%sl, %m1) {axis = 0 : si64} : (tensor<2xi64>, tensor<1xi64>) -> tensor<3xi64>
    %r = "onnx.Reshape"(%x, %shp) : (tensor<?x1152x16x16xf16>, tensor<3xi64>) -> tensor<?x?x?xf16>
    %t = "onnx.Transpose"(%r) {perm = [0, 2, 1]} : (tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
    %mul = "onnx.Mul"(%t, %b) : (tensor<?x?x?xf16>, tensor<?x256x1xf16>) -> tensor<?x?x?xf16>
    %shp2 = "onnx.Constant"() {value = dense<[-1, 294912]> : tensor<2xi64>} : () -> tensor<2xi64>
    %out = "onnx.Reshape"(%mul, %shp2) : (tensor<?x?x?xf16>, tensor<2xi64>) -> tensor<?x294912xf16>
    return %out : tensor<?x294912xf16>
  }
}
