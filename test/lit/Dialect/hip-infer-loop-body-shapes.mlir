// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-infer-loop-body-shapes %s | FileCheck %s

// What this file tests
// --------------------
// The `--hip-infer-loop-body-shapes` pass (lib/Dialect/Transforms/
// InferLoopBodyShapesPass.cpp), which rank-establishes `tensor<*xT>`
// values inside outlined `hip.loop` body functions BEFORE
// `convert-onnx-to-hip`:
//
//   - forward `onnx.Concat` shape rule ranks an unranked loop-carried
//     output from its ranked operands (`forward_concat`),
//   - loop-contract backstop ranks a returned value from $v_init when no
//     forward rule applies (`backstop_no_rule`),
//   - non-passthrough loops put cond_out at return slot 0 and v_carry at
//     slot 1 (`non_passthrough`),
//   - no-op when the body is already fully ranked (`already_ranked`),
//   - a main-graph Reshape whose shape vector has a static length gets
//     that rank, and ops that copy the shape pick it up
//     (`reshape_rank_from_shape`). A dynamic-length shape vector does
//     not (`reshape_rank_unknown`).

// -----------------------------------------------------------------------
// Forward Concat rule: Concat((1x?x1152, ?x?x?), axis=1) -> 1x?x1152.
// Non-axis dims prefer any operand's literal extent; axis dim is dynamic
// because one operand is dynamic there.
// -----------------------------------------------------------------------
func.func @forward_concat(%ctx: !hip.context, %M: index, %cond: i1,
                          %acc: tensor<1x?x1152xf16>,
                          %other: tensor<?x?x?xf16>) -> tensor<1x?x1152xf16> {
  %r = hip.loop(%ctx, %M, %cond)
                 iter_args(%acc : tensor<1x?x1152xf16>)
                 captures(%other : tensor<?x?x?xf16>)
                 -> (tensor<1x?x1152xf16>)
                 body @body_concat
                 {num_loop_carried = 1 : i32, cond_is_passthrough}
  return %r : tensor<1x?x1152xf16>
}

// CHECK-LABEL: func.func private @body_concat
// CHECK-SAME:    -> tensor<1x?x1152xf16>
// CHECK:         %[[R:.*]] = "onnx.Concat"
// CHECK-SAME:      -> tensor<1x?x1152xf16>
// CHECK:         return %[[R]] : tensor<1x?x1152xf16>
func.func private @body_concat(%ctx: !hip.context, %iter: tensor<i64>,
                               %cond_in: tensor<ui8>,
                               %acc: tensor<1x?x1152xf16>,
                               %other: tensor<?x?x?xf16>) -> tensor<*xf16> {
  %r = "onnx.Concat"(%acc, %other) {axis = 1 : si64} :
         (tensor<1x?x1152xf16>, tensor<?x?x?xf16>) -> tensor<*xf16>
  return %r : tensor<*xf16>
}

// -----------------------------------------------------------------------
// Loop-contract backstop: an op with no forward rule still gets its
// returned (loop-carried) result ranked from $v_init.
// -----------------------------------------------------------------------
func.func @backstop_no_rule(%ctx: !hip.context, %M: index, %cond: i1,
                            %acc: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %r = hip.loop(%ctx, %M, %cond)
                 iter_args(%acc : tensor<4x8xf32>)
                 -> (tensor<4x8xf32>)
                 body @body_backstop
                 {num_loop_carried = 1 : i32, cond_is_passthrough}
  return %r : tensor<4x8xf32>
}

// CHECK-LABEL: func.func private @body_backstop
// CHECK-SAME:    -> tensor<4x8xf32>
// CHECK:         %[[R:.*]] = "onnx.NoRuleOp"
// CHECK-SAME:      -> tensor<4x8xf32>
// CHECK:         return %[[R]] : tensor<4x8xf32>
func.func private @body_backstop(%ctx: !hip.context, %iter: tensor<i64>,
                                 %cond_in: tensor<ui8>,
                                 %acc: tensor<4x8xf32>) -> tensor<*xf32> {
  %r = "onnx.NoRuleOp"(%acc) : (tensor<4x8xf32>) -> tensor<*xf32>
  return %r : tensor<*xf32>
}

// -----------------------------------------------------------------------
// Non-passthrough loop: return slot 0 is cond_out, v_carry is slot 1.
// -----------------------------------------------------------------------
func.func @non_passthrough(%ctx: !hip.context, %M: index, %cond: i1,
                           %acc: tensor<1x?x1152xf16>,
                           %other: tensor<?x?x?xf16>) -> tensor<1x?x1152xf16> {
  %r = hip.loop(%ctx, %M, %cond)
                 iter_args(%acc : tensor<1x?x1152xf16>)
                 captures(%other : tensor<?x?x?xf16>)
                 -> (tensor<1x?x1152xf16>)
                 body @body_non_passthrough
                 {num_loop_carried = 1 : i32}
  return %r : tensor<1x?x1152xf16>
}

// CHECK-LABEL: func.func private @body_non_passthrough
// CHECK-SAME:    -> (tensor<ui8>, tensor<1x?x1152xf16>)
// CHECK:         %[[R:.*]] = "onnx.Concat"
// CHECK-SAME:      -> tensor<1x?x1152xf16>
// CHECK:         return %{{.*}}, %[[R]] : tensor<ui8>, tensor<1x?x1152xf16>
func.func private @body_non_passthrough(%ctx: !hip.context, %iter: tensor<i64>,
                                        %cond_in: tensor<ui8>,
                                        %acc: tensor<1x?x1152xf16>,
                                        %other: tensor<?x?x?xf16>)
    -> (tensor<ui8>, tensor<*xf16>) {
  %r = "onnx.Concat"(%acc, %other) {axis = 1 : si64} :
         (tensor<1x?x1152xf16>, tensor<?x?x?xf16>) -> tensor<*xf16>
  return %cond_in, %r : tensor<ui8>, tensor<*xf16>
}

// -----------------------------------------------------------------------
// Already fully ranked: pass is a no-op (no type changes).
// -----------------------------------------------------------------------
func.func @already_ranked(%ctx: !hip.context, %M: index, %cond: i1,
                          %acc: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %r = hip.loop(%ctx, %M, %cond)
                 iter_args(%acc : tensor<4x8xf32>)
                 -> (tensor<4x8xf32>)
                 body @body_ranked
                 {num_loop_carried = 1 : i32, cond_is_passthrough}
  return %r : tensor<4x8xf32>
}

// CHECK-LABEL: func.func private @body_ranked
// CHECK-SAME:    -> tensor<4x8xf32>
// CHECK:         return %{{.*}} : tensor<4x8xf32>
func.func private @body_ranked(%ctx: !hip.context, %iter: tensor<i64>,
                               %cond_in: tensor<ui8>,
                               %acc: tensor<4x8xf32>) -> tensor<4x8xf32> {
  return %acc : tensor<4x8xf32>
}

// -----------------------------------------------------------------------
// Main graph, no loop. The shape vector length is the Reshape result rank.
// Cast copies that shape, MatMul keeps the weight's trailing extent, and
// ReduceMean drops nothing because keepdims = 1.
// -----------------------------------------------------------------------
// CHECK-LABEL: func.func @reshape_rank_from_shape
// CHECK-SAME:    -> tensor<?x?x1xf32>
// CHECK:         "onnx.Reshape"{{.*}} -> tensor<?x?x?xi64>
// CHECK:         "onnx.Cast"{{.*}} -> tensor<?x?x?xf32>
// CHECK:         "onnx.MatMul"{{.*}} -> tensor<?x?x768xf32>
// CHECK:         "onnx.ReduceMean"{{.*}} -> tensor<?x?x1xf32>
func.func @reshape_rank_from_shape(%x: tensor<?x256xi64>, %shape: tensor<3xi64>,
                                   %weight: tensor<768x768xf32>) -> tensor<*xf32> {
  %r = "onnx.Reshape"(%x, %shape) : (tensor<?x256xi64>, tensor<3xi64>) -> tensor<*xi64>
  %c = "onnx.Cast"(%r) {to = f32} : (tensor<*xi64>) -> tensor<*xf32>
  %m = "onnx.MatMul"(%c, %weight) : (tensor<*xf32>, tensor<768x768xf32>) -> tensor<*xf32>
  %y = "onnx.ReduceMean"(%m) {axes = [2], keepdims = 1 : si64} : (tensor<*xf32>) -> tensor<*xf32>
  return %y : tensor<*xf32>
}

// A shape vector of unknown length does not fix the rank.
// CHECK-LABEL: func.func @reshape_rank_unknown
// CHECK:         -> tensor<*xi64>
func.func @reshape_rank_unknown(%x: tensor<?x256xi64>, %shape: tensor<?xi64>) -> tensor<*xi64> {
  %r = "onnx.Reshape"(%x, %shape) : (tensor<?x256xi64>, tensor<?xi64>) -> tensor<*xi64>
  return %r : tensor<*xi64>
}

// Constant entries of the shape vector are static dims. The runtime entry
// stays dynamic. Transpose then permutes those dims.
// CHECK-LABEL: func.func @reshape_static_shape_entries
// CHECK:         "onnx.Reshape"{{.*}} -> tensor<?x256x2xf32>
// CHECK:         "onnx.Transpose"{{.*}} -> tensor<2x?x256xf32>
func.func @reshape_static_shape_entries(%x: tensor<?x2xf32>, %batch: tensor<i64>) -> tensor<*xf32> {
  %c256 = "onnx.Constant"() {value = dense<256> : tensor<i64>} : () -> tensor<i64>
  %c2 = "onnx.Constant"() {value = dense<2> : tensor<i64>} : () -> tensor<i64>
  %b = "onnx.Unsqueeze"(%batch) {axes = [0]} : (tensor<i64>) -> tensor<1xi64>
  %d256 = "onnx.Unsqueeze"(%c256) {axes = [0]} : (tensor<i64>) -> tensor<1xi64>
  %d2 = "onnx.Unsqueeze"(%c2) {axes = [0]} : (tensor<i64>) -> tensor<1xi64>
  %shape = "onnx.Concat"(%b, %d256, %d2) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
  %r = "onnx.Reshape"(%x, %shape) : (tensor<?x2xf32>, tensor<3xi64>) -> tensor<*xf32>
  %t = "onnx.Transpose"(%r) {perm = [2, 0, 1]} : (tensor<*xf32>) -> tensor<*xf32>
  return %t : tensor<*xf32>
}

// Pad then Conv. Pads and the conv formula are static, so the unranked
// results become the concrete FNS-Candy shapes (224 padded by 4, 9x9 conv).
// CHECK-LABEL: func.func @pad_conv_static
// CHECK:         "onnx.Pad"{{.*}} -> tensor<1x3x232x232xf32>
// CHECK:         "onnx.Conv"{{.*}} -> tensor<1x32x224x224xf32>
// CHECK:         "onnx.InstanceNormalization"{{.*}} -> tensor<1x32x224x224xf32>
func.func @pad_conv_static(%x: tensor<1x3x224x224xf32>,
                          %w: tensor<32x3x9x9xf32>,
                          %b: tensor<32xf32>,
                          %scale: tensor<32xf32>,
                          %bias: tensor<32xf32>) -> tensor<*xf32> {
  %p = "onnx.Pad"(%x) {mode = "reflect", pads = [0, 0, 4, 4, 0, 0, 4, 4]}
      : (tensor<1x3x224x224xf32>) -> tensor<*xf32>
  %c = "onnx.Conv"(%p, %w, %b) {dilations = [1, 1], group = 1 : si64,
                                kernel_shape = [9, 9], pads = [0, 0, 0, 0],
                                strides = [1, 1]}
      : (tensor<*xf32>, tensor<32x3x9x9xf32>, tensor<32xf32>) -> tensor<*xf32>
  %n = "onnx.InstanceNormalization"(%c, %scale, %bias)
      : (tensor<*xf32>, tensor<32xf32>, tensor<32xf32>) -> tensor<*xf32>
  return %n : tensor<*xf32>
}

// Upsample scales folded from Shape(x) * 2 / Shape(x), i.e. [1, 1, 2, 2].
// CHECK-LABEL: func.func @upsample_scales_from_shape
// CHECK:         "onnx.Upsample"{{.*}} -> tensor<1x128x112x112xf32>
func.func @upsample_scales_from_shape(%x: tensor<1x128x56x56xf32>) -> tensor<*xf32> {
  %two = "onnx.Constant"() {value = dense<2> : tensor<i64>} : () -> tensor<i64>
  %i2 = "onnx.Constant"() {value = dense<2> : tensor<i64>} : () -> tensor<i64>
  %i3 = "onnx.Constant"() {value = dense<3> : tensor<i64>} : () -> tensor<i64>
  %ones = "onnx.Constant"() {value = dense<[1.0, 1.0]> : tensor<2xf32>} : () -> tensor<2xf32>
  %sh = "onnx.Shape"(%x) : (tensor<1x128x56x56xf32>) -> tensor<*xi64>
  %h = "onnx.Gather"(%sh, %i2) {axis = 0 : si64} : (tensor<*xi64>, tensor<i64>) -> tensor<*xi64>
  %w = "onnx.Gather"(%sh, %i3) {axis = 0 : si64} : (tensor<*xi64>, tensor<i64>) -> tensor<*xi64>
  %h2 = "onnx.Mul"(%h, %two) : (tensor<*xi64>, tensor<i64>) -> tensor<*xi64>
  %w2 = "onnx.Mul"(%w, %two) : (tensor<*xi64>, tensor<i64>) -> tensor<*xi64>
  %hf = "onnx.Cast"(%h2) {to = 1 : si64} : (tensor<*xi64>) -> tensor<*xf32>
  %wf = "onnx.Cast"(%w2) {to = 1 : si64} : (tensor<*xi64>) -> tensor<*xf32>
  %hfloor = "onnx.Floor"(%hf) : (tensor<*xf32>) -> tensor<*xf32>
  %wfloor = "onnx.Floor"(%wf) : (tensor<*xf32>) -> tensor<*xf32>
  %hu = "onnx.Unsqueeze"(%hfloor) {axes = [0]} : (tensor<*xf32>) -> tensor<*xf32>
  %wu = "onnx.Unsqueeze"(%wfloor) {axes = [0]} : (tensor<*xf32>) -> tensor<*xf32>
  %hw = "onnx.Concat"(%hu, %wu) {axis = 0 : si64} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  %spatial = "onnx.Slice"(%sh) {axes = [0], starts = [2], ends = [4]}
      : (tensor<*xi64>) -> tensor<*xi64>
  %spatialf = "onnx.Cast"(%spatial) {to = 1 : si64} : (tensor<*xi64>) -> tensor<*xf32>
  %scaleshw = "onnx.Div"(%hw, %spatialf) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  %scales = "onnx.Concat"(%ones, %scaleshw) {axis = 0 : si64}
      : (tensor<2xf32>, tensor<*xf32>) -> tensor<*xf32>
  %y = "onnx.Upsample"(%x, %scales) {mode = "nearest"}
      : (tensor<1x128x56x56xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %y : tensor<*xf32>
}
