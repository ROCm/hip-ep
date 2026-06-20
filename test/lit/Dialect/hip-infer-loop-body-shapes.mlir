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
//   - no-op when the body is already fully ranked (`already_ranked`).

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
