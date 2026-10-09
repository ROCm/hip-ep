// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-infer-shapes | FileCheck %s --check-prefix=INFER
// RUN: hip-mlir-opt %s --hip-resolve-tensor-dims | FileCheck %s --check-prefix=DIM

// Whole-shape reification remains semantic: the static leading input extent
// refines both SkipRMSNorm results and their DPS destinations.
// INFER-LABEL: func.func @semantic_whole_shape
// INFER: %[[OUT:.*]] = tensor.empty(%{{.*}}) : tensor<2x?x64xf16>
// INFER: %[[RES:.*]] = tensor.empty(%{{.*}}) : tensor<2x?x64xf16>
// INFER: %[[RESULT:.*]]:2 = hip.skip_rms_norm
// INFER-SAME: outs(%[[OUT]], %[[RES]] : tensor<2x?x64xf16>, tensor<2x?x64xf16>)
// INFER-SAME: : tensor<2x?x64xf16>, tensor<2x?x64xf16>
func.func @semantic_whole_shape(
    %ctx: !hip.context,
    %input: tensor<2x?x64xf16>,
    %skip: tensor<2x?x64xf16>,
    %gamma: tensor<64xf16>,
    %d0: index, %d1: index)
    -> (tensor<?x?x64xf16>, tensor<?x?x64xf16>) {
  %out = tensor.empty(%d0, %d1) : tensor<?x?x64xf16>
  %residual = tensor.empty(%d0, %d1) : tensor<?x?x64xf16>
  %result:2 = hip.skip_rms_norm(%ctx)
      ins(%input, %skip, %gamma :
          tensor<2x?x64xf16>, tensor<2x?x64xf16>, tensor<64xf16>)
      outs(%out, %residual : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
      {epsilon = 1.0e-05 : f32}
      : tensor<?x?x64xf16>, tensor<?x?x64xf16>
  return %result#0, %result#1 : tensor<?x?x64xf16>, tensor<?x?x64xf16>
}

// Direct per-dimension reification follows the exact result/init ties.
// DIM-LABEL: func.func @direct_two_output_dims
// DIM-SAME: %[[INPUT:[^,]+]]: tensor<?x?x64xf16>,
// DIM-SAME: %[[SKIP:[^,]+]]: tensor<?x?x64xf16>,
// DIM-SAME: %[[GAMMA:[^,]+]]: tensor<64xf16>,
// DIM-SAME: %[[OUTPUT:[^,]+]]: tensor<?x?x64xf16>,
// DIM-SAME: %[[RESIDUAL:[^)]+]]: tensor<?x?x64xf16>)
// DIM: %[[OUT_DIM:.*]] = tensor.dim %[[OUTPUT]], %{{.*}}
// DIM: %[[RES_DIM:.*]] = tensor.dim %[[RESIDUAL]], %{{.*}}
// DIM: return %[[OUT_DIM]], %[[RES_DIM]]
func.func @direct_two_output_dims(
    %ctx: !hip.context,
    %input: tensor<?x?x64xf16>,
    %skip: tensor<?x?x64xf16>,
    %gamma: tensor<64xf16>,
    %output: tensor<?x?x64xf16>,
    %residual: tensor<?x?x64xf16>) -> (index, index) {
  %result:2 = hip.skip_rms_norm(%ctx)
      ins(%input, %skip, %gamma :
          tensor<?x?x64xf16>, tensor<?x?x64xf16>, tensor<64xf16>)
      outs(%output, %residual : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
      {epsilon = 1.0e-05 : f32}
      : tensor<?x?x64xf16>, tensor<?x?x64xf16>
  %c0 = arith.constant 0 : index
  %out_dim = tensor.dim %result#0, %c0 : tensor<?x?x64xf16>
  %res_dim = tensor.dim %result#1, %c0 : tensor<?x?x64xf16>
  return %out_dim, %res_dim : index, index
}

// A long residual chain used to expand one tail query through every dimension
// of both results at every preceding layer. The direct hook terminates at the
// tied residual destination instead.
// DIM-LABEL: func.func @deep_residual_chain
// DIM-SAME: %[[INPUT:[^,]+]]: tensor<?x?x64xf16>,
// DIM-SAME: %[[SKIP:[^,]+]]: tensor<?x?x64xf16>,
// DIM-SAME: %[[GAMMA:[^,]+]]: tensor<64xf16>,
// DIM-SAME: %[[OUTPUT:[^,]+]]: tensor<?x?x64xf16>,
// DIM-SAME: %[[RESIDUAL:[^)]+]]: tensor<?x?x64xf16>)
// DIM-NOT: tensor.dim %{{.*}}#
// DIM: %[[TAIL_DIM:.*]] = tensor.dim %[[RESIDUAL]], %{{.*}}
// DIM: return %[[TAIL_DIM]]
func.func @deep_residual_chain(
    %ctx: !hip.context,
    %input: tensor<?x?x64xf16>,
    %skip: tensor<?x?x64xf16>,
    %gamma: tensor<64xf16>,
    %output: tensor<?x?x64xf16>,
    %residual: tensor<?x?x64xf16>) -> index {
  %r0:2 = hip.skip_rms_norm(%ctx)
      ins(%input, %skip, %gamma :
          tensor<?x?x64xf16>, tensor<?x?x64xf16>, tensor<64xf16>)
      outs(%output, %residual : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
      {epsilon = 1.0e-05 : f32}
      : tensor<?x?x64xf16>, tensor<?x?x64xf16>
  %r1:2 = hip.skip_rms_norm(%ctx)
      ins(%r0#0, %r0#1, %gamma :
          tensor<?x?x64xf16>, tensor<?x?x64xf16>, tensor<64xf16>)
      outs(%output, %residual : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
      {epsilon = 1.0e-05 : f32}
      : tensor<?x?x64xf16>, tensor<?x?x64xf16>
  %r2:2 = hip.skip_rms_norm(%ctx)
      ins(%r1#0, %r1#1, %gamma :
          tensor<?x?x64xf16>, tensor<?x?x64xf16>, tensor<64xf16>)
      outs(%output, %residual : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
      {epsilon = 1.0e-05 : f32}
      : tensor<?x?x64xf16>, tensor<?x?x64xf16>
  %r3:2 = hip.skip_rms_norm(%ctx)
      ins(%r2#0, %r2#1, %gamma :
          tensor<?x?x64xf16>, tensor<?x?x64xf16>, tensor<64xf16>)
      outs(%output, %residual : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
      {epsilon = 1.0e-05 : f32}
      : tensor<?x?x64xf16>, tensor<?x?x64xf16>
  %r4:2 = hip.skip_rms_norm(%ctx)
      ins(%r3#0, %r3#1, %gamma :
          tensor<?x?x64xf16>, tensor<?x?x64xf16>, tensor<64xf16>)
      outs(%output, %residual : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
      {epsilon = 1.0e-05 : f32}
      : tensor<?x?x64xf16>, tensor<?x?x64xf16>
  %r5:2 = hip.skip_rms_norm(%ctx)
      ins(%r4#0, %r4#1, %gamma :
          tensor<?x?x64xf16>, tensor<?x?x64xf16>, tensor<64xf16>)
      outs(%output, %residual : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
      {epsilon = 1.0e-05 : f32}
      : tensor<?x?x64xf16>, tensor<?x?x64xf16>
  %r6:2 = hip.skip_rms_norm(%ctx)
      ins(%r5#0, %r5#1, %gamma :
          tensor<?x?x64xf16>, tensor<?x?x64xf16>, tensor<64xf16>)
      outs(%output, %residual : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
      {epsilon = 1.0e-05 : f32}
      : tensor<?x?x64xf16>, tensor<?x?x64xf16>
  %r7:2 = hip.skip_rms_norm(%ctx)
      ins(%r6#0, %r6#1, %gamma :
          tensor<?x?x64xf16>, tensor<?x?x64xf16>, tensor<64xf16>)
      outs(%output, %residual : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
      {epsilon = 1.0e-05 : f32}
      : tensor<?x?x64xf16>, tensor<?x?x64xf16>
  %r8:2 = hip.skip_rms_norm(%ctx)
      ins(%r7#0, %r7#1, %gamma :
          tensor<?x?x64xf16>, tensor<?x?x64xf16>, tensor<64xf16>)
      outs(%output, %residual : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
      {epsilon = 1.0e-05 : f32}
      : tensor<?x?x64xf16>, tensor<?x?x64xf16>
  %r9:2 = hip.skip_rms_norm(%ctx)
      ins(%r8#0, %r8#1, %gamma :
          tensor<?x?x64xf16>, tensor<?x?x64xf16>, tensor<64xf16>)
      outs(%output, %residual : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
      {epsilon = 1.0e-05 : f32}
      : tensor<?x?x64xf16>, tensor<?x?x64xf16>
  %r10:2 = hip.skip_rms_norm(%ctx)
      ins(%r9#0, %r9#1, %gamma :
          tensor<?x?x64xf16>, tensor<?x?x64xf16>, tensor<64xf16>)
      outs(%output, %residual : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
      {epsilon = 1.0e-05 : f32}
      : tensor<?x?x64xf16>, tensor<?x?x64xf16>
  %r11:2 = hip.skip_rms_norm(%ctx)
      ins(%r10#0, %r10#1, %gamma :
          tensor<?x?x64xf16>, tensor<?x?x64xf16>, tensor<64xf16>)
      outs(%output, %residual : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
      {epsilon = 1.0e-05 : f32}
      : tensor<?x?x64xf16>, tensor<?x?x64xf16>
  %r12:2 = hip.skip_rms_norm(%ctx)
      ins(%r11#0, %r11#1, %gamma :
          tensor<?x?x64xf16>, tensor<?x?x64xf16>, tensor<64xf16>)
      outs(%output, %residual : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
      {epsilon = 1.0e-05 : f32}
      : tensor<?x?x64xf16>, tensor<?x?x64xf16>
  %r13:2 = hip.skip_rms_norm(%ctx)
      ins(%r12#0, %r12#1, %gamma :
          tensor<?x?x64xf16>, tensor<?x?x64xf16>, tensor<64xf16>)
      outs(%output, %residual : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
      {epsilon = 1.0e-05 : f32}
      : tensor<?x?x64xf16>, tensor<?x?x64xf16>
  %r14:2 = hip.skip_rms_norm(%ctx)
      ins(%r13#0, %r13#1, %gamma :
          tensor<?x?x64xf16>, tensor<?x?x64xf16>, tensor<64xf16>)
      outs(%output, %residual : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
      {epsilon = 1.0e-05 : f32}
      : tensor<?x?x64xf16>, tensor<?x?x64xf16>
  %r15:2 = hip.skip_rms_norm(%ctx)
      ins(%r14#0, %r14#1, %gamma :
          tensor<?x?x64xf16>, tensor<?x?x64xf16>, tensor<64xf16>)
      outs(%output, %residual : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
      {epsilon = 1.0e-05 : f32}
      : tensor<?x?x64xf16>, tensor<?x?x64xf16>
  %r16:2 = hip.skip_rms_norm(%ctx)
      ins(%r15#0, %r15#1, %gamma :
          tensor<?x?x64xf16>, tensor<?x?x64xf16>, tensor<64xf16>)
      outs(%output, %residual : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
      {epsilon = 1.0e-05 : f32}
      : tensor<?x?x64xf16>, tensor<?x?x64xf16>
  %r17:2 = hip.skip_rms_norm(%ctx)
      ins(%r16#0, %r16#1, %gamma :
          tensor<?x?x64xf16>, tensor<?x?x64xf16>, tensor<64xf16>)
      outs(%output, %residual : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
      {epsilon = 1.0e-05 : f32}
      : tensor<?x?x64xf16>, tensor<?x?x64xf16>
  %r18:2 = hip.skip_rms_norm(%ctx)
      ins(%r17#0, %r17#1, %gamma :
          tensor<?x?x64xf16>, tensor<?x?x64xf16>, tensor<64xf16>)
      outs(%output, %residual : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
      {epsilon = 1.0e-05 : f32}
      : tensor<?x?x64xf16>, tensor<?x?x64xf16>
  %r19:2 = hip.skip_rms_norm(%ctx)
      ins(%r18#0, %r18#1, %gamma :
          tensor<?x?x64xf16>, tensor<?x?x64xf16>, tensor<64xf16>)
      outs(%output, %residual : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
      {epsilon = 1.0e-05 : f32}
      : tensor<?x?x64xf16>, tensor<?x?x64xf16>
  %r20:2 = hip.skip_rms_norm(%ctx)
      ins(%r19#0, %r19#1, %gamma :
          tensor<?x?x64xf16>, tensor<?x?x64xf16>, tensor<64xf16>)
      outs(%output, %residual : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
      {epsilon = 1.0e-05 : f32}
      : tensor<?x?x64xf16>, tensor<?x?x64xf16>
  %r21:2 = hip.skip_rms_norm(%ctx)
      ins(%r20#0, %r20#1, %gamma :
          tensor<?x?x64xf16>, tensor<?x?x64xf16>, tensor<64xf16>)
      outs(%output, %residual : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
      {epsilon = 1.0e-05 : f32}
      : tensor<?x?x64xf16>, tensor<?x?x64xf16>
  %r22:2 = hip.skip_rms_norm(%ctx)
      ins(%r21#0, %r21#1, %gamma :
          tensor<?x?x64xf16>, tensor<?x?x64xf16>, tensor<64xf16>)
      outs(%output, %residual : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
      {epsilon = 1.0e-05 : f32}
      : tensor<?x?x64xf16>, tensor<?x?x64xf16>
  %r23:2 = hip.skip_rms_norm(%ctx)
      ins(%r22#0, %r22#1, %gamma :
          tensor<?x?x64xf16>, tensor<?x?x64xf16>, tensor<64xf16>)
      outs(%output, %residual : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
      {epsilon = 1.0e-05 : f32}
      : tensor<?x?x64xf16>, tensor<?x?x64xf16>
  %r24:2 = hip.skip_rms_norm(%ctx)
      ins(%r23#0, %r23#1, %gamma :
          tensor<?x?x64xf16>, tensor<?x?x64xf16>, tensor<64xf16>)
      outs(%output, %residual : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
      {epsilon = 1.0e-05 : f32}
      : tensor<?x?x64xf16>, tensor<?x?x64xf16>
  %r25:2 = hip.skip_rms_norm(%ctx)
      ins(%r24#0, %r24#1, %gamma :
          tensor<?x?x64xf16>, tensor<?x?x64xf16>, tensor<64xf16>)
      outs(%output, %residual : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
      {epsilon = 1.0e-05 : f32}
      : tensor<?x?x64xf16>, tensor<?x?x64xf16>
  %r26:2 = hip.skip_rms_norm(%ctx)
      ins(%r25#0, %r25#1, %gamma :
          tensor<?x?x64xf16>, tensor<?x?x64xf16>, tensor<64xf16>)
      outs(%output, %residual : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
      {epsilon = 1.0e-05 : f32}
      : tensor<?x?x64xf16>, tensor<?x?x64xf16>
  %r27:2 = hip.skip_rms_norm(%ctx)
      ins(%r26#0, %r26#1, %gamma :
          tensor<?x?x64xf16>, tensor<?x?x64xf16>, tensor<64xf16>)
      outs(%output, %residual : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
      {epsilon = 1.0e-05 : f32}
      : tensor<?x?x64xf16>, tensor<?x?x64xf16>
  %r28:2 = hip.skip_rms_norm(%ctx)
      ins(%r27#0, %r27#1, %gamma :
          tensor<?x?x64xf16>, tensor<?x?x64xf16>, tensor<64xf16>)
      outs(%output, %residual : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
      {epsilon = 1.0e-05 : f32}
      : tensor<?x?x64xf16>, tensor<?x?x64xf16>
  %r29:2 = hip.skip_rms_norm(%ctx)
      ins(%r28#0, %r28#1, %gamma :
          tensor<?x?x64xf16>, tensor<?x?x64xf16>, tensor<64xf16>)
      outs(%output, %residual : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
      {epsilon = 1.0e-05 : f32}
      : tensor<?x?x64xf16>, tensor<?x?x64xf16>
  %r30:2 = hip.skip_rms_norm(%ctx)
      ins(%r29#0, %r29#1, %gamma :
          tensor<?x?x64xf16>, tensor<?x?x64xf16>, tensor<64xf16>)
      outs(%output, %residual : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
      {epsilon = 1.0e-05 : f32}
      : tensor<?x?x64xf16>, tensor<?x?x64xf16>
  %r31:2 = hip.skip_rms_norm(%ctx)
      ins(%r30#0, %r30#1, %gamma :
          tensor<?x?x64xf16>, tensor<?x?x64xf16>, tensor<64xf16>)
      outs(%output, %residual : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
      {epsilon = 1.0e-05 : f32}
      : tensor<?x?x64xf16>, tensor<?x?x64xf16>
  %c0 = arith.constant 0 : index
  %tail_dim = tensor.dim %r31#1, %c0 : tensor<?x?x64xf16>
  return %tail_dim : index
}
