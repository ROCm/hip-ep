// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Verify hip.gemm expands into the tosa.matmul + tosa.add chain that ONNX Gemm
// means: optional transposes, a batch-of-one around the batched tosa.matmul,
// the alpha/beta scales, and C broadcast to [M, N].
//
// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file \
// RUN:   --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @gemm
// CHECK: %[[A:.*]] = tosa.reshape %arg1, {{.*}} -> tensor<1x32x64xf32>
// CHECK: %[[B:.*]] = tosa.reshape %arg2, {{.*}} -> tensor<1x64x128xf32>
// CHECK: %[[MM:.*]] = tosa.matmul %[[A]], %[[B]]
// CHECK-SAME: -> tensor<1x32x128xf32>
// CHECK: %[[FLAT:.*]] = tosa.reshape %[[MM]], {{.*}} -> tensor<32x128xf32>
// CHECK: %[[C:.*]] = tosa.reshape %arg3, {{.*}} -> tensor<1x128xf32>
// CHECK: tosa.add %[[FLAT]], %[[C]]
// CHECK-NOT: hip.gemm
func.func @gemm(%ctx: !hip.context, %a: tensor<32x64xf32>,
                %b: tensor<64x128xf32>, %c: tensor<128xf32>,
                %init: tensor<32x128xf32>) -> tensor<32x128xf32>
    attributes {rock.kernel} {
  %r = hip.gemm(%ctx) ins(%a, %b, %c :
      tensor<32x64xf32>, tensor<64x128xf32>, tensor<128xf32>)
      outs(%init : tensor<32x128xf32>) : tensor<32x128xf32>
  return %r : tensor<32x128xf32>
}

// -----

// C is optional; without it the matmul result is returned directly.
// CHECK-LABEL: func.func @gemm_no_c
// CHECK: tosa.matmul
// CHECK-NOT: tosa.add
// CHECK-NOT: hip.gemm
func.func @gemm_no_c(%ctx: !hip.context, %a: tensor<4x8xf32>,
                     %b: tensor<8x16xf32>, %init: tensor<4x16xf32>)
    -> tensor<4x16xf32> attributes {rock.kernel} {
  %r = hip.gemm(%ctx) ins(%a, %b : tensor<4x8xf32>, tensor<8x16xf32>)
      outs(%init : tensor<4x16xf32>) : tensor<4x16xf32>
  return %r : tensor<4x16xf32>
}

// -----

// transA swaps A's dims before the matmul; A is [K, M] on the way in.
// CHECK-LABEL: func.func @gemm_trans_a
// CHECK: %[[AT:.*]] = tosa.transpose %arg1 {perms = array<i32: 1, 0>} : (tensor<8x4xf32>) -> tensor<4x8xf32>
// CHECK: tosa.reshape %[[AT]], {{.*}} -> tensor<1x4x8xf32>
// CHECK: tosa.matmul
// CHECK-NOT: hip.gemm
func.func @gemm_trans_a(%ctx: !hip.context, %a: tensor<8x4xf32>,
                        %b: tensor<8x16xf32>, %init: tensor<4x16xf32>)
    -> tensor<4x16xf32> attributes {rock.kernel} {
  %r = hip.gemm(%ctx) ins(%a, %b : tensor<8x4xf32>, tensor<8x16xf32>)
      outs(%init : tensor<4x16xf32>) {transA = 1 : i64} : tensor<4x16xf32>
  return %r : tensor<4x16xf32>
}

// -----

// transB swaps B's dims; B is [N, K] on the way in.
// CHECK-LABEL: func.func @gemm_trans_b
// CHECK: %[[BT:.*]] = tosa.transpose %arg2 {perms = array<i32: 1, 0>} : (tensor<16x8xf32>) -> tensor<8x16xf32>
// CHECK: tosa.reshape %[[BT]], {{.*}} -> tensor<1x8x16xf32>
// CHECK: tosa.matmul
// CHECK-NOT: hip.gemm
func.func @gemm_trans_b(%ctx: !hip.context, %a: tensor<4x8xf32>,
                        %b: tensor<16x8xf32>, %init: tensor<4x16xf32>)
    -> tensor<4x16xf32> attributes {rock.kernel} {
  %r = hip.gemm(%ctx) ins(%a, %b : tensor<4x8xf32>, tensor<16x8xf32>)
      outs(%init : tensor<4x16xf32>) {transB = 1 : i64} : tensor<4x16xf32>
  return %r : tensor<4x16xf32>
}

// -----

// alpha scales the product and beta scales C, each as a splat multiply.
// CHECK-LABEL: func.func @gemm_alpha_beta
// CHECK: %[[ALPHA:.*]] = "tosa.const"() <{values = dense<2.000000e+00> : tensor<4x16xf32>}>
// CHECK: %[[SCALED:.*]] = tosa.mul {{.*}}, %[[ALPHA]]
// CHECK: %[[BETA:.*]] = "tosa.const"() <{values = dense<5.000000e-01> : tensor<1x16xf32>}>
// CHECK: %[[CB:.*]] = tosa.mul {{.*}}, %[[BETA]]
// CHECK: tosa.add %[[SCALED]], %[[CB]]
// CHECK-NOT: hip.gemm
func.func @gemm_alpha_beta(%ctx: !hip.context, %a: tensor<4x8xf32>,
                           %b: tensor<8x16xf32>, %c: tensor<16xf32>,
                           %init: tensor<4x16xf32>) -> tensor<4x16xf32>
    attributes {rock.kernel} {
  %r = hip.gemm(%ctx) ins(%a, %b, %c :
      tensor<4x8xf32>, tensor<8x16xf32>, tensor<16xf32>)
      outs(%init : tensor<4x16xf32>)
      {alpha = 2.0 : f32, beta = 0.5 : f32} : tensor<4x16xf32>
  return %r : tensor<4x16xf32>
}

// -----

// A C that already has the result's rank and shape needs no reshape.
// CHECK-LABEL: func.func @gemm_full_c
// CHECK: %[[MM:.*]] = tosa.matmul
// CHECK: %[[FLAT:.*]] = tosa.reshape %[[MM]], {{.*}} -> tensor<4x16xf32>
// CHECK: tosa.add %[[FLAT]], %arg3 : (tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<4x16xf32>
// CHECK-NOT: hip.gemm
func.func @gemm_full_c(%ctx: !hip.context, %a: tensor<4x8xf32>,
                       %b: tensor<8x16xf32>, %c: tensor<4x16xf32>,
                       %init: tensor<4x16xf32>) -> tensor<4x16xf32>
    attributes {rock.kernel} {
  %r = hip.gemm(%ctx) ins(%a, %b, %c :
      tensor<4x8xf32>, tensor<8x16xf32>, tensor<4x16xf32>)
      outs(%init : tensor<4x16xf32>) : tensor<4x16xf32>
  return %r : tensor<4x16xf32>
}

// -----

// A column vector C broadcasts along N.
// CHECK-LABEL: func.func @gemm_column_c
// CHECK: tosa.add {{.*}} : (tensor<4x16xf32>, tensor<4x1xf32>) -> tensor<4x16xf32>
// CHECK-NOT: hip.gemm
func.func @gemm_column_c(%ctx: !hip.context, %a: tensor<4x8xf32>,
                         %b: tensor<8x16xf32>, %c: tensor<4x1xf32>,
                         %init: tensor<4x16xf32>) -> tensor<4x16xf32>
    attributes {rock.kernel} {
  %r = hip.gemm(%ctx) ins(%a, %b, %c :
      tensor<4x8xf32>, tensor<8x16xf32>, tensor<4x1xf32>)
      outs(%init : tensor<4x16xf32>) : tensor<4x16xf32>
  return %r : tensor<4x16xf32>
}

// -----

// Outside a rock.kernel the pass is a no-op.
// CHECK-LABEL: func.func @gemm_not_a_kernel
// CHECK: hip.gemm
// CHECK-NOT: tosa.matmul
func.func @gemm_not_a_kernel(%ctx: !hip.context, %a: tensor<4x8xf32>,
                             %b: tensor<8x16xf32>, %init: tensor<4x16xf32>)
    -> tensor<4x16xf32> {
  %r = hip.gemm(%ctx) ins(%a, %b : tensor<4x8xf32>, tensor<8x16xf32>)
      outs(%init : tensor<4x16xf32>) : tensor<4x16xf32>
  return %r : tensor<4x16xf32>
}

// -----

func.func @gemm_dynamic(%ctx: !hip.context, %a: tensor<?x8xf32>,
                        %b: tensor<8x16xf32>, %init: tensor<?x16xf32>)
    -> tensor<?x16xf32> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.gemm'}}
  %r = hip.gemm(%ctx) ins(%a, %b : tensor<?x8xf32>, tensor<8x16xf32>)
      outs(%init : tensor<?x16xf32>) : tensor<?x16xf32>
  return %r : tensor<?x16xf32>
}

// -----

func.func @gemm_integer(%ctx: !hip.context, %a: tensor<4x8xi32>,
                        %b: tensor<8x16xi32>, %init: tensor<4x16xi32>)
    -> tensor<4x16xi32> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.gemm'}}
  %r = hip.gemm(%ctx) ins(%a, %b : tensor<4x8xi32>, tensor<8x16xi32>)
      outs(%init : tensor<4x16xi32>) : tensor<4x16xi32>
  return %r : tensor<4x16xi32>
}

// -----

// C must broadcast to [M, N]; a length that matches neither 1 nor N is a
// mismatch ONNX would have rejected at shape inference.
func.func @gemm_bad_c(%ctx: !hip.context, %a: tensor<4x8xf32>,
                      %b: tensor<8x16xf32>, %c: tensor<7xf32>,
                      %init: tensor<4x16xf32>) -> tensor<4x16xf32>
    attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.gemm'}}
  %r = hip.gemm(%ctx) ins(%a, %b, %c :
      tensor<4x8xf32>, tensor<8x16xf32>, tensor<7xf32>)
      outs(%init : tensor<4x16xf32>) : tensor<4x16xf32>
  return %r : tensor<4x16xf32>
}
