// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX Pow with a constant scalar exponent is decomposed at compile
// time into already-supported HIP ops (hip.mul / hip.sqrt / hip.reciprocal),
// so no dedicated Pow kernel/runtime is needed. The decomposition runs in
// PRE-lowering (alongside FastGelu/ErfGelu fusion) and produces ONNX
// primitives, which then flow through their own ONNX→HIP converters.
// Running pre-lowering is required: with externalization enabled (production
// default), every onnx.Constant — including 1-element scalars — is replaced
// by bufferization.to_tensor(memref.get_global) whose value lives in the
// constants sidecar, and a post-lowering matcher could not recover it.
//
// Covered exponents (all forms):
//   2    -> single hip.mul (x*x)        — dominant RMS/LayerNorm variance case
//   3    -> two chained hip.mul (x*x*x) — tanh-Gelu chain
//   0.5  -> hip.sqrt
//   -1   -> hip.reciprocal
//   dynamic-shape Pow(x, 2) -> tensor.dim + tensor.empty(%d0, %d1) + hip.mul
//
// Covered exponent sources (pre-lowering peeks through Cast chains to either):
//   * arith.constant (this file's hand-written tests)
//   * onnx.Constant (production form, before lowerOnnxConstants externalizes)
//   * onnx.Cast / onnx.CastLike wrapping either of the above (Gemma-3 inlined
//     Gelu, SAM LayerNorm2d — ORT emits the literal as f32 and casts to the
//     activation dtype)
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s
// RUN: hip-mlir-opt %s --hip-add-context-arg \
// RUN:   --convert-onnx-to-hip="externalize-min-num-elements=1" \
// RUN:   | FileCheck %s --check-prefix=EXTERN

module {
  // Dummy entry point required by generateModuleMetadata.
  func.func @main_graph(%arg0: tensor<4x8xf16>) -> tensor<4x8xf16> {
    return %arg0 : tensor<4x8xf16>
  }

  // Test 1: Pow(x, 2) -> x*x via a single hip.mul.
  func.func @test_pow_square(%arg0: tensor<4x8xf16>) -> tensor<4x8xf16> {
    %exp = arith.constant dense<2.0> : tensor<f16>
    %0 = "onnx.Pow"(%arg0, %exp) : (tensor<4x8xf16>, tensor<f16>) -> tensor<4x8xf16>
    return %0 : tensor<4x8xf16>
  }
  // CHECK-LABEL: func.func @test_pow_square
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[ARG:.*]]: tensor<4x8xf16>) -> tensor<4x8xf16>
  // CHECK-NOT: onnx.Pow
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<4x8xf16>
  // CHECK: %[[RES:.*]] = hip.mul(%[[CTX]]) ins(%[[ARG]], %[[ARG]] : tensor<4x8xf16>, tensor<4x8xf16>) outs(%[[INIT]] : tensor<4x8xf16>) -> tensor<4x8xf16>
  // CHECK: return %[[RES]] : tensor<4x8xf16>

  // Test 2: Pow(x, 3) -> x*x*x via two chained hip.mul.
  func.func @test_pow_cube(%arg0: tensor<2x16xf32>) -> tensor<2x16xf32> {
    %exp = arith.constant dense<3.0> : tensor<f32>
    %0 = "onnx.Pow"(%arg0, %exp) : (tensor<2x16xf32>, tensor<f32>) -> tensor<2x16xf32>
    return %0 : tensor<2x16xf32>
  }
  // CHECK-LABEL: func.func @test_pow_cube
  // CHECK-NOT: onnx.Pow
  // CHECK: %[[M1:.*]] = hip.mul(%{{.*}}) ins(%[[A:.*]], %[[A]] : tensor<2x16xf32>, tensor<2x16xf32>) outs({{.*}}) -> tensor<2x16xf32>
  // CHECK: %[[M2:.*]] = hip.mul(%{{.*}}) ins(%[[M1]], %[[A]] : tensor<2x16xf32>, tensor<2x16xf32>) outs({{.*}}) -> tensor<2x16xf32>
  // CHECK: return %[[M2]] : tensor<2x16xf32>

  // Test 3: Pow(x, 0.5) -> hip.sqrt.
  func.func @test_pow_half(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %exp = arith.constant dense<5.000000e-01> : tensor<f32>
    %0 = "onnx.Pow"(%arg0, %exp) : (tensor<128xf32>, tensor<f32>) -> tensor<128xf32>
    return %0 : tensor<128xf32>
  }
  // CHECK-LABEL: func.func @test_pow_half
  // CHECK-NOT: onnx.Pow
  // CHECK: hip.sqrt(%{{.*}}) ins(%{{.*}} : tensor<128xf32>) outs({{.*}} : tensor<128xf32>) : tensor<128xf32>

  // Test 4: Pow(x, -1) -> hip.reciprocal.
  func.func @test_pow_recip(%arg0: tensor<64xf16>) -> tensor<64xf16> {
    %exp = arith.constant dense<-1.0> : tensor<f16>
    %0 = "onnx.Pow"(%arg0, %exp) : (tensor<64xf16>, tensor<f16>) -> tensor<64xf16>
    return %0 : tensor<64xf16>
  }
  // CHECK-LABEL: func.func @test_pow_recip
  // CHECK-NOT: onnx.Pow
  // CHECK: hip.reciprocal(%{{.*}}) ins(%{{.*}} : tensor<64xf16>) outs({{.*}} : tensor<64xf16>) : tensor<64xf16>

  // Test 5: Dynamic-shape Pow(x, 2) -> tensor.dim + tensor.empty(dyn) + hip.mul.
  func.func @test_pow_dynamic(%arg0: tensor<?x?xf16>) -> tensor<?x?xf16> {
    %exp = arith.constant dense<2.0> : tensor<f16>
    %0 = "onnx.Pow"(%arg0, %exp) : (tensor<?x?xf16>, tensor<f16>) -> tensor<?x?xf16>
    return %0 : tensor<?x?xf16>
  }
  // CHECK-LABEL: func.func @test_pow_dynamic
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[ARG:.*]]: tensor<?x?xf16>) -> tensor<?x?xf16>
  // CHECK-NOT: onnx.Pow
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[DIM0:.*]] = tensor.dim %[[ARG]], %[[C0]] : tensor<?x?xf16>
  // CHECK: %[[DIM1:.*]] = tensor.dim %[[ARG]], %[[C1]] : tensor<?x?xf16>
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xf16>
  // CHECK: hip.mul(%[[CTX]]) ins(%[[ARG]], %[[ARG]] : tensor<?x?xf16>, tensor<?x?xf16>) outs(%[[INIT]] : tensor<?x?xf16>) -> tensor<?x?xf16>

  // Test 6: Cast-wrapped exponent (arith.constant source). The pre-lowering
  // PowDecompose peeks through `onnx.Cast` to read the underlying constant.
  // Pow(x, 2) -> single hip.mul.
  func.func @test_pow_cast_exp(%arg0: tensor<4x8xf16>) -> tensor<4x8xf16> {
    %c = arith.constant dense<2.0> : tensor<f32>
    %cast = "onnx.Cast"(%c) {to = 10 : si64} : (tensor<f32>) -> tensor<f16>
    %0 = "onnx.Pow"(%arg0, %cast) : (tensor<4x8xf16>, tensor<f16>) -> tensor<4x8xf16>
    return %0 : tensor<4x8xf16>
  }
  // CHECK-LABEL: func.func @test_pow_cast_exp
  // CHECK-NOT: onnx.Pow
  // CHECK: hip.mul(%{{.*}}) ins(%[[A:.*]], %[[A]] : tensor<4x8xf16>, tensor<4x8xf16>) outs({{.*}}) -> tensor<4x8xf16>

  // Test 7: Production form — onnx.Constant exponent (Pow(x, 3)) wrapped in
  // onnx.Cast. Validates the actual SAM `output_upscaling.1/Pow` pattern.
  // Verified BOTH with externalization off (CHECK) AND on (EXTERN), to lock
  // in that the decomposition runs before lowerOnnxConstants moves the value
  // into the sidecar.
  func.func @test_pow_onnx_const_cast(%arg0: tensor<1x64x128x128xf16>) -> tensor<1x64x128x128xf16> {
    %c = "onnx.Constant"() {value = dense<3.0> : tensor<f32>} : () -> tensor<f32>
    %ec = "onnx.Cast"(%c) {to = 10 : si64} : (tensor<f32>) -> tensor<f16>
    %0 = "onnx.Pow"(%arg0, %ec) {onnx_node_name = "/output_upscaling.1/Pow"} : (tensor<1x64x128x128xf16>, tensor<f16>) -> tensor<1x64x128x128xf16>
    return %0 : tensor<1x64x128x128xf16>
  }
  // CHECK-LABEL:  func.func @test_pow_onnx_const_cast
  // CHECK-NOT:    onnx.Pow
  // CHECK:        %[[M1:.*]] = hip.mul(%{{.*}}) ins(%[[A:.*]], %[[A]] : tensor<1x64x128x128xf16>, tensor<1x64x128x128xf16>) outs({{.*}}) -> tensor<1x64x128x128xf16>
  // CHECK:        hip.mul(%{{.*}}) ins(%[[M1]], %[[A]] : tensor<1x64x128x128xf16>, tensor<1x64x128x128xf16>) outs({{.*}}) -> tensor<1x64x128x128xf16>
  // EXTERN-LABEL: func.func @test_pow_onnx_const_cast
  // EXTERN-NOT:   onnx.Pow
  // EXTERN:       %[[EM1:.*]] = hip.mul(%{{.*}}) ins(%[[EA:.*]], %[[EA]] : tensor<1x64x128x128xf16>, tensor<1x64x128x128xf16>) outs({{.*}}) -> tensor<1x64x128x128xf16>
  // EXTERN:       hip.mul(%{{.*}}) ins(%[[EM1]], %[[EA]] : tensor<1x64x128x128xf16>, tensor<1x64x128x128xf16>) outs({{.*}}) -> tensor<1x64x128x128xf16>

  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
