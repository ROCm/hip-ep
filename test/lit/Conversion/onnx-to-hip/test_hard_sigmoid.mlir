// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// onnx.HardSigmoid has no HIP op of its own: it decomposes to
// Clip(alpha*x + beta, 0, 1), and Clip in turn lowers to hip.max then hip.min.
// So a converted HardSigmoid is observable as mul -> add -> max -> min, with
// no onnx.* op left behind.
//
// The decomposition is emitted from the pre-lowering loop rather than
// convertComputeOps; see HardSigmoidConversion.cpp for why that distinction
// decides whether the emitted primitives are ever converted.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // MobileNetV3's h-swish gate: alpha = 1/6, beta left at its schema default
  // of 0.5. Covers the default-application path, which is the form the model
  // actually exports.
  func.func @test_hard_sigmoid_default_beta(%input: tensor<1x64x14x14xf32>)
      -> tensor<1x64x14x14xf32> {
    // CHECK-LABEL: func.func @test_hard_sigmoid_default_beta
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x64x14x14xf32>)
    %y = "onnx.HardSigmoid"(%input) {alpha = 0.166666672 : f32}
        : (tensor<1x64x14x14xf32>) -> tensor<1x64x14x14xf32>

    // CHECK-NOT: onnx.HardSigmoid
    // CHECK-NOT: onnx.Clip
    // CHECK-NOT: onnx.Mul
    // CHECK-NOT: onnx.Add
    // alpha*x, then + beta, then the clamp to [0, 1] that Clip becomes. The
    // constants are asserted in emission order, so reading `alpha` where
    // `beta` was meant -- or dropping the attribute for the default -- fails
    // here rather than passing on the operation sequence alone.
    // CHECK: hip.constant {{.*}}value = dense<0.166666672> : tensor<f32>
    // CHECK: hip.mul
    // CHECK: hip.constant {{.*}}value = dense<5.000000e-01> : tensor<f32>
    // CHECK: hip.add
    // CHECK: hip.max
    // CHECK: hip.min

    return %y : tensor<1x64x14x14xf32>
  }

  // Both attributes present and non-default, and deliberately unequal, so an
  // alpha/beta swap is observable in the asserted constants.
  func.func @test_hard_sigmoid_explicit_attrs(%input: tensor<2x8xf32>)
      -> tensor<2x8xf32> {
    // CHECK-LABEL: func.func @test_hard_sigmoid_explicit_attrs
    %y = "onnx.HardSigmoid"(%input) {alpha = 0.25 : f32, beta = 0.75 : f32}
        : (tensor<2x8xf32>) -> tensor<2x8xf32>

    // CHECK-NOT: onnx.HardSigmoid
    // CHECK: hip.constant {{.*}}value = dense<2.500000e-01> : tensor<f32>
    // CHECK: hip.mul
    // CHECK: hip.constant {{.*}}value = dense<7.500000e-01> : tensor<f32>
    // CHECK: hip.add
    // CHECK: hip.max
    // CHECK: hip.min

    return %y : tensor<2x8xf32>
  }

  // Neither attribute present: ONNX defaults alpha to 0.2 as well as beta to
  // 0.5, so this must still convert rather than fall through unmatched.
  func.func @test_hard_sigmoid_all_defaults(%input: tensor<3x5xf32>)
      -> tensor<3x5xf32> {
    // CHECK-LABEL: func.func @test_hard_sigmoid_all_defaults
    %y = "onnx.HardSigmoid"(%input)
        : (tensor<3x5xf32>) -> tensor<3x5xf32>

    // CHECK-NOT: onnx.HardSigmoid
    // CHECK: hip.constant {{.*}}value = dense<2.000000e-01> : tensor<f32>
    // CHECK: hip.mul
    // CHECK: hip.constant {{.*}}value = dense<5.000000e-01> : tensor<f32>
    // CHECK: hip.add
    // CHECK: hip.max
    // CHECK: hip.min

    return %y : tensor<3x5xf32>
  }

  // f16, since the scalar constants are built by converting a double into the
  // result's own float semantics. The asserted alpha is the f32 attribute
  // narrowed to half, so a conversion that kept the wrong semantics is caught.
  func.func @test_hard_sigmoid_f16(%input: tensor<1x16xf16>) -> tensor<1x16xf16> {
    // CHECK-LABEL: func.func @test_hard_sigmoid_f16
    %y = "onnx.HardSigmoid"(%input) {alpha = 0.166666672 : f32}
        : (tensor<1x16xf16>) -> tensor<1x16xf16>

    // CHECK-NOT: onnx.HardSigmoid
    // CHECK: hip.constant {{.*}}value = dense<1.666260e-01> : tensor<f16>
    // CHECK: hip.mul
    // CHECK: hip.constant {{.*}}value = dense<5.000000e-01> : tensor<f16>
    // CHECK: hip.add
    // CHECK: hip.max
    // CHECK: hip.min

    return %y : tensor<1x16xf16>
  }

  // f64 with both attributes defaulted. ONNX types alpha/beta as float, so the
  // default must be the binary32 value widened to f64, not the binary64 0.2 --
  // the two differ from the 9th significant digit, and only f64 can observe it.
  func.func @test_hard_sigmoid_f64_defaults(%input: tensor<2x4xf64>)
      -> tensor<2x4xf64> {
    // CHECK-LABEL: func.func @test_hard_sigmoid_f64_defaults
    %y = "onnx.HardSigmoid"(%input)
        : (tensor<2x4xf64>) -> tensor<2x4xf64>

    // CHECK-NOT: onnx.HardSigmoid
    // CHECK: hip.constant {{.*}}value = dense<0.20000000298023224> : tensor<f64>
    // CHECK: hip.mul
    // CHECK: hip.constant {{.*}}value = dense<5.000000e-01> : tensor<f64>
    // CHECK: hip.add
    // CHECK: hip.max
    // CHECK: hip.min

    return %y : tensor<2x4xf64>
  }
}
