// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --onnx-to-hip-pipeline \
// RUN:     --mlir-print-ir-before=hip-externalize-constants 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=hip.batch_norm \
// RUN:       --implicit-check-not=hip.sub --implicit-check-not=hip.mul \
// RUN:       --implicit-check-not=hip.sqrt --implicit-check-not=hip.reciprocal

// A convolution followed by a batch normalization written as elementwise ops.
// hip-fusion-transform turns the chain into hip.batch_norm, and the
// hip-fold-batchnorm run after it folds that into the convolution, before
// hip-externalize-constants makes the parameters unreadable.
//
// scale = [1, 2], B = [0.5, 1], mean = [1, 0], var = [3, 0], epsilon = 1, so
// s = scale / sqrt(var + epsilon) = [0.5, 2] and t = B - mean * s = [0, 1],
// which becomes the convolution's bias.

// CHECK-LABEL: IR Dump Before ExternalizeConstantsPass
// CHECK:         %[[BIAS:.*]] = hip.constant {{.*}}value = dense<[0.000000e+00, 1.000000e+00]> : tensor<2xf32>
// CHECK:         hip.conv(%{{.*}}) ins(%{{.*}}, %{{.*}}, %[[BIAS]] : tensor<1x2x4x4xf32>, tensor<2x2x3x3xf32>, tensor<2xf32>)
module {
  func.func @main_graph(%x: tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32> {
    %weights = "onnx.Constant"() {value = dense<1.0> : tensor<2x2x3x3xf32>}
        : () -> tensor<2x2x3x3xf32>
    %scale = "onnx.Constant"() {value = dense<[1.0, 2.0]> : tensor<2xf32>}
        : () -> tensor<2xf32>
    %bias = "onnx.Constant"() {value = dense<[0.5, 1.0]> : tensor<2xf32>}
        : () -> tensor<2xf32>
    %mean = "onnx.Constant"() {value = dense<[1.0, 0.0]> : tensor<2xf32>}
        : () -> tensor<2xf32>
    %variance = "onnx.Constant"() {value = dense<[3.0, 0.0]> : tensor<2xf32>}
        : () -> tensor<2xf32>
    %epsilon = "onnx.Constant"() {value = dense<1.0> : tensor<f32>}
        : () -> tensor<f32>
    %axes = "onnx.Constant"() {value = dense<[1, 2]> : tensor<2xi64>}
        : () -> tensor<2xi64>

    %conv = "onnx.Conv"(%x, %weights) {auto_pad = "NOTSET", group = 1 : si64,
        kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]}
        : (tensor<1x2x4x4xf32>, tensor<2x2x3x3xf32>) -> tensor<1x2x4x4xf32>

    %variance_plus_epsilon = "onnx.Add"(%variance, %epsilon)
        : (tensor<2xf32>, tensor<f32>) -> tensor<2xf32>
    %standard_deviation = "onnx.Sqrt"(%variance_plus_epsilon)
        : (tensor<2xf32>) -> tensor<2xf32>
    %inverse = "onnx.Reciprocal"(%standard_deviation)
        : (tensor<2xf32>) -> tensor<2xf32>

    %mean_view = "onnx.Unsqueeze"(%mean, %axes)
        : (tensor<2xf32>, tensor<2xi64>) -> tensor<2x1x1xf32>
    %inverse_view = "onnx.Unsqueeze"(%inverse, %axes)
        : (tensor<2xf32>, tensor<2xi64>) -> tensor<2x1x1xf32>
    %scale_view = "onnx.Unsqueeze"(%scale, %axes)
        : (tensor<2xf32>, tensor<2xi64>) -> tensor<2x1x1xf32>
    %bias_view = "onnx.Unsqueeze"(%bias, %axes)
        : (tensor<2xf32>, tensor<2xi64>) -> tensor<2x1x1xf32>

    %centered = "onnx.Sub"(%conv, %mean_view)
        : (tensor<1x2x4x4xf32>, tensor<2x1x1xf32>) -> tensor<1x2x4x4xf32>
    %normalized = "onnx.Mul"(%centered, %inverse_view)
        : (tensor<1x2x4x4xf32>, tensor<2x1x1xf32>) -> tensor<1x2x4x4xf32>
    %scaled = "onnx.Mul"(%normalized, %scale_view)
        : (tensor<1x2x4x4xf32>, tensor<2x1x1xf32>) -> tensor<1x2x4x4xf32>
    %shifted = "onnx.Add"(%scaled, %bias_view)
        : (tensor<1x2x4x4xf32>, tensor<2x1x1xf32>) -> tensor<1x2x4x4xf32>
    return %shifted : tensor<1x2x4x4xf32>
  }
}
