// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Negative test: when onnx.Split is malformed (split_lengths don't sum
// to the axis dimension) the SplitConversion pattern refuses to match.
// In the old pipeline the surviving onnx.Split would silently propagate
// to bufferize, which would then abort with a terse "op was not
// bufferized" error. convert-onnx-to-hip now catches surviving onnx.*
// ops directly with a clear, op-named diagnostic — exercise that path
// here so the regression is caught at LIT time.

// RUN: not hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip 2>&1 | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1xf32>) -> tensor<1xf32> {
    return %arg0 : tensor<1xf32>
  }

  func.func @test_split_invalid_sum(%data: tensor<10x64xf32>)
      -> (tensor<3x64xf32>, tensor<3x64xf32>, tensor<3x64xf32>) {
    %split_lengths = "onnx.Constant"() {value = dense<[3, 3, 3]> : tensor<3xi64>}
        : () -> tensor<3xi64>
    %out0, %out1, %out2 = "onnx.Split"(%data, %split_lengths) {axis = 0 : si64}
        : (tensor<10x64xf32>, tensor<3xi64>)
        -> (tensor<3x64xf32>, tensor<3x64xf32>, tensor<3x64xf32>)
    return %out0, %out1, %out2 : tensor<3x64xf32>, tensor<3x64xf32>, tensor<3x64xf32>
  }
}

// CHECK: onnx.Split survived convert-onnx-to-hip
// CHECK-SAME: Add a conversion in lib/Conversion/OnnxToHip/
