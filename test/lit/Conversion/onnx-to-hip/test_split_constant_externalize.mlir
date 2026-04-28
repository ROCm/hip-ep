// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// Split sizes must stay as an inline dense constant through ConvertOnnxToHip:
// externalizing them to constants.bin removes the dense `value` from IR and
// Split lowering cannot run (onnx.Split survives -> bufferization error).
// With the split-size exception, externalize-min-num-elements=1 still lowers.

// RUN: mkdir -p %t.extsplit
// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip="externalize-min-num-elements=1 externalize-output-dir=%t.extsplit" | FileCheck %s

// CHECK-NOT: onnx.Split
// CHECK: hip.split
// CHECK: hip.split
// CHECK: hip.split

module {
  func.func @main_graph(%arg0: tensor<1x1x8192xf16>) -> (tensor<1x1x2048xf16>, tensor<1x1x2048xf16>, tensor<1x1x4096xf16>) {
    %split_lengths = "onnx.Constant"() {value = dense<[2048, 2048, 4096]> : tensor<3xi64>} : () -> tensor<3xi64>
    %o0, %o1, %o2 = "onnx.Split"(%arg0, %split_lengths) {axis = -1 : si64} : (tensor<1x1x8192xf16>, tensor<3xi64>) -> (tensor<1x1x2048xf16>, tensor<1x1x2048xf16>, tensor<1x1x4096xf16>)
    return %o0, %o1, %o2 : tensor<1x1x2048xf16>, tensor<1x1x2048xf16>, tensor<1x1x4096xf16>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
