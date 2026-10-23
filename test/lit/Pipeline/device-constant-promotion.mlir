// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: split-file %s %t
// RUN: mkdir -p %t/gather %t/concat
// RUN: hip-mlir-opt --onnx-to-hip-pipeline='externalize-min-num-elements=1 externalize-output-dir=%t/gather' %t/gather.mlir | FileCheck %s --check-prefix=GATHER
// RUN: hip-mlir-opt --onnx-to-hip-pipeline='externalize-min-num-elements=1 externalize-output-dir=%t/concat' %t/concat.mlir | FileCheck %s --check-prefix=CONCAT

// A ConstantOfShape fill folds to an arith.constant, which would bufferize to a
// host memref.global. hip-promote-device-constants gives every device read of
// it a hip.constant, so after the full pipeline the fill is read through
// hip.get_constant and no host global is left behind.

// An integer fill reaching hip.gather as indices through Unsqueeze. The
// tensor.expand_shape it becomes is folded into a 1x3 constant before
// promotion runs, so the kernel reads that device constant directly.
// GATHER-NOT:   memref.global "private" constant
// GATHER-LABEL: func.func @main_graph
// GATHER:         %[[D:.*]] = hip.get_constant(%{{.*}}, %{{.*}}) : memref<1x3xi64>
// GATHER-NOT:     expand_shape
// GATHER:         hip.gather(%{{.*}}) ins(%{{.*}}, %[[D]] :

// A zero fill concatenated after an input: Concat lowers to
// tensor.insert_slice, which bufferizes to a device-to-device memcpy, so the
// copy's source is the device constant.
// CONCAT-NOT:   memref.global "private" constant
// CONCAT-LABEL: func.func @main_graph
// CONCAT:         %[[D:.*]] = hip.get_constant(%{{.*}}, %{{.*}}) : memref<3x4xf32>
// CONCAT:         memref.copy %[[D]], %{{.*}} : memref<3x4xf32> to

//--- gather.mlir
module {
  func.func @main_graph(%table: tensor<8x4xf32> {onnx.name = "table"})
      -> tensor<1x3x4xf32> {
    %shape = "onnx.Constant"() {value = dense<[3]> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %fill = "onnx.ConstantOfShape"(%shape) {value = dense<1> : tensor<1xi64>}
        : (tensor<1xi64>) -> tensor<3xi64>
    %axes = "onnx.Constant"() {value = dense<[0]> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %indices = "onnx.Unsqueeze"(%fill, %axes)
        : (tensor<3xi64>, tensor<1xi64>) -> tensor<1x3xi64>
    %out = "onnx.Gather"(%table, %indices) {axis = 0 : si64}
        : (tensor<8x4xf32>, tensor<1x3xi64>) -> tensor<1x3x4xf32>
    return %out : tensor<1x3x4xf32>
  }
}

//--- concat.mlir
module {
  func.func @main_graph(%x: tensor<2x4xf32> {onnx.name = "x"})
      -> tensor<5x4xf32> {
    %shape = "onnx.Constant"() {value = dense<[3, 4]> : tensor<2xi64>}
        : () -> tensor<2xi64>
    %zeros = "onnx.ConstantOfShape"(%shape) {value = dense<0.0> : tensor<1xf32>}
        : (tensor<2xi64>) -> tensor<3x4xf32>
    %out = "onnx.Concat"(%x, %zeros) {axis = 0 : si64}
        : (tensor<2x4xf32>, tensor<3x4xf32>) -> tensor<5x4xf32>
    return %out : tensor<5x4xf32>
  }
}
