// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip --split-input-file %s 2>&1 | FileCheck %s

module {
  func.func @main_graph(%x: tensor<?x3x16x16xf16>,
                        %scales: tensor<4xf32>)
      -> tensor<2x3x32x32xf16> {
    %none = "onnx.NoValue"() {value} : () -> none
    // CHECK: error: resize output dimension 0 must remain dynamic because input N/C is dynamic
    %y = "onnx.Resize"(%x, %none, %scales)
        {mode = "linear", coordinate_transformation_mode = "half_pixel"}
        : (tensor<?x3x16x16xf16>, none, tensor<4xf32>)
        -> tensor<2x3x32x32xf16>
    return %y : tensor<2x3x32x32xf16>
  }
}

// -----

module {
  func.func @main_graph(%x: tensor<1x3x16x16xf16>,
                        %sizes: tensor<4xi64>)
      -> tensor<1x3x?x32xf16> {
    %none0 = "onnx.NoValue"() {value} : () -> none
    %none1 = "onnx.NoValue"() {value} : () -> none
    // CHECK: error: resize output spatial dimension 2 must be static because sizes/scales are not carried by hip.resize
    %y = "onnx.Resize"(%x, %none0, %none1, %sizes)
        {mode = "nearest"}
        : (tensor<1x3x16x16xf16>, none, none, tensor<4xi64>)
        -> tensor<1x3x?x32xf16>
    return %y : tensor<1x3x?x32xf16>
  }
}

// CHECK-NOT: tensor.empty
// CHECK-NOT: hip.resize
