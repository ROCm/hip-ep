// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: split-file %s %t
// RUN: hip-mlir-opt %t/swiglu.mlir --hip-add-context-arg --convert-ggml-to-hip | FileCheck %s --check-prefix=SWIGLU
// RUN: hip-mlir-opt %t/set-rows.mlir --hip-add-context-arg --convert-ggml-to-hip | FileCheck %s --check-prefix=SET-ROWS

// SWIGLU-LABEL: func.func @swiglu(
// SWIGLU: hip.swiglu
// SWIGLU-NOT: ggml.

//--- swiglu.mlir


func.func @swiglu(%gate: tensor<1x1x4x8xf32>,
                  %up: tensor<1x1x4x8xf32>) -> tensor<1x1x4x8xf32> {
  %result = "ggml.glu"(%gate, %up) <{
    glu_kind = #ggml.glu_kind<swiglu>
  }>
      : (tensor<1x1x4x8xf32>, tensor<1x1x4x8xf32>)
        -> tensor<1x1x4x8xf32>
  return %result : tensor<1x1x4x8xf32>
}


// F32 rows rounded to the F16 cache's type, then written into it.
// SET-ROWS-LABEL: func.func @set_rows(
// SET-ROWS: hip.cast
// SET-ROWS-SAME: to = 10
// SET-ROWS: hip.scatter_nd
// SET-ROWS-NOT: ggml.

//--- set-rows.mlir
#updates = #ggml.layout<ne = [8, 2, 1, 1], nb = [4, 32, 64, 64], offset = 0>
#indices = #ggml.layout<ne = [2, 1, 1, 1], nb = [8, 16, 16, 16], offset = 0>
#state = #ggml.layout<ne = [8, 16, 1, 1], nb = [2, 16, 256, 256], offset = 0>

func.func @set_rows(
    %updates: tensor<1x1x2x8xf32, #updates> ,
    %indices: tensor<1x1x1x2xi64, #indices> ,
    %state: tensor<1x1x16x8xf16, #state>)
    -> tensor<1x1x16x8xf16, #state> {
  %result = "ggml.set_rows"(%updates, %indices, %state)  : (tensor<1x1x2x8xf32, #updates>, tensor<1x1x1x2xi64, #indices>, tensor<1x1x16x8xf16, #state>) -> tensor<1x1x16x8xf16, #state>
  return %result : tensor<1x1x16x8xf16, #state>
}
