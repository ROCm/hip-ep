// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// hip-promote-device-constants must run after every pass that folds or reads
// the arith form of a constant: once a constant is a hip.constant, upstream
// folds and HIP's readback fold can no longer see its value.
//
// RUN: split-file %s %t
// RUN: hip-mlir-opt %t/graph.mlir --onnx-to-hip-pipeline --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=ORDER
// RUN: mkdir -p %t/out
// RUN: hip-mlir-opt %t/slice.mlir \
// RUN:   --pass-pipeline='builtin.module(hip-fusion-transform,func.func(hip-promote-device-constants),hip-externalize-constants{externalize-min-num-elements=1 externalize-output-dir=%t/out},canonicalize)' \
// RUN:   | FileCheck %s

// ORDER:      hip-infer-shapes
// ORDER-NEXT: hip-fusion-transform,
// ORDER-NEXT: func.func(
// ORDER-NEXT:   hip-promote-device-constants
// ORDER-NEXT: ),
// ORDER-NEXT: hip-externalize-constants
// ORDER:      canonicalize

// The pipeline slice above, from fusion to the first canonicalize, on IR
// shaped like a shape value read back on the host: a static tensor.dim packed
// into a rank-0 tensor, cast to i32 on the GPU and read back. Fusion's driver
// folds the dimension into an arith.constant; the readback fold must then
// still see it, leaving a host constant and no GPU cast, sync or D2H copy.
// CHECK-LABEL: func.func @static_dimension_readback
// CHECK-NOT:     hip.cast
// CHECK-NOT:     hip.readback_scalar
// CHECK-NOT:     hip.get_constant
// CHECK:         %[[C:.*]] = arith.constant 7 : i32
// CHECK:         return %[[C]] : i32

//--- graph.mlir
module {
  func.func @main_graph(%x: tensor<4xf32> {onnx.name = "x"}) -> tensor<4xf32> {
    return %x : tensor<4xf32>
  }
}

//--- slice.mlir
func.func @static_dimension_readback(%ctx: !hip.context, %data: tensor<4x7xi64>)
    -> i32 {
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %data, %c1 : tensor<4x7xi64>
  %wide = arith.index_cast %dim : index to i64
  %packed = tensor.from_elements %wide : tensor<i64>
  %init = tensor.empty() : tensor<i32>
  %gpu = hip.cast(%ctx) ins(%packed : tensor<i64>) outs(%init : tensor<i32>)
      {to = 6 : i64} : tensor<i32>
  %host = hip.readback_scalar(%ctx, %gpu : tensor<i32>) -> i32
  return %host : i32
}
