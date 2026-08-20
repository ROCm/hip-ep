// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" %s | FileCheck %s

// CHECK-LABEL: func.func @bufferize
// CHECK-SAME: %[[I32:[^,]+]]: memref<2xi32>
// CHECK-SAME: %[[I64:[^)]+]]: memref<i64>
// CHECK: %[[VALID:.*]], %[[VALUES:.*]]:3 = hip.readback_control
// CHECK-SAME: %[[I32]], %[[I64]] : memref<2xi32>, memref<i64>
// CHECK-SAME: -> (i1, i64, i64, i64)
func.func @bufferize(%ctx: !hip.context, %i32s: tensor<2xi32>,
                     %i64scalar: tensor<i64>) -> i1 {
  %r:4 = hip.readback_control(
      %ctx, %i32s, %i64scalar : tensor<2xi32>, tensor<i64>)
      -> (i1, i64, i64, i64)
  return %r#0 : i1
}
