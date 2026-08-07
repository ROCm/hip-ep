// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

// CHECK-LABEL: llvm.func @readback_shape
// CHECK: %[[HOST:.*]] = llvm.alloca
// CHECK-COUNT-1: llvm.call @hipdnn_ep_readback_shape_i64
// CHECK: llvm.getelementptr %[[HOST]]
// CHECK: llvm.load
// CHECK: llvm.getelementptr %[[HOST]]
// CHECK: llvm.load
// CHECK: llvm.getelementptr %[[HOST]]
// CHECK: llvm.load
func.func @readback_shape(%ctx: !hip.context,
                          %shape: memref<3xi64, 1>)
    -> (index, index, index) {
  %d:3 = hip.readback_shape(%ctx, %shape : memref<3xi64, 1>)
      {count = 3 : i64} -> (index, index, index)
  return %d#0, %d#1, %d#2 : index, index, index
}
