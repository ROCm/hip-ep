// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Pipeline test: onnx.Mul with two runtime f32 inputs through onnx-to-hip.
//
// Verifies that a simple element-wise Mul is converted end-to-end:
//   onnx.Mul -> hip.miopen.mul, bufferized, allocs pooled + lowered.
//===----------------------------------------------------------------------===//

// RUN: %hip-mlir-opt --onnx-to-hip-pipeline %s | %FileCheck %s

// CHECK-LABEL: func.func @mul

// onnx ops must be gone.
// CHECK-NOT: onnx.Mul
// CHECK-NOT: onnx.Return

// Handle lifecycle inserted.
// CHECK: hip.create_handle

// Mul lowered to HIP dialect.
// CHECK: hip.miopen.mul

// Allocs lowered to hip.alloc (no memref.alloc left).
// CHECK-NOT: memref.alloc

// Handle destroyed before return.
// CHECK: hip.destroy_handle
// CHECK: return

module {
  func.func @mul(%A: tensor<8xf32>, %B: tensor<8xf32>) -> tensor<8xf32> {
    %C = "onnx.Mul"(%A, %B) : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    "onnx.Return"(%C) : (tensor<8xf32>) -> ()
  }
}
