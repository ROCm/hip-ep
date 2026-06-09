// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// Expand + Shape -> from_elements: resolve dynamic empty() sizes via
// tensor.dim traceback (embedding-style), not tensor.extract on the shape
// vector.  Avoids bufferized memref.load from a shared shape buffer that
// breaks hip-pool-allocs dominance when multiple Expands share one Shape.
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<?x?x4096xi8>) -> tensor<?x?x4096xi8> {
    return %arg0 : tensor<?x?x4096xi8>
  }

  // Embedding-style: Shape(data) -> from_elements; two Expands share %shape.
  func.func @expand_shape_trace_shared(
      %mask: tensor<1x1xi8>, %data: tensor<?x?x4096xi8>)
      -> (tensor<?x?x4096xi8>, tensor<?x?x4096xi8>) {
    %shape = "onnx.Shape"(%data) : (tensor<?x?x4096xi8>) -> tensor<3xi64>
    %e0 = "onnx.Expand"(%mask, %shape)
        : (tensor<1x1xi8>, tensor<3xi64>) -> tensor<?x?x4096xi8>
    %e1 = "onnx.Expand"(%mask, %shape)
        : (tensor<1x1xi8>, tensor<3xi64>) -> tensor<?x?x4096xi8>
    return %e0, %e1 : tensor<?x?x4096xi8>, tensor<?x?x4096xi8>
  }

  // CHECK-LABEL: func.func @expand_shape_trace_shared
  // CHECK-NOT: tensor.extract
  // Two Expands: each empty() sizes dynamic dims via tensor.dim on %data.
  // CHECK-COUNT-4: tensor.dim %{{.*}} : tensor<?x?x4096xi8>
  // CHECK-COUNT-2: hip.expand

  // Opaque shape function argument: legacy tensor.extract path still allowed.
  func.func @expand_opaque_shape(%mask: tensor<1x1xi8>, %shape: tensor<3xi64>)
      -> tensor<?x?x4096xi8> {
    %e = "onnx.Expand"(%mask, %shape)
        : (tensor<1x1xi8>, tensor<3xi64>) -> tensor<?x?x4096xi8>
    return %e : tensor<?x?x4096xi8>
  }

  // CHECK-LABEL: func.func @expand_opaque_shape
  // CHECK: tensor.extract
  // CHECK: hip.expand
}
