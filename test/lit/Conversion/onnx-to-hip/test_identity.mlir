// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Verify that onnx.Identity is eliminated at the convert-onnx-to-hip stage
// by forwarding its input SSA value directly to all users (equivalent to a
// full-range memref.subview view, but cheaper — no view op materialised in
// the IR).  No HIP dialect op is produced and no runtime support is
// required.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // Test 1: static-shape Identity — the op is removed and the input is
  // returned directly.
  func.func @test_identity_static(%input: tensor<2x3x4xf32>)
      -> tensor<2x3x4xf32> {
    // CHECK-LABEL: func.func @test_identity_static
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<2x3x4xf32>) -> tensor<2x3x4xf32>

    %r = "onnx.Identity"(%input) : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>

    // CHECK-NOT: onnx.Identity
    // CHECK: return %[[IN]] : tensor<2x3x4xf32>

    return %r : tensor<2x3x4xf32>
  }

  // Test 2: integer element type — same forwarding behaviour, independent
  // of dtype.
  func.func @test_identity_i64(%input: tensor<3x5xi64>) -> tensor<3x5xi64> {
    // CHECK-LABEL: func.func @test_identity_i64
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<3x5xi64>) -> tensor<3x5xi64>

    %r = "onnx.Identity"(%input) : (tensor<3x5xi64>) -> tensor<3x5xi64>

    // CHECK-NOT: onnx.Identity
    // CHECK: return %[[IN]] : tensor<3x5xi64>

    return %r : tensor<3x5xi64>
  }

  // Test 3: chained Identity — every link folds out, so the final return
  // hands back the original input.
  func.func @test_identity_chain(%input: tensor<8xf16>) -> tensor<8xf16> {
    // CHECK-LABEL: func.func @test_identity_chain
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<8xf16>) -> tensor<8xf16>

    %a = "onnx.Identity"(%input) : (tensor<8xf16>) -> tensor<8xf16>
    %b = "onnx.Identity"(%a) : (tensor<8xf16>) -> tensor<8xf16>
    %c = "onnx.Identity"(%b) : (tensor<8xf16>) -> tensor<8xf16>

    // CHECK-NOT: onnx.Identity
    // CHECK: return %[[IN]] : tensor<8xf16>

    return %c : tensor<8xf16>
  }

  // Test 4: dynamic shape — works identically because the rewrite is type-
  // agnostic and never inspects shape information.
  func.func @test_identity_dynamic(%input: tensor<?x?xf16>) -> tensor<?x?xf16> {
    // CHECK-LABEL: func.func @test_identity_dynamic
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<?x?xf16>) -> tensor<?x?xf16>

    %r = "onnx.Identity"(%input) : (tensor<?x?xf16>) -> tensor<?x?xf16>

    // CHECK-NOT: onnx.Identity
    // CHECK: return %[[IN]] : tensor<?x?xf16>

    return %r : tensor<?x?xf16>
  }
}
