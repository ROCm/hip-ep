// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Verify the unified ONNX global-pool lowering: GlobalAveragePool /
// GlobalMaxPool / GlobalLpPool all lower into the same hip.global_pool DPS
// op, distinguished only by the `mode` attribute (and `p` for LP). The
// reduction itself runs in the runtime custom HIP kernel, not via a
// decomposition.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // Test 1: classic 4D NCHW image, AVERAGE. (1, 3, 5, 5) -> (1, 3, 1, 1).
  func.func @test_gap_nchw_f32(%arg0: tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32> {
    // CHECK-LABEL: func.func @test_gap_nchw_f32
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[ARG:.*]]: tensor<1x3x5x5xf32>)
    %r = "onnx.GlobalAveragePool"(%arg0)
        : (tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32>

    // CHECK-NOT: onnx.GlobalAveragePool
    // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<1x3x1x1xf32>
    // CHECK: hip.global_pool(%[[CTX]]) ins(%[[ARG]] : tensor<1x3x5x5xf32>)
    // CHECK-SAME: outs(%[[INIT]] : tensor<1x3x1x1xf32>)
    // CHECK-SAME: {mode = 0 : i64}

    return %r : tensor<1x3x1x1xf32>
  }

  // Test 2: 3D (1D-spatial) f16, AVERAGE. (2, 4, 8) -> (2, 4, 1).
  func.func @test_gap_3d_f16(%arg0: tensor<2x4x8xf16>) -> tensor<2x4x1xf16> {
    // CHECK-LABEL: func.func @test_gap_3d_f16
    %r = "onnx.GlobalAveragePool"(%arg0)
        : (tensor<2x4x8xf16>) -> tensor<2x4x1xf16>

    // CHECK-NOT: onnx.GlobalAveragePool
    // CHECK: tensor.empty() : tensor<2x4x1xf16>
    // CHECK: hip.global_pool
    // CHECK-SAME: {mode = 0 : i64}

    return %r : tensor<2x4x1xf16>
  }

  // Test 3: 5D (3D-spatial volumetric), AVERAGE.
  // (1, 2, 2, 3, 4) -> (1, 2, 1, 1, 1).
  func.func @test_gap_5d_f32(%arg0: tensor<1x2x2x3x4xf32>) -> tensor<1x2x1x1x1xf32> {
    // CHECK-LABEL: func.func @test_gap_5d_f32
    %r = "onnx.GlobalAveragePool"(%arg0)
        : (tensor<1x2x2x3x4xf32>) -> tensor<1x2x1x1x1xf32>

    // CHECK-NOT: onnx.GlobalAveragePool
    // CHECK: tensor.empty() : tensor<1x2x1x1x1xf32>
    // CHECK: hip.global_pool
    // CHECK-SAME: {mode = 0 : i64}

    return %r : tensor<1x2x1x1x1xf32>
  }

  // Test 4: dynamic batch + dynamic channel for AVERAGE. Spatial dims stay
  // static (5x5). The init tensor must carry the runtime N and C via
  // tensor.dim — verified explicitly so a regression that hardcodes a static
  // shape would fail here.
  func.func @test_gap_dynamic_nc(%arg0: tensor<?x?x5x5xf16>) -> tensor<?x?x1x1xf16> {
    // CHECK-LABEL: func.func @test_gap_dynamic_nc
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[ARG:.*]]: tensor<?x?x5x5xf16>)
    %r = "onnx.GlobalAveragePool"(%arg0)
        : (tensor<?x?x5x5xf16>) -> tensor<?x?x1x1xf16>

    // CHECK-NOT: onnx.GlobalAveragePool
    // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
    // CHECK: %[[DN:.*]] = tensor.dim %[[ARG]], %[[C0]] : tensor<?x?x5x5xf16>
    // CHECK: %[[DC:.*]] = tensor.dim %[[ARG]], %[[C1]] : tensor<?x?x5x5xf16>
    // CHECK: %[[INIT:.*]] = tensor.empty(%[[DN]], %[[DC]]) : tensor<?x?x1x1xf16>
    // CHECK: hip.global_pool(%[[CTX]]) ins(%[[ARG]] : tensor<?x?x5x5xf16>)
    // CHECK-SAME: outs(%[[INIT]] : tensor<?x?x1x1xf16>)
    // CHECK-SAME: {mode = 0 : i64}

    return %r : tensor<?x?x1x1xf16>
  }

  // Test 5: GlobalMaxPool 4D f32. mode == 1, p stays at the default 2.
  func.func @test_gmp_nchw_f32(%arg0: tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32> {
    // CHECK-LABEL: func.func @test_gmp_nchw_f32
    %r = "onnx.GlobalMaxPool"(%arg0)
        : (tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32>

    // CHECK-NOT: onnx.GlobalMaxPool
    // CHECK: tensor.empty() : tensor<1x3x1x1xf32>
    // CHECK: hip.global_pool
    // CHECK-SAME: {mode = 1 : i64}

    return %r : tensor<1x3x1x1xf32>
  }

  // Test 6: GlobalMaxPool dynamic NC, 3D f16. mode == 1.
  func.func @test_gmp_dynamic_3d_f16(%arg0: tensor<?x?x?xf16>) -> tensor<?x?x1xf16> {
    // CHECK-LABEL: func.func @test_gmp_dynamic_3d_f16
    %r = "onnx.GlobalMaxPool"(%arg0)
        : (tensor<?x?x?xf16>) -> tensor<?x?x1xf16>

    // CHECK-NOT: onnx.GlobalMaxPool
    // CHECK: tensor.empty(%{{.*}}, %{{.*}}) : tensor<?x?x1xf16>
    // CHECK: hip.global_pool
    // CHECK-SAME: {mode = 1 : i64}

    return %r : tensor<?x?x1xf16>
  }

  // Test 7: GlobalLpPool with default p (no `p` attribute). mode == 2, p == 2.
  func.func @test_glp_default_p(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x3x1x1xf32> {
    // CHECK-LABEL: func.func @test_glp_default_p
    %r = "onnx.GlobalLpPool"(%arg0)
        : (tensor<1x3x4x4xf32>) -> tensor<1x3x1x1xf32>

    // CHECK-NOT: onnx.GlobalLpPool
    // CHECK: tensor.empty() : tensor<1x3x1x1xf32>
    // CHECK: hip.global_pool
    // CHECK-SAME: {mode = 2 : i64}

    return %r : tensor<1x3x1x1xf32>
  }

  // Test 8: GlobalLpPool with explicit p=3, dynamic batch. mode == 2, p == 3.
  func.func @test_glp_p3_dynamic(%arg0: tensor<?x4x6x6xf32>) -> tensor<?x4x1x1xf32> {
    // CHECK-LABEL: func.func @test_glp_p3_dynamic
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[ARG:.*]]: tensor<?x4x6x6xf32>)
    %r = "onnx.GlobalLpPool"(%arg0) {p = 3 : i64}
        : (tensor<?x4x6x6xf32>) -> tensor<?x4x1x1xf32>

    // CHECK-NOT: onnx.GlobalLpPool
    // CHECK: %[[INIT:.*]] = tensor.empty(%{{.*}}) : tensor<?x4x1x1xf32>
    // CHECK: hip.global_pool(%[[CTX]]) ins(%[[ARG]] : tensor<?x4x6x6xf32>)
    // CHECK-SAME: outs(%[[INIT]] : tensor<?x4x1x1xf32>)
    // CHECK-SAME: {mode = 2 : i64, p = 3 : i64}

    return %r : tensor<?x4x1x1xf32>
  }

  // Even if imported spatial result dimensions remain dynamic, GlobalPool's
  // semantic destination extents are 1 rather than positional input extents.
  func.func @test_gap_dynamic_result_spatial(
      %arg0: tensor<?x?x?x?xf16>) -> tensor<?x?x?x?xf16> {
    %r = "onnx.GlobalAveragePool"(%arg0)
        : (tensor<?x?x?x?xf16>) -> tensor<?x?x?x?xf16>

    // CHECK-LABEL: func.func @test_gap_dynamic_result_spatial
    // CHECK-SAME: %[[ARG:.*]]: tensor<?x?x?x?xf16>
    // CHECK: %[[ONE:.*]] = arith.constant 1 : index
    // CHECK: tensor.empty(%{{.*}}, %{{.*}}, %[[ONE]], %[[ONE]]) : tensor<?x?x?x?xf16>
    // CHECK: hip.global_pool

    return %r : tensor<?x?x?x?xf16>
  }
}
