// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Verify the ONNX MaxPool / AveragePool / LpPool conversions all produce a
// single hip.pool DPS op (distinguished only by pool_mode) with fully
// resolved attributes.  Compile-time work covered:
//   * default-fill of strides / dilations / pads when missing
//   * resolution of `auto_pad = SAME_UPPER` to explicit pads (requires
//     static spatial dims)
//   * pass-through of `ceil_mode` and `storage_order`
//   * variadic outputs: 1 (Y only) or 2 (Y + Indices i64, MAX only)
//   * per-mode attributes: count_include_pad (AVERAGE) / p (LP)

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // Test 1: classic 2D NCHW MaxPool with explicit pads.
  // (1, 3, 32, 32) -> (1, 3, 16, 16) with kernel 2, stride 2, no padding.
  func.func @test_maxpool_2d_static(%arg0: tensor<1x3x32x32xf16>)
      -> tensor<1x3x16x16xf16> {
    // CHECK-LABEL: func.func @test_maxpool_2d_static
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[ARG:.*]]: tensor<1x3x32x32xf16>)
    %y = "onnx.MaxPool"(%arg0)
        {kernel_shape = [2, 2], strides = [2, 2], pads = [0, 0, 0, 0]}
        : (tensor<1x3x32x32xf16>) -> tensor<1x3x16x16xf16>

    // CHECK-NOT: onnx.MaxPool
    // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<1x3x16x16xf16>
    // CHECK: hip.pool(%[[CTX]]) ins(%[[ARG]] : tensor<1x3x32x32xf16>)
    // CHECK-SAME: outs(%[[INIT]] : tensor<1x3x16x16xf16>)
    // CHECK-SAME: ceil_mode = 0
    // CHECK-SAME: dilations = [1, 1]
    // CHECK-SAME: kernel_shape = [2, 2]
    // CHECK-SAME: pads = [0, 0, 0, 0]
    // CHECK-SAME: pool_mode = 1
    // CHECK-SAME: storage_order = 0
    // CHECK-SAME: strides = [2, 2]

    return %y : tensor<1x3x16x16xf16>
  }

  // Test 2: 1D MaxPool, defaults for strides / dilations / pads.
  // kernel=2 -> output spatial = in - 1.
  func.func @test_maxpool_1d_default(%arg0: tensor<1x3x32xf32>)
      -> tensor<1x3x31xf32> {
    // CHECK-LABEL: func.func @test_maxpool_1d_default
    %y = "onnx.MaxPool"(%arg0) {kernel_shape = [2]}
        : (tensor<1x3x32xf32>) -> tensor<1x3x31xf32>

    // CHECK-NOT: onnx.MaxPool
    // Defaults: strides = [1], pads = [0, 0], dilations = [1].
    // CHECK: hip.pool
    // CHECK-SAME: dilations = [1]
    // CHECK-SAME: kernel_shape = [2]
    // CHECK-SAME: pads = [0, 0]
    // CHECK-SAME: pool_mode = 1
    // CHECK-SAME: strides = [1]

    return %y : tensor<1x3x31xf32>
  }

  // Test 3: 3D MaxPool (volumetric). Verifies spatial_rank=3 path through
  // attribute parsing.  No padding, kernel 2, stride 2.
  func.func @test_maxpool_3d(%arg0: tensor<1x2x4x4x4xf32>)
      -> tensor<1x2x2x2x2xf32> {
    // CHECK-LABEL: func.func @test_maxpool_3d
    %y = "onnx.MaxPool"(%arg0)
        {kernel_shape = [2, 2, 2], strides = [2, 2, 2],
         pads = [0, 0, 0, 0, 0, 0]}
        : (tensor<1x2x4x4x4xf32>) -> tensor<1x2x2x2x2xf32>

    // CHECK-NOT: onnx.MaxPool
    // CHECK: hip.pool
    // CHECK-SAME: kernel_shape = [2, 2, 2]
    // CHECK-SAME: pads = [0, 0, 0, 0, 0, 0]
    // CHECK-SAME: pool_mode = 1
    // CHECK-SAME: strides = [2, 2, 2]

    return %y : tensor<1x2x2x2x2xf32>
  }

  // Test 4: auto_pad = SAME_UPPER must be resolved to explicit pads.
  // (1, 3, 7, 7), k=3, s=2 -> out=ceil(7/2)=4. pad_total = (4-1)*2 + 3 - 7 = 2.
  // SAME_UPPER splits as begin=1, end=1 along each axis.
  func.func @test_maxpool_same_upper(%arg0: tensor<1x3x7x7xf32>)
      -> tensor<1x3x4x4xf32> {
    // CHECK-LABEL: func.func @test_maxpool_same_upper
    %y = "onnx.MaxPool"(%arg0)
        {kernel_shape = [3, 3], strides = [2, 2], auto_pad = "SAME_UPPER"}
        : (tensor<1x3x7x7xf32>) -> tensor<1x3x4x4xf32>

    // CHECK-NOT: onnx.MaxPool
    // CHECK: hip.pool
    // CHECK-SAME: pads = [1, 1, 1, 1]
    // The conversion must NOT leave the deprecated `auto_pad` attribute on
    // the resulting hip op — only explicit pads survive.
    // CHECK-NOT: auto_pad

    return %y : tensor<1x3x4x4xf32>
  }

  // Test 5: 2D MaxPool with Indices (i64) second output.
  func.func @test_maxpool_with_indices(%arg0: tensor<1x3x32x32xf32>)
      -> (tensor<1x3x16x16xf32>, tensor<1x3x16x16xi64>) {
    // CHECK-LABEL: func.func @test_maxpool_with_indices
    %y, %idx = "onnx.MaxPool"(%arg0)
        {kernel_shape = [2, 2], strides = [2, 2], pads = [0, 0, 0, 0]}
        : (tensor<1x3x32x32xf32>) -> (tensor<1x3x16x16xf32>,
                                       tensor<1x3x16x16xi64>)

    // CHECK-NOT: onnx.MaxPool
    // CHECK: %[[Y_INIT:.*]] = tensor.empty() : tensor<1x3x16x16xf32>
    // CHECK: %[[I_INIT:.*]] = tensor.empty() : tensor<1x3x16x16xi64>
    // CHECK: hip.pool
    // CHECK-SAME: outs(%[[Y_INIT]], %[[I_INIT]]
    // CHECK-SAME: tensor<1x3x16x16xf32>, tensor<1x3x16x16xi64>

    return %y, %idx : tensor<1x3x16x16xf32>, tensor<1x3x16x16xi64>
  }

  // Test 6: dynamic batch (N), static C and spatial.  Verifies the
  // tensor.dim path for dynamic N flows into tensor.empty(...).
  func.func @test_maxpool_dynamic_n(%arg0: tensor<?x3x16x16xf16>)
      -> tensor<?x3x8x8xf16> {
    // CHECK-LABEL: func.func @test_maxpool_dynamic_n
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[ARG:.*]]: tensor<?x3x16x16xf16>)
    %y = "onnx.MaxPool"(%arg0)
        {kernel_shape = [2, 2], strides = [2, 2], pads = [0, 0, 0, 0]}
        : (tensor<?x3x16x16xf16>) -> tensor<?x3x8x8xf16>

    // CHECK-NOT: onnx.MaxPool
    // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
    // CHECK: %[[DN:.*]] = tensor.dim %[[ARG]], %[[C0]] : tensor<?x3x16x16xf16>
    // CHECK: %[[INIT:.*]] = tensor.empty(%[[DN]]) : tensor<?x3x8x8xf16>
    // CHECK: hip.pool(%[[CTX]]) ins(%[[ARG]] : tensor<?x3x16x16xf16>)
    // CHECK-SAME: outs(%[[INIT]] : tensor<?x3x8x8xf16>)

    return %y : tensor<?x3x8x8xf16>
  }

  // Test 7: non-overlapping 2D AveragePool (kernel == stride, no pad).
  // Pre-lowering decomposes to expand_shape + transpose + reduce_mean
  // instead of hip.pool — the fast path for projector-style patch pooling.
  // The mean division happens in the reduce_mean kernel (no separate Mul).
  func.func @test_averagepool_2d(%arg0: tensor<1x3x32x32xf32>)
      -> tensor<1x3x16x16xf32> {
    // CHECK-LABEL: func.func @test_averagepool_2d
    %y = "onnx.AveragePool"(%arg0)
        {kernel_shape = [2, 2], strides = [2, 2], pads = [0, 0, 0, 0],
         count_include_pad = 1 : si64}
        : (tensor<1x3x32x32xf32>) -> tensor<1x3x16x16xf32>

    // CHECK-NOT: onnx.AveragePool
    // CHECK-NOT: hip.pool
    // CHECK: tensor.expand_shape
    // CHECK-SAME: output_shape [1, 3, 16, 2, 16, 2]
    // CHECK: hip.transpose
    // CHECK-SAME: perm = [0, 1, 2, 4, 3, 5]
    // CHECK: hip.reduce_mean
    // CHECK-SAME: keepdims = 0

    return %y : tensor<1x3x16x16xf32>
  }

  // Test 7b: overlapping 2D AveragePool (stride < kernel). Outside the
  // decomposition preconditions — falls back to hip.pool (pool_mode = 0).
  func.func @test_averagepool_overlap(%arg0: tensor<1x3x32x32xf32>)
      -> tensor<1x3x31x31xf32> {
    // CHECK-LABEL: func.func @test_averagepool_overlap
    %y = "onnx.AveragePool"(%arg0)
        {kernel_shape = [2, 2], strides = [1, 1], pads = [0, 0, 0, 0],
         count_include_pad = 1 : si64}
        : (tensor<1x3x32x32xf32>) -> tensor<1x3x31x31xf32>

    // CHECK-NOT: onnx.AveragePool
    // CHECK: hip.pool
    // CHECK-SAME: count_include_pad = 1
    // CHECK-SAME: kernel_shape = [2, 2]
    // CHECK-SAME: pool_mode = 0
    // CHECK-SAME: strides = [1, 1]

    return %y : tensor<1x3x31x31xf32>
  }

  // Test 7c: dynamic-spatial AveragePool. The decomposition fast path bails
  // (needs static spatial dims), so it falls back to hip.pool, which now
  // materializes the dynamic output spatial extents with arith from the
  // input dims: out = floordiv(in + pad_b + pad_e - ((k-1)*d+1), stride) + 1.
  // For kernel=stride=4, no pad, dilation=1: out = floordiv(in - 4, 4) + 1.
  func.func @test_averagepool_dynamic_spatial(%arg0: tensor<?x?x?x?xf16>)
      -> tensor<?x?x?x?xf16> {
    // CHECK-LABEL: func.func @test_averagepool_dynamic_spatial
    %y = "onnx.AveragePool"(%arg0)
        {kernel_shape = [4, 4], strides = [4, 4], pads = [0, 0, 0, 0],
         count_include_pad = 1 : si64}
        : (tensor<?x?x?x?xf16>) -> tensor<?x?x?x?xf16>

    // CHECK-NOT: onnx.AveragePool
    // N and C are passthrough dims; the two spatial dims use the arith
    // output-size formula (in - 4) floordiv 4 + 1.
    // CHECK-DAG: %[[H:.*]] = tensor.dim %{{.*}}, %{{.*}} : tensor<?x?x?x?xf16>
    // CHECK-DAG: arith.floordivsi
    // CHECK: tensor.empty
    // CHECK: hip.pool
    // CHECK-SAME: kernel_shape = [4, 4]
    // CHECK-SAME: pool_mode = 0
    // CHECK-SAME: strides = [4, 4]

    return %y : tensor<?x?x?x?xf16>
  }

  // Test 8: 2D LpPool with p = 3. Lowers to pool_mode = 2 and carries p.
  func.func @test_lppool_2d(%arg0: tensor<1x3x32x32xf32>)
      -> tensor<1x3x16x16xf32> {
    // CHECK-LABEL: func.func @test_lppool_2d
    %y = "onnx.LpPool"(%arg0)
        {kernel_shape = [2, 2], strides = [2, 2], pads = [0, 0, 0, 0],
         p = 3 : si64}
        : (tensor<1x3x32x32xf32>) -> tensor<1x3x16x16xf32>

    // CHECK-NOT: onnx.LpPool
    // CHECK: hip.pool
    // CHECK-SAME: kernel_shape = [2, 2]
    // CHECK-SAME: p = 3
    // CHECK-SAME: pool_mode = 2

    return %y : tensor<1x3x16x16xf32>
  }
}
