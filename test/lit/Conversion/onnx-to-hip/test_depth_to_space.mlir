// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// onnx.DepthToSpace decomposes to tensor.expand_shape + hip.transpose +
// tensor.collapse_shape. blocksize == 1 is the identity in both modes.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x4x2x2xf32>) -> tensor<1x4x2x2xf32> {
    return %arg0 : tensor<1x4x2x2xf32>
  }

  // DCR: [1, 18, 3, 5] -> expand [1, 3, 3, 2, 3, 5]
  //      -> transpose perm [0, 3, 4, 1, 5, 2] -> [1, 2, 3, 3, 5, 3]
  //      -> collapse [1, 2, 9, 15].
  func.func @test_depth_to_space_dcr(%arg0: tensor<1x18x3x5xf32>) -> tensor<1x2x9x15xf32> {
    // CHECK-LABEL: func.func @test_depth_to_space_dcr
    %r = "onnx.DepthToSpace"(%arg0) {blocksize = 3 : si64, mode = "DCR"}
        : (tensor<1x18x3x5xf32>) -> tensor<1x2x9x15xf32>

    // CHECK-NOT: onnx.DepthToSpace
    // CHECK: tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2, 3], [4], [5]] output_shape [1, 3, 3, 2, 3, 5]
    // CHECK-SAME: tensor<1x18x3x5xf32> into tensor<1x3x3x2x3x5xf32>
    // CHECK: tensor.empty() : tensor<1x2x3x3x5x3xf32>
    // CHECK: hip.transpose
    // CHECK-SAME: perm = [0, 3, 4, 1, 5, 2]
    // CHECK: tensor.collapse_shape %{{.*}} {{\[\[}}0], [1], [2, 3], [4, 5]]
    // CHECK-SAME: tensor<1x2x3x3x5x3xf32> into tensor<1x2x9x15xf32>

    return %r : tensor<1x2x9x15xf32>
  }

  // Omitted mode is DCR.
  func.func @test_depth_to_space_default_mode(%arg0: tensor<1x18x3x5xf16>) -> tensor<1x2x9x15xf16> {
    // CHECK-LABEL: func.func @test_depth_to_space_default_mode
    %r = "onnx.DepthToSpace"(%arg0) {blocksize = 3 : si64}
        : (tensor<1x18x3x5xf16>) -> tensor<1x2x9x15xf16>

    // CHECK-NOT: onnx.DepthToSpace
    // CHECK: tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2, 3], [4], [5]] output_shape [1, 3, 3, 2, 3, 5]
    // CHECK-SAME: tensor<1x18x3x5xf16> into tensor<1x3x3x2x3x5xf16>
    // CHECK: hip.transpose
    // CHECK-SAME: perm = [0, 3, 4, 1, 5, 2]

    return %r : tensor<1x2x9x15xf16>
  }

  // CRD: channel split is [Cout, B, B] and perm is [0, 1, 4, 2, 5, 3].
  func.func @test_depth_to_space_crd(%arg0: tensor<1x18x3x5xi32>) -> tensor<1x2x9x15xi32> {
    // CHECK-LABEL: func.func @test_depth_to_space_crd
    %r = "onnx.DepthToSpace"(%arg0) {blocksize = 3 : si64, mode = "CRD"}
        : (tensor<1x18x3x5xi32>) -> tensor<1x2x9x15xi32>

    // CHECK-NOT: onnx.DepthToSpace
    // CHECK: tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2, 3], [4], [5]] output_shape [1, 2, 3, 3, 3, 5]
    // CHECK-SAME: tensor<1x18x3x5xi32> into tensor<1x2x3x3x3x5xi32>
    // CHECK: tensor.empty() : tensor<1x2x3x3x5x3xi32>
    // CHECK: hip.transpose
    // CHECK-SAME: perm = [0, 1, 4, 2, 5, 3]
    // CHECK: tensor.collapse_shape %{{.*}} {{\[\[}}0], [1], [2, 3], [4, 5]]
    // CHECK-SAME: tensor<1x2x3x3x5x3xi32> into tensor<1x2x9x15xi32>

    return %r : tensor<1x2x9x15xi32>
  }

  // blocksize 1 is an identity for CRD (the dc-ae decoder shape).
  func.func @test_depth_to_space_blocksize1_crd(%arg0: tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xf32> {
    // CHECK-LABEL: func.func @test_depth_to_space_blocksize1_crd
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<1x1024x16x16xf32>)
    %r = "onnx.DepthToSpace"(%arg0) {blocksize = 1 : si64, mode = "CRD"}
        : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xf32>

    // CHECK-NOT: onnx.DepthToSpace
    // CHECK-NOT: hip.transpose
    // CHECK: return %[[IN]]

    return %r : tensor<1x1024x16x16xf32>
  }

  // blocksize 1, DCR, dynamic batch made static on the result: a cast.
  func.func @test_depth_to_space_blocksize1_cast(%arg0: tensor<?x4x2x2xf32>) -> tensor<1x4x2x2xf32> {
    // CHECK-LABEL: func.func @test_depth_to_space_blocksize1_cast
    // CHECK-SAME: %[[IN:arg[0-9]+]]: tensor<?x4x2x2xf32>
    %r = "onnx.DepthToSpace"(%arg0) {blocksize = 1 : si64, mode = "DCR"}
        : (tensor<?x4x2x2xf32>) -> tensor<1x4x2x2xf32>

    // CHECK-NOT: onnx.DepthToSpace
    // CHECK-NOT: hip.transpose
    // CHECK: tensor.cast %[[IN]] : tensor<?x4x2x2xf32> to tensor<1x4x2x2xf32>

    return %r : tensor<1x4x2x2xf32>
  }

  // Dynamic batch and height. Static C still splits, and W * blocksize is static.
  func.func @test_depth_to_space_dynamic(%arg0: tensor<?x8x?x5xf32>) -> tensor<?x2x?x10xf32> {
    // CHECK-LABEL: func.func @test_depth_to_space_dynamic
    // CHECK-SAME: %[[IN:arg[0-9]+]]: tensor<?x8x?x5xf32>
    %r = "onnx.DepthToSpace"(%arg0) {blocksize = 2 : si64, mode = "DCR"}
        : (tensor<?x8x?x5xf32>) -> tensor<?x2x?x10xf32>

    // CHECK-NOT: onnx.DepthToSpace
    // CHECK: %[[N:.*]] = tensor.dim %[[IN]], %{{.*}} : tensor<?x8x?x5xf32>
    // CHECK: %[[H:.*]] = tensor.dim %[[IN]], %{{.*}} : tensor<?x8x?x5xf32>
    // CHECK: tensor.expand_shape %[[IN]] {{\[\[}}0], [1, 2, 3], [4], [5]] output_shape [%[[N]], 2, 2, 2, %[[H]], 5]
    // CHECK-SAME: tensor<?x8x?x5xf32> into tensor<?x2x2x2x?x5xf32>
    // CHECK: tensor.empty(%[[N]], %[[H]]) : tensor<?x2x?x2x5x2xf32>
    // CHECK: hip.transpose
    // CHECK-SAME: perm = [0, 3, 4, 1, 5, 2]
    // CHECK: tensor.collapse_shape
    // CHECK-SAME: into tensor<?x2x?x10xf32>

    return %r : tensor<?x2x?x10xf32>
  }

  // Both batch and channels dynamic: Cout = C / blocksize^2 at runtime.
  func.func @test_depth_to_space_dynamic_channels(%arg0: tensor<?x?x2x2xf32>) -> tensor<?x?x4x4xf32> {
    // CHECK-LABEL: func.func @test_depth_to_space_dynamic_channels
    %r = "onnx.DepthToSpace"(%arg0) {blocksize = 2 : si64, mode = "CRD"}
        : (tensor<?x?x2x2xf32>) -> tensor<?x?x4x4xf32>

    // CHECK-NOT: onnx.DepthToSpace
    // CHECK: arith.divsi
    // CHECK: tensor.expand_shape {{.*}} into tensor<?x?x2x2x2x2xf32>
    // CHECK: hip.transpose
    // CHECK-SAME: perm = [0, 1, 4, 2, 5, 3]
    // CHECK: tensor.collapse_shape {{.*}} into tensor<?x?x4x4xf32>

    return %r : tensor<?x?x4x4xf32>
  }

  // Dynamic input channels, static result channels: the input is pinned to
  // C = Cout * blocksize^2 before the channel split.
  func.func @test_depth_to_space_refine_channels(%arg0: tensor<1x?x2x2xf32>) -> tensor<1x2x4x4xf32> {
    // CHECK-LABEL: func.func @test_depth_to_space_refine_channels
    // CHECK-SAME: %[[IN:arg[0-9]+]]: tensor<1x?x2x2xf32>
    %r = "onnx.DepthToSpace"(%arg0) {blocksize = 2 : si64, mode = "DCR"}
        : (tensor<1x?x2x2xf32>) -> tensor<1x2x4x4xf32>

    // CHECK-NOT: onnx.DepthToSpace
    // CHECK: %[[PINNED:.*]] = tensor.cast %[[IN]] : tensor<1x?x2x2xf32> to tensor<1x8x2x2xf32>
    // CHECK: tensor.expand_shape %[[PINNED]] {{\[\[}}0], [1, 2, 3], [4], [5]] output_shape [1, 2, 2, 2, 2, 2]
    // CHECK: hip.transpose
    // CHECK: tensor.collapse_shape {{.*}} into tensor<1x2x4x4xf32>
    // CHECK-NOT: tensor.cast

    return %r : tensor<1x2x4x4xf32>
  }

  // --- rejected forms stay onnx.DepthToSpace ---

  func.func @test_depth_to_space_blocksize_zero(%arg0: tensor<1x4x2x2xf32>) -> tensor<1x4x2x2xf32> {
    // CHECK-LABEL: func.func @test_depth_to_space_blocksize_zero
    %r = "onnx.DepthToSpace"(%arg0) {blocksize = 0 : si64, mode = "DCR"}
        : (tensor<1x4x2x2xf32>) -> tensor<1x4x2x2xf32>
    // CHECK: onnx.DepthToSpace
    return %r : tensor<1x4x2x2xf32>
  }

  func.func @test_depth_to_space_negative_blocksize(%arg0: tensor<1x4x2x2xf32>) -> tensor<1x4x2x2xf32> {
    // CHECK-LABEL: func.func @test_depth_to_space_negative_blocksize
    %r = "onnx.DepthToSpace"(%arg0) {blocksize = -2 : si64, mode = "DCR"}
        : (tensor<1x4x2x2xf32>) -> tensor<1x4x2x2xf32>
    // CHECK: onnx.DepthToSpace
    return %r : tensor<1x4x2x2xf32>
  }

  func.func @test_depth_to_space_channels_not_divisible(%arg0: tensor<1x6x2x2xf32>) -> tensor<1x6x4x4xf32> {
    // CHECK-LABEL: func.func @test_depth_to_space_channels_not_divisible
    %r = "onnx.DepthToSpace"(%arg0) {blocksize = 2 : si64, mode = "DCR"}
        : (tensor<1x6x2x2xf32>) -> tensor<1x6x4x4xf32>
    // CHECK: onnx.DepthToSpace
    return %r : tensor<1x6x4x4xf32>
  }

  func.func @test_depth_to_space_bad_output_channels(%arg0: tensor<1x8x2x2xf32>) -> tensor<1x4x4x4xf32> {
    // CHECK-LABEL: func.func @test_depth_to_space_bad_output_channels
    %r = "onnx.DepthToSpace"(%arg0) {blocksize = 2 : si64, mode = "DCR"}
        : (tensor<1x8x2x2xf32>) -> tensor<1x4x4x4xf32>
    // CHECK: onnx.DepthToSpace
    return %r : tensor<1x4x4x4xf32>
  }

  func.func @test_depth_to_space_bad_output_spatial(%arg0: tensor<1x8x2x2xf32>) -> tensor<1x2x3x4xf32> {
    // CHECK-LABEL: func.func @test_depth_to_space_bad_output_spatial
    %r = "onnx.DepthToSpace"(%arg0) {blocksize = 2 : si64, mode = "DCR"}
        : (tensor<1x8x2x2xf32>) -> tensor<1x2x3x4xf32>
    // CHECK: onnx.DepthToSpace
    return %r : tensor<1x2x3x4xf32>
  }

  func.func @test_depth_to_space_bad_batch(%arg0: tensor<1x8x2x2xf32>) -> tensor<2x2x4x4xf32> {
    // CHECK-LABEL: func.func @test_depth_to_space_bad_batch
    %r = "onnx.DepthToSpace"(%arg0) {blocksize = 2 : si64, mode = "DCR"}
        : (tensor<1x8x2x2xf32>) -> tensor<2x2x4x4xf32>
    // CHECK: onnx.DepthToSpace
    return %r : tensor<2x2x4x4xf32>
  }

  func.func @test_depth_to_space_rank3(%arg0: tensor<4x2x2xf32>) -> tensor<1x4x4xf32> {
    // CHECK-LABEL: func.func @test_depth_to_space_rank3
    %r = "onnx.DepthToSpace"(%arg0) {blocksize = 2 : si64, mode = "DCR"}
        : (tensor<4x2x2xf32>) -> tensor<1x4x4xf32>
    // CHECK: onnx.DepthToSpace
    return %r : tensor<1x4x4xf32>
  }

  func.func @test_depth_to_space_bad_mode(%arg0: tensor<1x8x2x2xf32>) -> tensor<1x2x4x4xf32> {
    // CHECK-LABEL: func.func @test_depth_to_space_bad_mode
    %r = "onnx.DepthToSpace"(%arg0) {blocksize = 2 : si64, mode = "XYZ"}
        : (tensor<1x8x2x2xf32>) -> tensor<1x2x4x4xf32>
    // CHECK: onnx.DepthToSpace
    return %r : tensor<1x2x4x4xf32>
  }

  func.func @test_depth_to_space_bool(%arg0: tensor<1x4x2x2xi1>) -> tensor<1x1x4x4xi1> {
    // CHECK-LABEL: func.func @test_depth_to_space_bool
    %r = "onnx.DepthToSpace"(%arg0) {blocksize = 2 : si64, mode = "DCR"}
        : (tensor<1x4x2x2xi1>) -> tensor<1x1x4x4xi1>
    // CHECK: onnx.DepthToSpace
    return %r : tensor<1x1x4x4xi1>
  }

  func.func @test_depth_to_space_static_output_not_multiple(%arg0: tensor<?x4x?x2xf32>) -> tensor<?x1x3x4xf32> {
    // CHECK-LABEL: func.func @test_depth_to_space_static_output_not_multiple
    %r = "onnx.DepthToSpace"(%arg0) {blocksize = 2 : si64, mode = "DCR"}
        : (tensor<?x4x?x2xf32>) -> tensor<?x1x3x4xf32>
    // CHECK: onnx.DepthToSpace
    return %r : tensor<?x1x3x4xf32>
  }
}
