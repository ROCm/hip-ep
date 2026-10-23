// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// onnx.SpaceToDepth decomposes to tensor.expand_shape + hip.transpose +
// tensor.collapse_shape. It is the inverse of DepthToSpace DCR.
// blocksize == 1 is the identity.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x4x2x2xf32>) -> tensor<1x4x2x2xf32> {
    return %arg0 : tensor<1x4x2x2xf32>
  }

  // [1, 3, 8, 8] -> expand [1, 3, 4, 2, 4, 2]
  // -> transpose perm [0, 3, 5, 1, 2, 4] -> [1, 2, 2, 3, 4, 4]
  // -> collapse [1, 12, 4, 4].
  func.func @test_space_to_depth(%arg0: tensor<1x3x8x8xf32>) -> tensor<1x12x4x4xf32> {
    // CHECK-LABEL: func.func @test_space_to_depth
    %r = "onnx.SpaceToDepth"(%arg0) {blocksize = 2 : si64}
        : (tensor<1x3x8x8xf32>) -> tensor<1x12x4x4xf32>

    // CHECK-NOT: onnx.SpaceToDepth
    // CHECK: tensor.expand_shape %{{.*}} {{\[\[}}0], [1], [2, 3], [4, 5]] output_shape [1, 3, 4, 2, 4, 2]
    // CHECK-SAME: tensor<1x3x8x8xf32> into tensor<1x3x4x2x4x2xf32>
    // CHECK: tensor.empty() : tensor<1x2x2x3x4x4xf32>
    // CHECK: hip.transpose
    // CHECK-SAME: perm = [0, 3, 5, 1, 2, 4]
    // CHECK: tensor.collapse_shape %{{.*}} {{\[\[}}0], [1, 2, 3], [4], [5]]
    // CHECK-SAME: tensor<1x2x2x3x4x4xf32> into tensor<1x12x4x4xf32>

    return %r : tensor<1x12x4x4xf32>
  }

  // The AAA-v10 generator node: blocksize 4, no mode, f16 NCHW.
  func.func @test_space_to_depth_blocksize4_f16(%arg0: tensor<1x3x1152x1152xf16>) -> tensor<1x48x288x288xf16> {
    // CHECK-LABEL: func.func @test_space_to_depth_blocksize4_f16
    %r = "onnx.SpaceToDepth"(%arg0) {blocksize = 4 : si64}
        : (tensor<1x3x1152x1152xf16>) -> tensor<1x48x288x288xf16>

    // CHECK-NOT: onnx.SpaceToDepth
    // CHECK: tensor.expand_shape %{{.*}} {{\[\[}}0], [1], [2, 3], [4, 5]] output_shape [1, 3, 288, 4, 288, 4]
    // CHECK-SAME: tensor<1x3x1152x1152xf16> into tensor<1x3x288x4x288x4xf16>
    // CHECK: tensor.empty() : tensor<1x4x4x3x288x288xf16>
    // CHECK: hip.transpose
    // CHECK-SAME: perm = [0, 3, 5, 1, 2, 4]
    // CHECK: tensor.collapse_shape %{{.*}} {{\[\[}}0], [1, 2, 3], [4], [5]]
    // CHECK-SAME: tensor<1x4x4x3x288x288xf16> into tensor<1x48x288x288xf16>

    return %r : tensor<1x48x288x288xf16>
  }

  // blocksize 1 is an identity.
  func.func @test_space_to_depth_blocksize1(%arg0: tensor<1x4x2x2xf32>) -> tensor<1x4x2x2xf32> {
    // CHECK-LABEL: func.func @test_space_to_depth_blocksize1
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<1x4x2x2xf32>)
    %r = "onnx.SpaceToDepth"(%arg0) {blocksize = 1 : si64}
        : (tensor<1x4x2x2xf32>) -> tensor<1x4x2x2xf32>

    // CHECK-NOT: onnx.SpaceToDepth
    // CHECK-NOT: hip.transpose
    // CHECK: return %[[IN]]

    return %r : tensor<1x4x2x2xf32>
  }

  // blocksize 1, dynamic batch made static on the result: a cast.
  func.func @test_space_to_depth_blocksize1_cast(%arg0: tensor<?x4x2x2xf32>) -> tensor<1x4x2x2xf32> {
    // CHECK-LABEL: func.func @test_space_to_depth_blocksize1_cast
    // CHECK-SAME: %[[IN:arg[0-9]+]]: tensor<?x4x2x2xf32>
    %r = "onnx.SpaceToDepth"(%arg0) {blocksize = 1 : si64}
        : (tensor<?x4x2x2xf32>) -> tensor<1x4x2x2xf32>

    // CHECK-NOT: onnx.SpaceToDepth
    // CHECK-NOT: hip.transpose
    // CHECK: tensor.cast %[[IN]] : tensor<?x4x2x2xf32> to tensor<1x4x2x2xf32>

    return %r : tensor<1x4x2x2xf32>
  }

  // Dynamic batch and height. Static C grows by blocksize^2, and W / blocksize
  // is static.
  func.func @test_space_to_depth_dynamic(%arg0: tensor<?x3x?x8xf16>) -> tensor<?x12x?x4xf16> {
    // CHECK-LABEL: func.func @test_space_to_depth_dynamic
    // CHECK-SAME: %[[IN:arg[0-9]+]]: tensor<?x3x?x8xf16>
    %r = "onnx.SpaceToDepth"(%arg0) {blocksize = 2 : si64}
        : (tensor<?x3x?x8xf16>) -> tensor<?x12x?x4xf16>

    // CHECK-NOT: onnx.SpaceToDepth
    // CHECK: %[[N:.*]] = tensor.dim %[[IN]], %{{.*}} : tensor<?x3x?x8xf16>
    // CHECK: %[[H:.*]] = tensor.dim %[[IN]], %{{.*}} : tensor<?x3x?x8xf16>
    // CHECK: %[[HINNER:.*]] = arith.divsi %[[H]]
    // CHECK: tensor.expand_shape %[[IN]] {{\[\[}}0], [1], [2, 3], [4, 5]] output_shape [%[[N]], 3, %[[HINNER]], 2, 4, 2]
    // CHECK-SAME: tensor<?x3x?x8xf16> into tensor<?x3x?x2x4x2xf16>
    // CHECK: tensor.empty(%[[N]], %[[HINNER]]) : tensor<?x2x2x3x?x4xf16>
    // CHECK: hip.transpose
    // CHECK-SAME: perm = [0, 3, 5, 1, 2, 4]
    // CHECK: tensor.collapse_shape
    // CHECK-SAME: into tensor<?x12x?x4xf16>

    return %r : tensor<?x12x?x4xf16>
  }

  // Dynamic input height, static result height: the input is pinned to
  // H = output_H * blocksize before the spatial split.
  func.func @test_space_to_depth_refine_height(%arg0: tensor<1x3x?x8xf32>) -> tensor<1x12x4x4xf32> {
    // CHECK-LABEL: func.func @test_space_to_depth_refine_height
    // CHECK-SAME: %[[IN:arg[0-9]+]]: tensor<1x3x?x8xf32>
    %r = "onnx.SpaceToDepth"(%arg0) {blocksize = 2 : si64}
        : (tensor<1x3x?x8xf32>) -> tensor<1x12x4x4xf32>

    // CHECK-NOT: onnx.SpaceToDepth
    // CHECK: %[[PINNED:.*]] = tensor.cast %[[IN]] : tensor<1x3x?x8xf32> to tensor<1x3x8x8xf32>
    // CHECK: tensor.expand_shape %[[PINNED]] {{\[\[}}0], [1], [2, 3], [4, 5]] output_shape [1, 3, 4, 2, 4, 2]
    // CHECK: hip.transpose
    // CHECK: tensor.collapse_shape {{.*}} into tensor<1x12x4x4xf32>
    // CHECK-NOT: tensor.cast

    return %r : tensor<1x12x4x4xf32>
  }

  // --- rejected forms stay onnx.SpaceToDepth ---

  func.func @test_space_to_depth_blocksize_zero(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32> {
    // CHECK-LABEL: func.func @test_space_to_depth_blocksize_zero
    %r = "onnx.SpaceToDepth"(%arg0) {blocksize = 0 : si64}
        : (tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
    // CHECK: onnx.SpaceToDepth
    return %r : tensor<1x3x4x4xf32>
  }

  func.func @test_space_to_depth_negative_blocksize(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32> {
    // CHECK-LABEL: func.func @test_space_to_depth_negative_blocksize
    %r = "onnx.SpaceToDepth"(%arg0) {blocksize = -2 : si64}
        : (tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
    // CHECK: onnx.SpaceToDepth
    return %r : tensor<1x3x4x4xf32>
  }

  func.func @test_space_to_depth_height_not_divisible(%arg0: tensor<1x3x5x4xf32>) -> tensor<1x12x2x2xf32> {
    // CHECK-LABEL: func.func @test_space_to_depth_height_not_divisible
    %r = "onnx.SpaceToDepth"(%arg0) {blocksize = 2 : si64}
        : (tensor<1x3x5x4xf32>) -> tensor<1x12x2x2xf32>
    // CHECK: onnx.SpaceToDepth
    return %r : tensor<1x12x2x2xf32>
  }

  func.func @test_space_to_depth_bad_output_channels(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x6x2x2xf32> {
    // CHECK-LABEL: func.func @test_space_to_depth_bad_output_channels
    %r = "onnx.SpaceToDepth"(%arg0) {blocksize = 2 : si64}
        : (tensor<1x3x4x4xf32>) -> tensor<1x6x2x2xf32>
    // CHECK: onnx.SpaceToDepth
    return %r : tensor<1x6x2x2xf32>
  }

  func.func @test_space_to_depth_bad_output_spatial(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x12x3x2xf32> {
    // CHECK-LABEL: func.func @test_space_to_depth_bad_output_spatial
    %r = "onnx.SpaceToDepth"(%arg0) {blocksize = 2 : si64}
        : (tensor<1x3x4x4xf32>) -> tensor<1x12x3x2xf32>
    // CHECK: onnx.SpaceToDepth
    return %r : tensor<1x12x3x2xf32>
  }

  func.func @test_space_to_depth_bad_batch(%arg0: tensor<1x3x4x4xf32>) -> tensor<2x12x2x2xf32> {
    // CHECK-LABEL: func.func @test_space_to_depth_bad_batch
    %r = "onnx.SpaceToDepth"(%arg0) {blocksize = 2 : si64}
        : (tensor<1x3x4x4xf32>) -> tensor<2x12x2x2xf32>
    // CHECK: onnx.SpaceToDepth
    return %r : tensor<2x12x2x2xf32>
  }

  func.func @test_space_to_depth_rank3(%arg0: tensor<3x4x4xf32>) -> tensor<12x2x2xf32> {
    // CHECK-LABEL: func.func @test_space_to_depth_rank3
    %r = "onnx.SpaceToDepth"(%arg0) {blocksize = 2 : si64}
        : (tensor<3x4x4xf32>) -> tensor<12x2x2xf32>
    // CHECK: onnx.SpaceToDepth
    return %r : tensor<12x2x2xf32>
  }

  func.func @test_space_to_depth_bool(%arg0: tensor<1x1x2x2xi1>) -> tensor<1x4x1x1xi1> {
    // CHECK-LABEL: func.func @test_space_to_depth_bool
    %r = "onnx.SpaceToDepth"(%arg0) {blocksize = 2 : si64}
        : (tensor<1x1x2x2xi1>) -> tensor<1x4x1x1xi1>
    // CHECK: onnx.SpaceToDepth
    return %r : tensor<1x4x1x1xi1>
  }
}
