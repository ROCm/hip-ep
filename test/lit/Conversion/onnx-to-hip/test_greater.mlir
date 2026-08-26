// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // Greater(A, B) decomposes into Less(B, A). The original onnx.Greater must
  // disappear.
  func.func @greater_static(%a: tensor<3x4xf32>, %b: tensor<3x4xf32>) -> tensor<3x4xi1> {
    %r = "onnx.Greater"(%a, %b) : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xi1>
    return %r : tensor<3x4xi1>
  }

  // CHECK-LABEL: func.func @greater_static
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<3x4xf32>, %[[B:.*]]: tensor<3x4xf32>)
  // CHECK: tensor.empty() : tensor<3x4xi1>
  // CHECK: hip.less(%[[CTX]]) ins(%[[B]], %[[A]] : tensor<3x4xf32>, tensor<3x4xf32>) outs({{.*}} : tensor<3x4xi1>)
  // CHECK-NOT: onnx.Greater

  func.func @greater_i32(%a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<4xi1> {
    %r = "onnx.Greater"(%a, %b) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
    return %r : tensor<4xi1>
  }

  // CHECK-LABEL: func.func @greater_i32
  // CHECK: hip.less({{.*}}) ins({{.*}}, {{.*}} : tensor<4xi32>, tensor<4xi32>) outs({{.*}} : tensor<4xi1>)
  // CHECK-NOT: onnx.Greater

  func.func @greater_dynamic(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>) -> tensor<?x?xi1> {
    %r = "onnx.Greater"(%a, %b) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xi1>
    return %r : tensor<?x?xi1>
  }

  // CHECK-LABEL: func.func @greater_dynamic
  // CHECK: tensor.dim
  // CHECK: hip.less({{.*}}) ins({{.*}}, {{.*}} : tensor<?x?xf32>, tensor<?x?xf32>) outs({{.*}} : tensor<?x?xi1>)
  // CHECK-NOT: onnx.Greater

  // A rank-6 Greater against a scalar threshold must be packed to rank four,
  // because hip.less only has a 4-D lowering. The scalar operand already
  // broadcasts through that lowering and stays rank zero.
  func.func @greater_6d_scalar(
      %a: tensor<1x6x128x200x8x200xf32>, %b: tensor<f32>)
      -> tensor<1x6x128x200x8x200xui8> {
    // CHECK-LABEL: func.func @greater_6d_scalar
    // CHECK: tensor.collapse_shape {{.*}} {{\[\[}}0], [1], [2], [3, 4, 5]] : tensor<1x6x128x200x8x200xf32> into tensor<1x6x128x320000xf32>
    // CHECK: hip.less({{.*}}) ins({{.*}}, {{.*}} : tensor<f32>, tensor<1x6x128x320000xf32>) outs({{.*}} : tensor<1x6x128x320000xui8>)
    // CHECK: tensor.expand_shape {{.*}} {{\[\[}}0], [1], [2], [3, 4, 5]] output_shape [1, 6, 128, 200, 8, 200] : tensor<1x6x128x320000xui8> into tensor<1x6x128x200x8x200xui8>
    // CHECK-NOT: onnx.Greater
    %r = "onnx.Greater"(%a, %b) :
        (tensor<1x6x128x200x8x200xf32>, tensor<f32>)
        -> tensor<1x6x128x200x8x200xui8>
    return %r : tensor<1x6x128x200x8x200xui8>
  }
}
