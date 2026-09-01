// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// onnx.ScatterND becomes hipsr.scatter_nd. Rejected forms are in
// scatter_nd-invalid.mlir.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --onnx-dialect=modeled --split-input-file -allow-unregistered-dialect -convert-onnx-to-hipsr | FileCheck %s

// The embedding graph writes image features into the token embeddings at the
// positions it found. The result takes the data's shape, but the placeholder
// still lists every scatter operand so the two stay in one pool domain. Its
// shape region stays empty for hipsr-populate-shape-region to fill in.
// CHECK-LABEL: func.func @scatter_features(
// CHECK-SAME:    %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME:    %[[EMBEDS:.+]]: tensor<?x?x4096xf16, #hipsr.mem<device>>,
// CHECK-SAME:    %[[POSITIONS:.+]]: tensor<?x3xi64, #hipsr.mem<device>>,
// CHECK-SAME:    %[[FEATURES:.+]]: tensor<?xf16, #hipsr.mem<device>>) -> tensor<?x?x4096xf16, #hipsr.mem<device>> {
// CHECK-NEXT:    %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[EMBEDS]], %[[POSITIONS]], %[[FEATURES]] : tensor<?x?x4096xf16, #hipsr.mem<device>>, tensor<?x3xi64, #hipsr.mem<device>>, tensor<?xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:    %[[RESULT:.+]] = hipsr.scatter_nd(%[[CTX]]) ins(%[[EMBEDS]], %[[POSITIONS]], %[[FEATURES]] : tensor<?x?x4096xf16, #hipsr.mem<device>>, tensor<?x3xi64, #hipsr.mem<device>>, tensor<?xf16, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<?x?x4096xf16, #hipsr.mem<device>>) : tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:    return %[[RESULT]] : tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:  }
func.func @scatter_features(%ctx: !hipsr.context,
                            %embeds: tensor<?x?x4096xf16>,
                            %positions: tensor<?x3xi64>,
                            %features: tensor<?xf16>) -> tensor<?x?x4096xf16> {
  %0 = "onnx.ScatterND"(%embeds, %positions, %features) {reduction = "none"}
      : (tensor<?x?x4096xf16>, tensor<?x3xi64>, tensor<?xf16>)
      -> tensor<?x?x4096xf16>
  "onnx.Return"(%0) : (tensor<?x?x4096xf16>) -> ()
}

// -----

// ONNX defaults the reduction to overwriting, so an absent attribute converts
// like an explicit one. A row here addresses one axis instead of all of them,
// so the updates carry the data extents it does not reach.
// CHECK-LABEL: func.func @default_reduction(
// CHECK-SAME:    %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME:    %[[DATA:.+]]: tensor<4x8x2xf16, #hipsr.mem<device>>,
// CHECK-SAME:    %[[IDS:.+]]: tensor<5x1xi64, #hipsr.mem<device>>,
// CHECK-SAME:    %[[UPDATES:.+]]: tensor<5x8x2xf16, #hipsr.mem<device>>) -> tensor<4x8x2xf16, #hipsr.mem<device>> {
// CHECK-NEXT:    %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[DATA]], %[[IDS]], %[[UPDATES]] : tensor<4x8x2xf16, #hipsr.mem<device>>, tensor<5x1xi64, #hipsr.mem<device>>, tensor<5x8x2xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4x8x2xf16, #hipsr.mem<device>>
// CHECK-NEXT:    %[[RESULT:.+]] = hipsr.scatter_nd(%[[CTX]]) ins(%[[DATA]], %[[IDS]], %[[UPDATES]] : tensor<4x8x2xf16, #hipsr.mem<device>>, tensor<5x1xi64, #hipsr.mem<device>>, tensor<5x8x2xf16, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<4x8x2xf16, #hipsr.mem<device>>) : tensor<4x8x2xf16, #hipsr.mem<device>>
// CHECK-NEXT:    return %[[RESULT]] : tensor<4x8x2xf16, #hipsr.mem<device>>
// CHECK-NEXT:  }
func.func @default_reduction(%ctx: !hipsr.context, %data: tensor<4x8x2xf16>,
                             %ids: tensor<5x1xi64>,
                             %updates: tensor<5x8x2xf16>) -> tensor<4x8x2xf16> {
  %0 = "onnx.ScatterND"(%data, %ids, %updates)
      : (tensor<4x8x2xf16>, tensor<5x1xi64>, tensor<5x8x2xf16>)
      -> tensor<4x8x2xf16>
  "onnx.Return"(%0) : (tensor<4x8x2xf16>) -> ()
}
