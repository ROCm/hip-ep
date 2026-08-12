// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// Converts onnx.NonZero to hipsr.nonzero plus the hipsr.compute that narrows
// its worst-case destination. Rejected forms live in nonzero-invalid.mlir.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --split-input-file -allow-unregistered-dialect -convert-onnx-to-hipsr | FileCheck %s

// The search half is DPS, so its placeholder stays empty for
// hipsr-populate-shape-region. The narrowing half is a compute, so this
// conversion fills its barrier region: the column count is a host read of the
// count the search published, which the placeholder takes as its input once
// the pass rewires it onto the shape graph.
// CHECK-LABEL: func.func @nonzero_mask(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context, %[[MASK:.*]]: tensor<?x?xi8>) -> tensor<2x?xi64> {
// CHECK-NEXT:    %[[INITS:.*]]:2 = hipsr.placeholder(%[[CTX]]) ins(%[[MASK]] : tensor<?x?xi8>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<2x?xi64>, tensor<1xi64>
// CHECK-NEXT:    %[[SEARCH:.*]]:2 = hipsr.nonzero(%[[CTX]]) ins(%[[MASK]] : tensor<?x?xi8>) outs(%[[INITS]]#0, %[[INITS]]#1 : tensor<2x?xi64>, tensor<1xi64>) : tensor<2x?xi64>, tensor<1xi64>
// CHECK-NEXT:    %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) ins(%[[INITS]]#1 : tensor<1xi64>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<2x?xi64> shape_region {
// CHECK-NEXT:    ^bb0(%{{.*}}: !hipsr.context, %[[COUNT:.*]]: tensor<1xi64>):
// CHECK-NEXT:      %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[FOUND:.*]] = tensor.extract %[[COUNT]]{{\[}}%[[ZERO]]] : tensor<1xi64>
// CHECK-NEXT:      %[[COLUMNS:.*]] = arith.index_cast %[[FOUND]] : i64 to index
// CHECK-NEXT:      %[[ROWS:.*]] = arith.constant 2 : index
// CHECK-NEXT:      %[[SHAPE:.*]] = shape.from_extents %[[ROWS]], %[[COLUMNS]] : index, index
// CHECK-NEXT:      hipsr.shape_yield %[[SHAPE]] : !shape.shape
// CHECK:         %[[NARROWED:.*]] = hipsr.compute(%[[CTX]]) ins(%[[SEARCH]]#0 : tensor<2x?xi64>) outs(%[[INIT]] : tensor<2x?xi64>) {
// CHECK-NEXT:    ^bb0(%{{.*}}: !hipsr.context, %[[POSITIONS:.*]]: tensor<2x?xi64>, %[[DEST:.*]]: tensor<2x?xi64>):
// CHECK-NEXT:      %[[ONE:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[DIM:.*]] = tensor.dim %[[DEST]], %[[ONE]] : tensor<2x?xi64>
// CHECK-NEXT:      %[[SLICE:.*]] = tensor.extract_slice %[[POSITIONS]][0, 0] [2, %[[DIM]]] [1, 1] : tensor<2x?xi64> to tensor<2x?xi64>
// CHECK-NEXT:      hipsr.compute_yield %[[SLICE]] : tensor<2x?xi64>
// CHECK:         return %[[NARROWED]] : tensor<2x?xi64>
func.func @nonzero_mask(%ctx: !hipsr.context,
                        %mask: tensor<?x?xi8>) -> tensor<2x?xi64> {
  %0 = "onnx.NonZero"(%mask) : (tensor<?x?xi8>) -> tensor<2x?xi64>
  return %0 : tensor<2x?xi64>
}

// -----

// A static input pins the worst case at its element count, so the search
// destination is static even though the published result stays dynamic.
// CHECK-LABEL: func.func @static_capacity(
// CHECK:         hipsr.nonzero(%{{.*}}) ins(%{{.*}} : tensor<4x3xi1>) outs(%{{.*}}, %{{.*}} : tensor<2x12xi64>, tensor<1xi64>) : tensor<2x12xi64>, tensor<1xi64>
// CHECK:         tensor.extract_slice %{{.*}}[0, 0] [2, %{{.*}}] [1, 1] : tensor<2x12xi64> to tensor<2x?xi64>
func.func @static_capacity(%ctx: !hipsr.context,
                           %mask: tensor<4x3xi1>) -> tensor<2x?xi64> {
  %0 = "onnx.NonZero"(%mask) : (tensor<4x3xi1>) -> tensor<2x?xi64>
  return %0 : tensor<2x?xi64>
}

// -----

// A rank-1 input names a position with a single row, and the row count in the
// barrier region follows.
// CHECK-LABEL: func.func @rank_one_input(
// CHECK:         hipsr.nonzero(%{{.*}}) ins(%{{.*}} : tensor<?xi64>) outs(%{{.*}}, %{{.*}} : tensor<1x?xi64>, tensor<1xi64>) : tensor<1x?xi64>, tensor<1xi64>
// CHECK:         %[[ROWS:.*]] = arith.constant 1 : index
// CHECK:         shape.from_extents %[[ROWS]], %{{.*}} : index, index
func.func @rank_one_input(%ctx: !hipsr.context,
                          %values: tensor<?xi64>) -> tensor<1x?xi64> {
  %0 = "onnx.NonZero"(%values) : (tensor<?xi64>) -> tensor<1x?xi64>
  return %0 : tensor<1x?xi64>
}
