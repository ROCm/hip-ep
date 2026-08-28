// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --split-input-file -hipsr-populate-shape-region | FileCheck %s

// A runtime shape longer than the input rank uses the barrier layout and reads
// each requested extent from the raw shape tensor.
// CHECK-LABEL: func.func @expand_runtime_shape(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME: %[[INPUT:.+]]: tensor<?x3xf16, #hipsr.mem<device>>,
// CHECK-SAME: %[[REQUEST:.+]]: tensor<3xi64, #hipsr.mem<host>>) {
// CHECK-NEXT: %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]], %[[REQUEST]] : tensor<?x3xf16, #hipsr.mem<device>>, tensor<3xi64, #hipsr.mem<host>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<?x?x?xf16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT: ^bb0(%[[SHAPE_CTX:.+]]: !hipsr.context, %[[SHAPE_INPUT:.+]]: tensor<?x3xf16, #hipsr.mem<device>>, %[[SHAPE_REQUEST:.+]]: tensor<3xi64, #hipsr.mem<host>>):
// CHECK-NEXT: %[[INPUT_SHAPE:.+]] = shape.shape_of %[[SHAPE_INPUT]] : tensor<?x3xf16, #hipsr.mem<device>> -> tensor<2xindex>
// CHECK-NEXT: %[[INDEX0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[REQUEST0_I64:.+]] = tensor.extract %[[SHAPE_REQUEST]][%[[INDEX0]]] : tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT: %[[REQUEST0:.+]] = arith.index_cast %[[REQUEST0_I64]] : i64 to index
// CHECK-NEXT: %[[INDEX1:.+]] = arith.constant 1 : index
// CHECK-NEXT: %[[REQUEST1_I64:.+]] = tensor.extract %[[SHAPE_REQUEST]][%[[INDEX1]]] : tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT: %[[REQUEST1:.+]] = arith.index_cast %[[REQUEST1_I64]] : i64 to index
// CHECK-NEXT: %[[INDEX2:.+]] = arith.constant 2 : index
// CHECK-NEXT: %[[REQUEST2_I64:.+]] = tensor.extract %[[SHAPE_REQUEST]][%[[INDEX2]]] : tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT: %[[REQUEST2:.+]] = arith.index_cast %[[REQUEST2_I64]] : i64 to index
// CHECK-NEXT: %[[REQUEST_SHAPE:.+]] = shape.from_extents %[[REQUEST0]], %[[REQUEST1]], %[[REQUEST2]] : index, index, index
// CHECK-NEXT: %[[WITNESS:.+]] = shape.cstr_broadcastable %[[INPUT_SHAPE]], %[[REQUEST_SHAPE]] : tensor<2xindex>, !shape.shape
// CHECK-NEXT: %[[RESULT_SHAPE:.+]] = shape.assuming %[[WITNESS]] -> (!shape.shape) {
// CHECK-NEXT: %[[BROADCAST:.+]] = shape.broadcast %[[INPUT_SHAPE]], %[[REQUEST_SHAPE]] : tensor<2xindex>, !shape.shape -> !shape.shape
// CHECK-NEXT: shape.assuming_yield %[[BROADCAST]] : !shape.shape
// CHECK-NEXT: }
// CHECK-NEXT: hipsr.shape_yield %[[RESULT_SHAPE]] : !shape.shape
// CHECK-NEXT: }
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.expand(%[[CTX]]) ins(%[[INPUT]], %[[REQUEST]] : tensor<?x3xf16, #hipsr.mem<device>>, tensor<3xi64, #hipsr.mem<host>>) outs(%[[INIT]] : tensor<?x?x?xf16, #hipsr.mem<device>>) : tensor<?x?x?xf16, #hipsr.mem<device>>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @expand_runtime_shape(
    %ctx: !hipsr.context, %input: tensor<?x3xf16, #hipsr.mem<device>>, %shape: tensor<3xi64, #hipsr.mem<host>>) {
  %init = hipsr.placeholder(%ctx)
      ins(%input, %shape : tensor<?x3xf16, #hipsr.mem<device>>, tensor<3xi64, #hipsr.mem<host>>)
      {placeholder_type = #hipsr.placeholder_type<barrier>}
      : tensor<?x?x?xf16, #hipsr.mem<device>>
  %result = hipsr.expand(%ctx)
      ins(%input, %shape : tensor<?x3xf16, #hipsr.mem<device>>, tensor<3xi64, #hipsr.mem<host>>)
      outs(%init : tensor<?x?x?xf16, #hipsr.mem<device>>) : tensor<?x?x?xf16, #hipsr.mem<device>>
  return
}

// -----

// A shape attribute shorter than the input rank uses the normal layout and
// broadcasts directly from the input's shape value.
// CHECK-LABEL: func.func @expand_shape_attr(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[INPUT:.+]]: tensor<?x3xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]] : tensor<?x3xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x3xf16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT: ^bb0(%[[INPUT_SHAPE:.+]]: !shape.shape):
// CHECK-NEXT: %[[REQUEST0:.+]] = arith.constant 3 : index
// CHECK-NEXT: %[[REQUEST_SHAPE:.+]] = shape.from_extents %[[REQUEST0]] : index
// CHECK-NEXT: %[[WITNESS:.+]] = shape.cstr_broadcastable %[[INPUT_SHAPE]], %[[REQUEST_SHAPE]] : !shape.shape, !shape.shape
// CHECK-NEXT: %[[RESULT_SHAPE:.+]] = shape.assuming %[[WITNESS]] -> (!shape.shape) {
// CHECK-NEXT: %[[BROADCAST:.+]] = shape.broadcast %[[INPUT_SHAPE]], %[[REQUEST_SHAPE]] : !shape.shape, !shape.shape -> !shape.shape
// CHECK-NEXT: shape.assuming_yield %[[BROADCAST]] : !shape.shape
// CHECK-NEXT: }
// CHECK-NEXT: hipsr.shape_yield %[[RESULT_SHAPE]] : !shape.shape
// CHECK-NEXT: }
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.expand(%[[CTX]]) ins(%[[INPUT]] : tensor<?x3xf16, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<?x3xf16, #hipsr.mem<device>>) {shape_attr = array<i64: 3>} : tensor<?x3xf16, #hipsr.mem<device>>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @expand_shape_attr(%ctx: !hipsr.context,
                             %input: tensor<?x3xf16, #hipsr.mem<device>>) {
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<?x3xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>}
      : tensor<?x3xf16, #hipsr.mem<device>>
  %result = hipsr.expand(%ctx)
      ins(%input : tensor<?x3xf16, #hipsr.mem<device>>)
      outs(%init : tensor<?x3xf16, #hipsr.mem<device>>)
      {shape_attr = array<i64: 3>} : tensor<?x3xf16, #hipsr.mem<device>>
  return
}
