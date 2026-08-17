// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// Converts onnx.Shape to a hipsr.compute that reads the input extents, and fills
// the placeholder's shape region while it builds the body. Rejected forms live
// in shape-invalid.mlir.
//
// Extents are read on the host, so the chain holding them names
// #hipsr.mem<host>: the placeholder, the destination, the yielded value and the
// result. Data keeps the #hipsr.mem<device> the pass gives a tensor that names
// no space.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --split-input-file -allow-unregistered-dialect -convert-onnx-to-hipsr | FileCheck %s

// Only a dynamic axis needs a query; a static one is a constant. The shape
// region is constant too, since the input's rank fixes the result length.
// CHECK-LABEL:   func.func @shape_full(
// CHECK-SAME:      %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:      %[[INPUT:.*]]: tensor<?x?x4096xf16, #hipsr.mem<device>>) -> tensor<3xi64, #hipsr.mem<host>> {
// CHECK-NEXT:      %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<3xi64, #hipsr.mem<host>> shape_region {
// CHECK-NEXT:        %[[LEN:.*]] = arith.constant 3 : index
// CHECK-NEXT:        %[[SHAPE:.*]] = shape.from_extents %[[LEN]] : index
// CHECK-NEXT:        hipsr.shape_yield %[[SHAPE]] : !shape.shape
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[RESULT:.*]] = hipsr.compute(%[[CTX]]) ins(%[[INPUT]] : tensor<?x?x4096xf16, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<3xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:      ^bb0(%{{.*}}: !hipsr.context, %[[IN:.*]]: tensor<?x?x4096xf16, #hipsr.mem<device>>, %{{.*}}: tensor<3xi64, #hipsr.mem<host>>):
// CHECK-NEXT:        %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[DIM0:.*]] = tensor.dim %[[IN]], %[[C0]] : tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[EXT0:.*]] = arith.index_cast %[[DIM0]] : index to i64
// CHECK-NEXT:        %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[DIM1:.*]] = tensor.dim %[[IN]], %[[C1]] : tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[EXT1:.*]] = arith.index_cast %[[DIM1]] : index to i64
// CHECK-NEXT:        %[[EXT2:.*]] = arith.constant 4096 : i64
// CHECK-NEXT:        %[[EXTENTS:.*]] = tensor.from_elements %[[EXT0]], %[[EXT1]], %[[EXT2]] : tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:        hipsr.compute_yield %[[EXTENTS]] : tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:      } : tensor<3xi64, #hipsr.mem<host>>{{$}}
// CHECK-NEXT:      return %[[RESULT]] : tensor<3xi64, #hipsr.mem<host>>
func.func @shape_full(%ctx: !hipsr.context, %input: tensor<?x?x4096xf16>)
    -> tensor<3xi64, #hipsr.mem<host>> {
  %0 = "onnx.Shape"(%input)
      : (tensor<?x?x4096xf16>) -> tensor<3xi64, #hipsr.mem<host>>
  return %0 : tensor<3xi64, #hipsr.mem<host>>
}

// -----

// start drops the leading axes, so axis 0 is never queried.
// CHECK-LABEL:   func.func @shape_start(
// CHECK-SAME:      %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:      %[[INPUT:.*]]: tensor<?x?x4096xf16, #hipsr.mem<device>>) -> tensor<2xi64, #hipsr.mem<host>> {
// CHECK-NEXT:      %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<2xi64, #hipsr.mem<host>> shape_region {
// CHECK-NEXT:        %[[LEN:.*]] = arith.constant 2 : index
// CHECK-NEXT:        %[[SHAPE:.*]] = shape.from_extents %[[LEN]] : index
// CHECK-NEXT:        hipsr.shape_yield %[[SHAPE]] : !shape.shape
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[RESULT:.*]] = hipsr.compute(%[[CTX]]) ins(%[[INPUT]] : tensor<?x?x4096xf16, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<2xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:      ^bb0(%{{.*}}: !hipsr.context, %[[IN:.*]]: tensor<?x?x4096xf16, #hipsr.mem<device>>, %{{.*}}: tensor<2xi64, #hipsr.mem<host>>):
// CHECK-NEXT:        %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[DIM1:.*]] = tensor.dim %[[IN]], %[[C1]] : tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[EXT0:.*]] = arith.index_cast %[[DIM1]] : index to i64
// CHECK-NEXT:        %[[EXT1:.*]] = arith.constant 4096 : i64
// CHECK-NEXT:        %[[EXTENTS:.*]] = tensor.from_elements %[[EXT0]], %[[EXT1]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:        hipsr.compute_yield %[[EXTENTS]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      } : tensor<2xi64, #hipsr.mem<host>>{{$}}
// CHECK-NEXT:      return %[[RESULT]] : tensor<2xi64, #hipsr.mem<host>>
func.func @shape_start(%ctx: !hipsr.context, %input: tensor<?x?x4096xf16>)
    -> tensor<2xi64, #hipsr.mem<host>> {
  %0 = "onnx.Shape"(%input) {start = 1 : si64}
      : (tensor<?x?x4096xf16>) -> tensor<2xi64, #hipsr.mem<host>>
  return %0 : tensor<2xi64, #hipsr.mem<host>>
}

// -----

// A negative start counts back from the end, selecting the last axis alone. That
// axis is static, so the body queries nothing.
// CHECK-LABEL:   func.func @shape_negative_start(
// CHECK-SAME:      %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:      %[[INPUT:.*]]: tensor<?x?x4096xf16, #hipsr.mem<device>>) -> tensor<1xi64, #hipsr.mem<host>> {
// CHECK-NEXT:      %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<1xi64, #hipsr.mem<host>> shape_region {
// CHECK-NEXT:        %[[LEN:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[SHAPE:.*]] = shape.from_extents %[[LEN]] : index
// CHECK-NEXT:        hipsr.shape_yield %[[SHAPE]] : !shape.shape
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[RESULT:.*]] = hipsr.compute(%[[CTX]]) ins(%[[INPUT]] : tensor<?x?x4096xf16, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<1xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:      ^bb0(%{{.*}}: !hipsr.context, %{{.*}}: tensor<?x?x4096xf16, #hipsr.mem<device>>, %{{.*}}: tensor<1xi64, #hipsr.mem<host>>):
// CHECK-NEXT:        %[[EXT0:.*]] = arith.constant 4096 : i64
// CHECK-NEXT:        %[[EXTENTS:.*]] = tensor.from_elements %[[EXT0]] : tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:        hipsr.compute_yield %[[EXTENTS]] : tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:      } : tensor<1xi64, #hipsr.mem<host>>{{$}}
// CHECK-NEXT:      return %[[RESULT]] : tensor<1xi64, #hipsr.mem<host>>
func.func @shape_negative_start(%ctx: !hipsr.context,
                                %input: tensor<?x?x4096xf16>)
    -> tensor<1xi64, #hipsr.mem<host>> {
  %0 = "onnx.Shape"(%input) {start = -1 : si64}
      : (tensor<?x?x4096xf16>) -> tensor<1xi64, #hipsr.mem<host>>
  return %0 : tensor<1xi64, #hipsr.mem<host>>
}

// -----

// end truncates the window, leaving out the trailing static axis.
// CHECK-LABEL:   func.func @shape_end(
// CHECK-SAME:      %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:      %[[INPUT:.*]]: tensor<?x?x4096xf16, #hipsr.mem<device>>) -> tensor<2xi64, #hipsr.mem<host>> {
// CHECK-NEXT:      %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<2xi64, #hipsr.mem<host>> shape_region {
// CHECK-NEXT:        %[[LEN:.*]] = arith.constant 2 : index
// CHECK-NEXT:        %[[SHAPE:.*]] = shape.from_extents %[[LEN]] : index
// CHECK-NEXT:        hipsr.shape_yield %[[SHAPE]] : !shape.shape
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[RESULT:.*]] = hipsr.compute(%[[CTX]]) ins(%[[INPUT]] : tensor<?x?x4096xf16, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<2xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:      ^bb0(%{{.*}}: !hipsr.context, %[[IN:.*]]: tensor<?x?x4096xf16, #hipsr.mem<device>>, %{{.*}}: tensor<2xi64, #hipsr.mem<host>>):
// CHECK-NEXT:        %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[DIM0:.*]] = tensor.dim %[[IN]], %[[C0]] : tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[EXT0:.*]] = arith.index_cast %[[DIM0]] : index to i64
// CHECK-NEXT:        %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[DIM1:.*]] = tensor.dim %[[IN]], %[[C1]] : tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[EXT1:.*]] = arith.index_cast %[[DIM1]] : index to i64
// CHECK-NEXT:        %[[EXTENTS:.*]] = tensor.from_elements %[[EXT0]], %[[EXT1]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:        hipsr.compute_yield %[[EXTENTS]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      } : tensor<2xi64, #hipsr.mem<host>>{{$}}
// CHECK-NEXT:      return %[[RESULT]] : tensor<2xi64, #hipsr.mem<host>>
func.func @shape_end(%ctx: !hipsr.context, %input: tensor<?x?x4096xf16>)
    -> tensor<2xi64, #hipsr.mem<host>> {
  %0 = "onnx.Shape"(%input) {end = 2 : si64}
      : (tensor<?x?x4096xf16>) -> tensor<2xi64, #hipsr.mem<host>>
  return %0 : tensor<2xi64, #hipsr.mem<host>>
}

// -----

// ONNX clamps a bound that runs past either end, so this selects every axis
// instead of being rejected.
// CHECK-LABEL:   func.func @shape_bounds_past_the_ends(
// CHECK-SAME:      %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:      %[[INPUT:.*]]: tensor<?x?x4096xf16, #hipsr.mem<device>>) -> tensor<3xi64, #hipsr.mem<host>> {
// CHECK-NEXT:      %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<3xi64, #hipsr.mem<host>> shape_region {
// CHECK-NEXT:        %[[LEN:.*]] = arith.constant 3 : index
// CHECK-NEXT:        %[[SHAPE:.*]] = shape.from_extents %[[LEN]] : index
// CHECK-NEXT:        hipsr.shape_yield %[[SHAPE]] : !shape.shape
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[RESULT:.*]] = hipsr.compute(%[[CTX]]) ins(%[[INPUT]] : tensor<?x?x4096xf16, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<3xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:      ^bb0(%{{.*}}: !hipsr.context, %[[IN:.*]]: tensor<?x?x4096xf16, #hipsr.mem<device>>, %{{.*}}: tensor<3xi64, #hipsr.mem<host>>):
// CHECK-NEXT:        %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[DIM0:.*]] = tensor.dim %[[IN]], %[[C0]] : tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[EXT0:.*]] = arith.index_cast %[[DIM0]] : index to i64
// CHECK-NEXT:        %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[DIM1:.*]] = tensor.dim %[[IN]], %[[C1]] : tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[EXT1:.*]] = arith.index_cast %[[DIM1]] : index to i64
// CHECK-NEXT:        %[[EXT2:.*]] = arith.constant 4096 : i64
// CHECK-NEXT:        %[[EXTENTS:.*]] = tensor.from_elements %[[EXT0]], %[[EXT1]], %[[EXT2]] : tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:        hipsr.compute_yield %[[EXTENTS]] : tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:      } : tensor<3xi64, #hipsr.mem<host>>{{$}}
// CHECK-NEXT:      return %[[RESULT]] : tensor<3xi64, #hipsr.mem<host>>
func.func @shape_bounds_past_the_ends(%ctx: !hipsr.context,
                                      %input: tensor<?x?x4096xf16>)
    -> tensor<3xi64, #hipsr.mem<host>> {
  %0 = "onnx.Shape"(%input) {start = -8 : si64, end = 8 : si64}
      : (tensor<?x?x4096xf16>) -> tensor<3xi64, #hipsr.mem<host>>
  return %0 : tensor<3xi64, #hipsr.mem<host>>
}

// -----

// A result naming no space, as an ONNX graph writes it, still puts the extents
// in host memory. Nothing reads them here, since a consumer carrying the type
// converter would ask for this result in the space that converter picks.
// CHECK-LABEL:   func.func @result_names_no_space(
// CHECK-SAME:      %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:      %[[INPUT:.*]]: tensor<2x3xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:      %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<2xi64, #hipsr.mem<host>> shape_region {
// CHECK-NEXT:        %[[LEN:.*]] = arith.constant 2 : index
// CHECK-NEXT:        %[[SHAPE:.*]] = shape.from_extents %[[LEN]] : index
// CHECK-NEXT:        hipsr.shape_yield %[[SHAPE]] : !shape.shape
// CHECK-NEXT:      }
// CHECK-NEXT:      hipsr.compute(%[[CTX]]) ins(%[[INPUT]] : tensor<2x3xf16, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<2xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:      ^bb0(%{{.*}}: !hipsr.context, %{{.*}}: tensor<2x3xf16, #hipsr.mem<device>>, %{{.*}}: tensor<2xi64, #hipsr.mem<host>>):
// CHECK-NEXT:        %[[EXT0:.*]] = arith.constant 2 : i64
// CHECK-NEXT:        %[[EXT1:.*]] = arith.constant 3 : i64
// CHECK-NEXT:        %[[EXTENTS:.*]] = tensor.from_elements %[[EXT0]], %[[EXT1]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:        hipsr.compute_yield %[[EXTENTS]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      } : tensor<2xi64, #hipsr.mem<host>>{{$}}
// CHECK-NEXT:      return{{$}}
func.func @result_names_no_space(%ctx: !hipsr.context,
                                 %input: tensor<2x3xf16>) {
  %0 = "onnx.Shape"(%input) : (tensor<2x3xf16>) -> tensor<2xi64>
  return
}

// -----

// The extents feed the shape graph as well as the data graph, and each takes a
// different value: the barrier placeholder's input names the buffer holding
// them, which is the compute's destination, while the expand reads what the
// compute wrote. Reading host extents says nothing about where the expand keeps
// its own data, which stays in the #hipsr.mem<device> its input and init
// require.
// CHECK-LABEL:   func.func @extents_feed_an_expand(
// CHECK-SAME:      %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:      %[[INPUT:.*]]: tensor<?x3xf16, #hipsr.mem<device>>) -> tensor<?x?xf16, #hipsr.mem<device>> {
// CHECK-NEXT:      %[[EXTENTS_INIT:.*]] = hipsr.placeholder(%[[CTX]]) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<2xi64, #hipsr.mem<host>> shape_region {
// CHECK-NEXT:        %[[LEN:.*]] = arith.constant 2 : index
// CHECK-NEXT:        %[[SHAPE:.*]] = shape.from_extents %[[LEN]] : index
// CHECK-NEXT:        hipsr.shape_yield %[[SHAPE]] : !shape.shape
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[RESULT:.*]] = hipsr.compute(%[[CTX]]) ins(%[[INPUT]] : tensor<?x3xf16, #hipsr.mem<device>>) outs(%[[EXTENTS_INIT]] : tensor<2xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:      ^bb0(%{{.*}}: !hipsr.context, %[[IN:.*]]: tensor<?x3xf16, #hipsr.mem<device>>, %{{.*}}: tensor<2xi64, #hipsr.mem<host>>):
// CHECK-NEXT:        %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[DIM0:.*]] = tensor.dim %[[IN]], %[[C0]] : tensor<?x3xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[EXT0:.*]] = arith.index_cast %[[DIM0]] : index to i64
// CHECK-NEXT:        %[[EXT1:.*]] = arith.constant 3 : i64
// CHECK-NEXT:        %[[EXTENTS:.*]] = tensor.from_elements %[[EXT0]], %[[EXT1]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:        hipsr.compute_yield %[[EXTENTS]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      } : tensor<2xi64, #hipsr.mem<host>>{{$}}
// CHECK-NEXT:      %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]], %[[EXTENTS_INIT]] : tensor<?x3xf16, #hipsr.mem<device>>, tensor<2xi64, #hipsr.mem<host>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<?x?xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[EXPANDED:.*]] = hipsr.expand(%[[CTX]]) ins(%[[INPUT]], %[[RESULT]] : tensor<?x3xf16, #hipsr.mem<device>>, tensor<2xi64, #hipsr.mem<host>>) outs(%[[INIT]] : tensor<?x?xf16, #hipsr.mem<device>>) : tensor<?x?xf16, #hipsr.mem<device>>
// CHECK-NEXT:      return %[[EXPANDED]] : tensor<?x?xf16, #hipsr.mem<device>>
func.func @extents_feed_an_expand(%ctx: !hipsr.context,
                                  %input: tensor<?x3xf16>) -> tensor<?x?xf16> {
  %0 = "onnx.Shape"(%input)
      : (tensor<?x3xf16>) -> tensor<2xi64, #hipsr.mem<host>>
  %1 = "onnx.Expand"(%input, %0)
      : (tensor<?x3xf16>, tensor<2xi64, #hipsr.mem<host>>) -> tensor<?x?xf16>
  return %1 : tensor<?x?xf16>
}
