// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// onnx.Unsqueeze becomes a hipsr.compute holding a tensor.expand_shape, with a
// placeholder whose shape region resolves the destination. The result type comes
// from the operands, not from the type ONNX declared: the axes place the unit
// dimensions and the input gives the rest. Rejected forms are in
// unsqueeze-invalid.mlir.
//
// The result aliases the input, so it keeps the input's memory space.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --onnx-dialect=modeled --split-input-file -allow-unregistered-dialect -convert-onnx-to-hipsr | FileCheck %s

// A trailing axis on a fully dynamic input, which is what an embedding graph
// inserts on a token mask. Every dynamic dimension is read off the input's
// shape, and the expand takes its dimensions off the destination.
// CHECK-LABEL: func.func @trailing_axis(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<?x?xi1, #hipsr.mem<device>>) -> tensor<?x?x1xi1, #hipsr.mem<device>> {
// CHECK-NEXT:    %{{.*}} = hipsr.constant {value = dense<-1> : tensor<1xi64>} : tensor<1xi64, #hipsr.mem<device>>
// CHECK-NEXT:    %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]] : tensor<?x?xi1, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x?x1xi1, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:    ^bb0(%[[IN_SHAPE:.*]]: tensor<2xindex>):
// CHECK-NEXT:      %[[AXIS0:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[ROWS:.*]] = shape.get_extent %[[IN_SHAPE]], %[[AXIS0]] : tensor<2xindex>, index -> index
// CHECK-NEXT:      %[[AXIS1:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[COLS:.*]] = shape.get_extent %[[IN_SHAPE]], %[[AXIS1]] : tensor<2xindex>, index -> index
// One unused divisor per dynamic group, left by the expand's shape inference.
// CHECK-NEXT:      arith.constant 1 : index
// CHECK-NEXT:      arith.constant 1 : index
// CHECK-NEXT:      %[[ONE:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[EXTENTS:.*]] = tensor.from_elements %[[ROWS]], %[[COLS]], %[[ONE]] : tensor<3xindex>
// CHECK-NEXT:      hipsr.shape_yield %[[EXTENTS]] : tensor<3xindex>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[RESULT:.*]] = hipsr.compute(%[[CTX]]) ins(%[[INPUT]] : tensor<?x?xi1, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<?x?x1xi1, #hipsr.mem<device>>) {
// CHECK-NEXT:    ^bb0(%{{.*}}: !hipsr.context, %[[IN:.*]]: tensor<?x?xi1, #hipsr.mem<device>>, %[[DEST:.*]]: tensor<?x?x1xi1, #hipsr.mem<device>>):
// CHECK-NEXT:      %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[DEST_ROWS:.*]] = tensor.dim %[[DEST]], %[[C0]] : tensor<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT:      %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[DEST_COLS:.*]] = tensor.dim %[[DEST]], %[[C1]] : tensor<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT:      %[[OUT:.*]] = tensor.expand_shape %[[IN]] {{\[}}[0], [1, 2]] output_shape {{\[}}%[[DEST_ROWS]], %[[DEST_COLS]], 1] : tensor<?x?xi1, #hipsr.mem<device>> into tensor<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.compute_yield %[[OUT]] : tensor<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT:    } : tensor<?x?x1xi1, #hipsr.mem<device>>{{$}}
// CHECK-NEXT:    return %[[RESULT]] : tensor<?x?x1xi1, #hipsr.mem<device>>
func.func @trailing_axis(%ctx: !hipsr.context,
                         %input: tensor<?x?xi1>) -> tensor<?x?x1xi1> {
  %axes = "onnx.Constant"() {value = dense<-1> : tensor<1xi64>}
      : () -> tensor<1xi64>
  %0 = "onnx.Unsqueeze"(%input, %axes)
      : (tensor<?x?xi1>, tensor<1xi64>) -> tensor<?x?x1xi1>
  "onnx.Return"(%0) : (tensor<?x?x1xi1>) -> ()
}

// -----

// The axes resolve a dimension the declared tensor<?x?x?xi1> leaves dynamic.
// The unit lands between the two dimensions the input gives, so the destination
// carries its dynamic dimensions at axes 0 and 2.
// CHECK-LABEL: func.func @interior_axis(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<?x?xi1, #hipsr.mem<device>>) -> tensor<?x1x?xi1, #hipsr.mem<device>> {
// CHECK-NEXT:    %{{.*}} = hipsr.constant {value = dense<1> : tensor<1xi64>} : tensor<1xi64, #hipsr.mem<device>>
// CHECK-NEXT:    %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]] : tensor<?x?xi1, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x1x?xi1, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:    ^bb0(%[[IN_SHAPE:.*]]: tensor<2xindex>):
// CHECK-NEXT:      %[[AXIS0:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[ROWS:.*]] = shape.get_extent %[[IN_SHAPE]], %[[AXIS0]] : tensor<2xindex>, index -> index
// CHECK-NEXT:      %[[AXIS1:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[COLS:.*]] = shape.get_extent %[[IN_SHAPE]], %[[AXIS1]] : tensor<2xindex>, index -> index
// One unused divisor per dynamic group, then the inserted unit.
// CHECK-NEXT:      arith.constant 1 : index
// CHECK-NEXT:      arith.constant 1 : index
// CHECK-NEXT:      %[[ONE:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[EXTENTS:.*]] = tensor.from_elements %[[ROWS]], %[[ONE]], %[[COLS]] : tensor<3xindex>
// CHECK-NEXT:      hipsr.shape_yield %[[EXTENTS]] : tensor<3xindex>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[RESULT:.*]] = hipsr.compute(%[[CTX]]) ins(%[[INPUT]] : tensor<?x?xi1, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<?x1x?xi1, #hipsr.mem<device>>) {
// CHECK-NEXT:    ^bb0(%{{.*}}: !hipsr.context, %[[IN:.*]]: tensor<?x?xi1, #hipsr.mem<device>>, %[[DEST:.*]]: tensor<?x1x?xi1, #hipsr.mem<device>>):
// CHECK-NEXT:      %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[DEST_ROWS:.*]] = tensor.dim %[[DEST]], %[[C0]] : tensor<?x1x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:      %[[C2:.*]] = arith.constant 2 : index
// CHECK-NEXT:      %[[DEST_COLS:.*]] = tensor.dim %[[DEST]], %[[C2]] : tensor<?x1x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:      %[[OUT:.*]] = tensor.expand_shape %[[IN]] {{\[}}[0], [1, 2]] output_shape {{\[}}%[[DEST_ROWS]], 1, %[[DEST_COLS]]] : tensor<?x?xi1, #hipsr.mem<device>> into tensor<?x1x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.compute_yield %[[OUT]] : tensor<?x1x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:    } : tensor<?x1x?xi1, #hipsr.mem<device>>{{$}}
// CHECK-NEXT:    return %[[RESULT]] : tensor<?x1x?xi1, #hipsr.mem<device>>
func.func @interior_axis(%ctx: !hipsr.context,
                         %input: tensor<?x?xi1>) -> tensor<?x?x?xi1> {
  %axes = "onnx.Constant"() {value = dense<1> : tensor<1xi64>}
      : () -> tensor<1xi64>
  %0 = "onnx.Unsqueeze"(%input, %axes)
      : (tensor<?x?xi1>, tensor<1xi64>) -> tensor<?x?x?xi1>
  "onnx.Return"(%0) : (tensor<?x?x?xi1>) -> ()
}

// -----

// A leading axis on a static host input. The chain stays in #hipsr.mem<host>,
// and a static result reads nothing off the input's shape: the region is
// constants and the expand states its dimensions.
// CHECK-LABEL: func.func @host_input(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<2xi64, #hipsr.mem<host>>) -> tensor<1x2xi64, #hipsr.mem<host>> {
// CHECK-NEXT:    %{{.*}} = hipsr.constant {value = dense<0> : tensor<1xi64>} : tensor<1xi64, #hipsr.mem<device>>
// CHECK-NEXT:    %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]] : tensor<2xi64, #hipsr.mem<host>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<1x2xi64, #hipsr.mem<host>> shape_region {
// CHECK-NEXT:    ^bb0(%{{.*}}: tensor<1xindex>):
// CHECK-NEXT:      %[[D0:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[D1:.*]] = arith.constant 2 : index
// CHECK-NEXT:      %[[EXTENTS:.*]] = tensor.from_elements %[[D0]], %[[D1]] : tensor<2xindex>
// CHECK-NEXT:      hipsr.shape_yield %[[EXTENTS]] : tensor<2xindex>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[RESULT:.*]] = hipsr.compute(%[[CTX]]) ins(%[[INPUT]] : tensor<2xi64, #hipsr.mem<host>>) outs(%[[INIT]] : tensor<1x2xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:    ^bb0(%{{.*}}: !hipsr.context, %[[IN:.*]]: tensor<2xi64, #hipsr.mem<host>>, %{{.*}}: tensor<1x2xi64, #hipsr.mem<host>>):
// CHECK-NEXT:      %[[OUT:.*]] = tensor.expand_shape %[[IN]] {{\[}}[0, 1]] output_shape {{\[}}1, 2] : tensor<2xi64, #hipsr.mem<host>> into tensor<1x2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      hipsr.compute_yield %[[OUT]] : tensor<1x2xi64, #hipsr.mem<host>>
// CHECK-NEXT:    } : tensor<1x2xi64, #hipsr.mem<host>>{{$}}
// CHECK-NEXT:    return %[[RESULT]] : tensor<1x2xi64, #hipsr.mem<host>>
func.func @host_input(%ctx: !hipsr.context,
                      %input: tensor<2xi64, #hipsr.mem<host>>)
    -> tensor<1x2xi64> {
  %axes = "onnx.Constant"() {value = dense<0> : tensor<1xi64>}
      : () -> tensor<1xi64>
  %0 = "onnx.Unsqueeze"(%input, %axes)
      : (tensor<2xi64, #hipsr.mem<host>>, tensor<1xi64>) -> tensor<1x2xi64>
  "onnx.Return"(%0) : (tensor<1x2xi64>) -> ()
}

// -----

// Two axes given out of order, which ONNX allows. They bracket the input, so
// each of its dimensions groups with one unit.
// CHECK-LABEL: func.func @two_axes(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<3x4xf16, #hipsr.mem<device>>) -> tensor<1x3x4x1xf16, #hipsr.mem<device>> {
// CHECK-NEXT:    %{{.*}} = hipsr.constant {value = dense<[3, 0]> : tensor<2xi64>} : tensor<2xi64, #hipsr.mem<device>>
// CHECK-NEXT:    %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]] : tensor<3x4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<1x3x4x1xf16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:    ^bb0(%{{.*}}: tensor<2xindex>):
// CHECK-NEXT:      %[[D0:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[D1:.*]] = arith.constant 3 : index
// CHECK-NEXT:      %[[D2:.*]] = arith.constant 4 : index
// CHECK-NEXT:      %[[D3:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[EXTENTS:.*]] = tensor.from_elements %[[D0]], %[[D1]], %[[D2]], %[[D3]] : tensor<4xindex>
// CHECK-NEXT:      hipsr.shape_yield %[[EXTENTS]] : tensor<4xindex>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[RESULT:.*]] = hipsr.compute(%[[CTX]]) ins(%[[INPUT]] : tensor<3x4xf16, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<1x3x4x1xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:    ^bb0(%{{.*}}: !hipsr.context, %[[IN:.*]]: tensor<3x4xf16, #hipsr.mem<device>>, %{{.*}}: tensor<1x3x4x1xf16, #hipsr.mem<device>>):
// CHECK-NEXT:      %[[OUT:.*]] = tensor.expand_shape %[[IN]] {{\[}}[0, 1], [2, 3]] output_shape {{\[}}1, 3, 4, 1] : tensor<3x4xf16, #hipsr.mem<device>> into tensor<1x3x4x1xf16, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.compute_yield %[[OUT]] : tensor<1x3x4x1xf16, #hipsr.mem<device>>
// CHECK-NEXT:    } : tensor<1x3x4x1xf16, #hipsr.mem<device>>{{$}}
// CHECK-NEXT:    return %[[RESULT]] : tensor<1x3x4x1xf16, #hipsr.mem<device>>
func.func @two_axes(%ctx: !hipsr.context,
                    %input: tensor<3x4xf16>) -> tensor<1x3x4x1xf16> {
  %axes = "onnx.Constant"() {value = dense<[3, 0]> : tensor<2xi64>}
      : () -> tensor<2xi64>
  %0 = "onnx.Unsqueeze"(%input, %axes)
      : (tensor<3x4xf16>, tensor<2xi64>) -> tensor<1x3x4x1xf16>
  "onnx.Return"(%0) : (tensor<1x3x4x1xf16>) -> ()
}

// -----

// Empty axes insert nothing, so the input stands as the result and no
// destination is built.
// CHECK-LABEL: func.func @no_axes(
// CHECK-SAME:    %{{.*}}: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<2x3xf16, #hipsr.mem<device>>) -> tensor<2x3xf16, #hipsr.mem<device>> {
// CHECK-NEXT:    %{{.*}} = hipsr.constant {value = dense<> : tensor<0xi64>} : tensor<0xi64, #hipsr.mem<device>>
// CHECK-NEXT:    return %[[INPUT]] : tensor<2x3xf16, #hipsr.mem<device>>
func.func @no_axes(%ctx: !hipsr.context,
                   %input: tensor<2x3xf16>) -> tensor<2x3xf16> {
  %axes = "onnx.Constant"() {value = dense<> : tensor<0xi64>}
      : () -> tensor<0xi64>
  %0 = "onnx.Unsqueeze"(%input, %axes)
      : (tensor<2x3xf16>, tensor<0xi64>) -> tensor<2x3xf16>
  "onnx.Return"(%0) : (tensor<2x3xf16>) -> ()
}
