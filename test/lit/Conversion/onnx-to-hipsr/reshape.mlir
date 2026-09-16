// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// onnx.Reshape becomes a hipsr.compute holding a collapse, an expand, or both,
// with a placeholder whose shape region resolves the destination. The result
// type comes from the operands, not from the type ONNX declared, and the region
// never reads memory. Rejected forms are in reshape-invalid.mlir.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --onnx-dialect=modeled --split-input-file -allow-unregistered-dialect -convert-onnx-to-hipsr | FileCheck %s

// A constant [-1] flattens a dynamic batch, as an embedding graph does to its
// image features. Nothing else is stated, so the divisor is 1, and the collapse
// alone already reaches the 1-D result.
// CHECK-LABEL: func.func @flatten_dynamic(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<?x4096xf16, #hipsr.mem<device>>) -> tensor<?xf16, #hipsr.mem<device>> {
// CHECK-NEXT:    %{{.*}} = hipsr.constant {value = dense<-1> : tensor<1xi64>} : tensor<1xi64, #hipsr.mem<device>>
// CHECK-NEXT:    %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]] : tensor<?x4096xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?xf16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:    ^bb0(%[[IN_SHAPE:.*]]: !shape.shape):
// CHECK-NEXT:      %[[COUNT:.*]] = shape.num_elements %[[IN_SHAPE]] : !shape.shape -> !shape.size
// CHECK-NEXT:      %[[DIVIDEND:.*]] = shape.size_to_index %[[COUNT]] : !shape.size
// CHECK-NEXT:      %[[DIVISOR:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[DIM:.*]] = arith.divui %[[DIVIDEND]], %[[DIVISOR]] : index
// CHECK-NEXT:      %[[SHAPE:.*]] = shape.from_extents %[[DIM]] : index
// CHECK-NEXT:      hipsr.shape_yield %[[SHAPE]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[RESULT:.*]] = hipsr.compute(%[[CTX]]) ins(%[[INPUT]] : tensor<?x4096xf16, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<?xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:    ^bb0(%{{.*}}: !hipsr.context, %[[IN:.*]]: tensor<?x4096xf16, #hipsr.mem<device>>, %{{.*}}: tensor<?xf16, #hipsr.mem<device>>):
// CHECK-NEXT:      %[[FLAT:.*]] = tensor.collapse_shape %[[IN]] {{\[}}[0, 1]] : tensor<?x4096xf16, #hipsr.mem<device>> into tensor<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.compute_yield %[[FLAT]] : tensor<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:    } : tensor<?xf16, #hipsr.mem<device>>{{$}}
// CHECK-NEXT:    return %[[RESULT]] : tensor<?xf16, #hipsr.mem<device>>
func.func @flatten_dynamic(%ctx: !hipsr.context,
                           %input: tensor<?x4096xf16>) -> tensor<?xf16> {
  %shape = "onnx.Constant"() {value = dense<-1> : tensor<1xi64>}
      : () -> tensor<1xi64>
  %0 = "onnx.Reshape"(%input, %shape) {allowzero = 0 : si64}
      : (tensor<?x4096xf16>, tensor<1xi64>) -> tensor<?xf16>
  "onnx.Return"(%0) : (tensor<?xf16>) -> ()
}

// -----

// A rank-1 input is already flat, so no collapse comes first. The expand takes
// its dynamic dimension off the destination rather than dividing again.
// CHECK-LABEL: func.func @expand_dynamic(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<?xf16, #hipsr.mem<device>>) -> tensor<?x4096xf16, #hipsr.mem<device>> {
// CHECK-NEXT:    %{{.*}} = hipsr.constant {value = dense<[-1, 4096]> : tensor<2xi64>} : tensor<2xi64, #hipsr.mem<device>>
// CHECK-NEXT:    %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]] : tensor<?xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x4096xf16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:    ^bb0(%[[IN_SHAPE:.*]]: !shape.shape):
// CHECK-NEXT:      %[[COUNT:.*]] = shape.num_elements %[[IN_SHAPE]] : !shape.shape -> !shape.size
// CHECK-NEXT:      %[[DIVIDEND:.*]] = shape.size_to_index %[[COUNT]] : !shape.size
// CHECK-NEXT:      %[[DIVISOR:.*]] = arith.constant 4096 : index
// CHECK-NEXT:      %[[ROWS:.*]] = arith.divui %[[DIVIDEND]], %[[DIVISOR]] : index
// CHECK-NEXT:      %[[COLS:.*]] = arith.constant 4096 : index
// CHECK-NEXT:      %[[SHAPE:.*]] = shape.from_extents %[[ROWS]], %[[COLS]] : index, index
// CHECK-NEXT:      hipsr.shape_yield %[[SHAPE]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[RESULT:.*]] = hipsr.compute(%[[CTX]]) ins(%[[INPUT]] : tensor<?xf16, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<?x4096xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:    ^bb0(%{{.*}}: !hipsr.context, %[[IN:.*]]: tensor<?xf16, #hipsr.mem<device>>, %[[DEST:.*]]: tensor<?x4096xf16, #hipsr.mem<device>>):
// CHECK-NEXT:      %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[DEST_ROWS:.*]] = tensor.dim %[[DEST]], %[[C0]] : tensor<?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[OUT:.*]] = tensor.expand_shape %[[IN]] {{\[}}[0, 1]] output_shape {{\[}}%[[DEST_ROWS]], 4096] : tensor<?xf16, #hipsr.mem<device>> into tensor<?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.compute_yield %[[OUT]] : tensor<?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:    } : tensor<?x4096xf16, #hipsr.mem<device>>{{$}}
// CHECK-NEXT:    return %[[RESULT]] : tensor<?x4096xf16, #hipsr.mem<device>>
func.func @expand_dynamic(%ctx: !hipsr.context,
                          %input: tensor<?xf16>) -> tensor<?x4096xf16> {
  %shape = "onnx.Constant"() {value = dense<[-1, 4096]> : tensor<2xi64>}
      : () -> tensor<2xi64>
  %0 = "onnx.Reshape"(%input, %shape) {allowzero = 0 : si64}
      : (tensor<?xf16>, tensor<2xi64>) -> tensor<?x4096xf16>
  "onnx.Return"(%0) : (tensor<?x4096xf16>) -> ()
}

// -----

// A dynamic input against a target shape that names every dimension. Neither
// collapse nor expand can make the flat dimension static, so a cast stands
// between them.
// CHECK-LABEL: func.func @cast_and_expand(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<?x4xi64, #hipsr.mem<device>>) -> tensor<6x4xi64, #hipsr.mem<device>> {
// CHECK-NEXT:    %{{.*}} = hipsr.constant {value = dense<[6, 4]> : tensor<2xi64>} : tensor<2xi64, #hipsr.mem<device>>
// CHECK-NEXT:    %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]] : tensor<?x4xi64, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<6x4xi64, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:    ^bb0(%{{.*}}: !shape.shape):
// CHECK-NEXT:      %[[ROWS:.*]] = arith.constant 6 : index
// CHECK-NEXT:      %[[COLS:.*]] = arith.constant 4 : index
// CHECK-NEXT:      %[[SHAPE:.*]] = shape.from_extents %[[ROWS]], %[[COLS]] : index, index
// CHECK-NEXT:      hipsr.shape_yield %[[SHAPE]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[RESULT:.*]] = hipsr.compute(%[[CTX]]) ins(%[[INPUT]] : tensor<?x4xi64, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<6x4xi64, #hipsr.mem<device>>) {
// CHECK-NEXT:    ^bb0(%{{.*}}: !hipsr.context, %[[IN:.*]]: tensor<?x4xi64, #hipsr.mem<device>>, %{{.*}}: tensor<6x4xi64, #hipsr.mem<device>>):
// CHECK-NEXT:      %[[FLAT:.*]] = tensor.collapse_shape %[[IN]] {{\[}}[0, 1]] : tensor<?x4xi64, #hipsr.mem<device>> into tensor<?xi64, #hipsr.mem<device>>
// CHECK-NEXT:      %[[REFINED:.*]] = tensor.cast %[[FLAT]] : tensor<?xi64, #hipsr.mem<device>> to tensor<24xi64, #hipsr.mem<device>>
// CHECK-NEXT:      %[[OUT:.*]] = tensor.expand_shape %[[REFINED]] {{\[}}[0, 1]] output_shape {{\[}}6, 4] : tensor<24xi64, #hipsr.mem<device>> into tensor<6x4xi64, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.compute_yield %[[OUT]] : tensor<6x4xi64, #hipsr.mem<device>>
// CHECK-NEXT:    } : tensor<6x4xi64, #hipsr.mem<device>>{{$}}
// CHECK-NEXT:    return %[[RESULT]] : tensor<6x4xi64, #hipsr.mem<device>>
func.func @cast_and_expand(%ctx: !hipsr.context,
                           %input: tensor<?x4xi64>) -> tensor<6x4xi64> {
  %shape = "onnx.Constant"() {value = dense<[6, 4]> : tensor<2xi64>}
      : () -> tensor<2xi64>
  %0 = "onnx.Reshape"(%input, %shape) {allowzero = 0 : si64}
      : (tensor<?x4xi64>, tensor<2xi64>) -> tensor<6x4xi64>
  "onnx.Return"(%0) : (tensor<6x4xi64>) -> ()
}

// -----

// The identity check runs on the inferred type, so a declared tensor<?x?xf16>
// cannot hide one. The reshape is replaced by its input, with no destination.
// CHECK-LABEL: func.func @refines_to_input(
// CHECK-SAME:    %{{.*}}: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<?x32xf16, #hipsr.mem<device>>) -> tensor<?x32xf16, #hipsr.mem<device>> {
// CHECK-NEXT:    %{{.*}} = hipsr.constant {value = dense<[-1, 32]> : tensor<2xi64>} : tensor<2xi64, #hipsr.mem<device>>
// CHECK-NEXT:    return %[[INPUT]] : tensor<?x32xf16, #hipsr.mem<device>>
func.func @refines_to_input(%ctx: !hipsr.context,
                            %input: tensor<?x32xf16>) -> tensor<?x?xf16> {
  %shape = "onnx.Constant"() {value = dense<[-1, 32]> : tensor<2xi64>}
      : () -> tensor<2xi64>
  %0 = "onnx.Reshape"(%input, %shape) {allowzero = 0 : si64}
      : (tensor<?x32xf16>, tensor<2xi64>) -> tensor<?x?xf16>
  "onnx.Return"(%0) : (tensor<?x?xf16>) -> ()
}

// -----

// A static input resolves the -1 at compile time, so the result type is static,
// the shape region is constants alone, and the expand states its dimensions.
// CHECK-LABEL: func.func @static_infers_dim(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<2x3x4xf16, #hipsr.mem<device>>) -> tensor<6x4xf16, #hipsr.mem<device>> {
// CHECK-NEXT:    %{{.*}} = hipsr.constant {value = dense<[-1, 4]> : tensor<2xi64>} : tensor<2xi64, #hipsr.mem<device>>
// CHECK-NEXT:    %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]] : tensor<2x3x4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<6x4xf16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:    ^bb0(%{{.*}}: !shape.shape):
// CHECK-NEXT:      %[[ROWS:.*]] = arith.constant 6 : index
// CHECK-NEXT:      %[[COLS:.*]] = arith.constant 4 : index
// CHECK-NEXT:      %[[SHAPE:.*]] = shape.from_extents %[[ROWS]], %[[COLS]] : index, index
// CHECK-NEXT:      hipsr.shape_yield %[[SHAPE]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[RESULT:.*]] = hipsr.compute(%[[CTX]]) ins(%[[INPUT]] : tensor<2x3x4xf16, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<6x4xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:    ^bb0(%{{.*}}: !hipsr.context, %[[IN:.*]]: tensor<2x3x4xf16, #hipsr.mem<device>>, %{{.*}}: tensor<6x4xf16, #hipsr.mem<device>>):
// CHECK-NEXT:      %[[FLAT:.*]] = tensor.collapse_shape %[[IN]] {{\[}}[0, 1, 2]] : tensor<2x3x4xf16, #hipsr.mem<device>> into tensor<24xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[OUT:.*]] = tensor.expand_shape %[[FLAT]] {{\[}}[0, 1]] output_shape {{\[}}6, 4] : tensor<24xf16, #hipsr.mem<device>> into tensor<6x4xf16, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.compute_yield %[[OUT]] : tensor<6x4xf16, #hipsr.mem<device>>
// CHECK-NEXT:    } : tensor<6x4xf16, #hipsr.mem<device>>{{$}}
// CHECK-NEXT:    return %[[RESULT]] : tensor<6x4xf16, #hipsr.mem<device>>
func.func @static_infers_dim(%ctx: !hipsr.context,
                             %input: tensor<2x3x4xf16>) -> tensor<?x?xf16> {
  %shape = "onnx.Constant"() {value = dense<[-1, 4]> : tensor<2xi64>}
      : () -> tensor<2xi64>
  %0 = "onnx.Reshape"(%input, %shape) {allowzero = 0 : si64}
      : (tensor<2x3x4xf16>, tensor<2xi64>) -> tensor<?x?xf16>
  "onnx.Return"(%0) : (tensor<?x?xf16>) -> ()
}
