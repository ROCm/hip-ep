// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// onnx.Reshape becomes a hipsr.compute holding a tensor.collapse_shape and a
// tensor.expand_shape, with the placeholder's shape region filled in as the
// body is built. Rejected forms are in reshape-invalid.mlir.
//
// The result type is inferred from the operands: a target shape operand that
// folds to a constant states the extents, and the input states the element type
// and memory space. The shape region never reads memory.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --onnx-dialect=modeled --split-input-file -allow-unregistered-dialect -convert-onnx-to-hipsr | FileCheck %s

// The reshape an embedding graph flattens its image features with: a constant
// [-1] against a batch that stays symbolic. Nothing else is stated, so the
// inferred extent is the whole element count with no divisor, and the body
// stops at the collapse that already reaches the 1-D result.
// CHECK-LABEL: func.func @flatten_dynamic(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<?x4096xf16, #hipsr.mem<device>>) -> tensor<?xf16, #hipsr.mem<device>> {
// CHECK-NEXT:    %{{.*}} = hipsr.constant {value = dense<-1> : tensor<1xi64>} : tensor<1xi64, #hipsr.mem<device>>
// CHECK-NEXT:    %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]] : tensor<?x4096xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?xf16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:    ^bb0(%[[IN_SHAPE:.*]]: !shape.shape):
// CHECK-NEXT:      %[[AXIS:.*]] = shape.const_size 0
// CHECK-NEXT:      %[[SIZE:.*]] = shape.get_extent %[[IN_SHAPE]], %[[AXIS]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:      %[[ROWS:.*]] = shape.size_to_index %[[SIZE]] : !shape.size
// CHECK-NEXT:      %[[COLS:.*]] = arith.constant 4096 : index
// CHECK-NEXT:      %[[COUNT:.*]] = arith.muli %[[ROWS]], %[[COLS]] : index
// CHECK-NEXT:      %[[SHAPE:.*]] = shape.from_extents %[[COUNT]] : index
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

// Expanding splits one input extent over a group, which the input type cannot
// pin down. The destination already carries what the shape region resolved, so
// output_shape reads the extent off it rather than dividing for it again. A
// rank-1 input is already flat, so no collapse comes first.
// CHECK-LABEL: func.func @expand_dynamic(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<?xf16, #hipsr.mem<device>>) -> tensor<?x4096xf16, #hipsr.mem<device>> {
// CHECK-NEXT:    %{{.*}} = hipsr.constant {value = dense<[-1, 4096]> : tensor<2xi64>} : tensor<2xi64, #hipsr.mem<device>>
// CHECK-NEXT:    %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]] : tensor<?xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x4096xf16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:    ^bb0(%[[IN_SHAPE:.*]]: !shape.shape):
// CHECK-NEXT:      %[[COLS:.*]] = arith.constant 4096 : index
// CHECK-NEXT:      %[[AXIS:.*]] = shape.const_size 0
// CHECK-NEXT:      %[[SIZE:.*]] = shape.get_extent %[[IN_SHAPE]], %[[AXIS]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:      %[[COUNT:.*]] = shape.size_to_index %[[SIZE]] : !shape.size
// CHECK-NEXT:      %[[DIVISOR:.*]] = arith.constant 4096 : index
// CHECK-NEXT:      %[[ROWS:.*]] = arith.divui %[[COUNT]], %[[DIVISOR]] : index
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

// The input leaves the element count dynamic while the target shape pins it
// down. Neither reshape op takes that mismatch across a group, so a cast
// refines the flat form first. The result is already 1-D, so no expansion
// follows.
// CHECK-LABEL: func.func @collapse_and_cast(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<?x4xi64, #hipsr.mem<device>>) -> tensor<24xi64, #hipsr.mem<device>> {
// CHECK-NEXT:    %{{.*}} = hipsr.constant {value = dense<24> : tensor<1xi64>} : tensor<1xi64, #hipsr.mem<device>>
// CHECK-NEXT:    %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]] : tensor<?x4xi64, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<24xi64, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:    ^bb0(%{{.*}}: !shape.shape):
// CHECK-NEXT:      %[[COUNT:.*]] = arith.constant 24 : index
// CHECK-NEXT:      %[[SHAPE:.*]] = shape.from_extents %[[COUNT]] : index
// CHECK-NEXT:      hipsr.shape_yield %[[SHAPE]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[RESULT:.*]] = hipsr.compute(%[[CTX]]) ins(%[[INPUT]] : tensor<?x4xi64, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<24xi64, #hipsr.mem<device>>) {
// CHECK-NEXT:    ^bb0(%{{.*}}: !hipsr.context, %[[IN:.*]]: tensor<?x4xi64, #hipsr.mem<device>>, %{{.*}}: tensor<24xi64, #hipsr.mem<device>>):
// CHECK-NEXT:      %[[FLAT:.*]] = tensor.collapse_shape %[[IN]] {{\[}}[0, 1]] : tensor<?x4xi64, #hipsr.mem<device>> into tensor<?xi64, #hipsr.mem<device>>
// CHECK-NEXT:      %[[REFINED:.*]] = tensor.cast %[[FLAT]] : tensor<?xi64, #hipsr.mem<device>> to tensor<24xi64, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.compute_yield %[[REFINED]] : tensor<24xi64, #hipsr.mem<device>>
// CHECK-NEXT:    } : tensor<24xi64, #hipsr.mem<device>>{{$}}
// CHECK-NEXT:    return %[[RESULT]] : tensor<24xi64, #hipsr.mem<device>>
func.func @collapse_and_cast(%ctx: !hipsr.context,
                             %input: tensor<?x4xi64>) -> tensor<24xi64> {
  %shape = "onnx.Constant"() {value = dense<24> : tensor<1xi64>}
      : () -> tensor<1xi64>
  %0 = "onnx.Reshape"(%input, %shape) {allowzero = 0 : si64}
      : (tensor<?x4xi64>, tensor<1xi64>) -> tensor<24xi64>
  "onnx.Return"(%0) : (tensor<24xi64>) -> ()
}
// -----

// The identity check runs on the inferred type, so it catches the reshapes only
// the target shape reveals to be identities: [-1, 32] against a rank-2 input
// names the extents back to the input's own. The reshape is replaced by its
// input, leaving no destination.
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

// Shape inference does not always fold a constant shape operand into the result
// type, which leaves both extents dynamic here. The operand names the second
// one, and the inferred type is what the compute carries, so it reaches the
// signature instead of being widened back to the one ONNX declared.
//
// The stated 32 and the input's 8x4 cancel, so the inferred extent is the batch
// read straight off the input's shape with nothing left to multiply. An
// expansion cannot take its extents from the flat form, so it reads them off
// the destination the shape region resolved.
// CHECK-LABEL: func.func @collapse_and_expand(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<?x8x4xf16, #hipsr.mem<device>>) -> tensor<?x32xf16, #hipsr.mem<device>> {
// CHECK-NEXT:    %{{.*}} = hipsr.constant {value = dense<[-1, 32]> : tensor<2xi64>} : tensor<2xi64, #hipsr.mem<device>>
// CHECK-NEXT:    %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]] : tensor<?x8x4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x32xf16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:    ^bb0(%[[IN_SHAPE:.*]]: !shape.shape):
// CHECK-NEXT:      %[[COLS:.*]] = arith.constant 32 : index
// CHECK-NEXT:      %[[AXIS:.*]] = shape.const_size 0
// CHECK-NEXT:      %[[SIZE:.*]] = shape.get_extent %[[IN_SHAPE]], %[[AXIS]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:      %[[ROWS:.*]] = shape.size_to_index %[[SIZE]] : !shape.size
// CHECK-NEXT:      %[[SHAPE:.*]] = shape.from_extents %[[ROWS]], %[[COLS]] : index, index
// CHECK-NEXT:      hipsr.shape_yield %[[SHAPE]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[RESULT:.*]] = hipsr.compute(%[[CTX]]) ins(%[[INPUT]] : tensor<?x8x4xf16, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<?x32xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:    ^bb0(%{{.*}}: !hipsr.context, %[[IN:.*]]: tensor<?x8x4xf16, #hipsr.mem<device>>, %[[DEST:.*]]: tensor<?x32xf16, #hipsr.mem<device>>):
// CHECK-NEXT:      %[[FLAT:.*]] = tensor.collapse_shape %[[IN]] {{\[}}[0, 1, 2]] : tensor<?x8x4xf16, #hipsr.mem<device>> into tensor<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[DEST_ROWS:.*]] = tensor.dim %[[DEST]], %[[C0]] : tensor<?x32xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[OUT:.*]] = tensor.expand_shape %[[FLAT]] {{\[}}[0, 1]] output_shape {{\[}}%[[DEST_ROWS]], 32] : tensor<?xf16, #hipsr.mem<device>> into tensor<?x32xf16, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.compute_yield %[[OUT]] : tensor<?x32xf16, #hipsr.mem<device>>
// CHECK-NEXT:    } : tensor<?x32xf16, #hipsr.mem<device>>{{$}}
// CHECK-NEXT:    return %[[RESULT]] : tensor<?x32xf16, #hipsr.mem<device>>
func.func @collapse_and_expand(%ctx: !hipsr.context,
                               %input: tensor<?x8x4xf16>) -> tensor<?x?xf16> {
  %shape = "onnx.Constant"() {value = dense<[-1, 32]> : tensor<2xi64>}
      : () -> tensor<2xi64>
  %0 = "onnx.Reshape"(%input, %shape) {allowzero = 0 : si64}
      : (tensor<?x8x4xf16>, tensor<2xi64>) -> tensor<?x?xf16>
  "onnx.Return"(%0) : (tensor<?x?xf16>) -> ()
}
