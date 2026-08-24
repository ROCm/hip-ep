// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// onnx.Reshape becomes a hipsr.compute holding a tensor.collapse_shape, a
// tensor.expand_shape, or both, with a placeholder whose shape region resolves
// the destination. Rejected forms are in reshape-invalid.mlir.
//
// The result type is inferred from the operands: a target shape operand that
// folds to a constant states the dimensions, and the input states the element
// type and memory space. The shape region never reads memory.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --onnx-dialect=modeled --split-input-file -allow-unregistered-dialect -convert-onnx-to-hipsr | FileCheck %s

// How an embedding graph flattens its image features: a constant [-1] against
// a batch that stays dynamic. Nothing else is stated, so the inferred dimension
// is the whole element count over a divisor of 1, and the body stops at the
// collapse that already reaches the 1-D result.
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

// Expanding splits the flat form over several dimensions, and the input type
// does not say what they are. The destination already carries what the shape
// region resolved, so output_shape reads them off there rather than dividing
// again. A rank-1 input is already flat, so no collapse comes first.
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

// The input leaves the element count dynamic while the target shape names every
// dimension. Neither reshape op can make that flat dimension static: a
// collapsed dimension stays dynamic with its group, and an expansion needs a
// static source to give a static result. A cast in between does it. The
// destination is static, so the expansion states its dimensions outright.
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

// The identity check runs on the inferred type, so the declared type cannot
// hide one: [-1, 32] against a rank-2 input names the input's own dimensions
// back. The reshape is replaced by its input, and no destination is built.
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
// type, which leaves both dimensions dynamic here. The operand names the second
// one, and the inferred type is what the compute carries, so it reaches the
// signature instead of being widened back to the one ONNX declared.
//
// Regrouping 8x4 into 32 takes both a collapse and an expand. An expansion
// cannot take its dimensions from the flat form, so it reads them off the
// destination the shape region resolved.
// CHECK-LABEL: func.func @collapse_and_expand(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<?x8x4xf16, #hipsr.mem<device>>) -> tensor<?x32xf16, #hipsr.mem<device>> {
// CHECK-NEXT:    %{{.*}} = hipsr.constant {value = dense<[-1, 32]> : tensor<2xi64>} : tensor<2xi64, #hipsr.mem<device>>
// CHECK-NEXT:    %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]] : tensor<?x8x4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x32xf16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:    ^bb0(%[[IN_SHAPE:.*]]: !shape.shape):
// CHECK-NEXT:      %[[COUNT:.*]] = shape.num_elements %[[IN_SHAPE]] : !shape.shape -> !shape.size
// CHECK-NEXT:      %[[DIVIDEND:.*]] = shape.size_to_index %[[COUNT]] : !shape.size
// CHECK-NEXT:      %[[DIVISOR:.*]] = arith.constant 32 : index
// CHECK-NEXT:      %[[ROWS:.*]] = arith.divui %[[DIVIDEND]], %[[DIVISOR]] : index
// CHECK-NEXT:      %[[COLS:.*]] = arith.constant 32 : index
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

// -----

// The same [-1, N] form as above, but against an input that states its element
// count. The -1 is resolved while the type is inferred, so nothing about the
// destination is left to runtime: the shape region is constants alone, and the
// expansion states its dimensions rather than reading them back. Neither
// dimension comes from the type ONNX declared -- the operand states the 4 and
// the input's count gives the 6.
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
