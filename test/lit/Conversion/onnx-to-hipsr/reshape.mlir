// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// onnx.Reshape becomes a hipsr.compute holding a tensor.collapse_shape and a
// tensor.expand_shape, with the placeholder's shape region filled in as the
// body is built. Rejected forms are in reshape-invalid.mlir.
//
// The conversion does not look for the single grouping that would express a
// regroup on its own. reshape-compose.mlir checks that canonicalization
// recovers it where one exists.
//
// The destination comes either from the result type plus a target shape operand
// that folds to a constant, or at runtime from a host operand the shape region
// reads.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --onnx-dialect=modeled --split-input-file -allow-unregistered-dialect -convert-onnx-to-hipsr | FileCheck %s

// A shape operand left as a runtime value goes unread: the result type already
// carries the extents, and a dynamic one follows from the preserved element
// count.
// CHECK-LABEL: func.func @collapse_dynamic(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<?x4096xf16, #hipsr.mem<device>>,
// CHECK-SAME:    %{{.*}}: tensor<1xi64, #hipsr.mem<device>>) -> tensor<?xf16, #hipsr.mem<device>> {
// CHECK-NEXT:    %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]] : tensor<?x4096xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?xf16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:    ^bb0(%[[IN_SHAPE:.*]]: !shape.shape):
// CHECK-NEXT:      %[[AXIS:.*]] = shape.const_size 0
// CHECK-NEXT:      %[[SIZE:.*]] = shape.get_extent %[[IN_SHAPE]], %[[AXIS]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:      %[[EXTENT:.*]] = shape.size_to_index %[[SIZE]] : !shape.size
// CHECK-NEXT:      %[[COLS:.*]] = arith.constant 4096 : index
// CHECK-NEXT:      %[[COUNT:.*]] = arith.muli %[[EXTENT]], %[[COLS]] : index
// CHECK-NEXT:      %[[SHAPE:.*]] = shape.from_extents %[[COUNT]] : index
// CHECK-NEXT:      hipsr.shape_yield %[[SHAPE]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[RESULT:.*]] = hipsr.compute(%[[CTX]]) ins(%[[INPUT]] : tensor<?x4096xf16, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<?xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:    ^bb0(%{{.*}}: !hipsr.context, %[[IN:.*]]: tensor<?x4096xf16, #hipsr.mem<device>>, %{{.*}}: tensor<?xf16, #hipsr.mem<device>>):
// CHECK-NEXT:      %[[FLAT:.*]] = tensor.collapse_shape %[[IN]] {{\[}}[0, 1]] : tensor<?x4096xf16, #hipsr.mem<device>> into tensor<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.compute_yield %[[FLAT]] : tensor<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:    } : tensor<?xf16, #hipsr.mem<device>>{{$}}
// CHECK-NEXT:    return %[[RESULT]] : tensor<?xf16, #hipsr.mem<device>>
func.func @collapse_dynamic(%ctx: !hipsr.context, %input: tensor<?x4096xf16>,
                            %shape: tensor<1xi64>) -> tensor<?xf16> {
  %0 = "onnx.Reshape"(%input, %shape) {allowzero = 0 : si64}
      : (tensor<?x4096xf16>, tensor<1xi64>) -> tensor<?xf16>
  "onnx.Return"(%0) : (tensor<?xf16>) -> ()
}

// -----

// Static shapes need no extent arithmetic, so the region is two constants. The
// body is the 1-D pair, the same as every other regroup.
// CHECK-LABEL: func.func @static_regroup(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<2x3x4xf16, #hipsr.mem<device>>,
// CHECK-SAME:    %{{.*}}: tensor<2xi64, #hipsr.mem<device>>) -> tensor<6x4xf16, #hipsr.mem<device>> {
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
// CHECK-NEXT:      %[[OUT:.*]] = tensor.expand_shape %[[FLAT]] {{\[}}[0, 1]] output_shape [6, 4] : tensor<24xf16, #hipsr.mem<device>> into tensor<6x4xf16, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.compute_yield %[[OUT]] : tensor<6x4xf16, #hipsr.mem<device>>
// CHECK-NEXT:    } : tensor<6x4xf16, #hipsr.mem<device>>{{$}}
// CHECK-NEXT:    return %[[RESULT]] : tensor<6x4xf16, #hipsr.mem<device>>
func.func @static_regroup(%ctx: !hipsr.context, %input: tensor<2x3x4xf16>,
                          %shape: tensor<2xi64>) -> tensor<6x4xf16> {
  %0 = "onnx.Reshape"(%input, %shape)
      : (tensor<2x3x4xf16>, tensor<2xi64>) -> tensor<6x4xf16>
  "onnx.Return"(%0) : (tensor<6x4xf16>) -> ()
}

// -----

// Expanding splits one input extent over a group, which the input type cannot
// pin down. The destination already carries what the shape region resolved, so
// output_shape reads the extent off it rather than dividing for it again. A
// rank-1 input is already flat, so no collapse comes first.
// CHECK-LABEL: func.func @expand_dynamic(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<?xf16, #hipsr.mem<device>>,
// CHECK-SAME:    %{{.*}}: tensor<2xi64, #hipsr.mem<device>>) -> tensor<?x4096xf16, #hipsr.mem<device>> {
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
func.func @expand_dynamic(%ctx: !hipsr.context, %input: tensor<?xf16>,
                          %shape: tensor<2xi64>) -> tensor<?x4096xf16> {
  %0 = "onnx.Reshape"(%input, %shape)
      : (tensor<?xf16>, tensor<2xi64>) -> tensor<?x4096xf16>
  "onnx.Return"(%0) : (tensor<?x4096xf16>) -> ()
}

// -----

// The input leaves the element count dynamic while the result pins it down.
// Neither reshape op takes that mismatch across a group, so a cast refines the
// flat form first. The result is already 1-D, so no expansion follows.
// CHECK-LABEL: func.func @regroup_refined(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<?x4xf16, #hipsr.mem<device>>,
// CHECK-SAME:    %{{.*}}: tensor<1xi64, #hipsr.mem<device>>) -> tensor<24xf16, #hipsr.mem<device>> {
// CHECK-NEXT:    %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]] : tensor<?x4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<24xf16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:    ^bb0(%{{.*}}: !shape.shape):
// CHECK-NEXT:      %[[COUNT:.*]] = arith.constant 24 : index
// CHECK-NEXT:      %[[SHAPE:.*]] = shape.from_extents %[[COUNT]] : index
// CHECK-NEXT:      hipsr.shape_yield %[[SHAPE]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[RESULT:.*]] = hipsr.compute(%[[CTX]]) ins(%[[INPUT]] : tensor<?x4xf16, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<24xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:    ^bb0(%{{.*}}: !hipsr.context, %[[IN:.*]]: tensor<?x4xf16, #hipsr.mem<device>>, %{{.*}}: tensor<24xf16, #hipsr.mem<device>>):
// CHECK-NEXT:      %[[FLAT:.*]] = tensor.collapse_shape %[[IN]] {{\[}}[0, 1]] : tensor<?x4xf16, #hipsr.mem<device>> into tensor<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[REFINED:.*]] = tensor.cast %[[FLAT]] : tensor<?xf16, #hipsr.mem<device>> to tensor<24xf16, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.compute_yield %[[REFINED]] : tensor<24xf16, #hipsr.mem<device>>
// CHECK-NEXT:    } : tensor<24xf16, #hipsr.mem<device>>{{$}}
// CHECK-NEXT:    return %[[RESULT]] : tensor<24xf16, #hipsr.mem<device>>
func.func @regroup_refined(%ctx: !hipsr.context, %input: tensor<?x4xf16>,
                           %shape: tensor<1xi64>) -> tensor<24xf16> {
  %0 = "onnx.Reshape"(%input, %shape)
      : (tensor<?x4xf16>, tensor<1xi64>) -> tensor<24xf16>
  "onnx.Return"(%0) : (tensor<24xf16>) -> ()
}

// -----

// A rank-0 input has no axis to collapse, so there is no 1-D form to go through
// and the expansion regroups it on its own. Such an input also names no memory
// space, so the result falls back to device.
// CHECK-LABEL: func.func @rank0_input(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<f16>) -> tensor<1xf16, #hipsr.mem<device>> {
// CHECK-NEXT:    %{{.*}} = hipsr.constant {value = dense<1> : tensor<1xi64>} : tensor<1xi64, #hipsr.mem<device>>
// CHECK-NEXT:    %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]] : tensor<f16>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<1xf16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:    ^bb0(%{{.*}}: !shape.shape):
// CHECK-NEXT:      %[[EXTENT:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[SHAPE:.*]] = shape.from_extents %[[EXTENT]] : index
// CHECK-NEXT:      hipsr.shape_yield %[[SHAPE]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[RESULT:.*]] = hipsr.compute(%[[CTX]]) ins(%[[INPUT]] : tensor<f16>) outs(%[[INIT]] : tensor<1xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:    ^bb0(%{{.*}}: !hipsr.context, %[[IN:.*]]: tensor<f16>, %{{.*}}: tensor<1xf16, #hipsr.mem<device>>):
// CHECK-NEXT:      %[[OUT:.*]] = tensor.expand_shape %[[IN]] [] output_shape [1] : tensor<f16> into tensor<1xf16, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.compute_yield %[[OUT]] : tensor<1xf16, #hipsr.mem<device>>
// CHECK-NEXT:    } : tensor<1xf16, #hipsr.mem<device>>{{$}}
// CHECK-NEXT:    return %[[RESULT]] : tensor<1xf16, #hipsr.mem<device>>
func.func @rank0_input(%ctx: !hipsr.context,
                       %input: tensor<f16>) -> tensor<1xf16> {
  %shape = "onnx.Constant"() {value = dense<1> : tensor<1xi64>}
      : () -> tensor<1xi64>
  %0 = "onnx.Reshape"(%input, %shape) {allowzero = 0 : si64}
      : (tensor<f16>, tensor<1xi64>) -> tensor<1xf16>
  "onnx.Return"(%0) : (tensor<1xf16>) -> ()
}

// -----

// Shape inference does not always fold a constant shape operand into the result
// type, which leaves both extents dynamic here. The operand still names the
// first one, and without it the expansion would have two unknowns in one group.
// That refinement is the type the compute carries, so it reaches the signature
// instead of being widened back to the one ONNX declared.
// CHECK-LABEL: func.func @constant_shape(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<?x?x4xf16, #hipsr.mem<device>>) -> tensor<2x?xf16, #hipsr.mem<device>> {
// CHECK-NEXT:    %{{.*}} = hipsr.constant {value = dense<[2, -1]> : tensor<2xi64>} : tensor<2xi64, #hipsr.mem<device>>
// CHECK-NEXT:    %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]] : tensor<?x?x4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<2x?xf16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:    ^bb0(%[[IN_SHAPE:.*]]: !shape.shape):
// CHECK-NEXT:      %[[ROWS:.*]] = arith.constant 2 : index
// CHECK-NEXT:      %[[AXIS0:.*]] = shape.const_size 0
// CHECK-NEXT:      %[[SIZE0:.*]] = shape.get_extent %[[IN_SHAPE]], %[[AXIS0]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:      %[[D0:.*]] = shape.size_to_index %[[SIZE0]] : !shape.size
// CHECK-NEXT:      %[[AXIS1:.*]] = shape.const_size 1
// CHECK-NEXT:      %[[SIZE1:.*]] = shape.get_extent %[[IN_SHAPE]], %[[AXIS1]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:      %[[D1:.*]] = shape.size_to_index %[[SIZE1]] : !shape.size
// CHECK-NEXT:      %[[D2:.*]] = arith.constant 4 : index
// CHECK-NEXT:      %[[PARTIAL:.*]] = arith.muli %[[D0]], %[[D1]] : index
// CHECK-NEXT:      %[[COUNT:.*]] = arith.muli %[[PARTIAL]], %[[D2]] : index
// CHECK-NEXT:      %[[DIVISOR:.*]] = arith.constant 2 : index
// CHECK-NEXT:      %[[COLS:.*]] = arith.divui %[[COUNT]], %[[DIVISOR]] : index
// CHECK-NEXT:      %[[SHAPE:.*]] = shape.from_extents %[[ROWS]], %[[COLS]] : index, index
// CHECK-NEXT:      hipsr.shape_yield %[[SHAPE]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[RESULT:.*]] = hipsr.compute(%[[CTX]]) ins(%[[INPUT]] : tensor<?x?x4xf16, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<2x?xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:    ^bb0(%{{.*}}: !hipsr.context, %[[IN:.*]]: tensor<?x?x4xf16, #hipsr.mem<device>>, %[[DEST:.*]]: tensor<2x?xf16, #hipsr.mem<device>>):
// CHECK-NEXT:      %[[FLAT:.*]] = tensor.collapse_shape %[[IN]] {{\[}}[0, 1, 2]] : tensor<?x?x4xf16, #hipsr.mem<device>> into tensor<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[DEST_COLS:.*]] = tensor.dim %[[DEST]], %[[C1]] : tensor<2x?xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[OUT:.*]] = tensor.expand_shape %[[FLAT]] {{\[}}[0, 1]] output_shape [2, %[[DEST_COLS]]] : tensor<?xf16, #hipsr.mem<device>> into tensor<2x?xf16, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.compute_yield %[[OUT]] : tensor<2x?xf16, #hipsr.mem<device>>
// CHECK-NEXT:    } : tensor<2x?xf16, #hipsr.mem<device>>{{$}}
// CHECK-NEXT:    return %[[RESULT]] : tensor<2x?xf16, #hipsr.mem<device>>
func.func @constant_shape(%ctx: !hipsr.context,
                          %input: tensor<?x?x4xf16>) -> tensor<?x?xf16> {
  %shape = "onnx.Constant"() {value = dense<[2, -1]> : tensor<2xi64>}
      : () -> tensor<2xi64>
  %0 = "onnx.Reshape"(%input, %shape) {allowzero = 0 : si64}
      : (tensor<?x?x4xf16>, tensor<2xi64>) -> tensor<?x?xf16>
  "onnx.Return"(%0) : (tensor<?x?xf16>) -> ()
}

// -----

// A 0 keeps the input's extent and a -1 takes the rest, which is how ONNX
// graphs flatten everything but the batch. The copied axis cancels out of the
// element count, so the inferred extent folds to a constant even though the
// axis beside it stays dynamic, and no division survives.
// CHECK-LABEL: func.func @copy_and_infer(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<?x8x4xf16, #hipsr.mem<device>>) -> tensor<?x32xf16, #hipsr.mem<device>> {
// CHECK-NEXT:    %{{.*}} = hipsr.constant {value = dense<[0, -1]> : tensor<2xi64>} : tensor<2xi64, #hipsr.mem<device>>
// CHECK-NEXT:    %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]] : tensor<?x8x4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x32xf16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:    ^bb0(%[[IN_SHAPE:.*]]: !shape.shape):
// CHECK-NEXT:      %[[AXIS:.*]] = shape.const_size 0
// CHECK-NEXT:      %[[SIZE:.*]] = shape.get_extent %[[IN_SHAPE]], %[[AXIS]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:      %[[ROWS:.*]] = shape.size_to_index %[[SIZE]] : !shape.size
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
func.func @copy_and_infer(%ctx: !hipsr.context,
                          %input: tensor<?x8x4xf16>) -> tensor<?x?xf16> {
  %shape = "onnx.Constant"() {value = dense<[0, -1]> : tensor<2xi64>}
      : () -> tensor<2xi64>
  %0 = "onnx.Reshape"(%input, %shape) {allowzero = 0 : si64}
      : (tensor<?x8x4xf16>, tensor<2xi64>) -> tensor<?x?xf16>
  "onnx.Return"(%0) : (tensor<?x?xf16>) -> ()
}

// -----

// The identity check runs on the refined type, so it covers both a declared
// type that already matches the input and this one, where only the operand
// shows it: [0, -1] against a rank-2 input names the extents back to the
// input's own. The reshape is replaced by its input, leaving no destination.
// CHECK-LABEL: func.func @refines_to_input(
// CHECK-SAME:    %{{.*}}: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<?x32xf16, #hipsr.mem<device>>) -> tensor<?x32xf16, #hipsr.mem<device>> {
// CHECK-NEXT:    %{{.*}} = hipsr.constant {value = dense<[0, -1]> : tensor<2xi64>} : tensor<2xi64, #hipsr.mem<device>>
// CHECK-NEXT:    return %[[INPUT]] : tensor<?x32xf16, #hipsr.mem<device>>
func.func @refines_to_input(%ctx: !hipsr.context,
                            %input: tensor<?x32xf16>) -> tensor<?x?xf16> {
  %shape = "onnx.Constant"() {value = dense<[0, -1]> : tensor<2xi64>}
      : () -> tensor<2xi64>
  %0 = "onnx.Reshape"(%input, %shape) {allowzero = 0 : si64}
      : (tensor<?x32xf16>, tensor<2xi64>) -> tensor<?x?xf16>
  "onnx.Return"(%0) : (tensor<?x?xf16>) -> ()
}

// -----

// Two extents nothing names leave the target shape operand as the only place
// they are, so the shape region reads it. That is a host read, hence the
// barrier layout, which hands the region the operands themselves rather than
// their shapes. A 0 still copies the input's extent and a -1 is still the
// element count divided by the rest, only now against runtime values. The
// destination carries what the region resolved, so the body reads the extents
// off it rather than resolving them a second time.
// CHECK-LABEL: func.func @host_shape(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<?x?x4xf16, #hipsr.mem<device>>,
// CHECK-SAME:    %[[TARGET:.*]]: tensor<2xi64, #hipsr.mem<host>>) -> tensor<?x?xf16, #hipsr.mem<device>> {
// CHECK-NEXT:    %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]], %[[TARGET]] : tensor<?x?x4xf16, #hipsr.mem<device>>, tensor<2xi64, #hipsr.mem<host>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<?x?xf16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:    ^bb0(%{{.*}}: !hipsr.context, %[[IN:.*]]: tensor<?x?x4xf16, #hipsr.mem<device>>, %[[SHAPE_IN:.*]]: tensor<2xi64, #hipsr.mem<host>>):
// CHECK-NEXT:      %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[ONE:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[AXIS0:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[D0:.*]] = tensor.dim %[[IN]], %[[AXIS0]] : tensor<?x?x4xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[AXIS1:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[D1:.*]] = tensor.dim %[[IN]], %[[AXIS1]] : tensor<?x?x4xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[D2:.*]] = arith.constant 4 : index
// CHECK-NEXT:      %[[PARTIAL:.*]] = arith.muli %[[D0]], %[[D1]] : index
// CHECK-NEXT:      %[[COUNT:.*]] = arith.muli %[[PARTIAL]], %[[D2]] : index
// CHECK-NEXT:      %[[ENTRY0:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[RAW0:.*]] = tensor.extract %[[SHAPE_IN]]{{\[}}%[[ENTRY0]]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[E0:.*]] = arith.index_cast %[[RAW0]] : i64 to index
// CHECK-NEXT:      %[[COPIES0:.*]] = arith.cmpi eq, %[[E0]], %[[ZERO]] : index
// CHECK-NEXT:      %[[SRC0:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[IN_D0:.*]] = tensor.dim %[[IN]], %[[SRC0]] : tensor<?x?x4xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[STATED0:.*]] = arith.select %[[COPIES0]], %[[IN_D0]], %[[E0]] : index
// CHECK-NEXT:      %[[INFERS0:.*]] = arith.cmpi slt, %[[STATED0]], %[[ZERO]] : index
// CHECK-NEXT:      %[[FACTOR0:.*]] = arith.select %[[INFERS0]], %[[ONE]], %[[STATED0]] : index
// CHECK-NEXT:      %[[ENTRY1:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[RAW1:.*]] = tensor.extract %[[SHAPE_IN]]{{\[}}%[[ENTRY1]]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[E1:.*]] = arith.index_cast %[[RAW1]] : i64 to index
// CHECK-NEXT:      %[[COPIES1:.*]] = arith.cmpi eq, %[[E1]], %[[ZERO]] : index
// CHECK-NEXT:      %[[SRC1:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[IN_D1:.*]] = tensor.dim %[[IN]], %[[SRC1]] : tensor<?x?x4xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[STATED1:.*]] = arith.select %[[COPIES1]], %[[IN_D1]], %[[E1]] : index
// CHECK-NEXT:      %[[INFERS1:.*]] = arith.cmpi slt, %[[STATED1]], %[[ZERO]] : index
// CHECK-NEXT:      %[[FACTOR1:.*]] = arith.select %[[INFERS1]], %[[ONE]], %[[STATED1]] : index
// CHECK-NEXT:      %[[DIVISOR:.*]] = arith.muli %[[FACTOR0]], %[[FACTOR1]] : index
// CHECK-NEXT:      %[[INFERRED:.*]] = arith.divui %[[COUNT]], %[[DIVISOR]] : index
// CHECK-NEXT:      %[[EXTENT0:.*]] = arith.select %[[INFERS0]], %[[INFERRED]], %[[STATED0]] : index
// CHECK-NEXT:      %[[EXTENT1:.*]] = arith.select %[[INFERS1]], %[[INFERRED]], %[[STATED1]] : index
// CHECK-NEXT:      %[[SHAPE:.*]] = shape.from_extents %[[EXTENT0]], %[[EXTENT1]] : index, index
// CHECK-NEXT:      hipsr.shape_yield %[[SHAPE]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[RESULT:.*]] = hipsr.compute(%[[CTX]]) ins(%[[INPUT]] : tensor<?x?x4xf16, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<?x?xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:    ^bb0(%{{.*}}: !hipsr.context, %[[BODY_IN:.*]]: tensor<?x?x4xf16, #hipsr.mem<device>>, %[[DEST:.*]]: tensor<?x?xf16, #hipsr.mem<device>>):
// CHECK-NEXT:      %[[FLAT:.*]] = tensor.collapse_shape %[[BODY_IN]] {{\[}}[0, 1, 2]] : tensor<?x?x4xf16, #hipsr.mem<device>> into tensor<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[DEST_D0:.*]] = tensor.dim %[[DEST]], %[[C0]] : tensor<?x?xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[DEST_D1:.*]] = tensor.dim %[[DEST]], %[[C1]] : tensor<?x?xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[OUT:.*]] = tensor.expand_shape %[[FLAT]] {{\[}}[0, 1]] output_shape {{\[}}%[[DEST_D0]], %[[DEST_D1]]] : tensor<?xf16, #hipsr.mem<device>> into tensor<?x?xf16, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.compute_yield %[[OUT]] : tensor<?x?xf16, #hipsr.mem<device>>
// CHECK-NEXT:    } : tensor<?x?xf16, #hipsr.mem<device>>{{$}}
// CHECK-NEXT:    return %[[RESULT]] : tensor<?x?xf16, #hipsr.mem<device>>
func.func @host_shape(%ctx: !hipsr.context, %input: tensor<?x?x4xf16>,
                      %shape: tensor<2xi64, #hipsr.mem<host>>)
    -> tensor<?x?xf16> {
  %0 = "onnx.Reshape"(%input, %shape) {allowzero = 0 : si64}
      : (tensor<?x?x4xf16>, tensor<2xi64, #hipsr.mem<host>>)
      -> tensor<?x?xf16>
  "onnx.Return"(%0) : (tensor<?x?xf16>) -> ()
}

// -----

// The result aliases the input, so a host input keeps the whole chain in
// #hipsr.mem<host>. An ONNX result naming no space follows the input rather
// than defaulting to device.
// CHECK-LABEL: func.func @host_input(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<2x3xi64, #hipsr.mem<host>>,
// CHECK-SAME:    %{{.*}}: tensor<1xi64, #hipsr.mem<device>>) {
// CHECK-NEXT:    %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]] : tensor<2x3xi64, #hipsr.mem<host>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<6xi64, #hipsr.mem<host>> shape_region {
// CHECK-NEXT:    ^bb0(%{{.*}}: !shape.shape):
// CHECK-NEXT:      %[[COUNT:.*]] = arith.constant 6 : index
// CHECK-NEXT:      %[[SHAPE:.*]] = shape.from_extents %[[COUNT]] : index
// CHECK-NEXT:      hipsr.shape_yield %[[SHAPE]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %{{.*}} = hipsr.compute(%[[CTX]]) ins(%[[INPUT]] : tensor<2x3xi64, #hipsr.mem<host>>) outs(%[[INIT]] : tensor<6xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:    ^bb0(%{{.*}}: !hipsr.context, %[[IN:.*]]: tensor<2x3xi64, #hipsr.mem<host>>, %{{.*}}: tensor<6xi64, #hipsr.mem<host>>):
// CHECK-NEXT:      %[[FLAT:.*]] = tensor.collapse_shape %[[IN]] {{\[}}[0, 1]] : tensor<2x3xi64, #hipsr.mem<host>> into tensor<6xi64, #hipsr.mem<host>>
// CHECK-NEXT:      hipsr.compute_yield %[[FLAT]] : tensor<6xi64, #hipsr.mem<host>>
// CHECK-NEXT:    } : tensor<6xi64, #hipsr.mem<host>>{{$}}
// CHECK-NEXT:    return{{$}}
func.func @host_input(%ctx: !hipsr.context,
                      %input: tensor<2x3xi64, #hipsr.mem<host>>,
                      %shape: tensor<1xi64>) {
  %0 = "onnx.Reshape"(%input, %shape)
      : (tensor<2x3xi64, #hipsr.mem<host>>, tensor<1xi64>) -> tensor<6xi64>
  "onnx.Return"() : () -> ()
}
