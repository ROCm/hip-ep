// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --split-input-file -hipsr-populate-shape-region | FileCheck %s

// A window in attributes against a known axis folds to the entries it takes:
// axis 0 runs backwards, and axis 2, named as -2, has a negative start and an
// end past the axis, which is how ONNX spells "to the end". Axis 1 takes the
// same rules as index arithmetic, because its size only arrives with the shape.
// Axis 3, which the window leaves out, keeps the data's size.
// CHECK-LABEL: func.func @slice_constant_window(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[DATA:.+]]: tensor<8x?x6x?xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[DATA]] : tensor<8x?x6x?xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<3x?x2x?xf16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT: ^bb0(%[[DATA_SHAPE:.+]]: tensor<4xindex>):
// CHECK-NEXT: %[[SIZE0:.+]] = arith.constant 3 : index
// CHECK-NEXT: %[[AXIS1:.+]] = arith.constant 1 : index
// CHECK-NEXT: %[[DATA_SIZE1:.+]] = shape.get_extent %[[DATA_SHAPE]], %[[AXIS1]] : tensor<4xindex>, index -> index
// CHECK-NEXT: %[[ZERO:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[START:.+]] = arith.constant 1 : index
// CHECK-NEXT: %[[LOW:.+]] = arith.minsi %[[DATA_SIZE1]], %[[START]] : index
// CHECK-NEXT: %[[END:.+]] = arith.constant 5 : index
// CHECK-NEXT: %[[HIGH:.+]] = arith.minsi %[[DATA_SIZE1]], %[[END]] : index
// CHECK-NEXT: %[[SPAN:.+]] = arith.subi %[[HIGH]], %[[LOW]] : index
// CHECK-NEXT: %[[SIZE1:.+]] = arith.maxsi %[[SPAN]], %[[ZERO]] : index
// CHECK-NEXT: %[[SIZE2:.+]] = arith.constant 2 : index
// CHECK-NEXT: %[[AXIS3:.+]] = arith.constant 3 : index
// CHECK-NEXT: %[[SIZE3:.+]] = shape.get_extent %[[DATA_SHAPE]], %[[AXIS3]] : tensor<4xindex>, index -> index
// CHECK-NEXT: %[[EXTENTS:.+]] = tensor.from_elements %[[SIZE0]], %[[SIZE1]], %[[SIZE2]], %[[SIZE3]] : tensor<4xindex>
// CHECK-NEXT: hipsr.shape_yield %[[EXTENTS]] : tensor<4xindex>
// CHECK-NEXT: }
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.slice(%[[CTX]]) ins(%[[DATA]] : tensor<8x?x6x?xf16, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<3x?x2x?xf16, #hipsr.mem<device>>) {axes_attr = array<i64: 0, 1, -2>, ends_attr = array<i64: 1, 5, 9223372036854775807>, starts_attr = array<i64: 6, 1, -4>, steps_attr = array<i64: -2, 1, 2>} : tensor<3x?x2x?xf16, #hipsr.mem<device>>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @slice_constant_window(%ctx: !hipsr.context,
                                 %data: tensor<8x?x6x?xf16, #hipsr.mem<device>>) {
  %init = hipsr.placeholder(%ctx)
      ins(%data : tensor<8x?x6x?xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>}
      : tensor<3x?x2x?xf16, #hipsr.mem<device>>
  %result = hipsr.slice(%ctx)
      ins(%data : tensor<8x?x6x?xf16, #hipsr.mem<device>>)
      outs(%init : tensor<3x?x2x?xf16, #hipsr.mem<device>>)
      {starts_attr = array<i64: 6, 1, -4>,
       ends_attr = array<i64: 1, 5, 9223372036854775807>,
       axes_attr = array<i64: 0, 1, -2>,
       steps_attr = array<i64: -2, 1, 2>}
      : tensor<3x?x2x?xf16, #hipsr.mem<device>>
  return
}

// -----

// A bound the graph computes, in the shape a shape computation leaves behind: a
// hipsr.compute writes it into a host destination and the barrier holds that
// destination. The region reads the bound off the operand the barrier hands
// over and clamps it against the data's own size. The start is in an attribute,
// so it needs no read, and folding drops the branch it cannot take.
// CHECK-LABEL: func.func @slice_runtime_bound(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[IN:.+]]: tensor<?xf16, #hipsr.mem<device>>) {

// The bound's own destination is a rank-1 host buffer, so its region is a lone
// constant.
// CHECK-NEXT: %[[ENDS_INIT:.+]] = hipsr.placeholder(%[[CTX]]) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<1xi64, #hipsr.mem<host>> shape_region {
// CHECK-NEXT: %[[ONE:.+]] = arith.constant 1 : index
// CHECK-NEXT: %[[ENDS_SHAPE:.+]] = tensor.from_elements %[[ONE]] : tensor<1xindex>
// CHECK-NEXT: hipsr.shape_yield %[[ENDS_SHAPE]] : tensor<1xindex>
// CHECK-NEXT: }

// The compute that writes the bound is left as it was.
// CHECK-NEXT: %[[ENDS_VALUE:.+]] = hipsr.compute(%[[CTX]]) ins(%[[IN]] : tensor<?xf16, #hipsr.mem<device>>) outs(%[[ENDS_INIT]] : tensor<1xi64, #hipsr.mem<host>>) {
// CHECK-NEXT: ^bb0(%{{.+}}: !hipsr.context, %[[BODY_IN:.+]]: tensor<?xf16, #hipsr.mem<device>>, %{{.+}}: tensor<1xi64, #hipsr.mem<host>>):
// CHECK-NEXT: %[[BODY_AXIS:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[BODY_SIZE:.+]] = tensor.dim %[[BODY_IN]], %[[BODY_AXIS]] : tensor<?xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[BODY_ENTRY:.+]] = arith.index_cast %[[BODY_SIZE]] : index to i64
// CHECK-NEXT: %[[BODY_ENTRIES:.+]] = tensor.from_elements %[[BODY_ENTRY]] : tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT: hipsr.compute_yield %[[BODY_ENTRIES]] : tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT: } : tensor<1xi64, #hipsr.mem<host>>

// CHECK-NEXT: %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[IN]], %[[ENDS_INIT]] : tensor<?xf16, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<host>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<?xf16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT: ^bb0(%{{.+}}: !hipsr.context, %[[DATA:.+]]: tensor<?xf16, #hipsr.mem<device>>, %[[ENDS:.+]]: tensor<1xi64, #hipsr.mem<host>>):
// CHECK-NEXT: %[[AXIS:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[DATA_SIZE:.+]] = tensor.dim %[[DATA]], %[[AXIS]]
// CHECK-NEXT: %[[SLOT:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[ENTRY:.+]] = tensor.extract %[[ENDS]][%[[SLOT]]]
// CHECK-NEXT: %[[END:.+]] = arith.index_cast %[[ENTRY]] : i64 to index
// CHECK-NEXT: %[[ZERO:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[NEGATIVE:.+]] = arith.cmpi slt, %[[END]], %[[ZERO]] : index
// CHECK-NEXT: %[[COUNTED:.+]] = arith.addi %[[END]], %[[DATA_SIZE]] : index
// CHECK-NEXT: %[[LAST:.+]] = arith.select %[[NEGATIVE]], %[[COUNTED]], %[[END]] : index
// CHECK-NEXT: %[[START_ZERO:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[LOW:.+]] = arith.minsi %[[DATA_SIZE]], %[[START_ZERO]] : index
// CHECK-NEXT: %[[LAST_UP:.+]] = arith.maxsi %[[LAST]], %[[ZERO]] : index
// CHECK-NEXT: %[[HIGH:.+]] = arith.minsi %[[LAST_UP]], %[[DATA_SIZE]] : index
// CHECK-NEXT: %[[SPAN:.+]] = arith.subi %[[HIGH]], %[[LOW]] : index
// CHECK-NEXT: %[[SIZE:.+]] = arith.maxsi %[[SPAN]], %[[ZERO]] : index
// CHECK-NEXT: %[[EXTENTS:.+]] = tensor.from_elements %[[SIZE]] : tensor<1xindex>
// CHECK-NEXT: hipsr.shape_yield %[[EXTENTS]] : tensor<1xindex>
// CHECK-NEXT: }
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.slice(%[[CTX]]) ins(%[[IN]] : tensor<?xf16, #hipsr.mem<device>>) ends(%[[ENDS_VALUE]] : tensor<1xi64, #hipsr.mem<host>>) outs(%[[INIT]] : tensor<?xf16, #hipsr.mem<device>>) {axes_attr = array<i64: 0>, starts_attr = array<i64: 0>, steps_attr = array<i64: 1>} : tensor<?xf16, #hipsr.mem<device>>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @slice_runtime_bound(%ctx: !hipsr.context,
                               %data: tensor<?xf16, #hipsr.mem<device>>) {
  %ends_init = hipsr.placeholder(%ctx)
      {placeholder_type = #hipsr.placeholder_type<normal>}
      : tensor<1xi64, #hipsr.mem<host>> shape_region {
    %one = arith.constant 1 : index
    %ends_shape = tensor.from_elements %one : tensor<1xindex>
    hipsr.shape_yield %ends_shape : tensor<1xindex>
  }
  %ends = hipsr.compute(%ctx) ins(%data : tensor<?xf16, #hipsr.mem<device>>)
                              outs(%ends_init : tensor<1xi64, #hipsr.mem<host>>) {
  ^bb0(%body_ctx: !hipsr.context, %in: tensor<?xf16, #hipsr.mem<device>>,
       %dest: tensor<1xi64, #hipsr.mem<host>>):
    %axis = arith.constant 0 : index
    %size = tensor.dim %in, %axis : tensor<?xf16, #hipsr.mem<device>>
    %entry = arith.index_cast %size : index to i64
    %entries = tensor.from_elements %entry : tensor<1xi64, #hipsr.mem<host>>
    hipsr.compute_yield %entries : tensor<1xi64, #hipsr.mem<host>>
  } : tensor<1xi64, #hipsr.mem<host>>
  %init = hipsr.placeholder(%ctx)
      ins(%data, %ends_init : tensor<?xf16, #hipsr.mem<device>>,
                              tensor<1xi64, #hipsr.mem<host>>)
      {placeholder_type = #hipsr.placeholder_type<barrier>}
      : tensor<?xf16, #hipsr.mem<device>>
  %result = hipsr.slice(%ctx)
      ins(%data : tensor<?xf16, #hipsr.mem<device>>)
      ends(%ends : tensor<1xi64, #hipsr.mem<host>>)
      outs(%init : tensor<?xf16, #hipsr.mem<device>>)
      {starts_attr = array<i64: 0>, axes_attr = array<i64: 0>,
       steps_attr = array<i64: 1>}
      : tensor<?xf16, #hipsr.mem<device>>
  return
}
