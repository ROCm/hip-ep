// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// Device data makes onnx.Gather a hipsr.gather, host data a hipsr.compute of
// scalar reads. The host cases read what onnx.Shape leaves in
// #hipsr.mem<host>: a graph input is always device, so host data has to be
// built inside the function.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --onnx-dialect=modeled --split-input-file -allow-unregistered-dialect -convert-onnx-to-hipsr | FileCheck %s

// An embedding graph looks token rows up in a table. The placeholder gets no
// shape region here: hipsr.gather is DPS, so hipsr-populate-shape-region fills
// it in later.
// CHECK-LABEL: func.func @embedding_lookup(
// CHECK-SAME:    %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME:    %[[TABLE:.+]]: tensor<8x4096xf16, #hipsr.mem<device>>,
// CHECK-SAME:    %[[IDS:.+]]: tensor<?x?xi64, #hipsr.mem<device>>) -> tensor<?x?x4096xf16, #hipsr.mem<device>> {
// CHECK-NEXT:    %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[TABLE]], %[[IDS]] : tensor<8x4096xf16, #hipsr.mem<device>>, tensor<?x?xi64, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:    %[[RESULT:.+]] = hipsr.gather(%[[CTX]]) ins(%[[TABLE]], %[[IDS]] : tensor<8x4096xf16, #hipsr.mem<device>>, tensor<?x?xi64, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<?x?x4096xf16, #hipsr.mem<device>>) {axis = 0 : i64} : tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:    return %[[RESULT]] : tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:  }
func.func @embedding_lookup(%ctx: !hipsr.context, %table: tensor<8x4096xf16>,
                            %ids: tensor<?x?xi64>) -> tensor<?x?x4096xf16> {
  %0 = "onnx.Gather"(%table, %ids) {axis = 0 : si64}
      : (tensor<8x4096xf16>, tensor<?x?xi64>) -> tensor<?x?x4096xf16>
  "onnx.Return"(%0) : (tensor<?x?x4096xf16>) -> ()
}

// -----

// ONNX lets the axis count back from the end; hipsr.gather gets it normalized.
// CHECK-LABEL: func.func @negative_axis(
// CHECK-SAME:    %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME:    %[[DATA:.+]]: tensor<2x3x4xf16, #hipsr.mem<device>>,
// CHECK-SAME:    %[[INDICES:.+]]: tensor<5xi64, #hipsr.mem<device>>) -> tensor<2x3x5xf16, #hipsr.mem<device>> {
// CHECK-NEXT:    %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[DATA]], %[[INDICES]] : tensor<2x3x4xf16, #hipsr.mem<device>>, tensor<5xi64, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<2x3x5xf16, #hipsr.mem<device>>
// CHECK-NEXT:    %[[RESULT:.+]] = hipsr.gather(%[[CTX]]) ins(%[[DATA]], %[[INDICES]] : tensor<2x3x4xf16, #hipsr.mem<device>>, tensor<5xi64, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<2x3x5xf16, #hipsr.mem<device>>) {axis = 2 : i64} : tensor<2x3x5xf16, #hipsr.mem<device>>
// CHECK-NEXT:    return %[[RESULT]] : tensor<2x3x5xf16, #hipsr.mem<device>>
// CHECK-NEXT:  }
func.func @negative_axis(%ctx: !hipsr.context, %data: tensor<2x3x4xf16>,
                         %indices: tensor<5xi64>) -> tensor<2x3x5xf16> {
  %0 = "onnx.Gather"(%data, %indices) {axis = -1 : si64}
      : (tensor<2x3x4xf16>, tensor<5xi64>) -> tensor<2x3x5xf16>
  "onnx.Return"(%0) : (tensor<2x3x5xf16>) -> ()
}

// -----

// Reading the leading dimension out of a shape, which is what a graph gathers
// on the host for. onnx.Shape leaves its extents in host memory, and gathering
// one of them keeps the chain there. The result is rank 0, so one read fills it
// and the region names an empty shape. The index is rank 0, which the pass
// leaves as an arith.constant, and only its value is read, so it is left dead.
// CHECK-LABEL: func.func @host_leading_dim(
// CHECK-SAME:    %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.+]]: tensor<?x4096xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:    %[[EXTENTS_INIT:.+]] = hipsr.placeholder(%[[CTX]]) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<2xi64, #hipsr.mem<host>> shape_region {
// CHECK-NEXT:      %[[LEN:.+]] = arith.constant 2 : index
// CHECK-NEXT:      %[[EXTENTS_SHAPE:.+]] = shape.from_extents %[[LEN]] : index
// CHECK-NEXT:      hipsr.shape_yield %[[EXTENTS_SHAPE]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[SHAPE:.+]] = hipsr.compute(%[[CTX]]) ins(%[[INPUT]] : tensor<?x4096xf16, #hipsr.mem<device>>) outs(%[[EXTENTS_INIT]] : tensor<2xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:    ^bb0(%{{.+}}: !hipsr.context, %[[DATA:.+]]: tensor<?x4096xf16, #hipsr.mem<device>>, %{{.+}}: tensor<2xi64, #hipsr.mem<host>>):
// CHECK-NEXT:      %[[AXIS0:.+]] = arith.constant 0 : index
// CHECK-NEXT:      %[[DIM0:.+]] = tensor.dim %[[DATA]], %[[AXIS0]] : tensor<?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[EXT0:.+]] = arith.index_cast %[[DIM0]] : index to i64
// CHECK-NEXT:      %[[EXT1:.+]] = arith.constant 4096 : i64
// CHECK-NEXT:      %[[EXTENTS:.+]] = tensor.from_elements %[[EXT0]], %[[EXT1]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      hipsr.compute_yield %[[EXTENTS]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:    } : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:    %{{.+}} = arith.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<i64, #hipsr.mem<host>> shape_region {
// CHECK-NEXT:      %[[RESULT_SHAPE:.+]] = shape.const_shape [] : !shape.shape
// CHECK-NEXT:      hipsr.shape_yield %[[RESULT_SHAPE]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %{{.+}} = hipsr.compute(%[[CTX]]) ins(%[[SHAPE]] : tensor<2xi64, #hipsr.mem<host>>) outs(%[[INIT]] : tensor<i64, #hipsr.mem<host>>) {
// CHECK-NEXT:    ^bb0(%{{.+}}: !hipsr.context, %[[IN:.+]]: tensor<2xi64, #hipsr.mem<host>>, %{{.+}}: tensor<i64, #hipsr.mem<host>>):
// CHECK-NEXT:      %[[ZERO:.+]] = arith.constant 0 : index
// CHECK-NEXT:      %[[DIM:.+]] = tensor.extract %[[IN]][%[[ZERO]]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[OUT:.+]] = tensor.from_elements %[[DIM]] : tensor<i64, #hipsr.mem<host>>
// CHECK-NEXT:      hipsr.compute_yield %[[OUT]] : tensor<i64, #hipsr.mem<host>>
// CHECK-NEXT:    } : tensor<i64, #hipsr.mem<host>>
// CHECK-NEXT:    return{{$}}
// CHECK-NEXT:  }
func.func @host_leading_dim(%ctx: !hipsr.context,
                            %input: tensor<?x4096xf16>) {
  %shape = "onnx.Shape"(%input) : (tensor<?x4096xf16>) -> tensor<2xi64>
  %index = "onnx.Constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
  %0 = "onnx.Gather"(%shape, %index) {axis = 0 : si64}
      : (tensor<2xi64>, tensor<i64>) -> tensor<i64>
  return
}

// -----

// Two indices, one of them counting back from the end: -1 lands on the last of
// the three extents, so the reads run 2 then 0. The indices shape becomes the
// result shape, so the result is rank 1 and the region names it. The constant
// is rank 1, which the constant conversion parks on the device, and only its
// value is read here, so that buffer is left dead.
// CHECK-LABEL: func.func @host_negative_index(
// CHECK-SAME:    %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.+]]: tensor<?x?x4096xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:    %[[EXTENTS_INIT:.+]] = hipsr.placeholder(%[[CTX]]) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<3xi64, #hipsr.mem<host>> shape_region {
// CHECK-NEXT:      %[[LEN:.+]] = arith.constant 3 : index
// CHECK-NEXT:      %[[EXTENTS_SHAPE:.+]] = shape.from_extents %[[LEN]] : index
// CHECK-NEXT:      hipsr.shape_yield %[[EXTENTS_SHAPE]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[SHAPE:.+]] = hipsr.compute(%[[CTX]]) ins(%[[INPUT]] : tensor<?x?x4096xf16, #hipsr.mem<device>>) outs(%[[EXTENTS_INIT]] : tensor<3xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:    ^bb0(%{{.+}}: !hipsr.context, %[[DATA:.+]]: tensor<?x?x4096xf16, #hipsr.mem<device>>, %{{.+}}: tensor<3xi64, #hipsr.mem<host>>):
// CHECK-NEXT:      %[[AXIS0:.+]] = arith.constant 0 : index
// CHECK-NEXT:      %[[DIM0:.+]] = tensor.dim %[[DATA]], %[[AXIS0]] : tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[EXT0:.+]] = arith.index_cast %[[DIM0]] : index to i64
// CHECK-NEXT:      %[[AXIS1:.+]] = arith.constant 1 : index
// CHECK-NEXT:      %[[DIM1:.+]] = tensor.dim %[[DATA]], %[[AXIS1]] : tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[EXT1:.+]] = arith.index_cast %[[DIM1]] : index to i64
// CHECK-NEXT:      %[[EXT2:.+]] = arith.constant 4096 : i64
// CHECK-NEXT:      %[[EXTENTS:.+]] = tensor.from_elements %[[EXT0]], %[[EXT1]], %[[EXT2]] : tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:      hipsr.compute_yield %[[EXTENTS]] : tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:    } : tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:    %{{.+}} = hipsr.constant {value = dense<[-1, 0]> : tensor<2xi64>} : tensor<2xi64, #hipsr.mem<device>>
// CHECK-NEXT:    %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<2xi64, #hipsr.mem<host>> shape_region {
// CHECK-NEXT:      %[[RESULT_SHAPE:.+]] = shape.const_shape [2] : !shape.shape
// CHECK-NEXT:      hipsr.shape_yield %[[RESULT_SHAPE]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %{{.+}} = hipsr.compute(%[[CTX]]) ins(%[[SHAPE]] : tensor<3xi64, #hipsr.mem<host>>) outs(%[[INIT]] : tensor<2xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:    ^bb0(%{{.+}}: !hipsr.context, %[[IN:.+]]: tensor<3xi64, #hipsr.mem<host>>, %{{.+}}: tensor<2xi64, #hipsr.mem<host>>):
// CHECK-NEXT:      %[[LAST:.+]] = arith.constant 2 : index
// CHECK-NEXT:      %[[E0:.+]] = tensor.extract %[[IN]][%[[LAST]]] : tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[FIRST:.+]] = arith.constant 0 : index
// CHECK-NEXT:      %[[E1:.+]] = tensor.extract %[[IN]][%[[FIRST]]] : tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[OUT:.+]] = tensor.from_elements %[[E0]], %[[E1]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      hipsr.compute_yield %[[OUT]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:    } : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:    return{{$}}
// CHECK-NEXT:  }
func.func @host_negative_index(%ctx: !hipsr.context,
                               %input: tensor<?x?x4096xf16>) {
  %shape = "onnx.Shape"(%input) : (tensor<?x?x4096xf16>) -> tensor<3xi64>
  %indices = "onnx.Constant"() {value = dense<[-1, 0]> : tensor<2xi64>}
      : () -> tensor<2xi64>
  %0 = "onnx.Gather"(%shape, %indices) {axis = 0 : si64}
      : (tensor<3xi64>, tensor<2xi64>) -> tensor<2xi64>
  return
}
