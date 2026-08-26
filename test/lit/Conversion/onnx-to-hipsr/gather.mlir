// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// Device data makes onnx.Gather a hipsr.gather, host data a hipsr.compute of
// scalar reads.
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
// on the host for. The result is rank 0, so one read fills it and the region
// names an empty shape. The index constant is left behind dead.
// CHECK-LABEL: func.func @host_leading_dim(
// CHECK-SAME:    %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME:    %[[SHAPE:.+]]: tensor<2xi64, #hipsr.mem<host>>) -> tensor<i64, #hipsr.mem<host>> {
// CHECK-NEXT:    %{{.+}} = arith.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<i64, #hipsr.mem<host>> shape_region {
// CHECK-NEXT:      %[[RESULT_SHAPE:.+]] = shape.const_shape [] : !shape.shape
// CHECK-NEXT:      hipsr.shape_yield %[[RESULT_SHAPE]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[RESULT:.+]] = hipsr.compute(%[[CTX]]) ins(%[[SHAPE]] : tensor<2xi64, #hipsr.mem<host>>) outs(%[[INIT]] : tensor<i64, #hipsr.mem<host>>) {
// CHECK-NEXT:    ^bb0(%{{.+}}: !hipsr.context, %[[IN:.+]]: tensor<2xi64, #hipsr.mem<host>>, %{{.+}}: tensor<i64, #hipsr.mem<host>>):
// CHECK-NEXT:      %[[ZERO:.+]] = arith.constant 0 : index
// CHECK-NEXT:      %[[DIM:.+]] = tensor.extract %[[IN]][%[[ZERO]]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[OUT:.+]] = tensor.from_elements %[[DIM]] : tensor<i64, #hipsr.mem<host>>
// CHECK-NEXT:      hipsr.compute_yield %[[OUT]] : tensor<i64, #hipsr.mem<host>>
// CHECK-NEXT:    } : tensor<i64, #hipsr.mem<host>>
// CHECK-NEXT:    return %[[RESULT]] : tensor<i64, #hipsr.mem<host>>
// CHECK-NEXT:  }
func.func @host_leading_dim(%ctx: !hipsr.context,
                            %shape: tensor<2xi64, #hipsr.mem<host>>)
    -> tensor<i64> {
  %index = "onnx.Constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
  %0 = "onnx.Gather"(%shape, %index) {axis = 0 : si64}
      : (tensor<2xi64, #hipsr.mem<host>>, tensor<i64>) -> tensor<i64>
  "onnx.Return"(%0) : (tensor<i64>) -> ()
}

// -----

// An axis with dimensions on both sides of it. The axes around the gathered one
// run through their own dimensions while the index stays pinned at 2, which is
// where -1 lands in an axis of 3. Only the index's value is read here, so the
// device buffer the constant conversion gave it is left dead.
// CHECK-LABEL: func.func @host_inner_axis(
// CHECK-SAME:    %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME:    %[[DATA:.+]]: tensor<2x3x2xi64, #hipsr.mem<host>>) -> tensor<2x1x2xi64, #hipsr.mem<host>> {
// CHECK-NEXT:    %{{.+}} = hipsr.constant {value = dense<-1> : tensor<1xi64>} : tensor<1xi64, #hipsr.mem<device>>
// CHECK-NEXT:    %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<2x1x2xi64, #hipsr.mem<host>> shape_region {
// CHECK-NEXT:      %[[RESULT_SHAPE:.+]] = shape.const_shape [2, 1, 2] : !shape.shape
// CHECK-NEXT:      hipsr.shape_yield %[[RESULT_SHAPE]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[RESULT:.+]] = hipsr.compute(%[[CTX]]) ins(%[[DATA]] : tensor<2x3x2xi64, #hipsr.mem<host>>) outs(%[[INIT]] : tensor<2x1x2xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:    ^bb0(%{{.+}}: !hipsr.context, %[[IN:.+]]: tensor<2x3x2xi64, #hipsr.mem<host>>, %{{.+}}: tensor<2x1x2xi64, #hipsr.mem<host>>):
// CHECK-NEXT:      %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT:      %[[C2:.+]] = arith.constant 2 : index
// CHECK-NEXT:      %[[E0:.+]] = tensor.extract %[[IN]][%[[C0]], %[[C2]], %[[C0]]] : tensor<2x3x2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[C1:.+]] = arith.constant 1 : index
// CHECK-NEXT:      %[[E1:.+]] = tensor.extract %[[IN]][%[[C0]], %[[C2]], %[[C1]]] : tensor<2x3x2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[E2:.+]] = tensor.extract %[[IN]][%[[C1]], %[[C2]], %[[C0]]] : tensor<2x3x2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[E3:.+]] = tensor.extract %[[IN]][%[[C1]], %[[C2]], %[[C1]]] : tensor<2x3x2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[OUT:.+]] = tensor.from_elements %[[E0]], %[[E1]], %[[E2]], %[[E3]] : tensor<2x1x2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      hipsr.compute_yield %[[OUT]] : tensor<2x1x2xi64, #hipsr.mem<host>>
// CHECK-NEXT:    } : tensor<2x1x2xi64, #hipsr.mem<host>>
// CHECK-NEXT:    return %[[RESULT]] : tensor<2x1x2xi64, #hipsr.mem<host>>
// CHECK-NEXT:  }
func.func @host_inner_axis(%ctx: !hipsr.context,
                           %data: tensor<2x3x2xi64, #hipsr.mem<host>>)
    -> tensor<2x1x2xi64> {
  %index = "onnx.Constant"() {value = dense<-1> : tensor<1xi64>}
      : () -> tensor<1xi64>
  %0 = "onnx.Gather"(%data, %index) {axis = 1 : si64}
      : (tensor<2x3x2xi64, #hipsr.mem<host>>, tensor<1xi64>)
      -> tensor<2x1x2xi64>
  "onnx.Return"(%0) : (tensor<2x1x2xi64>) -> ()
}
