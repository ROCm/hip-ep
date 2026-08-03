// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --resolve-shaped-type-result-dims %s | FileCheck %s

// input/output accessor family. The dynamic extent must come from the named
// input, while the source's static trailing extent reifies as a constant.
// CHECK-LABEL: func.func @input_output
// CHECK-SAME: %[[INPUT:[A-Za-z0-9_]+]]: tensor<?x8xf32>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
// CHECK: %[[D0:.*]] = tensor.dim %[[INPUT]], %[[C0]]
// CHECK: return %[[D0]], %[[C8]]
func.func @input_output(%ctx: !hip.context, %input: tensor<?x8xf32>,
                        %output: tensor<?x?xf32>) -> (index, index) {
  %result = hip.miopen.softmax(%ctx)
    ins(%input : tensor<?x8xf32>)
    outs(%output : tensor<?x?xf32>) -> tensor<?x?xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %result, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %result, %c1 : tensor<?x?xf32>
  return %d0, %d1 : index, index
}

// -----

// x/y accessor family with an additional shaped input. CumSum's scalar axis
// must not become the shape source.
// CHECK-LABEL: func.func @x_y_with_axis
// CHECK-SAME: %[[X:[A-Za-z0-9_]+]]: tensor<?x16xf32>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
// CHECK: %[[D0:.*]] = tensor.dim %[[X]], %[[C0]]
// CHECK: return %[[D0]], %[[C16]]
func.func @x_y_with_axis(%ctx: !hip.context, %x: tensor<?x16xf32>,
                         %axis: tensor<i64>, %y: tensor<?x?xf32>)
    -> (index, index) {
  %result = hip.cumsum(%ctx)
    ins(%x, %axis : tensor<?x16xf32>, tensor<i64>)
    outs(%y : tensor<?x?xf32>) : tensor<?x?xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %result, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %result, %c1 : tensor<?x?xf32>
  return %d0, %d1 : index, index
}

// -----

// data/output accessor family: ScatterElements ignores indices/updates shapes
// when reifying the output.
// CHECK-LABEL: func.func @scatter_elements
// CHECK-SAME: %[[DATA:[A-Za-z0-9_]+]]: tensor<?x4xf32>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
// CHECK: %[[D0:.*]] = tensor.dim %[[DATA]], %[[C0]]
// CHECK: return %[[D0]], %[[C4]]
func.func @scatter_elements(
    %ctx: !hip.context, %data: tensor<?x4xf32>,
    %indices: tensor<2x3xi64>, %updates: tensor<2x3xf32>,
    %output: tensor<?x?xf32>) -> (index, index) {
  %result = hip.scatter_elements(%ctx)
    ins(%data, %indices, %updates :
        tensor<?x4xf32>, tensor<2x3xi64>, tensor<2x3xf32>)
    outs(%output : tensor<?x?xf32>) : tensor<?x?xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %result, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %result, %c1 : tensor<?x?xf32>
  return %d0, %d1 : index, index
}

// -----

// ScatterND has different index/update ranks, but output still follows data.
// CHECK-LABEL: func.func @scatter_nd
// CHECK-SAME: %[[DATA:[A-Za-z0-9_]+]]: tensor<?x6x?xf32>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C6:.*]] = arith.constant 6 : index
// CHECK: %[[D0:.*]] = tensor.dim %[[DATA]], %[[C0]]
// CHECK: %[[D2:.*]] = tensor.dim %[[DATA]], %[[C2]]
// CHECK: return %[[D0]], %[[C6]], %[[D2]]
func.func @scatter_nd(
    %ctx: !hip.context, %data: tensor<?x6x?xf32>,
    %indices: tensor<2x1xi64>, %updates: tensor<2x6x?xf32>,
    %output: tensor<?x?x?xf32>) -> (index, index, index) {
  %result = hip.scatter_nd(%ctx)
    ins(%data, %indices, %updates :
        tensor<?x6x?xf32>, tensor<2x1xi64>, tensor<2x6x?xf32>)
    outs(%output : tensor<?x?x?xf32>) : tensor<?x?x?xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d0 = tensor.dim %result, %c0 : tensor<?x?x?xf32>
  %d1 = tensor.dim %result, %c1 : tensor<?x?x?xf32>
  %d2 = tensor.dim %result, %c2 : tensor<?x?x?xf32>
  return %d0, %d1, %d2 : index, index, index
}

// -----

// GatherElements is the non-leading-source case: output follows indices, not
// data. The dynamic extent and static trailing extent both come from indices.
// CHECK-LABEL: func.func @gather_elements_indices_source
// CHECK-SAME: %[[DATA:[A-Za-z0-9_]+]]: tensor<9x?xf32>
// CHECK-SAME: %[[INDICES:[A-Za-z0-9_]+]]: tensor<?x5xi64>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C5:.*]] = arith.constant 5 : index
// CHECK: %[[D0:.*]] = tensor.dim %[[INDICES]], %[[C0]]
// CHECK-NOT: tensor.dim %[[DATA]]
// CHECK: return %[[D0]], %[[C5]]
func.func @gather_elements_indices_source(
    %ctx: !hip.context, %data: tensor<9x?xf32>,
    %indices: tensor<?x5xi64>, %output: tensor<?x?xf32>)
    -> (index, index) {
  %result = hip.gather_elements(%ctx)
    ins(%data, %indices : tensor<9x?xf32>, tensor<?x5xi64>)
    outs(%output : tensor<?x?xf32>) {axis = 1 : i64} : tensor<?x?xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %result, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %result, %c1 : tensor<?x?xf32>
  return %d0, %d1 : index, index
}
