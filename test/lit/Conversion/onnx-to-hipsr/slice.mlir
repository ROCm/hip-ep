// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// onnx.Slice becomes hipsr.slice. Entries the compiler can read become
// attributes and the placeholder is normal; entries the graph computes stay a
// host operand and the placeholder is a barrier, whose region reads them.
// Rejected forms are in slice-invalid.mlir.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --onnx-dialect=modeled --split-input-file -allow-unregistered-dialect -convert-onnx-to-hipsr | FileCheck %s

// A constant window goes over as attributes, leaving the device constants the
// constant conversion made for dead-code elimination to drop. The placeholder
// takes only the data, and its shape region stays empty for
// hipsr-populate-shape-region.
// CHECK-LABEL: func.func @strided_window(
// CHECK-SAME:    %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME:    %[[DATA:.+]]: tensor<8x4xf16, #hipsr.mem<device>>) -> tensor<3x4xf16, #hipsr.mem<device>> {
// CHECK-NEXT:    %{{.+}} = hipsr.constant {value = dense<1> : tensor<1xi64>} : tensor<1xi64, #hipsr.mem<device>>
// CHECK-NEXT:    %{{.+}} = hipsr.constant {value = dense<7> : tensor<1xi64>} : tensor<1xi64, #hipsr.mem<device>>
// CHECK-NEXT:    %{{.+}} = hipsr.constant {value = dense<0> : tensor<1xi64>} : tensor<1xi64, #hipsr.mem<device>>
// CHECK-NEXT:    %{{.+}} = hipsr.constant {value = dense<2> : tensor<1xi64>} : tensor<1xi64, #hipsr.mem<device>>
// CHECK-NEXT:    %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[DATA]] : tensor<8x4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<3x4xf16, #hipsr.mem<device>>
// CHECK-NEXT:    %[[RESULT:.+]] = hipsr.slice(%[[CTX]]) ins(%[[DATA]] : tensor<8x4xf16, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<3x4xf16, #hipsr.mem<device>>) {axes_attr = array<i64: 0>, ends_attr = array<i64: 7>, starts_attr = array<i64: 1>, steps_attr = array<i64: 2>} : tensor<3x4xf16, #hipsr.mem<device>>
// CHECK-NEXT:    return %[[RESULT]] : tensor<3x4xf16, #hipsr.mem<device>>
// CHECK-NEXT:  }
func.func @strided_window(%ctx: !hipsr.context,
                          %data: tensor<8x4xf16>) -> tensor<3x4xf16> {
  %starts = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
  %ends = "onnx.Constant"() {value = dense<7> : tensor<1xi64>} : () -> tensor<1xi64>
  %axes = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
  %steps = "onnx.Constant"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>
  %0 = "onnx.Slice"(%data, %starts, %ends, %axes, %steps)
      : (tensor<8x4xf16>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>,
         tensor<1xi64>) -> tensor<3x4xf16>
  "onnx.Return"(%0) : (tensor<3x4xf16>) -> ()
}

// -----

// A bound the graph computes stays an operand and leaves the sliced axis
// dynamic, so the init is a barrier placeholder holding the data and that one
// operand for its region to read. The barrier takes the destination the compute
// writes into, the same as an expand's shape operand. `axes` and `steps` arrive
// as onnx.NoValue and get ONNX's defaults, the leading axis and a unit step.
// CHECK-LABEL: func.func @computed_bound(
// CHECK-SAME:    %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME:    %[[DATA:.+]]: tensor<8xf16, #hipsr.mem<device>>,
// CHECK-SAME:    %[[OTHER:.+]]: tensor<?x4096xf16, #hipsr.mem<device>>) -> tensor<?xf16, #hipsr.mem<device>> {
// CHECK-NEXT:    %[[ENDS_INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[OTHER]] : tensor<?x4096xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<1xi64, #hipsr.mem<host>> shape_region {
// CHECK-NEXT:    ^bb0(%{{.+}}: !shape.shape):
// CHECK-NEXT:      %[[LEN:.+]] = arith.constant 1 : index
// CHECK-NEXT:      %[[ENDS_SHAPE:.+]] = shape.from_extents %[[LEN]] : index
// CHECK-NEXT:      hipsr.shape_yield %[[ENDS_SHAPE]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[ENDS:.+]] = hipsr.compute(%[[CTX]]) ins(%[[OTHER]] : tensor<?x4096xf16, #hipsr.mem<device>>) outs(%[[ENDS_INIT]] : tensor<1xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:    ^bb0(%{{.+}}: !hipsr.context, %[[BODY_OTHER:.+]]: tensor<?x4096xf16, #hipsr.mem<device>>, %{{.+}}: tensor<1xi64, #hipsr.mem<host>>):
// CHECK-NEXT:      %[[AXIS:.+]] = arith.constant 0 : index
// CHECK-NEXT:      %[[DIM:.+]] = tensor.dim %[[BODY_OTHER]], %[[AXIS]] : tensor<?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[BOUND:.+]] = arith.index_cast %[[DIM]] : index to i64
// CHECK-NEXT:      %[[BOUND_VECTOR:.+]] = tensor.from_elements %[[BOUND]] : tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:      hipsr.compute_yield %[[BOUND_VECTOR]] : tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:    } : tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:    %{{.+}} = hipsr.constant {value = dense<0> : tensor<1xi64>} : tensor<1xi64, #hipsr.mem<device>>
// CHECK-NEXT:    %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[DATA]], %[[ENDS_INIT]] : tensor<8xf16, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<host>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:    %[[RESULT:.+]] = hipsr.slice(%[[CTX]]) ins(%[[DATA]] : tensor<8xf16, #hipsr.mem<device>>) ends(%[[ENDS]] : tensor<1xi64, #hipsr.mem<host>>) outs(%[[INIT]] : tensor<?xf16, #hipsr.mem<device>>) {axes_attr = array<i64: 0>, starts_attr = array<i64: 0>, steps_attr = array<i64: 1>} : tensor<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:    return %[[RESULT]] : tensor<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:  }
func.func @computed_bound(%ctx: !hipsr.context, %data: tensor<8xf16>,
                          %other: tensor<?x4096xf16>) -> tensor<?xf16> {
  %ends = "onnx.Shape"(%other) {start = 0 : si64, end = 1 : si64}
      : (tensor<?x4096xf16>) -> tensor<1xi64>
  %starts = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Slice"(%data, %starts, %ends, %none, %none)
      : (tensor<8xf16>, tensor<1xi64>, tensor<1xi64>, none, none)
      -> tensor<?xf16>
  "onnx.Return"(%0) : (tensor<?xf16>) -> ()
}
