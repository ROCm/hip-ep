// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip --canonicalize --cse | FileCheck %s

func.func @main_graph(%data: tensor<1xf32>) -> tensor<1xf32> {
  return %data : tensor<1xf32>
}

// Compute conversion visits Reshape before Concat. The final shape walk must
// recover the scalars after the ONNX producers become tensor operations.
// CHECK-LABEL: func.func @onnx_shape_concat
// CHECK-NOT: hip.readback_scalar
// CHECK-NOT: arith.divsi
// CHECK: tensor.reshape
// CHECK-NOT: hip.readback_scalar
// CHECK-NOT: arith.divsi
// CHECK: return
func.func @onnx_shape_concat(%data: tensor<?x?x?xf32>, %ref: tensor<?x?xf32>) -> tensor<?x?x?x?xf32> {
  %dims = "onnx.Shape"(%ref) : (tensor<?x?xf32>) -> tensor<2xi64>
  %ones = "onnx.Constant"() {value = dense<1> : tensor<2xi64>} : () -> tensor<2xi64>
  %shape = "onnx.Concat"(%dims, %ones) {axis = 0 : i64} : (tensor<2xi64>, tensor<2xi64>) -> tensor<4xi64>
  %result = "onnx.Reshape"(%data, %shape) : (tensor<?x?x?xf32>, tensor<4xi64>) -> tensor<?x?x?x?xf32>
  return %result : tensor<?x?x?x?xf32>
}

// The late walk must preserve an actual -1 in the same producer chain.
// CHECK-LABEL: func.func @onnx_shape_concat_infer
// CHECK-NOT: hip.readback_scalar
// CHECK: arith.divsi
// CHECK: tensor.reshape
// CHECK: return
func.func @onnx_shape_concat_infer(%data: tensor<?x?x?xf32>, %ref: tensor<?x?xf32>) -> tensor<?x?x?x?xf32> {
  %dims = "onnx.Shape"(%ref) : (tensor<?x?xf32>) -> tensor<2xi64>
  %tail = "onnx.Constant"() {value = dense<[1, -1]> : tensor<2xi64>} : () -> tensor<2xi64>
  %shape = "onnx.Concat"(%dims, %tail) {axis = 0 : i64} : (tensor<2xi64>, tensor<2xi64>) -> tensor<4xi64>
  %result = "onnx.Reshape"(%data, %shape) : (tensor<?x?x?xf32>, tensor<4xi64>) -> tensor<?x?x?x?xf32>
  return %result : tensor<?x?x?x?xf32>
}

// Shape scalars remain available through static slices and insertions.
// No shape entry can be -1. The data tensor's element count is not needed.
// CHECK-LABEL: func.func @host_shape
// CHECK-NOT: hip.readback_scalar
// CHECK-NOT: arith.divsi
// CHECK: tensor.reshape
// CHECK-NOT: hip.readback_scalar
// CHECK-NOT: arith.divsi
// CHECK: return
func.func @host_shape(%data: tensor<?x?x?xf32>, %ref: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %d0 = tensor.dim %ref, %c0 : tensor<?x?x?x?xf32>
  %d1 = tensor.dim %ref, %c1 : tensor<?x?x?x?xf32>
  %d2 = tensor.dim %ref, %c2 : tensor<?x?x?x?xf32>
  %d3 = tensor.dim %ref, %c3 : tensor<?x?x?x?xf32>
  %a = arith.index_cast %d0 : index to i64
  %b = arith.index_cast %d1 : index to i64
  %c = arith.index_cast %d2 : index to i64
  %d = arith.index_cast %d3 : index to i64
  %base = tensor.from_elements %a, %b, %c, %d : tensor<4xi64>
  %slice = tensor.extract_slice %base[1] [2] [1] : tensor<4xi64> to tensor<2xi64>
  %shape = tensor.insert_slice %slice into %base[0] [2] [2] : tensor<2xi64> into tensor<4xi64>
  %result = "onnx.Reshape"(%data, %shape) : (tensor<?x?x?xf32>, tensor<4xi64>) -> tensor<?x?x?x?xf32>
  return %result : tensor<?x?x?x?xf32>
}

// A strided shape slice forwards the selected elements, not adjacent ones.
// CHECK-LABEL: func.func @strided_shape
// CHECK-NOT: hip.readback_scalar
// CHECK-NOT: arith.divsi
// CHECK: tensor.from_elements %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<4xi64>
// CHECK: tensor.reshape
// CHECK: return
func.func @strided_shape(%data: tensor<?x?x?xf32>, %ref: tensor<?xf32>) -> tensor<?x?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %ref, %c0 : tensor<?xf32>
  %dim = arith.index_cast %n : index to i64
  %minus_one = arith.constant -1 : i64
  %base = tensor.from_elements %minus_one, %dim, %minus_one, %dim, %minus_one, %dim, %minus_one, %dim : tensor<8xi64>
  %shape = tensor.extract_slice %base[1] [4] [2] : tensor<8xi64> to tensor<4xi64>
  %result = "onnx.Reshape"(%data, %shape) : (tensor<?x?x?xf32>, tensor<4xi64>) -> tensor<?x?x?x?xf32>
  return %result : tensor<?x?x?x?xf32>
}

// An inline HIP constant can contain a real -1 request.
// CHECK-LABEL: func.func @real_minus_one
// CHECK-NOT: hip.readback_scalar
// CHECK: arith.divsi
// CHECK: tensor.reshape
// CHECK: return
func.func @real_minus_one(%data: tensor<?x?x?xf32>, %ref: tensor<?xf32>) -> tensor<?x?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %ref, %c0 : tensor<?xf32>
  %dim = arith.index_cast %n : index to i64
  %head = tensor.from_elements %dim : tensor<1xi64>
  %tail = hip.constant {value = dense<[1, 1, -1]> : tensor<3xi64>} : tensor<3xi64>
  %empty = tensor.empty() : tensor<4xi64>
  %partial = tensor.insert_slice %head into %empty[0] [1] [1] : tensor<1xi64> into tensor<4xi64>
  %shape = tensor.insert_slice %tail into %partial[1] [3] [1] : tensor<3xi64> into tensor<4xi64>
  %result = "onnx.Reshape"(%data, %shape) : (tensor<?x?x?xf32>, tensor<4xi64>) -> tensor<?x?x?x?xf32>
  return %result : tensor<?x?x?x?xf32>
}

// A narrowed dimension can become negative. Keep the runtime inference path.
// CHECK-LABEL: func.func @narrowed_dimension
// CHECK: arith.trunci
// CHECK: arith.divsi
// CHECK: arith.cmpi eq
// CHECK: tensor.reshape
// CHECK: return
func.func @narrowed_dimension(%data: tensor<?x?x?xf32>, %ref: tensor<?xf32>) -> tensor<?x?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %one = arith.constant 1 : i64
  %n = tensor.dim %ref, %c0 : tensor<?xf32>
  %wide = arith.index_cast %n : index to i64
  %narrow = arith.trunci %wide : i64 to i32
  %dim = arith.extsi %narrow : i32 to i64
  %shape = tensor.from_elements %dim, %one, %one, %one : tensor<4xi64>
  %result = "onnx.Reshape"(%data, %shape) : (tensor<?x?x?xf32>, tensor<4xi64>) -> tensor<?x?x?x?xf32>
  return %result : tensor<?x?x?x?xf32>
}

// A host scalar needs no readback, but its value can still be -1.
// CHECK-LABEL: func.func @arbitrary_host_scalar
// CHECK-NOT: hip.readback_scalar
// CHECK: arith.divsi
// CHECK: arith.cmpi eq
// CHECK: tensor.reshape
// CHECK: return
func.func @arbitrary_host_scalar(%data: tensor<?x?x?xf32>, %entry: i64) -> tensor<?x?x?x?xf32> {
  %one = arith.constant 1 : i64
  %shape = tensor.from_elements %entry, %one, %one, %one : tensor<4xi64>
  %result = "onnx.Reshape"(%data, %shape) : (tensor<?x?x?xf32>, tensor<4xi64>) -> tensor<?x?x?x?xf32>
  return %result : tensor<?x?x?x?xf32>
}

// An index cast must preserve the index width before it can preserve the bound.
// CHECK-LABEL: func.func @narrow_index_cast
// CHECK: arith.index_cast {{.*}} : index to i32
// CHECK: arith.divsi
// CHECK: arith.cmpi eq
// CHECK: tensor.reshape
// CHECK: return
func.func @narrow_index_cast(%data: tensor<?x?x?xf32>, %ref: tensor<?xf32>) -> tensor<?x?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %one = arith.constant 1 : i32
  %n = tensor.dim %ref, %c0 : tensor<?xf32>
  %dim = arith.index_cast %n : index to i32
  %shape = tensor.from_elements %dim, %one, %one, %one : tensor<4xi32>
  %result = "onnx.Reshape"(%data, %shape) : (tensor<?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  return %result : tensor<?x?x?x?xf32>
}

// Unknown shape payloads still require synchronized reads.
// CHECK-LABEL: func.func @unknown_shape
// CHECK-COUNT-4: hip.readback_scalar
// CHECK: arith.divsi
// CHECK: tensor.reshape
// CHECK: return
func.func @unknown_shape(%data: tensor<?x?x?xf32>, %shape: tensor<4xi64>) -> tensor<?x?x?x?xf32> {
  %result = "onnx.Reshape"(%data, %shape) : (tensor<?x?x?xf32>, tensor<4xi64>) -> tensor<?x?x?x?xf32>
  return %result : tensor<?x?x?x?xf32>
}

// A dynamic slice is outside the bounded scalar lookup.
// CHECK-LABEL: func.func @dynamic_slice
// CHECK-COUNT-4: hip.readback_scalar
// CHECK: arith.divsi
// CHECK: tensor.reshape
// CHECK: return
func.func @dynamic_slice(%data: tensor<?x?x?xf32>, %offset: index, %entry: i64) -> tensor<?x?x?x?xf32> {
  %base = tensor.from_elements %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry : tensor<8xi64>
  %shape = tensor.extract_slice %base[%offset] [4] [1] : tensor<8xi64> to tensor<4xi64>
  %result = "onnx.Reshape"(%data, %shape) : (tensor<?x?x?xf32>, tensor<4xi64>) -> tensor<?x?x?x?xf32>
  return %result : tensor<?x?x?x?xf32>
}

// A constant index vector is applied on the host. Indices [2, 0, 3] select
// d2, d0, d3. No entry is read back from the device.
// CHECK-LABEL: func.func @gather_index_vector
// CHECK-SAME: %[[DATA:.*]]: tensor<?x4xf32>, %[[D0:.*]]: i64, %[[D1:.*]]: i64, %[[D2:.*]]: i64, %[[D3:.*]]: i64
// CHECK-NOT: hip.readback_scalar
// CHECK: arith.select %{{.*}}, %{{.*}}, %[[D2]] : i64
// CHECK: arith.select %{{.*}}, %{{.*}}, %[[D0]] : i64
// CHECK: arith.select %{{.*}}, %{{.*}}, %[[D3]] : i64
// CHECK: tensor.reshape
// CHECK-NOT: hip.readback_scalar
// CHECK: return
func.func @gather_index_vector(%data: tensor<?x4xf32>, %d0: i64, %d1: i64, %d2: i64, %d3: i64) -> tensor<?x?x?xf32> {
  %base = tensor.from_elements %d0, %d1, %d2, %d3 : tensor<4xi64>
  %indices = "onnx.Constant"() {value = dense<[2, 0, 3]> : tensor<3xi64>} : () -> tensor<3xi64>
  %shape = "onnx.Gather"(%base, %indices) {axis = 0 : si64} : (tensor<4xi64>, tensor<3xi64>) -> tensor<3xi64>
  %result = "onnx.Reshape"(%data, %shape) : (tensor<?x4xf32>, tensor<3xi64>) -> tensor<?x?x?xf32>
  return %result : tensor<?x?x?xf32>
}

// A one-element index vector. Index 2 selects d2.
// CHECK-LABEL: func.func @gather_scalar_index
// CHECK-SAME: %[[DATA:.*]]: tensor<8xf32>, %[[D0:.*]]: i64, %[[D1:.*]]: i64, %[[D2:.*]]: i64, %[[D3:.*]]: i64
// CHECK-NOT: hip.readback_scalar
// CHECK: arith.select %{{.*}}, %{{.*}}, %[[D2]] : i64
// CHECK: tensor.reshape
// CHECK-NOT: hip.readback_scalar
// CHECK: return
func.func @gather_scalar_index(%data: tensor<8xf32>, %d0: i64, %d1: i64, %d2: i64, %d3: i64) -> tensor<?xf32> {
  %base = tensor.from_elements %d0, %d1, %d2, %d3 : tensor<4xi64>
  %indices = "onnx.Constant"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>
  %shape = "onnx.Gather"(%base, %indices) {axis = 0 : si64} : (tensor<4xi64>, tensor<1xi64>) -> tensor<1xi64>
  %result = "onnx.Reshape"(%data, %shape) : (tensor<8xf32>, tensor<1xi64>) -> tensor<?xf32>
  return %result : tensor<?xf32>
}

// A rank-0 index is the scalar form. Index 1 selects d1, and unsqueeze
// makes the rank-1 shape tensor Reshape requires.
// CHECK-LABEL: func.func @gather_scalar_rank0
// CHECK-SAME: %[[DATA:.*]]: tensor<8xf32>, %[[D0:.*]]: i64, %[[D1:.*]]: i64, %[[D2:.*]]: i64, %[[D3:.*]]: i64
// CHECK-NOT: hip.readback_scalar
// CHECK: arith.select %{{.*}}, %{{.*}}, %[[D1]] : i64
// CHECK: tensor.reshape
// CHECK-NOT: hip.readback_scalar
// CHECK: return
func.func @gather_scalar_rank0(%data: tensor<8xf32>, %d0: i64, %d1: i64, %d2: i64, %d3: i64) -> tensor<?xf32> {
  %base = tensor.from_elements %d0, %d1, %d2, %d3 : tensor<4xi64>
  %indices = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
  %picked = "onnx.Gather"(%base, %indices) {axis = 0 : si64} : (tensor<4xi64>, tensor<i64>) -> tensor<i64>
  %axes = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
  %shape = "onnx.Unsqueeze"(%picked, %axes) : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
  %result = "onnx.Reshape"(%data, %shape) : (tensor<8xf32>, tensor<1xi64>) -> tensor<?xf32>
  return %result : tensor<?xf32>
}

// Index -1 is the last element of the length-4 vector, d3.
// CHECK-LABEL: func.func @gather_negative_index
// CHECK-SAME: %[[DATA:.*]]: tensor<8xf32>, %[[D0:.*]]: i64, %[[D1:.*]]: i64, %[[D2:.*]]: i64, %[[D3:.*]]: i64
// CHECK-NOT: hip.readback_scalar
// CHECK: arith.select %{{.*}}, %{{.*}}, %[[D3]] : i64
// CHECK: tensor.reshape
// CHECK-NOT: hip.readback_scalar
// CHECK: return
func.func @gather_negative_index(%data: tensor<8xf32>, %d0: i64, %d1: i64, %d2: i64, %d3: i64) -> tensor<?xf32> {
  %base = tensor.from_elements %d0, %d1, %d2, %d3 : tensor<4xi64>
  %indices = "onnx.Constant"() {value = dense<-1> : tensor<1xi64>} : () -> tensor<1xi64>
  %shape = "onnx.Gather"(%base, %indices) {axis = 0 : si64} : (tensor<4xi64>, tensor<1xi64>) -> tensor<1xi64>
  %result = "onnx.Reshape"(%data, %shape) : (tensor<8xf32>, tensor<1xi64>) -> tensor<?xf32>
  return %result : tensor<?xf32>
}

// Axis -1 on a rank-1 shape vector is the same axis as 0.
// CHECK-LABEL: func.func @gather_negative_axis
// CHECK-SAME: %[[DATA:.*]]: tensor<8xf32>, %[[D0:.*]]: i64, %[[D1:.*]]: i64, %[[D2:.*]]: i64, %[[D3:.*]]: i64
// CHECK-NOT: hip.readback_scalar
// CHECK: arith.select %{{.*}}, %{{.*}}, %[[D2]] : i64
// CHECK: tensor.reshape
// CHECK-NOT: hip.readback_scalar
// CHECK: return
func.func @gather_negative_axis(%data: tensor<8xf32>, %d0: i64, %d1: i64, %d2: i64, %d3: i64) -> tensor<?xf32> {
  %base = tensor.from_elements %d0, %d1, %d2, %d3 : tensor<4xi64>
  %indices = "onnx.Constant"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>
  %shape = "onnx.Gather"(%base, %indices) {axis = -1 : si64} : (tensor<4xi64>, tensor<1xi64>) -> tensor<1xi64>
  %result = "onnx.Reshape"(%data, %shape) : (tensor<8xf32>, tensor<1xi64>) -> tensor<?xf32>
  return %result : tensor<?xf32>
}

// A runtime index cannot be folded. The shape entry is read back.
// CHECK-LABEL: func.func @gather_runtime_index
// CHECK: hip.gather
// CHECK: hip.readback_scalar
// CHECK: tensor.reshape
func.func @gather_runtime_index(%data: tensor<8xf32>, %d0: i64, %d1: i64, %d2: i64, %d3: i64, %index: tensor<1xi64>) -> tensor<?xf32> {
  %base = tensor.from_elements %d0, %d1, %d2, %d3 : tensor<4xi64>
  %shape = "onnx.Gather"(%base, %index) {axis = 0 : si64} : (tensor<4xi64>, tensor<1xi64>) -> tensor<1xi64>
  %result = "onnx.Reshape"(%data, %shape) : (tensor<8xf32>, tensor<1xi64>) -> tensor<?xf32>
  return %result : tensor<?xf32>
}
