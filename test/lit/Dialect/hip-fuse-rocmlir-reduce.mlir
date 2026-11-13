// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// A last-axis hip.reduce_sum or hip.reduce_mean is a prologue-only rocMLIR
// anchor. Pointwise ops in front of it join the rock.reduce kernel. Pointwise
// ops after it stay an elementwise kernel. Anything the reduce pipeline cannot
// lower stays a HIP op.

// RUN: hip-mlir-opt --split-input-file --hip-rocmlir-prepare-quant \
// RUN:   --hip-fuse-rocmlir-elementwise --hip-fuse-rocmlir-anchor %s \
// RUN:   | hip-mlir-opt --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @mlir_reduce_sum_0
// CHECK-SAME: attributes {rock.kernel, rock.reduce}
// CHECK: hip.reduce_sum
// CHECK-LABEL: func.func @main_graph
// CHECK: hip.rocmlir
// CHECK-NOT: hip.reduce_sum

func.func @main_graph(%ctx: !hip.context, %data: tensor<8x16xf32>) -> tensor<8x1xf32> {
  %axes = hip.constant {value = dense<1> : tensor<i64>} : tensor<i64>
  %empty = tensor.empty() : tensor<8x1xf32>
  %sum = hip.reduce_sum(%ctx) ins(%data, %axes : tensor<8x16xf32>, tensor<i64>)
      outs(%empty : tensor<8x1xf32>)
      {keepdims = 1 : i64, noop_with_empty_axes = 0 : i64, reduced_axes = array<i64: 1>}
      : tensor<8x1xf32>
  return %sum : tensor<8x1xf32>
}

// -----

// CHECK-LABEL: func.func @mlir_neg_reduce_sum_0
// CHECK-SAME: attributes {rock.kernel, rock.reduce}
// CHECK: hip.neg
// CHECK: hip.reduce_sum
// CHECK-LABEL: func.func @main_graph
// CHECK: hip.rocmlir
// CHECK-NOT: hip.neg
// CHECK-NOT: hip.reduce_sum

func.func @main_graph(%ctx: !hip.context, %data: tensor<8x16xf32>) -> tensor<8x1xf32> {
  %ne = tensor.empty() : tensor<8x16xf32>
  %n = hip.neg(%ctx) ins(%data : tensor<8x16xf32>)
      outs(%ne : tensor<8x16xf32>) : tensor<8x16xf32>
  %axes = hip.constant {value = dense<1> : tensor<i64>} : tensor<i64>
  %empty = tensor.empty() : tensor<8x1xf32>
  %sum = hip.reduce_sum(%ctx) ins(%n, %axes : tensor<8x16xf32>, tensor<i64>)
      outs(%empty : tensor<8x1xf32>)
      {keepdims = 1 : i64, noop_with_empty_axes = 0 : i64, reduced_axes = array<i64: 1>}
      : tensor<8x1xf32>
  return %sum : tensor<8x1xf32>
}

// -----

// The trailing pointwise kernel is printed ahead of the reduce kernel.
// CHECK-LABEL: func.func @mlir_neg_abs_1
// CHECK-SAME: attributes {rock.elementwise, rock.kernel}
// CHECK: hip.neg
// CHECK: hip.abs
// CHECK-LABEL: func.func @mlir_reduce_sum_0
// CHECK-SAME: attributes {rock.kernel, rock.reduce}
// CHECK-NOT: hip.neg
// CHECK-LABEL: func.func @main_graph
// CHECK: hip.rocmlir
// CHECK: hip.rocmlir
// CHECK-NOT: hip.reduce_sum
// CHECK-NOT: hip.neg

func.func @main_graph(%ctx: !hip.context, %data: tensor<8x16xf32>) -> tensor<8x1xf32> {
  %axes = hip.constant {value = dense<1> : tensor<i64>} : tensor<i64>
  %empty = tensor.empty() : tensor<8x1xf32>
  %sum = hip.reduce_sum(%ctx) ins(%data, %axes : tensor<8x16xf32>, tensor<i64>)
      outs(%empty : tensor<8x1xf32>)
      {keepdims = 1 : i64, noop_with_empty_axes = 0 : i64, reduced_axes = array<i64: 1>}
      : tensor<8x1xf32>
  %ne = tensor.empty() : tensor<8x1xf32>
  %n = hip.neg(%ctx) ins(%sum : tensor<8x1xf32>)
      outs(%ne : tensor<8x1xf32>) : tensor<8x1xf32>
  %ae = tensor.empty() : tensor<8x1xf32>
  %a = hip.abs(%ctx) ins(%n : tensor<8x1xf32>)
      outs(%ae : tensor<8x1xf32>) : tensor<8x1xf32>
  return %a : tensor<8x1xf32>
}

// -----

// CHECK-LABEL: func.func @mlir_reduce_mean_0
// CHECK-SAME: attributes {rock.kernel, rock.reduce}
// CHECK: hip.reduce_mean
// CHECK-LABEL: func.func @main_graph
// CHECK: hip.rocmlir
// CHECK-NOT: hip.reduce_mean

func.func @main_graph(%ctx: !hip.context, %data: tensor<8x16xf32>) -> tensor<8x1xf32> {
  %axes = hip.constant {value = dense<1> : tensor<i64>} : tensor<i64>
  %empty = tensor.empty() : tensor<8x1xf32>
  %mean = hip.reduce_mean(%ctx) ins(%data, %axes : tensor<8x16xf32>, tensor<i64>)
      outs(%empty : tensor<8x1xf32>)
      {keepdims = 1 : i64, noop_with_empty_axes = 0 : i64, reduced_axes = array<i64: 1>}
      : tensor<8x1xf32>
  return %mean : tensor<8x1xf32>
}

// -----

// CHECK-NOT: rock.reduce
// CHECK-LABEL: func.func @main_graph
// CHECK: hip.reduce_max

func.func @main_graph(%ctx: !hip.context, %data: tensor<8x16xf32>) -> tensor<8x1xf32> {
  %axes = hip.constant {value = dense<1> : tensor<i64>} : tensor<i64>
  %empty = tensor.empty() : tensor<8x1xf32>
  %max = hip.reduce_max(%ctx) ins(%data, %axes : tensor<8x16xf32>, tensor<i64>)
      outs(%empty : tensor<8x1xf32>)
      {keepdims = 1 : i64, noop_with_empty_axes = 0 : i64, reduced_axes = array<i64: 1>}
      : tensor<8x1xf32>
  return %max : tensor<8x1xf32>
}

// -----

// CHECK-NOT: rock.reduce
// CHECK-LABEL: func.func @main_graph
// CHECK: hip.reduce_sum

func.func @main_graph(%ctx: !hip.context, %data: tensor<8x16xf32>) -> tensor<1x16xf32> {
  %axes = hip.constant {value = dense<0> : tensor<i64>} : tensor<i64>
  %empty = tensor.empty() : tensor<1x16xf32>
  %sum = hip.reduce_sum(%ctx) ins(%data, %axes : tensor<8x16xf32>, tensor<i64>)
      outs(%empty : tensor<1x16xf32>)
      {keepdims = 1 : i64, noop_with_empty_axes = 0 : i64, reduced_axes = array<i64: 0>}
      : tensor<1x16xf32>
  return %sum : tensor<1x16xf32>
}
