// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// Windows the pass cannot turn into an output shape, either because the region
// cannot reach the values or because no value would settle the shape. The
// runtime resolves all of these at launch, so only the shape region turns them
// down.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --split-input-file -hipsr-populate-shape-region --verify-diagnostics

// A normal region gets only the data's shape, so a bound the graph computes is
// out of reach, even though a barrier region could read it.
func.func @runtime_bound_without_barrier(%ctx: !hipsr.context,
                                         %data: tensor<8xf16, #hipsr.mem<device>>) {
  %axis = arith.constant 0 : index
  %size = tensor.dim %data, %axis : tensor<8xf16, #hipsr.mem<device>>
  %entry = arith.index_cast %size : index to i64
  %ends = tensor.from_elements %entry : tensor<1xi64, #hipsr.mem<host>>
  %init = hipsr.placeholder(%ctx)
      ins(%data : tensor<8xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>}
      : tensor<?xf16, #hipsr.mem<device>>
  // expected-error@+1 {{a bound the graph computes needs a barrier placeholder, whose shape region reads the operands}}
  %result = hipsr.slice(%ctx)
      ins(%data : tensor<8xf16, #hipsr.mem<device>>)
      ends(%ends : tensor<1xi64, #hipsr.mem<host>>)
      outs(%init : tensor<?xf16, #hipsr.mem<device>>)
      {starts_attr = array<i64: 0>, axes_attr = array<i64: 0>,
       steps_attr = array<i64: 1>}
      : tensor<?xf16, #hipsr.mem<device>>
  return
}

// -----

// Which axis each entry narrows decides where the sizes go, so even a barrier
// region, which does read the operand, cannot place them.
func.func @runtime_axes(%ctx: !hipsr.context,
                        %data: tensor<8x4xf16, #hipsr.mem<device>>,
                        %axes: tensor<1xi64, #hipsr.mem<host>>) {
  %init = hipsr.placeholder(%ctx)
      ins(%data, %axes : tensor<8x4xf16, #hipsr.mem<device>>,
                         tensor<1xi64, #hipsr.mem<host>>)
      {placeholder_type = #hipsr.placeholder_type<barrier>}
      : tensor<?x?xf16, #hipsr.mem<device>>
  // expected-error@+1 {{needs compile-time axes and steps to tell which axes narrow and which way each window runs}}
  %result = hipsr.slice(%ctx)
      ins(%data : tensor<8x4xf16, #hipsr.mem<device>>)
      axes(%axes : tensor<1xi64, #hipsr.mem<host>>)
      outs(%init : tensor<?x?xf16, #hipsr.mem<device>>)
      {starts_attr = array<i64: 1>, ends_attr = array<i64: 4>,
       steps_attr = array<i64: 1>}
      : tensor<?x?xf16, #hipsr.mem<device>>
  return
}

// -----

// A zero step would take no stride through the axis, which ONNX calls an error.
func.func @zero_step(%ctx: !hipsr.context,
                     %data: tensor<8xf16, #hipsr.mem<device>>) {
  %init = hipsr.placeholder(%ctx)
      ins(%data : tensor<8xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>}
      : tensor<3xf16, #hipsr.mem<device>>
  // expected-error@+1 {{steps must not be zero}}
  %result = hipsr.slice(%ctx)
      ins(%data : tensor<8xf16, #hipsr.mem<device>>)
      outs(%init : tensor<3xf16, #hipsr.mem<device>>)
      {starts_attr = array<i64: 1>, ends_attr = array<i64: 4>,
       axes_attr = array<i64: 0>, steps_attr = array<i64: 0>}
      : tensor<3xf16, #hipsr.mem<device>>
  return
}

// -----

// Two entries for one axis leave it two sizes, and for rank-1 data 0 and -1
// name the same axis.
func.func @repeated_axis(%ctx: !hipsr.context,
                         %data: tensor<8xf16, #hipsr.mem<device>>) {
  %init = hipsr.placeholder(%ctx)
      ins(%data : tensor<8xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>}
      : tensor<3xf16, #hipsr.mem<device>>
  // expected-error@+1 {{axes must be distinct; got axis 0 twice}}
  %result = hipsr.slice(%ctx)
      ins(%data : tensor<8xf16, #hipsr.mem<device>>)
      outs(%init : tensor<3xf16, #hipsr.mem<device>>)
      {starts_attr = array<i64: 1, 2>, ends_attr = array<i64: 4, 5>,
       axes_attr = array<i64: 0, -1>, steps_attr = array<i64: 1, 1>}
      : tensor<3xf16, #hipsr.mem<device>>
  return
}

// -----

// Only the data's own axes can be sliced.
func.func @axis_out_of_range(%ctx: !hipsr.context,
                             %data: tensor<8xf16, #hipsr.mem<device>>) {
  %init = hipsr.placeholder(%ctx)
      ins(%data : tensor<8xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>}
      : tensor<3xf16, #hipsr.mem<device>>
  // expected-error@+1 {{axes must be in [-data rank, data rank); got 2}}
  %result = hipsr.slice(%ctx)
      ins(%data : tensor<8xf16, #hipsr.mem<device>>)
      outs(%init : tensor<3xf16, #hipsr.mem<device>>)
      {starts_attr = array<i64: 1>, ends_attr = array<i64: 4>,
       axes_attr = array<i64: 2>, steps_attr = array<i64: 1>}
      : tensor<3xf16, #hipsr.mem<device>>
  return
}
