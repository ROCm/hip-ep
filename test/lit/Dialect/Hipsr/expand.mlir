// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// hipsr.expand round-trip, verification, and generated ONNX broadcast shape.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s
// RUN: hip-mlir-opt --split-input-file --verify-diagnostics -hipsr-populate-shape-region %s | FileCheck %s --check-prefix=POPULATE

// CHECK-LABEL: func.func @expand_tensor
// CHECK:      hipsr.expand(%{{.+}}) ins(%{{.+}}, %{{.+}} : tensor<?x3xf16>, tensor<2xi64>)
// CHECK-SAME:   outs(%{{.+}} : tensor<?x?xf16>) : tensor<?x?xf16>
// CHECK-NEXT: return
func.func @expand_tensor(%ctx: !hipsr.context, %input: tensor<?x3xf16>,
                         %shape: tensor<2xi64>,
                         %init: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %0 = hipsr.expand(%ctx) ins(%input, %shape : tensor<?x3xf16>, tensor<2xi64>)
                   outs(%init : tensor<?x?xf16>) : tensor<?x?xf16>
  return %0 : tensor<?x?xf16>
}

// -----

// A StartBarrier shape region receives ctx followed by its two data inputs.
// POPULATE-LABEL: func.func @expand_runtime_shape
// POPULATE: hipsr.expand(%{{.+}}) ins(%{{.+}}, %{{.+}} : tensor<?x3xf16>, tensor<2xi64>)
// POPULATE-SAME: shape_region {
// POPULATE:   ^bb0(%[[CTX:.+]]: !hipsr.context, %[[INPUT:.+]]: tensor<?x3xf16>, %[[REQUEST:.+]]: tensor<2xi64>):
// POPULATE:   %[[INPUT_SHAPE:.+]] = shape.shape_of %[[INPUT]]
// POPULATE:   %[[I0:.+]] = arith.constant 0 : index
// POPULATE:   %[[R0_I64:.+]] = tensor.extract %[[REQUEST]][%[[I0]]]
// POPULATE:   %[[R0:.+]] = arith.index_cast %[[R0_I64]] : i64 to index
// POPULATE:   %[[I1:.+]] = arith.constant 1 : index
// POPULATE:   %[[R1_I64:.+]] = tensor.extract %[[REQUEST]][%[[I1]]]
// POPULATE:   %[[R1:.+]] = arith.index_cast %[[R1_I64]] : i64 to index
// POPULATE:   %[[REQUEST_SHAPE:.+]] = shape.from_extents %[[R0]], %[[R1]]
// POPULATE:   %[[WITNESS:.+]] = shape.cstr_broadcastable %[[INPUT_SHAPE]], %[[REQUEST_SHAPE]]
// POPULATE:   %[[DIMS:.+]]:2 = shape.assuming %[[WITNESS]] -> (index, index) {
// POPULATE:     %[[BROADCAST:.+]] = shape.broadcast %[[INPUT_SHAPE]], %[[REQUEST_SHAPE]]
// POPULATE:     %[[S0:.+]] = shape.get_extent %[[BROADCAST]]
// POPULATE:     %[[D0:.+]] = shape.size_to_index %[[S0]]
// POPULATE:     %[[S1:.+]] = shape.get_extent %[[BROADCAST]]
// POPULATE:     %[[D1:.+]] = shape.size_to_index %[[S1]]
// POPULATE:     shape.assuming_yield %[[D0]], %[[D1]] : index, index
// POPULATE:   }
// POPULATE:   hipsr.shape_yield (%[[DIMS]]#0, %[[DIMS]]#1) : [f16]
func.func @expand_runtime_shape(%ctx: !hipsr.context,
                                %input: tensor<?x3xf16>,
                                %shape: tensor<2xi64>,
                                %init: tensor<?x?xf16>)
    -> tensor<?x?xf16> {
  %0 = hipsr.expand(%ctx) ins(%input, %shape : tensor<?x3xf16>, tensor<2xi64>)
                   outs(%init : tensor<?x?xf16>) : tensor<?x?xf16>
  return %0 : tensor<?x?xf16>
}

// -----

// Buffer form reads requested extents only from host-visible memory.
// CHECK-LABEL: func.func @expand_host_shape_memref
// CHECK: hipsr.expand(%{{.+}}) ins(%{{.+}}, %{{.+}} : memref<?x3xf16, #hipsr.mem<device>>, memref<2xi64, #hipsr.mem<host>>)
// CHECK-SAME: outs(%{{.+}} : memref<?x?xf16, #hipsr.mem<device>>)
// POPULATE-LABEL: func.func @expand_host_shape_memref
// POPULATE: shape_region {
// POPULATE:   ^bb0(%{{.+}}: !hipsr.context, %{{.+}}: memref<?x3xf16, #hipsr.mem<device>>, %[[REQUEST:.+]]: memref<2xi64, #hipsr.mem<host>>):
// POPULATE:   memref.load %[[REQUEST]]
// POPULATE:   memref.load %[[REQUEST]]
// POPULATE:   shape.cstr_broadcastable
// POPULATE:   shape.broadcast
// POPULATE:   hipsr.shape_yield
func.func @expand_host_shape_memref(
    %ctx: !hipsr.context,
    %input: memref<?x3xf16, #hipsr.mem<device>>,
    %shape: memref<2xi64, #hipsr.mem<host>>,
    %init: memref<?x?xf16, #hipsr.mem<device>>) {
  hipsr.expand(%ctx)
      ins(%input, %shape : memref<?x3xf16, #hipsr.mem<device>>,
                            memref<2xi64, #hipsr.mem<host>>)
      outs(%init : memref<?x?xf16, #hipsr.mem<device>>)
  return
}

// -----

// A requested shape shorter than the input keeps the leading input ranks.
// POPULATE-LABEL: func.func @expand_shorter_requested_shape
// POPULATE: %[[DIMS:.+]]:3 = shape.assuming %{{.+}} -> (index, index, index) {
// POPULATE:   shape.broadcast
// POPULATE:   shape.assuming_yield %{{.+}}, %{{.+}}, %{{.+}} : index, index, index
// POPULATE: }
// POPULATE: hipsr.shape_yield (%[[DIMS]]#0, %[[DIMS]]#1, %[[DIMS]]#2) : [f16]
func.func @expand_shorter_requested_shape(
    %ctx: !hipsr.context, %input: tensor<?x4x8xf16>,
    %shape: tensor<1xi64>, %init: tensor<?x4x8xf16>)
    -> tensor<?x4x8xf16> {
  %0 = hipsr.expand(%ctx)
      ins(%input, %shape : tensor<?x4x8xf16>, tensor<1xi64>)
      outs(%init : tensor<?x4x8xf16>) : tensor<?x4x8xf16>
  return %0 : tensor<?x4x8xf16>
}

// -----

// A requested extent of one still participates in multidirectional broadcast;
// shape.broadcast selects the aligned input extent when it is not one.
// POPULATE-LABEL: func.func @expand_requested_one
// POPULATE: tensor.extract
// POPULATE: shape.cstr_broadcastable
// POPULATE: shape.broadcast
// POPULATE: hipsr.shape_yield
func.func @expand_requested_one(%ctx: !hipsr.context,
                                %input: tensor<?x8xf16>,
                                %init: tensor<?x8xf16>)
    -> tensor<?x8xf16> {
  %shape = arith.constant dense<[1, 8]> : tensor<2xi64>
  %0 = hipsr.expand(%ctx)
      ins(%input, %shape : tensor<?x8xf16>, tensor<2xi64>)
      outs(%init : tensor<?x8xf16>) : tensor<?x8xf16>
  return %0 : tensor<?x8xf16>
}

// -----

// A longer requested shape adds leading output dimensions.
// POPULATE-LABEL: func.func @expand_leading_rank
// POPULATE: %[[DIMS:.+]]:4 = shape.assuming %{{.+}} -> (index, index, index, index) {
// POPULATE:   shape.broadcast
// POPULATE:   shape.assuming_yield %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : index, index, index, index
// POPULATE: }
// POPULATE: hipsr.shape_yield (%[[DIMS]]#0, %[[DIMS]]#1, %[[DIMS]]#2, %[[DIMS]]#3) : [f16]
func.func @expand_leading_rank(%ctx: !hipsr.context,
                               %input: tensor<?x8xf16>,
                               %shape: tensor<4xi64>,
                               %init: tensor<?x?x?x8xf16>)
    -> tensor<?x?x?x8xf16> {
  %0 = hipsr.expand(%ctx)
      ins(%input, %shape : tensor<?x8xf16>, tensor<4xi64>)
      outs(%init : tensor<?x?x?x8xf16>) : tensor<?x?x?x8xf16>
  return %0 : tensor<?x?x?x8xf16>
}

// -----

// Incompatible runtime extents are guarded before shape.broadcast.
// POPULATE-LABEL: func.func @expand_broadcast_constraint
// POPULATE: shape.cstr_broadcastable
// POPULATE-NEXT: shape.assuming
// POPULATE: shape.broadcast
func.func @expand_broadcast_constraint(%ctx: !hipsr.context,
                                       %input: tensor<2x3xf16>,
                                       %init: tensor<2x4xf16>)
    -> tensor<2x4xf16> {
  %shape = arith.constant dense<[2, 4]> : tensor<2xi64>
  %0 = hipsr.expand(%ctx)
      ins(%input, %shape : tensor<2x3xf16>, tensor<2xi64>)
      outs(%init : tensor<2x4xf16>) : tensor<2x4xf16>
  return %0 : tensor<2x4xf16>
}

// -----

func.func @expand_shape_rank(%ctx: !hipsr.context,
                             %input: tensor<2x3xf16>,
                             %shape: tensor<1x2xi64>,
                             %init: tensor<2x3xf16>) -> tensor<2x3xf16> {
  // expected-error@+1 {{shape must be rank-1}}
  %0 = hipsr.expand(%ctx)
      ins(%input, %shape : tensor<2x3xf16>, tensor<1x2xi64>)
      outs(%init : tensor<2x3xf16>) : tensor<2x3xf16>
  return %0 : tensor<2x3xf16>
}

// -----

func.func @expand_shape_element_type(%ctx: !hipsr.context,
                                     %input: tensor<2x3xf16>,
                                     %shape: tensor<2xi32>,
                                     %init: tensor<2x3xf16>)
    -> tensor<2x3xf16> {
  // expected-error@+1 {{shape element type must be i64}}
  %0 = hipsr.expand(%ctx)
      ins(%input, %shape : tensor<2x3xf16>, tensor<2xi32>)
      outs(%init : tensor<2x3xf16>) : tensor<2x3xf16>
  return %0 : tensor<2x3xf16>
}

// -----

func.func @expand_dynamic_shape_length(%ctx: !hipsr.context,
                                       %input: tensor<2x3xf16>,
                                       %shape: tensor<?xi64>,
                                       %init: tensor<2x3xf16>)
    -> tensor<2x3xf16> {
  // expected-error@+1 {{shape length must be static}}
  %0 = hipsr.expand(%ctx)
      ins(%input, %shape : tensor<2x3xf16>, tensor<?xi64>)
      outs(%init : tensor<2x3xf16>) : tensor<2x3xf16>
  return %0 : tensor<2x3xf16>
}

// -----

func.func @expand_element_mismatch(%ctx: !hipsr.context,
                                   %input: tensor<2x3xf16>,
                                   %shape: tensor<2xi64>,
                                   %init: tensor<2x3xf32>)
    -> tensor<2x3xf32> {
  // expected-error@+1 {{input and output element types must match}}
  %0 = hipsr.expand(%ctx)
      ins(%input, %shape : tensor<2x3xf16>, tensor<2xi64>)
      outs(%init : tensor<2x3xf32>) : tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// -----

func.func @expand_output_rank(%ctx: !hipsr.context,
                              %input: tensor<2x3xf16>,
                              %shape: tensor<4xi64>,
                              %init: tensor<2x3xf16>) -> tensor<2x3xf16> {
  // expected-error@+1 {{output rank must equal max(input rank, shape length); expected 4, got 2}}
  %0 = hipsr.expand(%ctx)
      ins(%input, %shape : tensor<2x3xf16>, tensor<4xi64>)
      outs(%init : tensor<2x3xf16>) : tensor<2x3xf16>
  return %0 : tensor<2x3xf16>
}

// -----

func.func @expand_init_result_mismatch(
    %ctx: !hipsr.context, %input: tensor<2x3xf16>,
    %shape: tensor<2xi64>, %init: tensor<2x3xf16>)
    -> tensor<2x4xf16> {
  // expected-error@+1 {{to match type of corresponding result}}
  %0 = hipsr.expand(%ctx)
      ins(%input, %shape : tensor<2x3xf16>, tensor<2xi64>)
      outs(%init : tensor<2x3xf16>) : tensor<2x4xf16>
  return %0 : tensor<2x4xf16>
}

// -----

func.func @expand_device_shape(
    %ctx: !hipsr.context,
    %input: memref<2x3xf16, #hipsr.mem<device>>,
    %shape: memref<2xi64, #hipsr.mem<device>>,
    %init: memref<2x3xf16, #hipsr.mem<device>>) {
  // expected-error@+1 {{operand #2 must be ranked tensor or host memref}}
  hipsr.expand(%ctx)
      ins(%input, %shape : memref<2x3xf16, #hipsr.mem<device>>,
                            memref<2xi64, #hipsr.mem<device>>)
      outs(%init : memref<2x3xf16, #hipsr.mem<device>>)
  return
}
