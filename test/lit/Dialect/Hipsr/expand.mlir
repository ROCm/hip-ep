// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// hipsr.expand round-trip, shape-region population, and verification.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s
// RUN: hip-mlir-opt --split-input-file --verify-diagnostics -hipsr-populate-shape-region %s | FileCheck %s --check-prefix=POPULATE

// Tensor form round-trips without a shape region. Population checks the
// complete rank-2 ONNX broadcast shape computation.
// CHECK-LABEL: func.func @expand_tensor
// CHECK:      %[[RESULT:.+]] = hipsr.expand(%{{.+}}) ins(%{{.+}}, %{{.+}} : tensor<?x3xf16>, tensor<2xi64>)
// CHECK-SAME:   outs(%{{.+}} : tensor<?x?xf16>) : tensor<?x?xf16>
// CHECK-NOT:  shape_region
// CHECK-NEXT: return %[[RESULT]] : tensor<?x?xf16>
// POPULATE-LABEL: func.func @expand_tensor
// POPULATE:      %[[RESULT:.+]] = hipsr.expand(%{{.+}}) ins(%{{.+}}, %{{.+}} : tensor<?x3xf16>, tensor<2xi64>)
// POPULATE-SAME:   shape_region {
// POPULATE-NEXT: ^bb0(%{{.+}}: !hipsr.context, %[[INPUT:.+]]: tensor<?x3xf16>, %[[REQUEST:.+]]: tensor<2xi64>):
// POPULATE-NEXT:   %[[INPUT_SHAPE:.+]] = shape.shape_of %[[INPUT]] : tensor<?x3xf16> -> tensor<2xindex>
// POPULATE-NEXT:   %[[INDEX0:.+]] = arith.constant 0 : index
// POPULATE-NEXT:   %[[REQUEST0_I64:.+]] = tensor.extract %[[REQUEST]][%[[INDEX0]]] : tensor<2xi64>
// POPULATE-NEXT:   %[[REQUEST0:.+]] = arith.index_cast %[[REQUEST0_I64]] : i64 to index
// POPULATE-NEXT:   %[[INDEX1:.+]] = arith.constant 1 : index
// POPULATE-NEXT:   %[[REQUEST1_I64:.+]] = tensor.extract %[[REQUEST]][%[[INDEX1]]] : tensor<2xi64>
// POPULATE-NEXT:   %[[REQUEST1:.+]] = arith.index_cast %[[REQUEST1_I64]] : i64 to index
// POPULATE-NEXT:   %[[REQUEST_SHAPE:.+]] = shape.from_extents %[[REQUEST0]], %[[REQUEST1]] : index, index
// POPULATE-NEXT:   %[[WITNESS:.+]] = shape.cstr_broadcastable %[[INPUT_SHAPE]], %[[REQUEST_SHAPE]] : tensor<2xindex>, !shape.shape
// POPULATE-NEXT:   %[[DIMS:.+]]:2 = shape.assuming %[[WITNESS]] -> (index, index) {
// POPULATE-NEXT:     %[[BROADCAST:.+]] = shape.broadcast %[[INPUT_SHAPE]], %[[REQUEST_SHAPE]] : tensor<2xindex>, !shape.shape -> !shape.shape
// POPULATE-NEXT:     %[[SIZE_INDEX0:.+]] = shape.const_size 0
// POPULATE-NEXT:     %[[SIZE0:.+]] = shape.get_extent %[[BROADCAST]], %[[SIZE_INDEX0]] : !shape.shape, !shape.size -> !shape.size
// POPULATE-NEXT:     %[[DIM0:.+]] = shape.size_to_index %[[SIZE0]] : !shape.size
// POPULATE-NEXT:     %[[SIZE_INDEX1:.+]] = shape.const_size 1
// POPULATE-NEXT:     %[[SIZE1:.+]] = shape.get_extent %[[BROADCAST]], %[[SIZE_INDEX1]] : !shape.shape, !shape.size -> !shape.size
// POPULATE-NEXT:     %[[DIM1:.+]] = shape.size_to_index %[[SIZE1]] : !shape.size
// POPULATE-NEXT:     shape.assuming_yield %[[DIM0]], %[[DIM1]] : index, index
// POPULATE-NEXT:   }
// POPULATE-NEXT:   hipsr.shape_yield (%[[DIMS]]#0, %[[DIMS]]#1) : [f16]
// POPULATE-NEXT: }
// POPULATE-NEXT: return %[[RESULT]] : tensor<?x?xf16>
func.func @expand_tensor(%ctx: !hipsr.context, %input: tensor<?x3xf16>,
                         %shape: tensor<2xi64>,
                         %init: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %0 = hipsr.expand(%ctx) ins(%input, %shape : tensor<?x3xf16>, tensor<2xi64>)
                   outs(%init : tensor<?x?xf16>) : tensor<?x?xf16>
  return %0 : tensor<?x?xf16>
}

// -----

// Buffer form reads the requested extents from host-visible memory.
// CHECK-LABEL: func.func @expand_host_shape_memref
// CHECK: hipsr.expand(%{{.+}}) ins(%{{.+}}, %{{.+}} : memref<?x3xf16, #hipsr.mem<device>>, memref<2xi64, #hipsr.mem<host>>)
// CHECK-SAME: outs(%{{.+}} : memref<?x?xf16, #hipsr.mem<device>>)
// CHECK-NOT: shape_region
// CHECK-NEXT: return
// POPULATE-LABEL: func.func @expand_host_shape_memref
// POPULATE:      hipsr.expand(%{{.+}}) ins(%{{.+}}, %{{.+}} : memref<?x3xf16, #hipsr.mem<device>>, memref<2xi64, #hipsr.mem<host>>)
// POPULATE-SAME:   shape_region {
// POPULATE-NEXT: ^bb0(%{{.+}}: !hipsr.context, %[[INPUT:.+]]: memref<?x3xf16, #hipsr.mem<device>>, %[[REQUEST:.+]]: memref<2xi64, #hipsr.mem<host>>):
// POPULATE-NEXT:   %{{.+}} = shape.shape_of %[[INPUT]]
// POPULATE-NEXT:   %[[INDEX0:.+]] = arith.constant 0 : index
// POPULATE-NEXT:   %[[REQUEST0_I64:.+]] = memref.load %[[REQUEST]][%[[INDEX0]]] : memref<2xi64, #hipsr.mem<host>>
// POPULATE-NEXT:   %[[REQUEST0:.+]] = arith.index_cast %[[REQUEST0_I64]] : i64 to index
// POPULATE-NEXT:   %[[INDEX1:.+]] = arith.constant 1 : index
// POPULATE-NEXT:   %[[REQUEST1_I64:.+]] = memref.load %[[REQUEST]][%[[INDEX1]]] : memref<2xi64, #hipsr.mem<host>>
// POPULATE-NEXT:   %[[REQUEST1:.+]] = arith.index_cast %[[REQUEST1_I64]] : i64 to index
// POPULATE-NEXT:   %{{.+}} = shape.from_extents %[[REQUEST0]], %[[REQUEST1]] : index, index
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
// POPULATE:   shape.assuming_yield %{{.+}}, %{{.+}}, %{{.+}} : index, index, index
// POPULATE-NEXT: }
// POPULATE-NEXT: hipsr.shape_yield (%[[DIMS]]#0, %[[DIMS]]#1, %[[DIMS]]#2) : [f16]
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

// A longer requested shape adds leading output dimensions.
// POPULATE-LABEL: func.func @expand_leading_rank
// POPULATE: %[[DIMS:.+]]:4 = shape.assuming %{{.+}} -> (index, index, index, index) {
// POPULATE:   shape.assuming_yield %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : index, index, index, index
// POPULATE-NEXT: }
// POPULATE-NEXT: hipsr.shape_yield (%[[DIMS]]#0, %[[DIMS]]#1, %[[DIMS]]#2, %[[DIMS]]#3) : [f16]
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

// Shape must be a rank-1 extent vector.
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

// ONNX shape extents use i64 elements.
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

// The static shape length determines the output rank.
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

// Expand preserves the input element type.
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

// Output rank is max(input rank, requested shape length).
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

// The shape is read on the host to determine the output allocation.
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
