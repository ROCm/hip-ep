// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// hipsr.nonzero round-trip and verifier rules.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// The capacity is dynamic because it follows the input's element count, while
// the row count is the input rank.
// CHECK-LABEL: func.func @nonzero_mask(
// CHECK-SAME:    %[[CTX:.+]]: !hipsr.context, %[[MASK:.+]]: tensor<?x?x?xui8>,
// CHECK-SAME:    %[[IDS_INIT:.+]]: tensor<3x?xi64>,
// CHECK-SAME:    %[[COUNT_INIT:.+]]: tensor<1xi64>) -> (tensor<3x?xi64>, tensor<1xi64>) {
// CHECK-NEXT:    %[[RESULT:.*]]:2 = hipsr.nonzero(%[[CTX]]) ins(%[[MASK]] : tensor<?x?x?xui8>) outs(%[[IDS_INIT]], %[[COUNT_INIT]] : tensor<3x?xi64>, tensor<1xi64>) : tensor<3x?xi64>, tensor<1xi64>
// CHECK-NEXT:    return %[[RESULT]]#0, %[[RESULT]]#1 : tensor<3x?xi64>, tensor<1xi64>
func.func @nonzero_mask(
    %ctx: !hipsr.context, %mask: tensor<?x?x?xui8>,
    %indices_init: tensor<3x?xi64>,
    %count_init: tensor<1xi64>) -> (tensor<3x?xi64>, tensor<1xi64>) {
  %indices, %count = hipsr.nonzero(%ctx)
      ins(%mask : tensor<?x?x?xui8>)
      outs(%indices_init, %count_init : tensor<3x?xi64>, tensor<1xi64>)
      : tensor<3x?xi64>, tensor<1xi64>
  return %indices, %count : tensor<3x?xi64>, tensor<1xi64>
}

// -----

// A statically sized input pins the capacity at its element count.
// CHECK-LABEL: func.func @nonzero_static_capacity(
// CHECK:         hipsr.nonzero(%{{.+}}) ins(%{{.+}} : tensor<2x3xi1>) outs(%{{.+}}, %{{.+}} : tensor<2x6xi64>, tensor<1xi64>) : tensor<2x6xi64>, tensor<1xi64>
func.func @nonzero_static_capacity(
    %ctx: !hipsr.context, %mask: tensor<2x3xi1>,
    %indices_init: tensor<2x6xi64>,
    %count_init: tensor<1xi64>) -> (tensor<2x6xi64>, tensor<1xi64>) {
  %indices, %count = hipsr.nonzero(%ctx)
      ins(%mask : tensor<2x3xi1>)
      outs(%indices_init, %count_init : tensor<2x6xi64>, tensor<1xi64>)
      : tensor<2x6xi64>, tensor<1xi64>
  return %indices, %count : tensor<2x6xi64>, tensor<1xi64>
}

// -----

// Positions are i64, whatever the input holds.
func.func @nonzero_indices_element_type(
    %ctx: !hipsr.context, %mask: tensor<2x3xi1>,
    %indices_init: tensor<2x6xi32>,
    %count_init: tensor<1xi64>) -> (tensor<2x6xi32>, tensor<1xi64>) {
  // expected-error@+1 {{indices element type must be i64}}
  %indices, %count = hipsr.nonzero(%ctx)
      ins(%mask : tensor<2x3xi1>)
      outs(%indices_init, %count_init : tensor<2x6xi32>, tensor<1xi64>)
      : tensor<2x6xi32>, tensor<1xi64>
  return %indices, %count : tensor<2x6xi32>, tensor<1xi64>
}

// -----

// So is the count.
func.func @nonzero_count_element_type(
    %ctx: !hipsr.context, %mask: tensor<2x3xi1>,
    %indices_init: tensor<2x6xi64>,
    %count_init: tensor<1xi32>) -> (tensor<2x6xi64>, tensor<1xi32>) {
  // expected-error@+1 {{count element type must be i64}}
  %indices, %count = hipsr.nonzero(%ctx)
      ins(%mask : tensor<2x3xi1>)
      outs(%indices_init, %count_init : tensor<2x6xi64>, tensor<1xi32>)
      : tensor<2x6xi64>, tensor<1xi32>
  return %indices, %count : tensor<2x6xi64>, tensor<1xi32>
}

// -----

// One row per axis and one column per position leaves no room for a third
// axis.
func.func @nonzero_indices_rank(
    %ctx: !hipsr.context, %mask: tensor<2x3xi1>,
    %indices_init: tensor<2x6x1xi64>,
    %count_init: tensor<1xi64>) -> (tensor<2x6x1xi64>, tensor<1xi64>) {
  // expected-error@+1 {{indices must be rank-2: one row per input axis, one column per position found}}
  %indices, %count = hipsr.nonzero(%ctx)
      ins(%mask : tensor<2x3xi1>)
      outs(%indices_init, %count_init : tensor<2x6x1xi64>, tensor<1xi64>)
      : tensor<2x6x1xi64>, tensor<1xi64>
  return %indices, %count : tensor<2x6x1xi64>, tensor<1xi64>
}

// -----

// The row count is the input rank, which is known at compile time.
func.func @nonzero_dynamic_rows(
    %ctx: !hipsr.context, %mask: tensor<2x3xi1>,
    %indices_init: tensor<?x6xi64>,
    %count_init: tensor<1xi64>) -> (tensor<?x6xi64>, tensor<1xi64>) {
  // expected-error@+1 {{indices must have a static row count}}
  %indices, %count = hipsr.nonzero(%ctx)
      ins(%mask : tensor<2x3xi1>)
      outs(%indices_init, %count_init : tensor<?x6xi64>, tensor<1xi64>)
      : tensor<?x6xi64>, tensor<1xi64>
  return %indices, %count : tensor<?x6xi64>, tensor<1xi64>
}

// -----

// A position names every axis, so there is a row for each.
func.func @nonzero_row_count(
    %ctx: !hipsr.context, %mask: tensor<2x3xi1>,
    %indices_init: tensor<3x6xi64>,
    %count_init: tensor<1xi64>) -> (tensor<3x6xi64>, tensor<1xi64>) {
  // expected-error@+1 {{indices must have one row per input axis; input rank is 2, got 3}}
  %indices, %count = hipsr.nonzero(%ctx)
      ins(%mask : tensor<2x3xi1>)
      outs(%indices_init, %count_init : tensor<3x6xi64>, tensor<1xi64>)
      : tensor<3x6xi64>, tensor<1xi64>
  return %indices, %count : tensor<3x6xi64>, tensor<1xi64>
}

// -----

// The count is a single number.
func.func @nonzero_count_shape(
    %ctx: !hipsr.context, %mask: tensor<2x3xi1>,
    %indices_init: tensor<2x6xi64>,
    %count_init: tensor<2xi64>) -> (tensor<2x6xi64>, tensor<2xi64>) {
  // expected-error@+1 {{count must be a static single-element vector}}
  %indices, %count = hipsr.nonzero(%ctx)
      ins(%mask : tensor<2x3xi1>)
      outs(%indices_init, %count_init : tensor<2x6xi64>, tensor<2xi64>)
      : tensor<2x6xi64>, tensor<2xi64>
  return %indices, %count : tensor<2x6xi64>, tensor<2xi64>
}

// -----

// Reading the count is what resolves the published extent, so it cannot be
// dynamic either.
func.func @nonzero_dynamic_count(
    %ctx: !hipsr.context, %mask: tensor<2x3xi1>,
    %indices_init: tensor<2x6xi64>,
    %count_init: tensor<?xi64>) -> (tensor<2x6xi64>, tensor<?xi64>) {
  // expected-error@+1 {{count must be a static single-element vector}}
  %indices, %count = hipsr.nonzero(%ctx)
      ins(%mask : tensor<2x3xi1>)
      outs(%indices_init, %count_init : tensor<2x6xi64>, tensor<?xi64>)
      : tensor<2x6xi64>, tensor<?xi64>
  return %indices, %count : tensor<2x6xi64>, tensor<?xi64>
}

// -----

// The search runs on the device.
func.func @nonzero_host_input(
    %ctx: !hipsr.context,
    %mask: memref<2x3xi1, #hipsr.mem<host>>,
    %indices_init: memref<2x6xi64, #hipsr.mem<device>>,
    %count_init: memref<1xi64, #hipsr.mem<host>>) {
  // expected-error@+1 {{operand #1 must be ranked tensor or device memref}}
  hipsr.nonzero(%ctx)
      ins(%mask : memref<2x3xi1, #hipsr.mem<host>>)
      outs(%indices_init, %count_init
           : memref<2x6xi64, #hipsr.mem<device>>,
             memref<1xi64, #hipsr.mem<host>>)
  return
}

// -----

// The host reads the count, so it cannot live on the device.
func.func @nonzero_device_count(
    %ctx: !hipsr.context,
    %mask: memref<2x3xi1, #hipsr.mem<device>>,
    %indices_init: memref<2x6xi64, #hipsr.mem<device>>,
    %count_init: memref<1xi64, #hipsr.mem<device>>) {
  // expected-error@+1 {{operand #3 must be ranked tensor or host memref}}
  hipsr.nonzero(%ctx)
      ins(%mask : memref<2x3xi1, #hipsr.mem<device>>)
      outs(%indices_init, %count_init
           : memref<2x6xi64, #hipsr.mem<device>>,
             memref<1xi64, #hipsr.mem<device>>)
  return
}
