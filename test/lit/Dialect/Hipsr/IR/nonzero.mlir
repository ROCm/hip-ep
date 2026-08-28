// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// hipsr.nonzero round-trip and verifier rules.
//
// The search runs on the device, so the input and both destinations are
// #hipsr.mem<device>. Reading the count on the host is hipsr.copy_d2h's job.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// The capacity is dynamic because it follows the input's element count, while
// the row count is the input rank.
// CHECK-LABEL: func.func @nonzero_mask(
// CHECK-SAME:    %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME:    %[[MASK:.+]]: tensor<?x?x?xui8, #hipsr.mem<device>>,
// CHECK-SAME:    %[[IDS_INIT:.+]]: tensor<3x?xi64, #hipsr.mem<device>>,
// CHECK-SAME:    %[[COUNT_INIT:.+]]: tensor<1xi64, #hipsr.mem<device>>) -> (tensor<3x?xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<device>>) {
// CHECK-NEXT:    %[[RESULT:.*]]:2 = hipsr.nonzero(%[[CTX]]) ins(%[[MASK]] : tensor<?x?x?xui8, #hipsr.mem<device>>) outs(%[[IDS_INIT]], %[[COUNT_INIT]] : tensor<3x?xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<device>>) : tensor<3x?xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<device>>
// CHECK-NEXT:    return %[[RESULT]]#0, %[[RESULT]]#1 : tensor<3x?xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<device>>
// CHECK-NEXT:  }
func.func @nonzero_mask(
    %ctx: !hipsr.context, %mask: tensor<?x?x?xui8, #hipsr.mem<device>>,
    %indices_init: tensor<3x?xi64, #hipsr.mem<device>>,
    %count_init: tensor<1xi64, #hipsr.mem<device>>)
    -> (tensor<3x?xi64, #hipsr.mem<device>>,
        tensor<1xi64, #hipsr.mem<device>>) {
  %indices, %count = hipsr.nonzero(%ctx)
      ins(%mask : tensor<?x?x?xui8, #hipsr.mem<device>>)
      outs(%indices_init, %count_init
           : tensor<3x?xi64, #hipsr.mem<device>>,
             tensor<1xi64, #hipsr.mem<device>>)
      : tensor<3x?xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<device>>
  return %indices, %count
      : tensor<3x?xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<device>>
}

// -----

// Positions are i64, whatever the input holds.
func.func @nonzero_indices_element_type(
    %ctx: !hipsr.context, %mask: tensor<2x3xi1, #hipsr.mem<device>>,
    %indices_init: tensor<2x6xi32, #hipsr.mem<device>>,
    %count_init: tensor<1xi64, #hipsr.mem<device>>)
    -> (tensor<2x6xi32, #hipsr.mem<device>>,
        tensor<1xi64, #hipsr.mem<device>>) {
  // expected-error@+1 {{indices element type must be i64}}
  %indices, %count = hipsr.nonzero(%ctx)
      ins(%mask : tensor<2x3xi1, #hipsr.mem<device>>)
      outs(%indices_init, %count_init
           : tensor<2x6xi32, #hipsr.mem<device>>,
             tensor<1xi64, #hipsr.mem<device>>)
      : tensor<2x6xi32, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<device>>
  return %indices, %count
      : tensor<2x6xi32, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<device>>
}

// -----

// So is the count.
func.func @nonzero_count_element_type(
    %ctx: !hipsr.context, %mask: tensor<2x3xi1, #hipsr.mem<device>>,
    %indices_init: tensor<2x6xi64, #hipsr.mem<device>>,
    %count_init: tensor<1xi32, #hipsr.mem<device>>)
    -> (tensor<2x6xi64, #hipsr.mem<device>>,
        tensor<1xi32, #hipsr.mem<device>>) {
  // expected-error@+1 {{count element type must be i64}}
  %indices, %count = hipsr.nonzero(%ctx)
      ins(%mask : tensor<2x3xi1, #hipsr.mem<device>>)
      outs(%indices_init, %count_init
           : tensor<2x6xi64, #hipsr.mem<device>>,
             tensor<1xi32, #hipsr.mem<device>>)
      : tensor<2x6xi64, #hipsr.mem<device>>, tensor<1xi32, #hipsr.mem<device>>
  return %indices, %count
      : tensor<2x6xi64, #hipsr.mem<device>>, tensor<1xi32, #hipsr.mem<device>>
}

// -----

// One row per axis and one column per position leaves no room for a third
// axis.
func.func @nonzero_indices_rank(
    %ctx: !hipsr.context, %mask: tensor<2x3xi1, #hipsr.mem<device>>,
    %indices_init: tensor<2x6x1xi64, #hipsr.mem<device>>,
    %count_init: tensor<1xi64, #hipsr.mem<device>>)
    -> (tensor<2x6x1xi64, #hipsr.mem<device>>,
        tensor<1xi64, #hipsr.mem<device>>) {
  // expected-error@+1 {{indices must be rank-2: one row per input axis, one column per position found}}
  %indices, %count = hipsr.nonzero(%ctx)
      ins(%mask : tensor<2x3xi1, #hipsr.mem<device>>)
      outs(%indices_init, %count_init
           : tensor<2x6x1xi64, #hipsr.mem<device>>,
             tensor<1xi64, #hipsr.mem<device>>)
      : tensor<2x6x1xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<device>>
  return %indices, %count
      : tensor<2x6x1xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<device>>
}

// -----

// A position names every axis, so there is a row for each. A dynamic row count
// fails the same rule.
func.func @nonzero_row_count(
    %ctx: !hipsr.context, %mask: tensor<2x3xi1, #hipsr.mem<device>>,
    %indices_init: tensor<3x6xi64, #hipsr.mem<device>>,
    %count_init: tensor<1xi64, #hipsr.mem<device>>)
    -> (tensor<3x6xi64, #hipsr.mem<device>>,
        tensor<1xi64, #hipsr.mem<device>>) {
  // expected-error@+1 {{indices must have one row per input axis; input rank is 2}}
  %indices, %count = hipsr.nonzero(%ctx)
      ins(%mask : tensor<2x3xi1, #hipsr.mem<device>>)
      outs(%indices_init, %count_init
           : tensor<3x6xi64, #hipsr.mem<device>>,
             tensor<1xi64, #hipsr.mem<device>>)
      : tensor<3x6xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<device>>
  return %indices, %count
      : tensor<3x6xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<device>>
}

// -----

// The count is a single number.
func.func @nonzero_count_shape(
    %ctx: !hipsr.context, %mask: tensor<2x3xi1, #hipsr.mem<device>>,
    %indices_init: tensor<2x6xi64, #hipsr.mem<device>>,
    %count_init: tensor<2xi64, #hipsr.mem<device>>)
    -> (tensor<2x6xi64, #hipsr.mem<device>>,
        tensor<2xi64, #hipsr.mem<device>>) {
  // expected-error@+1 {{count must be a static single-element vector}}
  %indices, %count = hipsr.nonzero(%ctx)
      ins(%mask : tensor<2x3xi1, #hipsr.mem<device>>)
      outs(%indices_init, %count_init
           : tensor<2x6xi64, #hipsr.mem<device>>,
             tensor<2xi64, #hipsr.mem<device>>)
      : tensor<2x6xi64, #hipsr.mem<device>>, tensor<2xi64, #hipsr.mem<device>>
  return %indices, %count
      : tensor<2x6xi64, #hipsr.mem<device>>, tensor<2xi64, #hipsr.mem<device>>
}

// -----

// The search writes the count, so it lands on the device with the positions.
func.func @nonzero_host_count(
    %ctx: !hipsr.context,
    %mask: memref<2x3xi1, #hipsr.mem<device>>,
    %indices_init: memref<2x6xi64, #hipsr.mem<device>>,
    %count_init: memref<1xi64, #hipsr.mem<host>>) {
  // expected-error@+1 {{operand #3 must be ranked device tensor or device memref}}
  hipsr.nonzero(%ctx)
      ins(%mask : memref<2x3xi1, #hipsr.mem<device>>)
      outs(%indices_init, %count_init
           : memref<2x6xi64, #hipsr.mem<device>>,
             memref<1xi64, #hipsr.mem<host>>)
  return
}
