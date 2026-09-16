// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -split-input-file -hipsr-inline-regions | FileCheck %s

// Operands replace block arguments. The yield operand replaces the domain
// result. get_pool and memref.view stay in the parent.
// CHECK-LABEL:   func.func @inline_pool_domain(
// CHECK-SAME:      %[[ARG0:.*]]: !hipsr.context,
// CHECK-SAME:      %[[ARG1:.*]]: memref<4xf16, #hipsr.mem<device>>) -> memref<4xf16, #hipsr.mem<device>> {
// CHECK-NEXT:      %[[VAL_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 256 : index
// CHECK-NEXT:      %[[VAL_2:.*]] = hipsr.get_pool(%[[ARG0]], %[[VAL_1]]) {bufferization.manual_deallocation, domain_id = 0 : i64} : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT:      %[[VAL_3:.*]] = memref.view %[[VAL_2]]{{\[}}%[[VAL_0]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.add(%[[ARG0]]) ins(%[[ARG1]], %[[ARG1]] : memref<4xf16, #hipsr.mem<device>>, memref<4xf16, #hipsr.mem<device>>) outs(%[[VAL_3]] : memref<4xf16, #hipsr.mem<device>>)
// CHECK-NEXT:      return %[[VAL_3]] : memref<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:    }
func.func @inline_pool_domain(%ctx: !hipsr.context,
                              %in: memref<4xf16, #hipsr.mem<device>>)
    -> memref<4xf16, #hipsr.mem<device>> {
  %r = hipsr.pool_domain(%ctx, %in : !hipsr.context, memref<4xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<4xf16, #hipsr.mem<device>>):
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %pool = hipsr.get_pool(%dctx, %c256) {bufferization.manual_deallocation, domain_id = 0 : i64}
        : memref<?xi8, #hipsr.mem<device>>
    %out = memref.view %pool[%c0][] : memref<?xi8, #hipsr.mem<device>>
        to memref<4xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<4xf16, #hipsr.mem<device>>, memref<4xf16, #hipsr.mem<device>>)
               outs(%out : memref<4xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield %out : memref<4xf16, #hipsr.mem<device>>
  } -> memref<4xf16, #hipsr.mem<device>> {domain_id = 0 : i64}
  return %r : memref<4xf16, #hipsr.mem<device>>
}

// -----

// The yield operand replaces the compute result, so later uses read %out.
// CHECK-LABEL:   func.func @inline_compute(
// CHECK-SAME:      %[[ARG0:.*]]: !hipsr.context,
// CHECK-SAME:      %[[ARG1:.*]]: memref<4xf32, #hipsr.mem<device>>,
// CHECK-SAME:      %[[ARG2:.*]]: memref<4xf32, #hipsr.mem<device>>) -> memref<4xf32, #hipsr.mem<device>> {
// CHECK-NEXT:      memref.copy %[[ARG1]], %[[ARG2]] : memref<4xf32, #hipsr.mem<device>> to memref<4xf32, #hipsr.mem<device>>
// CHECK-NEXT:      return %[[ARG2]] : memref<4xf32, #hipsr.mem<device>>
// CHECK-NEXT:    }
func.func @inline_compute(%ctx: !hipsr.context,
                         %in: memref<4xf32, #hipsr.mem<device>>,
                         %out: memref<4xf32, #hipsr.mem<device>>)
    -> memref<4xf32, #hipsr.mem<device>> {
  %r = hipsr.compute(%ctx) ins(%in : memref<4xf32, #hipsr.mem<device>>)
                           outs(%out : memref<4xf32, #hipsr.mem<device>>) {
  ^bb0(%c: !hipsr.context, %i: memref<4xf32, #hipsr.mem<device>>,
       %o: memref<4xf32, #hipsr.mem<device>>):
    memref.copy %i, %o : memref<4xf32, #hipsr.mem<device>> to memref<4xf32, #hipsr.mem<device>>
    hipsr.compute_yield %o : memref<4xf32, #hipsr.mem<device>>
  } : memref<4xf32, #hipsr.mem<device>>
  return %r : memref<4xf32, #hipsr.mem<device>>
}

// -----

// Innermost first: the compute body lands in the domain, then the domain
// lands in the function. The pooled view is the copy destination and the
// returned value.
// CHECK-LABEL:   func.func @inline_nested(
// CHECK-SAME:      %[[ARG0:.*]]: !hipsr.context,
// CHECK-SAME:      %[[ARG1:.*]]: memref<4xf32, #hipsr.mem<device>>) -> memref<4xf32, #hipsr.mem<device>> {
// CHECK-NEXT:      %[[VAL_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 256 : index
// CHECK-NEXT:      %[[VAL_2:.*]] = hipsr.get_pool(%[[ARG0]], %[[VAL_1]]) {bufferization.manual_deallocation, domain_id = 0 : i64} : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT:      %[[VAL_3:.*]] = memref.view %[[VAL_2]]{{\[}}%[[VAL_0]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4xf32, #hipsr.mem<device>>
// CHECK-NEXT:      memref.copy %[[ARG1]], %[[VAL_3]] : memref<4xf32, #hipsr.mem<device>> to memref<4xf32, #hipsr.mem<device>>
// CHECK-NEXT:      return %[[VAL_3]] : memref<4xf32, #hipsr.mem<device>>
// CHECK-NEXT:    }
func.func @inline_nested(%ctx: !hipsr.context,
                        %in: memref<4xf32, #hipsr.mem<device>>)
    -> memref<4xf32, #hipsr.mem<device>> {
  %r = hipsr.pool_domain(%ctx, %in : !hipsr.context, memref<4xf32, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<4xf32, #hipsr.mem<device>>):
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %pool = hipsr.get_pool(%dctx, %c256) {bufferization.manual_deallocation, domain_id = 0 : i64}
        : memref<?xi8, #hipsr.mem<device>>
    %tmp = memref.view %pool[%c0][] : memref<?xi8, #hipsr.mem<device>>
        to memref<4xf32, #hipsr.mem<device>>
    %inner = hipsr.compute(%dctx) ins(%din : memref<4xf32, #hipsr.mem<device>>)
                                  outs(%tmp : memref<4xf32, #hipsr.mem<device>>) {
    ^bb1(%c: !hipsr.context, %i: memref<4xf32, #hipsr.mem<device>>,
         %o: memref<4xf32, #hipsr.mem<device>>):
      memref.copy %i, %o : memref<4xf32, #hipsr.mem<device>> to memref<4xf32, #hipsr.mem<device>>
      hipsr.compute_yield %o : memref<4xf32, #hipsr.mem<device>>
    } : memref<4xf32, #hipsr.mem<device>>
    hipsr.pool_domain_yield %inner : memref<4xf32, #hipsr.mem<device>>
  } -> memref<4xf32, #hipsr.mem<device>> {domain_id = 0 : i64}
  return %r : memref<4xf32, #hipsr.mem<device>>
}
