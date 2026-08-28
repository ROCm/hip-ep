// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// Checks how hipsr.placeholder relates to a hipsr.compute consumer. Because
// compute is not DPS, its outs entry only describes the destination and may
// have a different type from the result held there.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// The placeholder gives compute a 2x3 destination while the body yields the
// flattened 6-element result. A DPS consumer would reject this pair.
// CHECK-LABEL: func.func @compute_outs_shape_differs(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[DATA:.+]]: tensor<2x3xf16, #hipsr.mem<device>>) -> tensor<6xf16, #hipsr.mem<device>> {
// CHECK-NEXT: %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[DATA]] : tensor<2x3xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<2x3xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.compute(%[[CTX]]) ins(%[[DATA]] : tensor<2x3xf16, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<2x3xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%{{.+}}: !hipsr.context, %[[IN:.+]]: tensor<2x3xf16, #hipsr.mem<device>>, %{{.+}}: tensor<2x3xf16, #hipsr.mem<device>>):
// CHECK-NEXT: %[[FLAT:.+]] = tensor.collapse_shape %[[IN]] {{\[\[}}0, 1]] : tensor<2x3xf16, #hipsr.mem<device>> into tensor<6xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.compute_yield %[[FLAT]] : tensor<6xf16, #hipsr.mem<device>>
// CHECK-NEXT: } : tensor<6xf16, #hipsr.mem<device>>{{$}}
// CHECK-NEXT: return %[[RESULT]] : tensor<6xf16, #hipsr.mem<device>>
// CHECK-NEXT: }
func.func @compute_outs_shape_differs(
    %ctx: !hipsr.context, %data: tensor<2x3xf16, #hipsr.mem<device>>) -> tensor<6xf16, #hipsr.mem<device>> {
  %init = hipsr.placeholder(%ctx)
      ins(%data : tensor<2x3xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<2x3xf16, #hipsr.mem<device>>
  %out = hipsr.compute(%ctx) ins(%data : tensor<2x3xf16, #hipsr.mem<device>>)
                             outs(%init : tensor<2x3xf16, #hipsr.mem<device>>) {
  ^bb0(%body_ctx: !hipsr.context, %in: tensor<2x3xf16, #hipsr.mem<device>>,
       %dest: tensor<2x3xf16, #hipsr.mem<device>>):
    %flat = tensor.collapse_shape %in [[0, 1]]
        : tensor<2x3xf16, #hipsr.mem<device>> into tensor<6xf16, #hipsr.mem<device>>
    hipsr.compute_yield %flat : tensor<6xf16, #hipsr.mem<device>>
  } : tensor<6xf16, #hipsr.mem<device>>
  return %out : tensor<6xf16, #hipsr.mem<device>>
}

// -----
// One placeholder fills every outs slot of a multi-result compute, and neither
// result keeps the destination shape.
// CHECK-LABEL: func.func @multi_result_compute_outs(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[DATA:.+]]: tensor<2x3xf16, #hipsr.mem<device>>) -> (tensor<6xf16, #hipsr.mem<device>>, tensor<3x2xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: %[[INITS:.+]]:2 = hipsr.placeholder(%[[CTX]]) ins(%[[DATA]] : tensor<2x3xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<2x3xf16, #hipsr.mem<device>>, tensor<2x3xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[RESULTS:.+]]:2 = hipsr.compute(%[[CTX]]) ins(%[[DATA]] : tensor<2x3xf16, #hipsr.mem<device>>) outs(%[[INITS]]#0, %[[INITS]]#1 : tensor<2x3xf16, #hipsr.mem<device>>, tensor<2x3xf16, #hipsr.mem<device>>) {
// CHECK: } : tensor<6xf16, #hipsr.mem<device>>, tensor<3x2xf16, #hipsr.mem<device>>{{$}}
// CHECK-NEXT: return %[[RESULTS]]#0, %[[RESULTS]]#1 : tensor<6xf16, #hipsr.mem<device>>, tensor<3x2xf16, #hipsr.mem<device>>
// CHECK-NEXT: }
func.func @multi_result_compute_outs(
    %ctx: !hipsr.context, %data: tensor<2x3xf16, #hipsr.mem<device>>)
    -> (tensor<6xf16, #hipsr.mem<device>>, tensor<3x2xf16, #hipsr.mem<device>>) {
  %inits:2 = hipsr.placeholder(%ctx)
      ins(%data : tensor<2x3xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>}
      : tensor<2x3xf16, #hipsr.mem<device>>, tensor<2x3xf16, #hipsr.mem<device>>
  %out:2 = hipsr.compute(%ctx) ins(%data : tensor<2x3xf16, #hipsr.mem<device>>)
      outs(%inits#0, %inits#1 : tensor<2x3xf16, #hipsr.mem<device>>, tensor<2x3xf16, #hipsr.mem<device>>) {
  ^bb0(%body_ctx: !hipsr.context, %in: tensor<2x3xf16, #hipsr.mem<device>>,
       %dest0: tensor<2x3xf16, #hipsr.mem<device>>, %dest1: tensor<2x3xf16, #hipsr.mem<device>>):
    %flat = tensor.collapse_shape %in [[0, 1]]
        : tensor<2x3xf16, #hipsr.mem<device>> into tensor<6xf16, #hipsr.mem<device>>
    %swapped = tensor.expand_shape %flat [[0, 1]] output_shape [3, 2]
        : tensor<6xf16, #hipsr.mem<device>> into tensor<3x2xf16, #hipsr.mem<device>>
    hipsr.compute_yield %flat, %swapped : tensor<6xf16, #hipsr.mem<device>>, tensor<3x2xf16, #hipsr.mem<device>>
  } : tensor<6xf16, #hipsr.mem<device>>, tensor<3x2xf16, #hipsr.mem<device>>
  return %out#0, %out#1 : tensor<6xf16, #hipsr.mem<device>>, tensor<3x2xf16, #hipsr.mem<device>>
}

// -----
// A placeholder still cannot feed data into a consumer: an input operand is not
// a destination, whether or not the consumer is DPS.
func.func @compute_input_placeholder(
    %ctx: !hipsr.context, %data: tensor<2x3xf16, #hipsr.mem<device>>) {
  // expected-error @+1 {{requires each result use to be a placeholder input, pool-domain yield, or an outs operand of a hipsr operation}}
  %init = hipsr.placeholder(%ctx)
      ins(%data : tensor<2x3xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<2x3xf16, #hipsr.mem<device>>
  hipsr.compute(%ctx) ins(%init : tensor<2x3xf16, #hipsr.mem<device>>) outs() {
  ^bb0(%body_ctx: !hipsr.context, %in: tensor<2x3xf16, #hipsr.mem<device>>):
    hipsr.compute_yield
  }
  return
}

// -----
// Mixing a compute consumer and a DPS consumer still splits the placeholder
// across two ops, which the pool-domain passes cannot place.
func.func @split_compute_and_dps_consumers(
    %ctx: !hipsr.context, %data: tensor<2x3xf16, #hipsr.mem<device>>)
    -> (tensor<6xf16, #hipsr.mem<device>>, tensor<2x3xf32, #hipsr.mem<device>>) {
  // expected-error @+1 {{requires all results to initialize the same hipsr operation}}
  %inits:2 = hipsr.placeholder(%ctx)
      ins(%data : tensor<2x3xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>}
      : tensor<2x3xf16, #hipsr.mem<device>>, tensor<2x3xf32, #hipsr.mem<device>>
  %out = hipsr.compute(%ctx) ins(%data : tensor<2x3xf16, #hipsr.mem<device>>)
                             outs(%inits#0 : tensor<2x3xf16, #hipsr.mem<device>>) {
  ^bb0(%body_ctx: !hipsr.context, %in: tensor<2x3xf16, #hipsr.mem<device>>,
       %dest: tensor<2x3xf16, #hipsr.mem<device>>):
    %flat = tensor.collapse_shape %in [[0, 1]]
        : tensor<2x3xf16, #hipsr.mem<device>> into tensor<6xf16, #hipsr.mem<device>>
    hipsr.compute_yield %flat : tensor<6xf16, #hipsr.mem<device>>
  } : tensor<6xf16, #hipsr.mem<device>>
  %cast = hipsr.cast(%ctx) ins(%data : tensor<2x3xf16, #hipsr.mem<device>>)
      outs(%inits#1 : tensor<2x3xf32, #hipsr.mem<device>>) : tensor<2x3xf32, #hipsr.mem<device>>
  return %out, %cast : tensor<6xf16, #hipsr.mem<device>>, tensor<2x3xf32, #hipsr.mem<device>>
}
