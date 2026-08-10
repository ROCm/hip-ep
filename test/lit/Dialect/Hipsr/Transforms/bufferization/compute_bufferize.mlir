// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// Checks that One-Shot Bufferize rewrites hipsr.compute, its body, and its
// hipsr.compute_yield terminator into device memrefs.
//
// Each case starts from device memrefs wrapped in `bufferization.to_tensor
// restrict`, rather than from `tensor.empty`, because the buffer a hipsr op
// takes must name its memory space and One-Shot Bufferize has no default one to
// give (`memref<6xf16>` would fail the Hipsr_TensorOrDeviceMemRef constraint).
// Bufferizing a view keeps the space, so device memrefs on the way in are what
// makes device memrefs on the way out. Assigning the space to allocations that
// the pipeline itself creates is a separate, still missing, stage.
//
// The cases spell out their whole function with CHECK-NEXT, so absences are
// load bearing: a leftover `tensor.*` op, or a `bufferization.to_tensor` that
// did not fold away, shows up as an extra line and breaks the chain.
//
// Nothing here should copy: every result is a view of a buffer that crossed the
// boundary, which is the point of implementing all three alias layers.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --split-input-file --one-shot-bufferize="bufferize-function-boundaries" %s | FileCheck %s
// RUN: hip-mlir-opt --split-input-file --one-shot-bufferize="bufferize-function-boundaries" %s | FileCheck %s --check-prefix=NOCOPY

// NOCOPY-NOT: memref.copy

// The non-DPS case a DPS op cannot express: the body flattens its input, so the
// result type differs from the output it was given. The result buffer therefore
// comes from the yielded value, and the op keeps that result as a memref
// instead of losing it the way a bufferized DPS op does.
// CHECK-LABEL: func.func @flatten(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[SHAPE:.+]]: !shape.shape, %[[DATA:.+]]: memref<2x3xf16, #hipsr.mem<device>>, %[[INIT:.+]]: memref<2x3xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.compute(%[[CTX]]) ins(%[[DATA]] : memref<2x3xf16, #hipsr.mem<device>>) outs(%[[INIT]] : memref<2x3xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%[[BODY_CTX:.+]]: !hipsr.context, %[[IN:.+]]: memref<2x3xf16, #hipsr.mem<device>>, %[[DEST:.+]]: memref<2x3xf16, #hipsr.mem<device>>):
// CHECK-NEXT: %[[FLAT:.+]] = memref.collapse_shape %[[IN]] {{\[\[}}0, 1]] : memref<2x3xf16, #hipsr.mem<device>> into memref<6xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.compute_yield %[[FLAT]] : memref<6xf16, #hipsr.mem<device>>
// CHECK-NEXT: } : memref<6xf16, #hipsr.mem<device>>{{$}}
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE]], %[[RESULT]] : memref<6xf16, #hipsr.mem<device>>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @flatten(%ctx: !hipsr.context, %shape: !shape.shape,
                   %data: memref<2x3xf16, #hipsr.mem<device>>,
                   %init: memref<2x3xf16, #hipsr.mem<device>>) {
  %in = bufferization.to_tensor %data restrict
      : memref<2x3xf16, #hipsr.mem<device>> to tensor<2x3xf16>
  %dest = bufferization.to_tensor %init restrict writable
      : memref<2x3xf16, #hipsr.mem<device>> to tensor<2x3xf16>
  %out = hipsr.compute(%ctx) ins(%in : tensor<2x3xf16>)
                             outs(%dest : tensor<2x3xf16>) {
  ^bb0(%body_ctx: !hipsr.context, %body_in: tensor<2x3xf16>,
       %body_dest: tensor<2x3xf16>):
    %flat = tensor.collapse_shape %body_in [[0, 1]]
        : tensor<2x3xf16> into tensor<6xf16>
    hipsr.compute_yield %flat : tensor<6xf16>
  } : tensor<6xf16>
  hipsr.preserve_shape %shape, %out : tensor<6xf16>
  return
}

// -----
// One result per output, with a reshape chain inside: the expand reads the
// collapse, so bufferizing the body has to walk the nested ops rather than just
// the boundary.
// CHECK-LABEL: func.func @multi_result(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[SHAPE:.+]]: !shape.shape, %[[DATA:.+]]: memref<2x3xf16, #hipsr.mem<device>>, %[[INIT0:.+]]: memref<6xf16, #hipsr.mem<device>>, %[[INIT1:.+]]: memref<3x2xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: %[[RESULTS:.+]]:2 = hipsr.compute(%[[CTX]]) ins(%[[DATA]] : memref<2x3xf16, #hipsr.mem<device>>) outs(%[[INIT0]], %[[INIT1]] : memref<6xf16, #hipsr.mem<device>>, memref<3x2xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%[[BODY_CTX:.+]]: !hipsr.context, %[[IN:.+]]: memref<2x3xf16, #hipsr.mem<device>>, %[[DEST0:.+]]: memref<6xf16, #hipsr.mem<device>>, %[[DEST1:.+]]: memref<3x2xf16, #hipsr.mem<device>>):
// CHECK-NEXT: %[[FLAT:.+]] = memref.collapse_shape %[[IN]] {{\[\[}}0, 1]] : memref<2x3xf16, #hipsr.mem<device>> into memref<6xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[SWAPPED:.+]] = memref.expand_shape %[[FLAT]] {{\[\[}}0, 1]] output_shape [3, 2] : memref<6xf16, #hipsr.mem<device>> into memref<3x2xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.compute_yield %[[FLAT]], %[[SWAPPED]] : memref<6xf16, #hipsr.mem<device>>, memref<3x2xf16, #hipsr.mem<device>>
// CHECK-NEXT: } : memref<6xf16, #hipsr.mem<device>>, memref<3x2xf16, #hipsr.mem<device>>{{$}}
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE]], %[[RESULTS]]#0 : memref<6xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE]], %[[RESULTS]]#1 : memref<3x2xf16, #hipsr.mem<device>>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @multi_result(%ctx: !hipsr.context, %shape: !shape.shape,
                        %data: memref<2x3xf16, #hipsr.mem<device>>,
                        %init0: memref<6xf16, #hipsr.mem<device>>,
                        %init1: memref<3x2xf16, #hipsr.mem<device>>) {
  %in = bufferization.to_tensor %data restrict
      : memref<2x3xf16, #hipsr.mem<device>> to tensor<2x3xf16>
  %dest0 = bufferization.to_tensor %init0 restrict writable
      : memref<6xf16, #hipsr.mem<device>> to tensor<6xf16>
  %dest1 = bufferization.to_tensor %init1 restrict writable
      : memref<3x2xf16, #hipsr.mem<device>> to tensor<3x2xf16>
  %out:2 = hipsr.compute(%ctx) ins(%in : tensor<2x3xf16>)
                               outs(%dest0, %dest1 : tensor<6xf16>,
                                                     tensor<3x2xf16>) {
  ^bb0(%body_ctx: !hipsr.context, %body_in: tensor<2x3xf16>,
       %body_dest0: tensor<6xf16>, %body_dest1: tensor<3x2xf16>):
    %flat = tensor.collapse_shape %body_in [[0, 1]]
        : tensor<2x3xf16> into tensor<6xf16>
    %swapped = tensor.expand_shape %flat [[0, 1]] output_shape [3, 2]
        : tensor<6xf16> into tensor<3x2xf16>
    hipsr.compute_yield %flat, %swapped : tensor<6xf16>, tensor<3x2xf16>
  } : tensor<6xf16>, tensor<3x2xf16>
  hipsr.preserve_shape %shape, %out#0 : tensor<6xf16>
  hipsr.preserve_shape %shape, %out#1 : tensor<3x2xf16>
  return
}

// -----
// The body yields the output it was handed, so the result buffer is the output
// buffer. This is the whole alias chain in its shortest form -- output operand,
// entry block argument, yield operand, result -- and every link has to be
// Equivalent for the op to bufferize in place instead of copying.
// CHECK-LABEL: func.func @yield_destination(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[SHAPE:.+]]: !shape.shape, %[[DATA:.+]]: memref<2x3xf16, #hipsr.mem<device>>, %[[INIT:.+]]: memref<2x3xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.compute(%[[CTX]]) ins(%[[DATA]] : memref<2x3xf16, #hipsr.mem<device>>) outs(%[[INIT]] : memref<2x3xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%[[BODY_CTX:.+]]: !hipsr.context, %[[IN:.+]]: memref<2x3xf16, #hipsr.mem<device>>, %[[DEST:.+]]: memref<2x3xf16, #hipsr.mem<device>>):
// CHECK-NEXT: hipsr.compute_yield %[[DEST]] : memref<2x3xf16, #hipsr.mem<device>>
// CHECK-NEXT: } : memref<2x3xf16, #hipsr.mem<device>>{{$}}
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE]], %[[RESULT]] : memref<2x3xf16, #hipsr.mem<device>>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @yield_destination(%ctx: !hipsr.context, %shape: !shape.shape,
                             %data: memref<2x3xf16, #hipsr.mem<device>>,
                             %init: memref<2x3xf16, #hipsr.mem<device>>) {
  %in = bufferization.to_tensor %data restrict
      : memref<2x3xf16, #hipsr.mem<device>> to tensor<2x3xf16>
  %dest = bufferization.to_tensor %init restrict writable
      : memref<2x3xf16, #hipsr.mem<device>> to tensor<2x3xf16>
  %out = hipsr.compute(%ctx) ins(%in : tensor<2x3xf16>)
                             outs(%dest : tensor<2x3xf16>) {
  ^bb0(%body_ctx: !hipsr.context, %body_in: tensor<2x3xf16>,
       %body_dest: tensor<2x3xf16>):
    hipsr.compute_yield %body_dest : tensor<2x3xf16>
  } : tensor<2x3xf16>
  hipsr.preserve_shape %shape, %out : tensor<2x3xf16>
  return
}
