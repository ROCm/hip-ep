// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// Checks that One-Shot Bufferize rewrites hipsr.compute, its body, and its
// hipsr.compute_yield terminator into memrefs.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --split-input-file --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" %s | FileCheck %s
// RUN: hip-mlir-opt --split-input-file --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" %s | FileCheck %s --check-prefix=NOCOPY

// NOCOPY-NOT: memref.copy

// The non-DPS case a DPS op cannot express: the body flattens its input, so the
// result type differs from the output it was given. The result buffer therefore
// comes from the yielded value, and the op keeps that result as a memref
// instead of losing it the way a bufferized DPS op does.
// CHECK-LABEL: func.func @flatten(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[SHAPE:.+]]: !shape.shape, %[[DATA:.+]]: memref<2x3xf16>, %[[INIT:.+]]: memref<2x3xf16>) {
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.compute(%[[CTX]]) ins(%[[DATA]] : memref<2x3xf16>) outs(%[[INIT]] : memref<2x3xf16>) {
// CHECK-NEXT: ^bb0(%[[BODY_CTX:.+]]: !hipsr.context, %[[IN:.+]]: memref<2x3xf16>, %[[DEST:.+]]: memref<2x3xf16>):
// CHECK-NEXT: %[[FLAT:.+]] = memref.collapse_shape %[[IN]] {{\[\[}}0, 1]] : memref<2x3xf16> into memref<6xf16>
// CHECK-NEXT: hipsr.compute_yield %[[FLAT]] : memref<6xf16>
// CHECK-NEXT: } : memref<6xf16>{{$}}
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE]], %[[RESULT]] : memref<6xf16>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @flatten(%ctx: !hipsr.context, %shape: !shape.shape,
                          %data: tensor<2x3xf16>,
                          %init: tensor<2x3xf16>) {
  %out = hipsr.compute(%ctx) ins(%data : tensor<2x3xf16>)
                             outs(%init : tensor<2x3xf16>) {
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
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[SHAPE:.+]]: !shape.shape, %[[DATA:.+]]: memref<2x3xf16>, %[[INIT0:.+]]: memref<6xf16>, %[[INIT1:.+]]: memref<3x2xf16>) {
// CHECK-NEXT: %[[RESULTS:.+]]:2 = hipsr.compute(%[[CTX]]) ins(%[[DATA]] : memref<2x3xf16>) outs(%[[INIT0]], %[[INIT1]] : memref<6xf16>, memref<3x2xf16>) {
// CHECK-NEXT: ^bb0(%[[BODY_CTX:.+]]: !hipsr.context, %[[IN:.+]]: memref<2x3xf16>, %[[DEST0:.+]]: memref<6xf16>, %[[DEST1:.+]]: memref<3x2xf16>):
// CHECK-NEXT: %[[FLAT:.+]] = memref.collapse_shape %[[IN]] {{\[\[}}0, 1]] : memref<2x3xf16> into memref<6xf16>
// CHECK-NEXT: %[[SWAPPED:.+]] = memref.expand_shape %[[FLAT]] {{\[\[}}0, 1]] output_shape [3, 2] : memref<6xf16> into memref<3x2xf16>
// CHECK-NEXT: hipsr.compute_yield %[[FLAT]], %[[SWAPPED]] : memref<6xf16>, memref<3x2xf16>
// CHECK-NEXT: } : memref<6xf16>, memref<3x2xf16>{{$}}
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE]], %[[RESULTS]]#0 : memref<6xf16>
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE]], %[[RESULTS]]#1 : memref<3x2xf16>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @multi_result(%ctx: !hipsr.context, %shape: !shape.shape,
                        %data: tensor<2x3xf16>,
                        %init0: tensor<6xf16>,
                        %init1: tensor<3x2xf16>) {
  %out:2 = hipsr.compute(%ctx) ins(%data : tensor<2x3xf16>)
                               outs(%init0, %init1 : tensor<6xf16>,
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
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[SHAPE:.+]]: !shape.shape, %[[DATA:.+]]: memref<2x3xf16>, %[[INIT:.+]]: memref<2x3xf16>) {
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.compute(%[[CTX]]) ins(%[[DATA]] : memref<2x3xf16>) outs(%[[INIT]] : memref<2x3xf16>) {
// CHECK-NEXT: ^bb0(%[[BODY_CTX:.+]]: !hipsr.context, %[[IN:.+]]: memref<2x3xf16>, %[[DEST:.+]]: memref<2x3xf16>):
// CHECK-NEXT: hipsr.compute_yield %[[DEST]] : memref<2x3xf16>
// CHECK-NEXT: } : memref<2x3xf16>{{$}}
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE]], %[[RESULT]] : memref<2x3xf16>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @yield_destination(%ctx: !hipsr.context, %shape: !shape.shape,
                             %data: tensor<2x3xf16>,
                             %init: tensor<2x3xf16>) {
  %out = hipsr.compute(%ctx) ins(%data : tensor<2x3xf16>)
                             outs(%init : tensor<2x3xf16>) {
  ^bb0(%body_ctx: !hipsr.context, %body_in: tensor<2x3xf16>,
       %body_dest: tensor<2x3xf16>):
    hipsr.compute_yield %body_dest : tensor<2x3xf16>
  } : tensor<2x3xf16>
  hipsr.preserve_shape %shape, %out : tensor<2x3xf16>
  return
}

// -----
// A pool_domain body with DPS matmul/add followed by non-DPS compute that
// flattens its input, entered from a pure tensor boundary. The body follows the
// intended pipeline layout: shape regions first, then the allocations they
// describe, then the data ops, then the preserve_shape links.
//
// compute takes %sum as both ins and outs. The body only builds a collapse view
// of that buffer, so the result lands in init2 and needs no buffer of its own
// and the domain allocates twice rather than three times. The shared value is
// only conflict-free because bufferizesToMemoryWrite asks the body, where the
// outs block argument has no uses; reporting every outs as written would make
// the analysis allocate a second buffer for it.
//
// matmul and add are hip ops rather than hipsr ones because Hipsr_MatMulOp and
// Hipsr_AddOp still require a device memref, which a tensor boundary cannot
// produce -- bufferizing a tensor argument yields a bare memref with no memory
// space. Hip_TensorOrMemRef already admits both forms.
//
// Each preserve_shape and the domain yield read the DPS result, not the init. In
// tensor form those are two different values, and reading the init would mean
// reading what the buffer held before the write, which forces a copy.
// CHECK-LABEL: func.func @pool_domain_mlp_flatten(
// CHECK-SAME: %[[HIP_CTX:.+]]: !hip.context, %[[CTX:.+]]: !hipsr.context, %[[INPUT:.+]]: memref<?x256xf16>, %[[WEIGHT:.+]]: memref<256x512xf16>, %[[BIAS:.+]]: memref<?x512xf16>) -> memref<?xf16> {
// CHECK-NEXT: %[[OUT:.+]] = hipsr.pool_domain(%[[HIP_CTX]], %[[CTX]], %[[INPUT]], %[[WEIGHT]], %[[BIAS]] : !hip.context, !hipsr.context, memref<?x256xf16>, memref<256x512xf16>, memref<?x512xf16>) {
// CHECK-NEXT: ^bb0(%[[DHIP:.+]]: !hip.context, %[[DCTX:.+]]: !hipsr.context, %[[IN:.+]]: memref<?x256xf16>, %[[W:.+]]: memref<256x512xf16>, %[[B:.+]]: memref<?x512xf16>):
// CHECK-NEXT: %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[C1:.+]] = arith.constant 1 : index
// CHECK-NEXT: %[[M:.+]] = memref.dim %[[IN]], %[[C0]] : memref<?x256xf16>
// CHECK-NEXT: %[[N:.+]] = memref.dim %[[W]], %[[C1]] : memref<256x512xf16>
// CHECK-NEXT: %[[FLAT_SIZE:.+]] = arith.muli %[[M]], %[[N]] : index
// CHECK-NEXT: %[[SHAPE1:.+]] = scf.execute_region -> !shape.shape {
// CHECK-NEXT: %[[S1:.+]] = shape.from_extents %[[M]], %[[N]] : index, index
// CHECK-NEXT: scf.yield %[[S1]] : !shape.shape
// CHECK-NEXT: }
// CHECK-NEXT: %[[SHAPE2:.+]] = scf.execute_region -> !shape.shape {
// CHECK-NEXT: %[[S2:.+]] = shape.from_extents %[[M]], %[[N]] : index, index
// CHECK-NEXT: scf.yield %[[S2]] : !shape.shape
// CHECK-NEXT: }
// CHECK-NEXT: %[[SHAPE3:.+]] = scf.execute_region -> !shape.shape {
// CHECK-NEXT: %[[S3:.+]] = shape.from_extents %[[FLAT_SIZE]] : index
// CHECK-NEXT: scf.yield %[[S3]] : !shape.shape
// CHECK-NEXT: }
// CHECK-NEXT: %[[INIT1:.+]] = memref.alloc(%[[M]]){{.*}} : memref<?x512xf16>
// CHECK-NEXT: hip.matmul(%[[DHIP]]) ins(%[[IN]], %[[W]] : memref<?x256xf16>, memref<256x512xf16>) outs(%[[INIT1]] : memref<?x512xf16>)
// CHECK-NEXT: %[[INIT2:.+]] = memref.alloc(%[[M]]){{.*}} : memref<?x512xf16>
// CHECK-NEXT: hip.add(%[[DHIP]]) ins(%[[INIT1]], %[[B]] : memref<?x512xf16>, memref<?x512xf16>) outs(%[[INIT2]] : memref<?x512xf16>)
// CHECK-NEXT: %[[FLAT:.+]] = hipsr.compute(%[[DCTX]]) ins(%[[INIT2]] : memref<?x512xf16>) outs(%[[INIT2]] : memref<?x512xf16>) {
// CHECK-NEXT: ^bb0(%[[BODY_CTX:.+]]: !hipsr.context, %[[BODY_IN:.+]]: memref<?x512xf16>, %[[BODY_DEST:.+]]: memref<?x512xf16>):
// CHECK-NEXT: %[[COLLAPSED:.+]] = memref.collapse_shape %[[BODY_IN]] {{\[\[}}0, 1]] : memref<?x512xf16> into memref<?xf16>
// CHECK-NEXT: hipsr.compute_yield %[[COLLAPSED]] : memref<?xf16>
// CHECK-NEXT: } : memref<?xf16>{{$}}
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE1]], %[[INIT1]] : memref<?x512xf16>
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE2]], %[[INIT2]] : memref<?x512xf16>
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE3]], %[[FLAT]] : memref<?xf16>
// CHECK-NEXT: hipsr.pool_domain_yield %[[FLAT]] : memref<?xf16>
// CHECK-NEXT: } -> memref<?xf16> {domain_id = 0 : i64}
// CHECK-NEXT: return %[[OUT]] : memref<?xf16>
// CHECK-NEXT: }
func.func @pool_domain_mlp_flatten(
    %hip_ctx: !hip.context,
    %ctx: !hipsr.context,
    %input: tensor<?x256xf16>,
    %weight: tensor<256x512xf16>,
    %bias: tensor<?x512xf16>) -> tensor<?xf16> {
  %out = hipsr.pool_domain(%hip_ctx, %ctx, %input, %weight, %bias
      : !hip.context, !hipsr.context, tensor<?x256xf16>,
        tensor<256x512xf16>, tensor<?x512xf16>) {
  ^bb0(%dhip: !hip.context, %dctx: !hipsr.context,
       %input_arg: tensor<?x256xf16>, %weight_arg: tensor<256x512xf16>,
       %bias_arg: tensor<?x512xf16>):
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %input_arg, %c0 : tensor<?x256xf16>
    %n = tensor.dim %weight_arg, %c1 : tensor<256x512xf16>
    %flat_size = arith.muli %m, %n : index
    %shape1 = scf.execute_region -> !shape.shape {
      %s = shape.from_extents %m, %n : index, index
      scf.yield %s : !shape.shape
    }
    %shape2 = scf.execute_region -> !shape.shape {
      %s = shape.from_extents %m, %n : index, index
      scf.yield %s : !shape.shape
    }
    %shape3 = scf.execute_region -> !shape.shape {
      %s = shape.from_extents %flat_size : index
      scf.yield %s : !shape.shape
    }
    %init1 = tensor.empty(%m) : tensor<?x512xf16>
    %init2 = tensor.empty(%m) : tensor<?x512xf16>
    %init3 = tensor.empty(%flat_size) : tensor<?xf16>
    %mm = hip.matmul(%dhip) ins(%input_arg, %weight_arg
                                : tensor<?x256xf16>, tensor<256x512xf16>)
                            outs(%init1 : tensor<?x512xf16>) : tensor<?x512xf16>
    %sum = hip.add(%dhip) ins(%mm, %bias_arg
                              : tensor<?x512xf16>, tensor<?x512xf16>)
                          outs(%init2 : tensor<?x512xf16>) -> tensor<?x512xf16>
    %flat = hipsr.compute(%dctx) ins(%sum : tensor<?x512xf16>)
                                 outs(%init3 : tensor<?xf16>) {
    ^bb0(%body_ctx: !hipsr.context, %in: tensor<?x512xf16>,
         %dest: tensor<?xf16>):
      %collapsed = tensor.collapse_shape %in [[0, 1]]
          : tensor<?x512xf16> into tensor<?xf16>
      hipsr.compute_yield %collapsed : tensor<?xf16>
    } : tensor<?xf16>
    hipsr.preserve_shape %shape1, %mm : tensor<?x512xf16>
    hipsr.preserve_shape %shape2, %sum : tensor<?x512xf16>
    hipsr.preserve_shape %shape3, %flat : tensor<?xf16>
    hipsr.pool_domain_yield %flat : tensor<?xf16>
  } -> tensor<?xf16> {domain_id = 0 : i64}
  return %out : tensor<?xf16>
}
