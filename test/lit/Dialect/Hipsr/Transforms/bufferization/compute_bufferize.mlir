// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// Checks that One-Shot Bufferize rewrites hipsr.compute, its body, and its
// hipsr.compute_yield terminator into device memrefs.
//
// Each positive case uses device memrefs at the function boundary (wrapped in
// `bufferization.to_tensor restrict` where the body still starts as tensors),
// rather than tensor.empty at the boundary, because the buffer a hipsr op takes
// takes must name its memory space and One-Shot Bufferize has no default one to
// give (`memref<6xf16>` would fail the Hipsr_TensorOrDeviceMemRef constraint).
// Bufferizing a view keeps the space, so device memrefs on the way in are what
// makes device memrefs on the way out. Assigning the space to allocations that
// the pipeline itself creates is a separate, still missing, stage.
//
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

// -----
// A pool_domain body with DPS matmul/add followed by non-DPS compute that
// flattens its input. The body follows the intended pipeline layout:
// allocations sized from the matmul input batch; init3 is the flattened 1D
// buffer (m * 512). Shape regions read each init buffer's extents, then the
// data ops, then preserve_shape links. Matmul and add are device memrefs with
// no results; compute collapses init2 into init3's rank via tensor.collapse_shape
// and bufferizes with init3 as the outs buffer.
// CHECK-LABEL: func.func @pool_domain_mlp_flatten(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[INPUT:.+]]: memref<?x256xf16, #hipsr.mem<device>>, %[[WEIGHT:.+]]: memref<256x512xf16, #hipsr.mem<device>>, %[[BIAS:.+]]: memref<?x512xf16, #hipsr.mem<device>>) -> memref<?xf16, #hipsr.mem<device>> {
// CHECK-NEXT: %[[OUT:.+]] = hipsr.pool_domain(%[[CTX]], %[[INPUT]], %[[WEIGHT]], %[[BIAS]] : !hipsr.context, memref<?x256xf16, #hipsr.mem<device>>, memref<256x512xf16, #hipsr.mem<device>>, memref<?x512xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%[[DCTX:.+]]: !hipsr.context, %[[IN:.+]]: memref<?x256xf16, #hipsr.mem<device>>, %[[W:.+]]: memref<256x512xf16, #hipsr.mem<device>>, %[[B:.+]]: memref<?x512xf16, #hipsr.mem<device>>):
// CHECK-NEXT: %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[C1:.+]] = arith.constant 1 : index
// CHECK-NEXT: %[[C512:.+]] = arith.constant 512 : index
// CHECK-NEXT: %[[M:.+]] = memref.dim %[[IN]], %[[C0]] : memref<?x256xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[FLAT_SIZE:.+]] = arith.muli %[[M]], %[[C512]] : index
// CHECK-NEXT: %[[INIT1:.+]] = memref.alloc(%[[M]]) : memref<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[INIT2:.+]] = memref.alloc(%[[M]]) : memref<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[INIT3:.+]] = memref.alloc(%[[FLAT_SIZE]]) : memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[SHAPE1:.+]] = scf.execute_region -> !shape.shape {
// CHECK-NEXT: %[[S1D0:.+]] = memref.dim %[[INIT1]], %[[C0]] : memref<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[S1D1:.+]] = memref.dim %[[INIT1]], %[[C1]] : memref<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[S1:.+]] = shape.from_extents %[[S1D0]], %[[S1D1]] : index, index
// CHECK-NEXT: scf.yield %[[S1]] : !shape.shape
// CHECK-NEXT: }
// CHECK-NEXT: %[[SHAPE2:.+]] = scf.execute_region -> !shape.shape {
// CHECK-NEXT: %[[S2D0:.+]] = memref.dim %[[INIT2]], %[[C0]] : memref<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[S2D1:.+]] = memref.dim %[[INIT2]], %[[C1]] : memref<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[S2:.+]] = shape.from_extents %[[S2D0]], %[[S2D1]] : index, index
// CHECK-NEXT: scf.yield %[[S2]] : !shape.shape
// CHECK-NEXT: }
// CHECK-NEXT: %[[SHAPE3:.+]] = scf.execute_region -> !shape.shape {
// CHECK-NEXT: %[[S3D0:.+]] = memref.dim %[[INIT3]], %[[C0]] : memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[S3:.+]] = shape.from_extents %[[S3D0]] : index
// CHECK-NEXT: scf.yield %[[S3]] : !shape.shape
// CHECK-NEXT: }
// CHECK-NEXT: hipsr.matmul(%[[DCTX]]) ins(%[[IN]], %[[W]] : memref<?x256xf16, #hipsr.mem<device>>, memref<256x512xf16, #hipsr.mem<device>>) outs(%[[INIT1]] : memref<?x512xf16, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%[[DCTX]]) ins(%[[INIT1]], %[[B]] : memref<?x512xf16, #hipsr.mem<device>>, memref<?x512xf16, #hipsr.mem<device>>) outs(%[[INIT2]] : memref<?x512xf16, #hipsr.mem<device>>)
// CHECK-NEXT: %[[FLAT:.+]] = hipsr.compute(%[[DCTX]]) ins(%[[INIT2]] : memref<?x512xf16, #hipsr.mem<device>>) outs(%[[INIT3]] : memref<?xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%[[BODY_CTX:.+]]: !hipsr.context, %[[BODY_IN:.+]]: memref<?x512xf16, #hipsr.mem<device>>, %[[BODY_DEST:.+]]: memref<?xf16, #hipsr.mem<device>>):
// CHECK-NEXT: %[[COLLAPSED:.+]] = memref.collapse_shape %[[BODY_IN]] {{\[\[}}0, 1]] : memref<?x512xf16, #hipsr.mem<device>> into memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.compute_yield %[[COLLAPSED]] : memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT: } : memref<?xf16, #hipsr.mem<device>>{{$}}
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE1]], %[[INIT1]] : memref<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE2]], %[[INIT2]] : memref<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE3]], %[[INIT3]] : memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.pool_domain_yield %[[INIT3]] : memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT: } -> memref<?xf16, #hipsr.mem<device>> {domain_id = 0 : i64}
// CHECK-NEXT: return %[[OUT]] : memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT: }
func.func @pool_domain_mlp_flatten(
    %ctx: !hipsr.context,
    %input: memref<?x256xf16, #hipsr.mem<device>>,
    %weight: memref<256x512xf16, #hipsr.mem<device>>,
    %bias: memref<?x512xf16, #hipsr.mem<device>>)
    -> memref<?xf16, #hipsr.mem<device>> {
  %out = hipsr.pool_domain(%ctx, %input, %weight, %bias
      : !hipsr.context,
        memref<?x256xf16, #hipsr.mem<device>>,
        memref<256x512xf16, #hipsr.mem<device>>,
        memref<?x512xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context,
       %input_arg: memref<?x256xf16, #hipsr.mem<device>>,
       %weight_arg: memref<256x512xf16, #hipsr.mem<device>>,
       %bias_arg: memref<?x512xf16, #hipsr.mem<device>>):
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %m = memref.dim %input_arg, %c0 : memref<?x256xf16, #hipsr.mem<device>>
    %flat_size = arith.muli %m, %c512 : index
    %init1 = memref.alloc(%m) : memref<?x512xf16, #hipsr.mem<device>>
    %init2 = memref.alloc(%m) : memref<?x512xf16, #hipsr.mem<device>>
    %init3 = memref.alloc(%flat_size) : memref<?xf16, #hipsr.mem<device>>
    %shape1 = scf.execute_region -> !shape.shape {
      %d0 = memref.dim %init1, %c0 : memref<?x512xf16, #hipsr.mem<device>>
      %d1 = memref.dim %init1, %c1 : memref<?x512xf16, #hipsr.mem<device>>
      %s = shape.from_extents %d0, %d1 : index, index
      scf.yield %s : !shape.shape
    }
    %shape2 = scf.execute_region -> !shape.shape {
      %d0 = memref.dim %init2, %c0 : memref<?x512xf16, #hipsr.mem<device>>
      %d1 = memref.dim %init2, %c1 : memref<?x512xf16, #hipsr.mem<device>>
      %s = shape.from_extents %d0, %d1 : index, index
      scf.yield %s : !shape.shape
    }
    %shape3 = scf.execute_region -> !shape.shape {
      %d0 = memref.dim %init3, %c0 : memref<?xf16, #hipsr.mem<device>>
      %s = shape.from_extents %d0 : index
      scf.yield %s : !shape.shape
    }
    hipsr.matmul(%dctx)
        ins(%input_arg, %weight_arg
            : memref<?x256xf16, #hipsr.mem<device>>,
              memref<256x512xf16, #hipsr.mem<device>>)
        outs(%init1 : memref<?x512xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx)
        ins(%init1, %bias_arg
            : memref<?x512xf16, #hipsr.mem<device>>,
              memref<?x512xf16, #hipsr.mem<device>>)
        outs(%init2 : memref<?x512xf16, #hipsr.mem<device>>)
    %add_t = bufferization.to_tensor %init2 restrict
        : memref<?x512xf16, #hipsr.mem<device>> to tensor<?x512xf16>
    %dest_t = bufferization.to_tensor %init3 restrict writable
        : memref<?xf16, #hipsr.mem<device>> to tensor<?xf16>
    %flat = hipsr.compute(%dctx) ins(%add_t : tensor<?x512xf16>)
                               outs(%dest_t : tensor<?xf16>) {
    ^bb0(%body_ctx: !hipsr.context, %in: tensor<?x512xf16>,
         %dest: tensor<?xf16>):
      %collapsed = tensor.collapse_shape %in [[0, 1]]
          : tensor<?x512xf16> into tensor<?xf16>
      hipsr.compute_yield %collapsed : tensor<?xf16>
    } : tensor<?xf16>
    hipsr.preserve_shape %shape1, %init1 : memref<?x512xf16, #hipsr.mem<device>>
    hipsr.preserve_shape %shape2, %init2 : memref<?x512xf16, #hipsr.mem<device>>
    hipsr.preserve_shape %shape3, %init3 : memref<?xf16, #hipsr.mem<device>>
    hipsr.pool_domain_yield %init3 : memref<?xf16, #hipsr.mem<device>>
  } -> memref<?xf16, #hipsr.mem<device>> {domain_id = 0 : i64}
  return %out : memref<?xf16, #hipsr.mem<device>>
}
