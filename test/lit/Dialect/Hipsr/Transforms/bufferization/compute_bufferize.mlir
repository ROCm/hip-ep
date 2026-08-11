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

// -----
// A pool_domain body with DPS matmul/add followed by non-DPS compute that
// flattens its input. Matmul and add are already device memrefs (virtual data
// region after allocation), so they carry no results: each writes through its
// outs buffer and the next reader takes that buffer. Compute still starts as
// tensors and is bufferized here. The body keeps the three regions of the
// intended pipeline layout in order: scf.execute_region ops compute the shapes,
// the data ops run, and preserve_shape links the two, attaching shapes to the
// DPS outs buffers (init1, init2) and to the compute result (non-DPS). Shapes
// are produced inside the domain rather than passed in, which is what the
// domain being IsolatedFromAbove asks for. The allocations they describe still
// arrive as operands: an in-body tensor.empty would bufferize to a memref with
// no memory space, which is the gap the space-assignment stage has to close.
//
// hipsr.pool_domain has no bufferization interface, so its boundary has to be
// memref on both sides already; the tensor island around compute is closed with
// bufferization.to_buffer, which folds away once compute yields a memref. That
// to_buffer is read_only: without it the analysis treats the yielded buffer as
// one it may write, and since the buffer is a view of a read-only ins operand
// it allocates and copies inside the body instead.
// CHECK-LABEL: func.func @pool_domain_mlp_flatten(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[INPUT:.+]]: memref<4x256xf16, #hipsr.mem<device>>, %[[WEIGHT:.+]]: memref<256x512xf16, #hipsr.mem<device>>, %[[BIAS:.+]]: memref<4x512xf16, #hipsr.mem<device>>, %[[INIT1:.+]]: memref<4x512xf16, #hipsr.mem<device>>, %[[INIT2:.+]]: memref<4x512xf16, #hipsr.mem<device>>, %[[INIT3:.+]]: memref<4x512xf16, #hipsr.mem<device>>) -> memref<2048xf16, #hipsr.mem<device>> {
// CHECK-NEXT: %[[OUT:.+]] = hipsr.pool_domain(%[[CTX]], %[[INPUT]], %[[WEIGHT]], %[[BIAS]], %[[INIT1]], %[[INIT2]], %[[INIT3]] : !hipsr.context, memref<4x256xf16, #hipsr.mem<device>>, memref<256x512xf16, #hipsr.mem<device>>, memref<4x512xf16, #hipsr.mem<device>>, memref<4x512xf16, #hipsr.mem<device>>, memref<4x512xf16, #hipsr.mem<device>>, memref<4x512xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%[[DCTX:.+]]: !hipsr.context, %[[IN:.+]]: memref<4x256xf16, #hipsr.mem<device>>, %[[W:.+]]: memref<256x512xf16, #hipsr.mem<device>>, %[[B:.+]]: memref<4x512xf16, #hipsr.mem<device>>, %[[DEST1:.+]]: memref<4x512xf16, #hipsr.mem<device>>, %[[DEST2:.+]]: memref<4x512xf16, #hipsr.mem<device>>, %[[DEST3:.+]]: memref<4x512xf16, #hipsr.mem<device>>):
// CHECK-NEXT: %[[SHAPE1:.+]] = scf.execute_region -> !shape.shape {
// CHECK-NEXT: %[[S1:.+]] = shape.const_shape {{\[}}4, 512] : !shape.shape
// CHECK-NEXT: scf.yield %[[S1]] : !shape.shape
// CHECK-NEXT: }
// CHECK-NEXT: %[[SHAPE2:.+]] = scf.execute_region -> !shape.shape {
// CHECK-NEXT: %[[S2:.+]] = shape.const_shape {{\[}}4, 512] : !shape.shape
// CHECK-NEXT: scf.yield %[[S2]] : !shape.shape
// CHECK-NEXT: }
// CHECK-NEXT: %[[SHAPE3:.+]] = scf.execute_region -> !shape.shape {
// CHECK-NEXT: %[[S3:.+]] = shape.const_shape {{\[}}2048] : !shape.shape
// CHECK-NEXT: scf.yield %[[S3]] : !shape.shape
// CHECK-NEXT: }
// CHECK-NEXT: hipsr.matmul(%[[DCTX]]) ins(%[[IN]], %[[W]] : memref<4x256xf16, #hipsr.mem<device>>, memref<256x512xf16, #hipsr.mem<device>>) outs(%[[DEST1]] : memref<4x512xf16, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%[[DCTX]]) ins(%[[DEST1]], %[[B]] : memref<4x512xf16, #hipsr.mem<device>>, memref<4x512xf16, #hipsr.mem<device>>) outs(%[[DEST2]] : memref<4x512xf16, #hipsr.mem<device>>)
// CHECK-NEXT: %[[FLAT:.+]] = hipsr.compute(%[[DCTX]]) ins(%[[DEST2]] : memref<4x512xf16, #hipsr.mem<device>>) outs(%[[DEST3]] : memref<4x512xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%[[BODY_CTX:.+]]: !hipsr.context, %[[BODY_IN:.+]]: memref<4x512xf16, #hipsr.mem<device>>, %[[BODY_DEST:.+]]: memref<4x512xf16, #hipsr.mem<device>>):
// CHECK-NEXT: %[[COLLAPSED:.+]] = memref.collapse_shape %[[BODY_IN]] {{\[\[}}0, 1]] : memref<4x512xf16, #hipsr.mem<device>> into memref<2048xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.compute_yield %[[COLLAPSED]] : memref<2048xf16, #hipsr.mem<device>>
// CHECK-NEXT: } : memref<2048xf16, #hipsr.mem<device>>{{$}}
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE1]], %[[DEST1]] : memref<4x512xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE2]], %[[DEST2]] : memref<4x512xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE3]], %[[FLAT]] : memref<2048xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.pool_domain_yield %[[FLAT]] : memref<2048xf16, #hipsr.mem<device>>
// CHECK-NEXT: } -> memref<2048xf16, #hipsr.mem<device>> {domain_id = 0 : i64}
// CHECK-NEXT: return %[[OUT]] : memref<2048xf16, #hipsr.mem<device>>
// CHECK-NEXT: }
func.func @pool_domain_mlp_flatten(
    %ctx: !hipsr.context,
    %input: memref<4x256xf16, #hipsr.mem<device>>,
    %weight: memref<256x512xf16, #hipsr.mem<device>>,
    %bias: memref<4x512xf16, #hipsr.mem<device>>,
    %init1: memref<4x512xf16, #hipsr.mem<device>>,
    %init2: memref<4x512xf16, #hipsr.mem<device>>,
    %init3: memref<4x512xf16, #hipsr.mem<device>>)
    -> memref<2048xf16, #hipsr.mem<device>> {
  %out = hipsr.pool_domain(%ctx, %input, %weight, %bias, %init1, %init2,
                           %init3
      : !hipsr.context,
        memref<4x256xf16, #hipsr.mem<device>>,
        memref<256x512xf16, #hipsr.mem<device>>,
        memref<4x512xf16, #hipsr.mem<device>>,
        memref<4x512xf16, #hipsr.mem<device>>,
        memref<4x512xf16, #hipsr.mem<device>>,
        memref<4x512xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context,
       %input_arg: memref<4x256xf16, #hipsr.mem<device>>,
       %weight_arg: memref<256x512xf16, #hipsr.mem<device>>,
       %bias_arg: memref<4x512xf16, #hipsr.mem<device>>,
       %init1_arg: memref<4x512xf16, #hipsr.mem<device>>,
       %init2_arg: memref<4x512xf16, #hipsr.mem<device>>,
       %init3_arg: memref<4x512xf16, #hipsr.mem<device>>):
    %shape1 = scf.execute_region -> !shape.shape {
      %s = shape.const_shape [4, 512] : !shape.shape
      scf.yield %s : !shape.shape
    }
    %shape2 = scf.execute_region -> !shape.shape {
      %s = shape.const_shape [4, 512] : !shape.shape
      scf.yield %s : !shape.shape
    }
    %shape3 = scf.execute_region -> !shape.shape {
      %s = shape.const_shape [2048] : !shape.shape
      scf.yield %s : !shape.shape
    }
    hipsr.matmul(%dctx)
        ins(%input_arg, %weight_arg
            : memref<4x256xf16, #hipsr.mem<device>>,
              memref<256x512xf16, #hipsr.mem<device>>)
        outs(%init1_arg : memref<4x512xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx)
        ins(%init1_arg, %bias_arg
            : memref<4x512xf16, #hipsr.mem<device>>,
              memref<4x512xf16, #hipsr.mem<device>>)
        outs(%init2_arg : memref<4x512xf16, #hipsr.mem<device>>)
    %add_t = bufferization.to_tensor %init2_arg restrict
        : memref<4x512xf16, #hipsr.mem<device>> to tensor<4x512xf16>
    %dest_t = bufferization.to_tensor %init3_arg restrict writable
        : memref<4x512xf16, #hipsr.mem<device>> to tensor<4x512xf16>
    %flat = hipsr.compute(%dctx) ins(%add_t : tensor<4x512xf16>)
                               outs(%dest_t : tensor<4x512xf16>) {
    ^bb0(%body_ctx: !hipsr.context, %in: tensor<4x512xf16>,
         %dest: tensor<4x512xf16>):
      %collapsed = tensor.collapse_shape %in [[0, 1]]
          : tensor<4x512xf16> into tensor<2048xf16>
      hipsr.compute_yield %collapsed : tensor<2048xf16>
    } : tensor<2048xf16>
    %flat_buf = bufferization.to_buffer %flat read_only
        : tensor<2048xf16> to memref<2048xf16, #hipsr.mem<device>>
    hipsr.preserve_shape %shape1, %init1_arg
        : memref<4x512xf16, #hipsr.mem<device>>
    hipsr.preserve_shape %shape2, %init2_arg
        : memref<4x512xf16, #hipsr.mem<device>>
    hipsr.preserve_shape %shape3, %flat_buf
        : memref<2048xf16, #hipsr.mem<device>>
    hipsr.pool_domain_yield %flat_buf : memref<2048xf16, #hipsr.mem<device>>
  } -> memref<2048xf16, #hipsr.mem<device>> {domain_id = 0 : i64}
  return %out : memref<2048xf16, #hipsr.mem<device>>
}
