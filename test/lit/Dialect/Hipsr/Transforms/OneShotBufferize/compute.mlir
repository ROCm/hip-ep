// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// Checks that One-Shot Bufferize rewrites hipsr.compute, its body, and its
// hipsr.compute_yield terminator into memrefs.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --split-input-file --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map use-encoding-for-memory-space" %s | FileCheck %s
// RUN: hip-mlir-opt --split-input-file --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map use-encoding-for-memory-space" %s | FileCheck %s --check-prefix=NOCOPY

// NOCOPY-NOT: memref.copy

// -----
// A pool_domain body with a DPS cast chain followed by a non-DPS compute that
// flattens its input. The body follows the intended pipeline layout: shape
// regions first, then the allocations they describe, then the data ops, then
// the preserve_shape links.
//
// The #hipsr.mem<device> tensor encoding is what supplies the memory space the
// hipsr operands require: use-encoding-for-memory-space carries the encoding of
// a ranked tensor onto the memref it becomes, so a tensor.empty init allocates
// in device space.
//
// cast is destination-passing, so a bufferized cast keeps no result and every
// use of its tensor result reads the buffer of its init instead.
//
// compute takes %cast2 as both ins and outs, so the two entry block arguments
// name one buffer and collapsing the ins argument still holds the result in the
// destination. The body only builds that collapse view, so the result lands in
// init2 and needs no buffer of its own and the domain allocates twice rather
// than three times. The shared value is
// only conflict-free because bufferizesToMemoryWrite asks the body, where the
// outs block argument has no uses; reporting every outs as written would make
// the analysis allocate a second buffer for it.
//
// Each preserve_shape and the domain yield read the DPS result, not the init. In
// tensor form those are two different values, and reading the init would mean
// reading what the buffer held before the write, which forces a copy.
// CHECK-LABEL: func.func @pool_domain_mlp_flatten(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME: %[[INPUT:.+]]: memref<?x256xf16, #hipsr.mem<device>>) -> memref<?xf16, #hipsr.mem<device>> {
// CHECK-NEXT: %[[OUT:.+]] = hipsr.pool_domain(%[[CTX]], %[[INPUT]] : !hipsr.context, memref<?x256xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%[[DCTX:.+]]: !hipsr.context, %[[IN:.+]]: memref<?x256xf16, #hipsr.mem<device>>):
// CHECK-NEXT: %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[C1:.+]] = arith.constant 1 : index
// CHECK-NEXT: %[[M:.+]] = memref.dim %[[IN]], %[[C0]] : memref<?x256xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[N:.+]] = memref.dim %[[IN]], %[[C1]] : memref<?x256xf16, #hipsr.mem<device>>
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
// CHECK-NEXT: %[[INIT1:.+]] = memref.alloc(%[[M]]){{.*}} : memref<?x256xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.cast(%[[DCTX]]) ins(%[[IN]] : memref<?x256xf16, #hipsr.mem<device>>)
// CHECK-SAME: outs(%[[INIT1]] : memref<?x256xf16, #hipsr.mem<device>>)
// CHECK-NEXT: %[[INIT2:.+]] = memref.alloc(%[[M]]){{.*}} : memref<?x256xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.cast(%[[DCTX]]) ins(%[[INIT1]] : memref<?x256xf16, #hipsr.mem<device>>)
// CHECK-SAME: outs(%[[INIT2]] : memref<?x256xf16, #hipsr.mem<device>>)
// CHECK-NEXT: %[[FLAT:.+]] = hipsr.compute(%[[DCTX]]) ins(%[[INIT2]] : memref<?x256xf16, #hipsr.mem<device>>)
// CHECK-SAME: outs(%[[INIT2]] : memref<?x256xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%[[BODY_CTX:.+]]: !hipsr.context, %[[BODY_IN:.+]]: memref<?x256xf16, #hipsr.mem<device>>,
// CHECK-SAME: %[[BODY_DEST:.+]]: memref<?x256xf16, #hipsr.mem<device>>):
// CHECK-NEXT: %[[COLLAPSED:.+]] = memref.collapse_shape %[[BODY_IN]] {{\[\[}}0, 1]]
// CHECK-SAME: : memref<?x256xf16, #hipsr.mem<device>> into memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.compute_yield %[[COLLAPSED]] : memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT: } : memref<?xf16, #hipsr.mem<device>>{{$}}
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE1]], %[[INIT1]] : !shape.shape, memref<?x256xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE2]], %[[INIT2]] : !shape.shape, memref<?x256xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE3]], %[[FLAT]] : !shape.shape, memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.pool_domain_yield %[[FLAT]] : memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT: } -> memref<?xf16, #hipsr.mem<device>> {domain_id = 0 : i64}
// CHECK-NEXT: return %[[OUT]] : memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT: }
func.func @pool_domain_mlp_flatten(
    %ctx: !hipsr.context,
    %input: tensor<?x256xf16, #hipsr.mem<device>>)
    -> tensor<?xf16, #hipsr.mem<device>> {
  %out = hipsr.pool_domain(%ctx, %input
      : !hipsr.context, tensor<?x256xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context,
       %input_arg: tensor<?x256xf16, #hipsr.mem<device>>):
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %input_arg, %c0 : tensor<?x256xf16, #hipsr.mem<device>>
    %n = tensor.dim %input_arg, %c1 : tensor<?x256xf16, #hipsr.mem<device>>
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
    %init1 = tensor.empty(%m) : tensor<?x256xf16, #hipsr.mem<device>>
    %init2 = tensor.empty(%m) : tensor<?x256xf16, #hipsr.mem<device>>
    %init3 = tensor.empty(%flat_size) : tensor<?xf16, #hipsr.mem<device>>
    %cast1 = hipsr.cast(%dctx)
        ins(%input_arg : tensor<?x256xf16, #hipsr.mem<device>>)
        outs(%init1 : tensor<?x256xf16, #hipsr.mem<device>>)
        : tensor<?x256xf16, #hipsr.mem<device>>
    %cast2 = hipsr.cast(%dctx)
        ins(%cast1 : tensor<?x256xf16, #hipsr.mem<device>>)
        outs(%init2 : tensor<?x256xf16, #hipsr.mem<device>>)
        : tensor<?x256xf16, #hipsr.mem<device>>
    %flat = hipsr.compute(%dctx)
        ins(%cast2 : tensor<?x256xf16, #hipsr.mem<device>>)
        outs(%init3 : tensor<?xf16, #hipsr.mem<device>>) {
    ^bb0(%body_ctx: !hipsr.context, %in: tensor<?x256xf16, #hipsr.mem<device>>,
         %dest: tensor<?xf16, #hipsr.mem<device>>):
      %collapsed = tensor.collapse_shape %in [[0, 1]]
          : tensor<?x256xf16, #hipsr.mem<device>>
          into tensor<?xf16, #hipsr.mem<device>>
      hipsr.compute_yield %collapsed : tensor<?xf16, #hipsr.mem<device>>
    } : tensor<?xf16, #hipsr.mem<device>>
    hipsr.preserve_shape %shape1, %cast1 : !shape.shape, tensor<?x256xf16, #hipsr.mem<device>>
    hipsr.preserve_shape %shape2, %cast2 : !shape.shape, tensor<?x256xf16, #hipsr.mem<device>>
    hipsr.preserve_shape %shape3, %flat : !shape.shape, tensor<?xf16, #hipsr.mem<device>>
    hipsr.pool_domain_yield %flat : tensor<?xf16, #hipsr.mem<device>>
  } -> tensor<?xf16, #hipsr.mem<device>> {domain_id = 0 : i64}
  return %out : tensor<?xf16, #hipsr.mem<device>>
}

// -----
// Two copies of one compute op that differ only in whether the body reads its
// destination entry argument. Holding a result in an input's buffer also
// retypes that argument, so bufferization only does it when the body never
// touches the argument.
//
// The destination is a rank wider than the input here. Reusing the input buffer
// for @compute_reads_destination would hand the body a memref<?x?xi1> where it
// reads a tensor<?x?x1xi1>, so the destination keeps an allocation of its own.
// CHECK-LABEL: func.func @compute_reads_destination(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[ROWS:.+]]: index, %[[COLS:.+]]: index,
// CHECK-SAME: %[[INPUT:.+]]: memref<?x?xi1, #hipsr.mem<device>>) -> memref<?x?x1xi1, #hipsr.mem<device>> {
// CHECK-NEXT: %[[DEST:.+]] = memref.alloc(%[[ROWS]], %[[COLS]]){{.*}} : memref<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT: %[[OUT:.+]] = hipsr.compute(%[[CTX]]) ins(%[[INPUT]] : memref<?x?xi1, #hipsr.mem<device>>)
// CHECK-SAME: outs(%[[DEST]] : memref<?x?x1xi1, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%{{.+}}: !hipsr.context, %[[IN:.+]]: memref<?x?xi1, #hipsr.mem<device>>,
// CHECK-SAME: %[[BODY_DEST:.+]]: memref<?x?x1xi1, #hipsr.mem<device>>):
// CHECK-NEXT: %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[C1:.+]] = arith.constant 1 : index
// CHECK-NEXT: %[[D0:.+]] = memref.dim %[[BODY_DEST]], %[[C0]] : memref<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT: %[[D1:.+]] = memref.dim %[[BODY_DEST]], %[[C1]] : memref<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT: %[[EXPANDED:.+]] = memref.expand_shape %[[IN]] {{\[\[}}0], [1, 2]] output_shape {{\[}}%[[D0]], %[[D1]], 1]
// CHECK-SAME: : memref<?x?xi1, #hipsr.mem<device>> into memref<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.compute_yield %[[EXPANDED]] : memref<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT: } : memref<?x?x1xi1, #hipsr.mem<device>>{{$}}
// CHECK-NEXT: return %[[OUT]] : memref<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT: }
func.func @compute_reads_destination(
    %ctx: !hipsr.context, %rows: index, %cols: index,
    %input: tensor<?x?xi1, #hipsr.mem<device>>)
    -> tensor<?x?x1xi1, #hipsr.mem<device>> {
  %init = tensor.empty(%rows, %cols) : tensor<?x?x1xi1, #hipsr.mem<device>>
  %out = hipsr.compute(%ctx)
      ins(%input : tensor<?x?xi1, #hipsr.mem<device>>)
      outs(%init : tensor<?x?x1xi1, #hipsr.mem<device>>) {
  ^bb0(%body_ctx: !hipsr.context, %in: tensor<?x?xi1, #hipsr.mem<device>>,
       %dest: tensor<?x?x1xi1, #hipsr.mem<device>>):
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %d0 = tensor.dim %dest, %c0 : tensor<?x?x1xi1, #hipsr.mem<device>>
    %d1 = tensor.dim %dest, %c1 : tensor<?x?x1xi1, #hipsr.mem<device>>
    %expanded = tensor.expand_shape %in [[0], [1, 2]] output_shape [%d0, %d1, 1]
        : tensor<?x?xi1, #hipsr.mem<device>>
        into tensor<?x?x1xi1, #hipsr.mem<device>>
    hipsr.compute_yield %expanded : tensor<?x?x1xi1, #hipsr.mem<device>>
  } : tensor<?x?x1xi1, #hipsr.mem<device>>
  return %out : tensor<?x?x1xi1, #hipsr.mem<device>>
}

// The dimensions come off the input here, which leaves the destination argument
// unused, so the result lands in the input's buffer and tensor.empty allocates
// nothing. The destination argument comes back a rank lower with it, which is
// what the body above could not have tolerated.
// CHECK-LABEL: func.func @compute_ignores_destination(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %{{.+}}: index, %{{.+}}: index,
// CHECK-SAME: %[[INPUT:.+]]: memref<?x?xi1, #hipsr.mem<device>>) -> memref<?x?x1xi1, #hipsr.mem<device>> {
// CHECK-NEXT: %[[OUT:.+]] = hipsr.compute(%[[CTX]]) ins(%[[INPUT]] : memref<?x?xi1, #hipsr.mem<device>>)
// CHECK-SAME: outs(%[[INPUT]] : memref<?x?xi1, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%{{.+}}: !hipsr.context, %[[IN:.+]]: memref<?x?xi1, #hipsr.mem<device>>,
// CHECK-SAME: %{{.+}}: memref<?x?xi1, #hipsr.mem<device>>):
// CHECK-NEXT: %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[C1:.+]] = arith.constant 1 : index
// CHECK-NEXT: %[[D0:.+]] = memref.dim %[[IN]], %[[C0]] : memref<?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT: %[[D1:.+]] = memref.dim %[[IN]], %[[C1]] : memref<?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT: %[[EXPANDED:.+]] = memref.expand_shape %[[IN]] {{\[\[}}0], [1, 2]] output_shape {{\[}}%[[D0]], %[[D1]], 1]
// CHECK-SAME: : memref<?x?xi1, #hipsr.mem<device>> into memref<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.compute_yield %[[EXPANDED]] : memref<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT: } : memref<?x?x1xi1, #hipsr.mem<device>>{{$}}
// CHECK-NEXT: return %[[OUT]] : memref<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT: }
func.func @compute_ignores_destination(
    %ctx: !hipsr.context, %rows: index, %cols: index,
    %input: tensor<?x?xi1, #hipsr.mem<device>>)
    -> tensor<?x?x1xi1, #hipsr.mem<device>> {
  %init = tensor.empty(%rows, %cols) : tensor<?x?x1xi1, #hipsr.mem<device>>
  %out = hipsr.compute(%ctx)
      ins(%input : tensor<?x?xi1, #hipsr.mem<device>>)
      outs(%init : tensor<?x?x1xi1, #hipsr.mem<device>>) {
  ^bb0(%body_ctx: !hipsr.context, %in: tensor<?x?xi1, #hipsr.mem<device>>,
       %dest: tensor<?x?x1xi1, #hipsr.mem<device>>):
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %d0 = tensor.dim %in, %c0 : tensor<?x?xi1, #hipsr.mem<device>>
    %d1 = tensor.dim %in, %c1 : tensor<?x?xi1, #hipsr.mem<device>>
    %expanded = tensor.expand_shape %in [[0], [1, 2]] output_shape [%d0, %d1, 1]
        : tensor<?x?xi1, #hipsr.mem<device>>
        into tensor<?x?x1xi1, #hipsr.mem<device>>
    hipsr.compute_yield %expanded : tensor<?x?x1xi1, #hipsr.mem<device>>
  } : tensor<?x?x1xi1, #hipsr.mem<device>>
  return %out : tensor<?x?x1xi1, #hipsr.mem<device>>
}
