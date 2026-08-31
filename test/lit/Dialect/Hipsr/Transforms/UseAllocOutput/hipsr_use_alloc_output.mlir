// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -hipsr-use-output-allocator | FileCheck %s

// The pass runs after bufferization, so every data value is a memref and the
// extent tensor reaches preserve_shape through bufferization.to_tensor.
//
// A returned dynamic allocation becomes hipsr.alloc_output. Dynamic sizes come
// from the preserved shape, not directly from the original alloc operands.
// CHECK-LABEL: func.func @dynamic_output(
// CHECK-SAME: %[[CTX:.*]]: !hipsr.context, %[[M:.*]]: index, %[[N:.*]]: index) -> memref<?x?xf16, #hipsr.mem<device>> {

// The buffer holding the extents is left alone. Only the data allocation the
// function returns is replaced.
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT: %[[EXTENTS_BUF:.*]] = memref.alloc() : memref<2xindex>
// CHECK-NEXT: memref.store %[[M]], %[[EXTENTS_BUF]]{{\[}}%[[C0]]] : memref<2xindex>
// CHECK-NEXT: memref.store %[[N]], %[[EXTENTS_BUF]]{{\[}}%[[C1]]] : memref<2xindex>
// CHECK-NEXT: %[[EXTENTS:.*]] = bufferization.to_tensor %[[EXTENTS_BUF]] : memref<2xindex> to tensor<2xindex>

// The pass reads the shape twice: once for the size it asks the runtime for,
// and once for the view it would build over the result. Both are the same here
// because the result needs no view, so the second pair has no user.
// CHECK-NEXT: %[[D0_INDEX:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[D0:.*]] = shape.get_extent %[[EXTENTS]], %[[D0_INDEX]] : tensor<2xindex>, index -> index
// CHECK-NEXT: %[[D1_INDEX:.*]] = arith.constant 1 : index
// CHECK-NEXT: %[[D1:.*]] = shape.get_extent %[[EXTENTS]], %[[D1_INDEX]] : tensor<2xindex>, index -> index
// CHECK-NEXT: %[[VIEW_D0_INDEX:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[VIEW_D0:.*]] = shape.get_extent %[[EXTENTS]], %[[VIEW_D0_INDEX]] : tensor<2xindex>, index -> index
// CHECK-NEXT: %[[VIEW_D1_INDEX:.*]] = arith.constant 1 : index
// CHECK-NEXT: %[[VIEW_D1:.*]] = shape.get_extent %[[EXTENTS]], %[[VIEW_D1_INDEX]] : tensor<2xindex>, index -> index

// CHECK-NEXT: %[[OUT:.*]] = hipsr.alloc_output(%[[CTX]], %[[D0]], %[[D1]]) {out_idx = 0 : i64} : memref<?x?xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.preserve_shape %[[EXTENTS]], %[[OUT]] : tensor<2xindex>, memref<?x?xf16, #hipsr.mem<device>>
// CHECK-NEXT: return %[[OUT]] : memref<?x?xf16, #hipsr.mem<device>>
// CHECK-NEXT: }
func.func @dynamic_output(
    %ctx: !hipsr.context, %M: index, %N: index)
    -> memref<?x?xf16, #hipsr.mem<device>> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %extents = memref.alloc() : memref<2xindex>
  memref.store %M, %extents[%c0] : memref<2xindex>
  memref.store %N, %extents[%c1] : memref<2xindex>
  %shape = bufferization.to_tensor %extents : memref<2xindex> to tensor<2xindex>
  %out = memref.alloc(%M, %N)
      : memref<?x?xf16, #hipsr.mem<device>>
  hipsr.preserve_shape %shape, %out : tensor<2xindex>, memref<?x?xf16, #hipsr.mem<device>>
  return %out : memref<?x?xf16, #hipsr.mem<device>>
}


// ---

// in onnx:
// %graph_input
//     ↓
// onnx.MatMul
//     ↓
// onnx.Add (?x256xf16)
//     ↓
// onnx.Reshape(?xf16) ──> func.return（%graph_output）
//     └──────────> onnx.Cast

// CHECK-LABEL: func.func @test_return_shape
// CHECK-NEXT: %[[OUT:.*]] = hipsr.pool_domain(%[[CTX:.*]], %[[INPUT:.*]], %[[WEIGHT:.*]], %[[BIAS:.*]] : !hipsr.context, memref<?x512xf16, #hipsr.mem<device>>, memref<512x256xf16, #hipsr.mem<device>>, memref<?x256xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%[[DCTX:.*]]: !hipsr.context, %[[IN:.*]]: memref<?x512xf16, #hipsr.mem<device>>, %[[W:.*]]: memref<512x256xf16, #hipsr.mem<device>>, %[[B:.*]]: memref<?x256xf16, #hipsr.mem<device>>):
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT: %[[M:.*]] = memref.dim %[[IN]], %[[C0]] : memref<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[N:.*]] = memref.dim %[[W]], %[[C1]] : memref<512x256xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[FLAT_SIZE:.*]] = arith.muli %[[M]], %[[N]] : index
// CHECK-NEXT: %[[E1:.*]] = memref.alloc() : memref<2xindex>
// CHECK-NEXT: memref.store %[[M]], %[[E1]]{{\[}}%[[C0]]] : memref<2xindex>
// CHECK-NEXT: memref.store %[[N]], %[[E1]]{{\[}}%[[C1]]] : memref<2xindex>
// CHECK-NEXT: %[[SHAPE1:.*]] = bufferization.to_tensor %[[E1]] : memref<2xindex> to tensor<2xindex>
// CHECK-NEXT: %[[E2:.*]] = memref.alloc() : memref<2xindex>
// CHECK-NEXT: memref.store %[[M]], %[[E2]]{{\[}}%[[C0]]] : memref<2xindex>
// CHECK-NEXT: memref.store %[[N]], %[[E2]]{{\[}}%[[C1]]] : memref<2xindex>
// CHECK-NEXT: %[[SHAPE2:.*]] = bufferization.to_tensor %[[E2]] : memref<2xindex> to tensor<2xindex>
// CHECK-NEXT: %[[E3:.*]] = memref.alloc() : memref<1xindex>
// CHECK-NEXT: memref.store %[[FLAT_SIZE]], %[[E3]]{{\[}}%[[C0]]] : memref<1xindex>
// CHECK-NEXT: %[[SHAPE3:.*]] = bufferization.to_tensor %[[E3]] : memref<1xindex> to tensor<1xindex>
// CHECK-NEXT: %[[INIT1:.*]] = memref.alloc(%[[M]]) : memref<?x256xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.matmul(%[[DCTX]]) ins(%[[IN]], %[[W]] : memref<?x512xf16, #hipsr.mem<device>>, memref<512x256xf16, #hipsr.mem<device>>) outs(%[[INIT1]] : memref<?x256xf16, #hipsr.mem<device>>)
// The external (rank-1) output size comes from SHAPE3; the internal (rank-2)
// view size comes from SHAPE2.
// CHECK-NEXT: %[[CS_EXT0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[EXT_SIZE:.*]] = shape.get_extent %[[SHAPE3]], %[[CS_EXT0]] : tensor<1xindex>, index -> index
// CHECK-NEXT: %[[CS_INT0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[INT_SIZE:.*]] = shape.get_extent %[[SHAPE2]], %[[CS_INT0]] : tensor<2xindex>, index -> index
// CHECK-NEXT: %[[C256:.*]] = arith.constant 256 : index
// CHECK-NEXT: %[[OUTBUF:.*]] = hipsr.alloc_output(%[[DCTX]], %[[EXT_SIZE]]) {out_idx = 0 : i64} : memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[RC:.*]] = memref.reinterpret_cast %[[OUTBUF]] to offset: [0], sizes: [%[[INT_SIZE]], 256], strides: [256, 1] : memref<?xf16, #hipsr.mem<device>> to memref<?x256xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.add(%[[DCTX]]) ins(%[[INIT1]], %[[B]] : memref<?x256xf16, #hipsr.mem<device>>, memref<?x256xf16, #hipsr.mem<device>>) outs(%[[RC]] : memref<?x256xf16, #hipsr.mem<device>>)
// CHECK-NEXT: %[[FLAT:.*]] = hipsr.compute(%[[DCTX]]) ins(%[[RC]] : memref<?x256xf16, #hipsr.mem<device>>) outs(%[[RC]] : memref<?x256xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%{{.*}}: !hipsr.context, %[[BODY_IN:.*]]: memref<?x256xf16, #hipsr.mem<device>>, %{{.*}}: memref<?x256xf16, #hipsr.mem<device>>):
// CHECK-NEXT: %[[COLLAPSED:.*]] = memref.collapse_shape %[[BODY_IN]] {{\[\[}}0, 1]] : memref<?x256xf16, #hipsr.mem<device>> into memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.compute_yield %[[COLLAPSED]] : memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT: } : memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[CAST_INIT:.*]] = memref.alloc(%[[FLAT_SIZE]]) : memref<?xf32, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.cast(%[[DCTX]]) ins(%[[FLAT]] : memref<?xf16, #hipsr.mem<device>>) outs(%[[CAST_INIT]] : memref<?xf32, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE1]], %[[INIT1]] : tensor<2xindex>, memref<?x256xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE2]], %[[RC]] : tensor<2xindex>, memref<?x256xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE3]], %[[FLAT]] : tensor<1xindex>, memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE3]], %[[CAST_INIT]] : tensor<1xindex>, memref<?xf32, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.pool_domain_yield %[[FLAT]] : memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT: } -> memref<?xf16, #hipsr.mem<device>> {domain_id = 0 : i64}
// CHECK-NEXT: return %[[OUT]] : memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT: }
func.func @test_return_shape(
    %ctx: !hipsr.context,
    %input: memref<?x512xf16, #hipsr.mem<device>>,
    %weight: memref<512x256xf16, #hipsr.mem<device>>,
    %bias: memref<?x256xf16, #hipsr.mem<device>>
) -> memref<?xf16, #hipsr.mem<device>> {
  %out = hipsr.pool_domain(
      %ctx, %input, %weight, %bias
      : !hipsr.context, memref<?x512xf16, #hipsr.mem<device>>,
        memref<512x256xf16, #hipsr.mem<device>>,
        memref<?x256xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context,
       %in: memref<?x512xf16, #hipsr.mem<device>>,
       %w: memref<512x256xf16, #hipsr.mem<device>>,
       %b: memref<?x256xf16, #hipsr.mem<device>>):
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %m = memref.dim %in, %c0 : memref<?x512xf16, #hipsr.mem<device>>
    %n = memref.dim %w, %c1 : memref<512x256xf16, #hipsr.mem<device>>
    %flat_size = arith.muli %m, %n : index

    %e1 = memref.alloc() : memref<2xindex>
    memref.store %m, %e1[%c0] : memref<2xindex>
    memref.store %n, %e1[%c1] : memref<2xindex>
    %shape1 = bufferization.to_tensor %e1 : memref<2xindex> to tensor<2xindex>

    %e2 = memref.alloc() : memref<2xindex>
    memref.store %m, %e2[%c0] : memref<2xindex>
    memref.store %n, %e2[%c1] : memref<2xindex>
    %shape2 = bufferization.to_tensor %e2 : memref<2xindex> to tensor<2xindex>

    %e3 = memref.alloc() : memref<1xindex>
    memref.store %flat_size, %e3[%c0] : memref<1xindex>
    %shape3 = bufferization.to_tensor %e3 : memref<1xindex> to tensor<1xindex>

    %init1 = memref.alloc(%m) : memref<?x256xf16, #hipsr.mem<device>>
    hipsr.matmul(%dctx)
      ins(%in, %w : memref<?x512xf16, #hipsr.mem<device>>,
                    memref<512x256xf16, #hipsr.mem<device>>)
      outs(%init1 : memref<?x256xf16, #hipsr.mem<device>>)
    %init2 = memref.alloc(%m) : memref<?x256xf16, #hipsr.mem<device>>
    hipsr.add(%dctx)
      ins(%init1, %b : memref<?x256xf16, #hipsr.mem<device>>,
                       memref<?x256xf16, #hipsr.mem<device>>)
      outs(%init2 : memref<?x256xf16, #hipsr.mem<device>>)
    %flat = hipsr.compute(%dctx)
      ins(%init2 : memref<?x256xf16, #hipsr.mem<device>>)
      outs(%init2 : memref<?x256xf16, #hipsr.mem<device>>) {
    ^bb0(
        %body_ctx: !hipsr.context,
        %body_in: memref<?x256xf16, #hipsr.mem<device>>,
        %body_dest: memref<?x256xf16, #hipsr.mem<device>>
    ):
      %collapsed = memref.collapse_shape %body_in [[0, 1]]
        : memref<?x256xf16, #hipsr.mem<device>>
          into memref<?xf16, #hipsr.mem<device>>
      hipsr.compute_yield %collapsed : memref<?xf16, #hipsr.mem<device>>
    } : memref<?xf16, #hipsr.mem<device>>

    %cast_init = memref.alloc(%flat_size)
      : memref<?xf32, #hipsr.mem<device>>
    hipsr.cast(%dctx)
      ins(%flat : memref<?xf16, #hipsr.mem<device>>)
      outs(%cast_init : memref<?xf32, #hipsr.mem<device>>)

    hipsr.preserve_shape %shape1, %init1
      : tensor<2xindex>, memref<?x256xf16, #hipsr.mem<device>>
    hipsr.preserve_shape %shape2, %init2
      : tensor<2xindex>, memref<?x256xf16, #hipsr.mem<device>>
    hipsr.preserve_shape %shape3, %flat
      : tensor<1xindex>, memref<?xf16, #hipsr.mem<device>>
    hipsr.preserve_shape %shape3, %cast_init
      : tensor<1xindex>, memref<?xf32, #hipsr.mem<device>>
    hipsr.pool_domain_yield %flat
      : memref<?xf16, #hipsr.mem<device>>
  } -> memref<?xf16, #hipsr.mem<device>> {
    domain_id = 0 : i64
  }

  return %out : memref<?xf16, #hipsr.mem<device>>
}
