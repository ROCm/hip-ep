// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -split-input-file -hipsr-use-output-allocator | FileCheck %s

// The pass runs after bufferization, so every preserved shape here is an index
// memref. Each case checks its whole function, because the extents the pass
// picks and the rewritten users only mean anything as one sequence.

// The extents come from the preserved shape, not from the memref.alloc
// operands.
// CHECK-LABEL:   func.func @dynamic_output(
// CHECK-SAME:      %[[ARG0:.*]]: !hipsr.context,
// CHECK-SAME:      %[[ARG1:.*]]: memref<2xindex>,
// CHECK-SAME:      %[[ARG2:.*]]: index,
// CHECK-SAME:      %[[ARG3:.*]]: index) -> memref<?x?xf16, #hipsr.mem<device>> {
// CHECK-NEXT:    %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:    %[[LOAD_0:.*]] = memref.load %[[ARG1]]{{\[}}%[[CONSTANT_0]]] : memref<2xindex>
// CHECK-NEXT:    %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:    %[[LOAD_1:.*]] = memref.load %[[ARG1]]{{\[}}%[[CONSTANT_1]]] : memref<2xindex>
// CHECK-NEXT:    %[[ALLOC_OUTPUT_0:.*]] = hipsr.alloc_output(%[[ARG0]], %[[LOAD_0]], %[[LOAD_1]]) {out_idx = 0 : i64} : memref<?x?xf16, #hipsr.mem<device>>
// CHECK-NEXT:    hipsr.preserve_shape %[[ARG1]], %[[ALLOC_OUTPUT_0]] : memref<2xindex>, memref<?x?xf16, #hipsr.mem<device>>
// CHECK-NEXT:    return %[[ALLOC_OUTPUT_0]] : memref<?x?xf16, #hipsr.mem<device>>
// CHECK-NEXT:  }
func.func @dynamic_output(
    %ctx: !hipsr.context, %shape: memref<2xindex>, %M: index, %N: index)
    -> memref<?x?xf16, #hipsr.mem<device>> {
  %out = memref.alloc(%M, %N)
      : memref<?x?xf16, #hipsr.mem<device>>
  hipsr.preserve_shape %shape, %out
    : memref<2xindex>, memref<?x?xf16, #hipsr.mem<device>>
  return %out : memref<?x?xf16, #hipsr.mem<device>>
}

// -----

// The graph is MatMul -> Add -> Reshape, and a Cast also reads the returned
// Reshape result. The return type is rank 1 while the allocation the Add
// writes is rank 2, so the pass adds a reinterpret_cast view.
// CHECK-LABEL:   func.func @test_return_shape(
// CHECK-SAME:      %[[ARG0:.*]]: !hipsr.context,
// CHECK-SAME:      %[[ARG1:.*]]: memref<?x512xf16, #hipsr.mem<device>>,
// CHECK-SAME:      %[[ARG2:.*]]: memref<512x256xf16, #hipsr.mem<device>>,
// CHECK-SAME:      %[[ARG3:.*]]: memref<?x256xf16, #hipsr.mem<device>>) -> memref<?xf16, #hipsr.mem<device>> {
// CHECK-NEXT:    %[[POOL_DOMAIN_0:.*]] = hipsr.pool_domain(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]] : !hipsr.context, memref<?x512xf16, #hipsr.mem<device>>, memref<512x256xf16, #hipsr.mem<device>>, memref<?x256xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:    ^bb0(%[[VAL_0:.*]]: !hipsr.context, %[[VAL_1:.*]]: memref<?x512xf16, #hipsr.mem<device>>, %[[VAL_2:.*]]: memref<512x256xf16, #hipsr.mem<device>>, %[[VAL_3:.*]]: memref<?x256xf16, #hipsr.mem<device>>):
// CHECK-NEXT:      %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[CONSTANT_2:.*]] = arith.constant 256 : index
// CHECK-NEXT:      %[[DIM_0:.*]] = memref.dim %[[VAL_1]], %[[CONSTANT_0]] : memref<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[DIM_1:.*]] = memref.dim %[[VAL_2]], %[[CONSTANT_1]] : memref<512x256xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[MULI_0:.*]] = arith.muli %[[DIM_0]], %[[DIM_1]] : index
// CHECK-NEXT:      %[[ALLOC_0:.*]] = memref.alloc() : memref<2xindex>
// CHECK-NEXT:      memref.store %[[DIM_0]], %[[ALLOC_0]]{{\[}}%[[CONSTANT_0]]] : memref<2xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_2]], %[[ALLOC_0]]{{\[}}%[[CONSTANT_1]]] : memref<2xindex>
// CHECK-NEXT:      %[[ALLOC_1:.*]] = memref.alloc() : memref<2xindex>
// CHECK-NEXT:      memref.store %[[DIM_0]], %[[ALLOC_1]]{{\[}}%[[CONSTANT_0]]] : memref<2xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_2]], %[[ALLOC_1]]{{\[}}%[[CONSTANT_1]]] : memref<2xindex>
// CHECK-NEXT:      %[[ALLOC_2:.*]] = memref.alloc() : memref<1xindex>
// CHECK-NEXT:      memref.store %[[MULI_0]], %[[ALLOC_2]]{{\[}}%[[CONSTANT_0]]] : memref<1xindex>
// CHECK-NEXT:      %[[ALLOC_3:.*]] = memref.alloc(%[[DIM_0]]) : memref<?x256xf16, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.matmul(%[[VAL_0]]) ins(%[[VAL_1]], %[[VAL_2]] : memref<?x512xf16, #hipsr.mem<device>>, memref<512x256xf16, #hipsr.mem<device>>) outs(%[[ALLOC_3]] : memref<?x256xf16, #hipsr.mem<device>>)
// CHECK-NEXT:      %[[CONSTANT_3:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[LOAD_0:.*]] = memref.load %[[ALLOC_2]]{{\[}}%[[CONSTANT_3]]] : memref<1xindex>
// CHECK-NEXT:      %[[ALLOC_OUTPUT_0:.*]] = hipsr.alloc_output(%[[VAL_0]], %[[LOAD_0]]) {out_idx = 0 : i64} : memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[CONSTANT_4:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[LOAD_1:.*]] = memref.load %[[ALLOC_1]]{{\[}}%[[CONSTANT_4]]] : memref<2xindex>
// CHECK-NEXT:      %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[ALLOC_OUTPUT_0]] to offset: [0], sizes: [%[[LOAD_1]], 256], strides: [256, 1] : memref<?xf16, #hipsr.mem<device>> to memref<?x256xf16, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.add(%[[VAL_0]]) ins(%[[ALLOC_3]], %[[VAL_3]] : memref<?x256xf16, #hipsr.mem<device>>, memref<?x256xf16, #hipsr.mem<device>>) outs(%[[REINTERPRET_CAST_0]] : memref<?x256xf16, #hipsr.mem<device>>)
// CHECK-NEXT:      %[[COMPUTE_0:.*]] = hipsr.compute(%[[VAL_0]]) ins(%[[REINTERPRET_CAST_0]] : memref<?x256xf16, #hipsr.mem<device>>) outs(%[[REINTERPRET_CAST_0]] : memref<?x256xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:      ^bb0(%[[VAL_4:.*]]: !hipsr.context, %[[VAL_5:.*]]: memref<?x256xf16, #hipsr.mem<device>>, %[[VAL_6:.*]]: memref<?x256xf16, #hipsr.mem<device>>):
// CHECK-NEXT:        %[[COLLAPSE_SHAPE_0:.*]] = memref.collapse_shape %[[VAL_5]] {{\[\[}}0, 1]] : memref<?x256xf16, #hipsr.mem<device>> into memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.compute_yield %[[COLLAPSE_SHAPE_0]] : memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:      } : memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[ALLOC_4:.*]] = memref.alloc(%[[MULI_0]]) : memref<?xf32, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.cast(%[[VAL_0]]) ins(%[[COMPUTE_0]] : memref<?xf16, #hipsr.mem<device>>) outs(%[[ALLOC_4]] : memref<?xf32, #hipsr.mem<device>>)
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_0]], %[[ALLOC_3]] : memref<2xindex>, memref<?x256xf16, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_1]], %[[REINTERPRET_CAST_0]] : memref<2xindex>, memref<?x256xf16, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_2]], %[[COMPUTE_0]] : memref<1xindex>, memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_2]], %[[ALLOC_4]] : memref<1xindex>, memref<?xf32, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.pool_domain_yield %[[COMPUTE_0]] : memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:    } -> memref<?xf16, #hipsr.mem<device>> {domain_id = 0 : i64}
// CHECK-NEXT:    return %[[POOL_DOMAIN_0]] : memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:  }
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
    %c256 = arith.constant 256 : index

    %m = memref.dim %in, %c0 : memref<?x512xf16, #hipsr.mem<device>>
    %n = memref.dim %w, %c1 : memref<512x256xf16, #hipsr.mem<device>>
    %flat_size = arith.muli %m, %n : index

    %shape1 = memref.alloc() : memref<2xindex>
    memref.store %m, %shape1[%c0] : memref<2xindex>
    memref.store %c256, %shape1[%c1] : memref<2xindex>
    %shape2 = memref.alloc() : memref<2xindex>
    memref.store %m, %shape2[%c0] : memref<2xindex>
    memref.store %c256, %shape2[%c1] : memref<2xindex>
    %shape3 = memref.alloc() : memref<1xindex>
    memref.store %flat_size, %shape3[%c0] : memref<1xindex>

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
      : memref<2xindex>, memref<?x256xf16, #hipsr.mem<device>>
    hipsr.preserve_shape %shape2, %init2
      : memref<2xindex>, memref<?x256xf16, #hipsr.mem<device>>
    hipsr.preserve_shape %shape3, %flat
      : memref<1xindex>, memref<?xf16, #hipsr.mem<device>>
    hipsr.preserve_shape %shape3, %cast_init
      : memref<1xindex>, memref<?xf32, #hipsr.mem<device>>
    hipsr.pool_domain_yield %flat
      : memref<?xf16, #hipsr.mem<device>>
  } -> memref<?xf16, #hipsr.mem<device>> {
    domain_id = 0 : i64
  }

  return %out : memref<?xf16, #hipsr.mem<device>>
}
