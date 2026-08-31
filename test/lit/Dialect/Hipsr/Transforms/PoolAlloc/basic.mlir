// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -split-input-file -hipsr-pool-alloc | FileCheck %s

// Each case checks its whole function. The pass decides a pool size, an offset
// per buffer and which allocations survive, and those only mean anything as one
// sequence, so a partial check would let any of them drift.

// Three f16 elements are six bytes, which round up to the pool alignment.
// CHECK-LABEL:   func.func @align_up_rounding(
// CHECK-SAME:      %[[ARG0:.*]]: !hipsr.context,
// CHECK-SAME:      %[[ARG1:.*]]: memref<3xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:           hipsr.pool_domain(%[[ARG0]], %[[ARG1]] : !hipsr.context, memref<3xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:           ^bb0(%[[VAL_0:.*]]: !hipsr.context, %[[VAL_1:.*]]: memref<3xf16, #hipsr.mem<device>>):
// CHECK-NEXT:             %[[CONSTANT_0:.*]] = arith.constant 6 : index
// CHECK-NEXT:             %[[CONSTANT_1:.*]] = arith.constant 256 : index
// CHECK-NEXT:             %[[CONSTANT_2:.*]] = arith.constant 255 : index
// CHECK-NEXT:             %[[ADDI_0:.*]] = arith.addi %[[CONSTANT_0]], %[[CONSTANT_2]] : index
// CHECK-NEXT:             %[[DIVUI_0:.*]] = arith.divui %[[ADDI_0]], %[[CONSTANT_1]] : index
// CHECK-NEXT:             %[[MULI_0:.*]] = arith.muli %[[DIVUI_0]], %[[CONSTANT_1]] : index
// CHECK-NEXT:             %[[CONSTANT_3:.*]] = arith.constant 0 : index
// CHECK-NEXT:             %[[GET_POOL_0:.*]] = hipsr.get_pool(%[[VAL_0]], %[[MULI_0]]) {domain_id = 0 : i64} : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT:             %[[VIEW_0:.*]] = memref.view %[[GET_POOL_0]]{{\[}}%[[CONSTANT_3]]][] : memref<?xi8, #hipsr.mem<device>> to memref<3xf16, #hipsr.mem<device>>
// CHECK-NEXT:             hipsr.add(%[[VAL_0]]) ins(%[[VAL_1]], %[[VAL_1]] : memref<3xf16, #hipsr.mem<device>>, memref<3xf16, #hipsr.mem<device>>) outs(%[[VIEW_0]] : memref<3xf16, #hipsr.mem<device>>)
// CHECK-NEXT:           } {domain_id = 0 : i64}
// CHECK-NEXT:           return
// CHECK-NEXT:         }
func.func @align_up_rounding(%ctx: !hipsr.context,
                             %in: memref<3xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %in : !hipsr.context, memref<3xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %d: memref<3xf16, #hipsr.mem<device>>):
    %a1 = memref.alloc() : memref<3xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%d, %d : memref<3xf16, #hipsr.mem<device>>, memref<3xf16, #hipsr.mem<device>>)
               outs(%a1 : memref<3xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  } {domain_id = 0 : i64}
  return
}

// -----

// A mask element gets a whole byte, so 4x256 reserves 1024 rather than the 0 a
// truncating bit-width-over-eight would give it.
// CHECK-LABEL:   func.func @sub_byte_element(
// CHECK-SAME:      %[[ARG0:.*]]: !hipsr.context,
// CHECK-SAME:      %[[ARG1:.*]]: memref<4x256xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:           hipsr.pool_domain(%[[ARG0]], %[[ARG1]] : !hipsr.context, memref<4x256xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:           ^bb0(%[[VAL_0:.*]]: !hipsr.context, %[[VAL_1:.*]]: memref<4x256xf16, #hipsr.mem<device>>):
// CHECK-NEXT:             %[[CONSTANT_0:.*]] = arith.constant 1024 : index
// CHECK-NEXT:             %[[CONSTANT_1:.*]] = arith.constant 256 : index
// CHECK-NEXT:             %[[CONSTANT_2:.*]] = arith.constant 255 : index
// CHECK-NEXT:             %[[ADDI_0:.*]] = arith.addi %[[CONSTANT_0]], %[[CONSTANT_2]] : index
// CHECK-NEXT:             %[[DIVUI_0:.*]] = arith.divui %[[ADDI_0]], %[[CONSTANT_1]] : index
// CHECK-NEXT:             %[[MULI_0:.*]] = arith.muli %[[DIVUI_0]], %[[CONSTANT_1]] : index
// CHECK-NEXT:             %[[CONSTANT_3:.*]] = arith.constant 0 : index
// CHECK-NEXT:             %[[GET_POOL_0:.*]] = hipsr.get_pool(%[[VAL_0]], %[[MULI_0]]) {domain_id = 0 : i64} : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT:             %[[VIEW_0:.*]] = memref.view %[[GET_POOL_0]]{{\[}}%[[CONSTANT_3]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x256xi1, #hipsr.mem<device>>
// CHECK-NEXT:             hipsr.equal(%[[VAL_0]]) ins(%[[VAL_1]], %[[VAL_1]] : memref<4x256xf16, #hipsr.mem<device>>, memref<4x256xf16, #hipsr.mem<device>>) outs(%[[VIEW_0]] : memref<4x256xi1, #hipsr.mem<device>>)
// CHECK-NEXT:           } {domain_id = 0 : i64}
// CHECK-NEXT:           return
// CHECK-NEXT:         }
func.func @sub_byte_element(%ctx: !hipsr.context,
                            %in: memref<4x256xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %in : !hipsr.context, memref<4x256xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<4x256xf16, #hipsr.mem<device>>):
    %a1 = memref.alloc() : memref<4x256xi1, #hipsr.mem<device>>
    hipsr.equal(%dctx) ins(%din, %din : memref<4x256xf16, #hipsr.mem<device>>, memref<4x256xf16, #hipsr.mem<device>>)
                 outs(%a1 : memref<4x256xi1, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  } {domain_id = 0 : i64}
  return
}

// -----

// An allocation nothing reads keeps its memref.alloc. Only the live one is
// sized into the pool, so the reserved bytes cover one buffer rather than two.
// CHECK-LABEL:   func.func @dead_alloc_skipped(
// CHECK-SAME:      %[[ARG0:.*]]: !hipsr.context,
// CHECK-SAME:      %[[ARG1:.*]]: memref<4x1024xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:           hipsr.pool_domain(%[[ARG0]], %[[ARG1]] : !hipsr.context, memref<4x1024xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:           ^bb0(%[[VAL_0:.*]]: !hipsr.context, %[[VAL_1:.*]]: memref<4x1024xf16, #hipsr.mem<device>>):
// CHECK-NEXT:             %[[ALLOC_0:.*]] = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT:             %[[CONSTANT_0:.*]] = arith.constant 8192 : index
// CHECK-NEXT:             %[[CONSTANT_1:.*]] = arith.constant 256 : index
// CHECK-NEXT:             %[[CONSTANT_2:.*]] = arith.constant 255 : index
// CHECK-NEXT:             %[[ADDI_0:.*]] = arith.addi %[[CONSTANT_0]], %[[CONSTANT_2]] : index
// CHECK-NEXT:             %[[DIVUI_0:.*]] = arith.divui %[[ADDI_0]], %[[CONSTANT_1]] : index
// CHECK-NEXT:             %[[MULI_0:.*]] = arith.muli %[[DIVUI_0]], %[[CONSTANT_1]] : index
// CHECK-NEXT:             %[[CONSTANT_3:.*]] = arith.constant 0 : index
// CHECK-NEXT:             %[[GET_POOL_0:.*]] = hipsr.get_pool(%[[VAL_0]], %[[MULI_0]]) {domain_id = 0 : i64} : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT:             %[[VIEW_0:.*]] = memref.view %[[GET_POOL_0]]{{\[}}%[[CONSTANT_3]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT:             hipsr.add(%[[VAL_0]]) ins(%[[VAL_1]], %[[VAL_1]] : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%[[VIEW_0]] : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT:           } {domain_id = 0 : i64}
// CHECK-NEXT:           return
// CHECK-NEXT:         }
func.func @dead_alloc_skipped(%ctx: !hipsr.context,
                              %in: memref<4x1024xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %in : !hipsr.context, memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<4x1024xf16, #hipsr.mem<device>>):
    %dead = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    %a1 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>)
               outs(%a1 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  } {domain_id = 0 : i64}
  return
}

// -----

// Buffers of different element types share one pool, so its size is the larger
// of a static f32 buffer and a dynamic f16 one.
// CHECK-LABEL:   func.func @mixed_dtypes(
// CHECK-SAME:      %[[ARG0:.*]]: !hipsr.context,
// CHECK-SAME:      %[[ARG1:.*]]: memref<4x256xf32, #hipsr.mem<device>>,
// CHECK-SAME:      %[[ARG2:.*]]: memref<?x512xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:           hipsr.pool_domain(%[[ARG0]], %[[ARG1]], %[[ARG2]] : !hipsr.context, memref<4x256xf32, #hipsr.mem<device>>, memref<?x512xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:           ^bb0(%[[VAL_0:.*]]: !hipsr.context, %[[VAL_1:.*]]: memref<4x256xf32, #hipsr.mem<device>>, %[[VAL_2:.*]]: memref<?x512xf16, #hipsr.mem<device>>):
// CHECK-NEXT:             %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:             %[[DIM_0:.*]] = memref.dim %[[VAL_2]], %[[CONSTANT_0]] : memref<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT:             %[[CONSTANT_1:.*]] = arith.constant 4096 : index
// CHECK-NEXT:             %[[CONSTANT_2:.*]] = arith.constant 1024 : index
// CHECK-NEXT:             %[[MULI_0:.*]] = arith.muli %[[CONSTANT_2]], %[[DIM_0]] : index
// CHECK-NEXT:             %[[MAXUI_0:.*]] = arith.maxui %[[CONSTANT_1]], %[[MULI_0]] : index
// CHECK-NEXT:             %[[CONSTANT_3:.*]] = arith.constant 256 : index
// CHECK-NEXT:             %[[CONSTANT_4:.*]] = arith.constant 255 : index
// CHECK-NEXT:             %[[ADDI_0:.*]] = arith.addi %[[MAXUI_0]], %[[CONSTANT_4]] : index
// CHECK-NEXT:             %[[DIVUI_0:.*]] = arith.divui %[[ADDI_0]], %[[CONSTANT_3]] : index
// CHECK-NEXT:             %[[MULI_1:.*]] = arith.muli %[[DIVUI_0]], %[[CONSTANT_3]] : index
// CHECK-NEXT:             %[[CONSTANT_5:.*]] = arith.constant 0 : index
// CHECK-NEXT:             %[[GET_POOL_0:.*]] = hipsr.get_pool(%[[VAL_0]], %[[MULI_1]]) {domain_id = 0 : i64} : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT:             %[[VIEW_0:.*]] = memref.view %[[GET_POOL_0]]{{\[}}%[[CONSTANT_5]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x256xf32, #hipsr.mem<device>>
// CHECK-NEXT:             %[[VIEW_1:.*]] = memref.view %[[GET_POOL_0]]{{\[}}%[[CONSTANT_5]]]{{\[}}%[[DIM_0]]] : memref<?xi8, #hipsr.mem<device>> to memref<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT:             hipsr.add(%[[VAL_0]]) ins(%[[VAL_1]], %[[VAL_1]] : memref<4x256xf32, #hipsr.mem<device>>, memref<4x256xf32, #hipsr.mem<device>>) outs(%[[VIEW_0]] : memref<4x256xf32, #hipsr.mem<device>>)
// CHECK-NEXT:             hipsr.add(%[[VAL_0]]) ins(%[[VAL_2]], %[[VAL_2]] : memref<?x512xf16, #hipsr.mem<device>>, memref<?x512xf16, #hipsr.mem<device>>) outs(%[[VIEW_1]] : memref<?x512xf16, #hipsr.mem<device>>)
// CHECK-NEXT:           } {domain_id = 0 : i64}
// CHECK-NEXT:           return
// CHECK-NEXT:         }
func.func @mixed_dtypes(%ctx: !hipsr.context,
                        %inf32: memref<4x256xf32, #hipsr.mem<device>>,
                        %inf16: memref<?x512xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %inf32, %inf16 :
      !hipsr.context,
      memref<4x256xf32, #hipsr.mem<device>>,
      memref<?x512xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context,
       %sf32: memref<4x256xf32, #hipsr.mem<device>>,
       %sf16: memref<?x512xf16, #hipsr.mem<device>>):
    %c0 = arith.constant 0 : index
    %d = memref.dim %sf16, %c0 : memref<?x512xf16, #hipsr.mem<device>>
    %a1 = memref.alloc() : memref<4x256xf32, #hipsr.mem<device>>
    %a2 = memref.alloc(%d) : memref<?x512xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%sf32, %sf32 : memref<4x256xf32, #hipsr.mem<device>>, memref<4x256xf32, #hipsr.mem<device>>)
               outs(%a1 : memref<4x256xf32, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%sf16, %sf16 : memref<?x512xf16, #hipsr.mem<device>>, memref<?x512xf16, #hipsr.mem<device>>)
               outs(%a2 : memref<?x512xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  } {domain_id = 0 : i64}
  return
}

// -----

// Neither size is known at compile time, so the pool takes a runtime max of
// the two products rather than folding to a constant.
// CHECK-LABEL:   func.func @mixed_dynamic_dims(
// CHECK-SAME:      %[[ARG0:.*]]: !hipsr.context,
// CHECK-SAME:      %[[ARG1:.*]]: memref<?x512xf16, #hipsr.mem<device>>,
// CHECK-SAME:      %[[ARG2:.*]]: memref<?x256xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:           hipsr.pool_domain(%[[ARG0]], %[[ARG1]], %[[ARG2]] : !hipsr.context, memref<?x512xf16, #hipsr.mem<device>>, memref<?x256xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:           ^bb0(%[[VAL_0:.*]]: !hipsr.context, %[[VAL_1:.*]]: memref<?x512xf16, #hipsr.mem<device>>, %[[VAL_2:.*]]: memref<?x256xf16, #hipsr.mem<device>>):
// CHECK-NEXT:             %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:             %[[DIM_0:.*]] = memref.dim %[[VAL_1]], %[[CONSTANT_0]] : memref<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT:             %[[DIM_1:.*]] = memref.dim %[[VAL_2]], %[[CONSTANT_0]] : memref<?x256xf16, #hipsr.mem<device>>
// CHECK-NEXT:             %[[CONSTANT_1:.*]] = arith.constant 1024 : index
// CHECK-NEXT:             %[[MULI_0:.*]] = arith.muli %[[CONSTANT_1]], %[[DIM_0]] : index
// CHECK-NEXT:             %[[CONSTANT_2:.*]] = arith.constant 512 : index
// CHECK-NEXT:             %[[MULI_1:.*]] = arith.muli %[[CONSTANT_2]], %[[DIM_1]] : index
// CHECK-NEXT:             %[[MAXUI_0:.*]] = arith.maxui %[[MULI_0]], %[[MULI_1]] : index
// CHECK-NEXT:             %[[CONSTANT_3:.*]] = arith.constant 256 : index
// CHECK-NEXT:             %[[CONSTANT_4:.*]] = arith.constant 255 : index
// CHECK-NEXT:             %[[ADDI_0:.*]] = arith.addi %[[MAXUI_0]], %[[CONSTANT_4]] : index
// CHECK-NEXT:             %[[DIVUI_0:.*]] = arith.divui %[[ADDI_0]], %[[CONSTANT_3]] : index
// CHECK-NEXT:             %[[MULI_2:.*]] = arith.muli %[[DIVUI_0]], %[[CONSTANT_3]] : index
// CHECK-NEXT:             %[[CONSTANT_5:.*]] = arith.constant 0 : index
// CHECK-NEXT:             %[[GET_POOL_0:.*]] = hipsr.get_pool(%[[VAL_0]], %[[MULI_2]]) {domain_id = 0 : i64} : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT:             %[[VIEW_0:.*]] = memref.view %[[GET_POOL_0]]{{\[}}%[[CONSTANT_5]]]{{\[}}%[[DIM_0]]] : memref<?xi8, #hipsr.mem<device>> to memref<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT:             %[[VIEW_1:.*]] = memref.view %[[GET_POOL_0]]{{\[}}%[[CONSTANT_5]]]{{\[}}%[[DIM_1]]] : memref<?xi8, #hipsr.mem<device>> to memref<?x256xf16, #hipsr.mem<device>>
// CHECK-NEXT:             hipsr.add(%[[VAL_0]]) ins(%[[VAL_1]], %[[VAL_1]] : memref<?x512xf16, #hipsr.mem<device>>, memref<?x512xf16, #hipsr.mem<device>>) outs(%[[VIEW_0]] : memref<?x512xf16, #hipsr.mem<device>>)
// CHECK-NEXT:             hipsr.add(%[[VAL_0]]) ins(%[[VIEW_0]], %[[VIEW_0]] : memref<?x512xf16, #hipsr.mem<device>>, memref<?x512xf16, #hipsr.mem<device>>) outs(%[[VAL_1]] : memref<?x512xf16, #hipsr.mem<device>>)
// CHECK-NEXT:             hipsr.add(%[[VAL_0]]) ins(%[[VAL_2]], %[[VAL_2]] : memref<?x256xf16, #hipsr.mem<device>>, memref<?x256xf16, #hipsr.mem<device>>) outs(%[[VIEW_1]] : memref<?x256xf16, #hipsr.mem<device>>)
// CHECK-NEXT:             hipsr.add(%[[VAL_0]]) ins(%[[VIEW_1]], %[[VIEW_1]] : memref<?x256xf16, #hipsr.mem<device>>, memref<?x256xf16, #hipsr.mem<device>>) outs(%[[VAL_2]] : memref<?x256xf16, #hipsr.mem<device>>)
// CHECK-NEXT:           } {domain_id = 0 : i64}
// CHECK-NEXT:           return
// CHECK-NEXT:         }
func.func @mixed_dynamic_dims(%ctx: !hipsr.context,
                              %ina: memref<?x512xf16, #hipsr.mem<device>>,
                              %inb: memref<?x256xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %ina, %inb :
      !hipsr.context,
      memref<?x512xf16, #hipsr.mem<device>>,
      memref<?x256xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context,
       %da: memref<?x512xf16, #hipsr.mem<device>>,
       %db: memref<?x256xf16, #hipsr.mem<device>>):
    %c0 = arith.constant 0 : index
    %d0 = memref.dim %da, %c0 : memref<?x512xf16, #hipsr.mem<device>>
    %d1 = memref.dim %db, %c0 : memref<?x256xf16, #hipsr.mem<device>>
    %a1 = memref.alloc(%d0) : memref<?x512xf16, #hipsr.mem<device>>
    %a2 = memref.alloc(%d1) : memref<?x256xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%da, %da : memref<?x512xf16, #hipsr.mem<device>>, memref<?x512xf16, #hipsr.mem<device>>) outs(%a1 : memref<?x512xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a1, %a1 : memref<?x512xf16, #hipsr.mem<device>>, memref<?x512xf16, #hipsr.mem<device>>) outs(%da : memref<?x512xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%db, %db : memref<?x256xf16, #hipsr.mem<device>>, memref<?x256xf16, #hipsr.mem<device>>) outs(%a2 : memref<?x256xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a2, %a2 : memref<?x256xf16, #hipsr.mem<device>>, memref<?x256xf16, #hipsr.mem<device>>) outs(%db : memref<?x256xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  } {domain_id = 0 : i64}
  return
}

// -----

// Two domains size and fetch their pools separately, each keyed by its own
// domain_id, so neither reuses the other's bytes.
// CHECK-LABEL:   func.func @two_domains(
// CHECK-SAME:      %[[ARG0:.*]]: !hipsr.context,
// CHECK-SAME:      %[[ARG1:.*]]: memref<4x1024xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:           hipsr.pool_domain(%[[ARG0]], %[[ARG1]] : !hipsr.context, memref<4x1024xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:           ^bb0(%[[VAL_0:.*]]: !hipsr.context, %[[VAL_1:.*]]: memref<4x1024xf16, #hipsr.mem<device>>):
// CHECK-NEXT:             %[[CONSTANT_0:.*]] = arith.constant 8192 : index
// CHECK-NEXT:             %[[CONSTANT_1:.*]] = arith.constant 256 : index
// CHECK-NEXT:             %[[CONSTANT_2:.*]] = arith.constant 255 : index
// CHECK-NEXT:             %[[ADDI_0:.*]] = arith.addi %[[CONSTANT_0]], %[[CONSTANT_2]] : index
// CHECK-NEXT:             %[[DIVUI_0:.*]] = arith.divui %[[ADDI_0]], %[[CONSTANT_1]] : index
// CHECK-NEXT:             %[[MULI_0:.*]] = arith.muli %[[DIVUI_0]], %[[CONSTANT_1]] : index
// CHECK-NEXT:             %[[CONSTANT_3:.*]] = arith.constant 0 : index
// CHECK-NEXT:             %[[GET_POOL_0:.*]] = hipsr.get_pool(%[[VAL_0]], %[[MULI_0]]) {domain_id = 0 : i64} : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT:             %[[VIEW_0:.*]] = memref.view %[[GET_POOL_0]]{{\[}}%[[CONSTANT_3]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT:             hipsr.add(%[[VAL_0]]) ins(%[[VAL_1]], %[[VAL_1]] : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%[[VIEW_0]] : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT:           } {domain_id = 0 : i64}
// CHECK-NEXT:           hipsr.pool_domain(%[[ARG0]], %[[ARG1]] : !hipsr.context, memref<4x1024xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:           ^bb0(%[[VAL_2:.*]]: !hipsr.context, %[[VAL_3:.*]]: memref<4x1024xf16, #hipsr.mem<device>>):
// CHECK-NEXT:             %[[CONSTANT_4:.*]] = arith.constant 8192 : index
// CHECK-NEXT:             %[[CONSTANT_5:.*]] = arith.constant 256 : index
// CHECK-NEXT:             %[[CONSTANT_6:.*]] = arith.constant 255 : index
// CHECK-NEXT:             %[[ADDI_1:.*]] = arith.addi %[[CONSTANT_4]], %[[CONSTANT_6]] : index
// CHECK-NEXT:             %[[DIVUI_1:.*]] = arith.divui %[[ADDI_1]], %[[CONSTANT_5]] : index
// CHECK-NEXT:             %[[MULI_1:.*]] = arith.muli %[[DIVUI_1]], %[[CONSTANT_5]] : index
// CHECK-NEXT:             %[[CONSTANT_7:.*]] = arith.constant 0 : index
// CHECK-NEXT:             %[[GET_POOL_1:.*]] = hipsr.get_pool(%[[VAL_2]], %[[MULI_1]]) {domain_id = 7 : i64} : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT:             %[[VIEW_1:.*]] = memref.view %[[GET_POOL_1]]{{\[}}%[[CONSTANT_7]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT:             hipsr.add(%[[VAL_2]]) ins(%[[VAL_3]], %[[VAL_3]] : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%[[VIEW_1]] : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT:           } {domain_id = 7 : i64}
// CHECK-NEXT:           return
// CHECK-NEXT:         }
func.func @two_domains(%ctx: !hipsr.context, %in: memref<4x1024xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %in : !hipsr.context, memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<4x1024xf16, #hipsr.mem<device>>):
    %a1 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a1 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  } {domain_id = 0 : i64}
  hipsr.pool_domain(%ctx, %in : !hipsr.context, memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<4x1024xf16, #hipsr.mem<device>>):
    %a2 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a2 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  } {domain_id = 7 : i64}
  return
}
