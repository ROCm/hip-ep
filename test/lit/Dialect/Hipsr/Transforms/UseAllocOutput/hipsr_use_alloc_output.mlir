// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -hipsr-use-output-allocator | FileCheck %s

// A returned dynamic allocation becomes hipsr.alloc_output. Its dynamic sizes
// and return position are preserved.
// CHECK-LABEL: func.func @dynamic_output(
// CHECK-SAME: %[[CTX:.*]]: !hipsr.context, %[[M:.*]]: index, %[[N:.*]]: index
// CHECK-NOT: memref.alloc
// CHECK: %[[OUT:.*]] = hipsr.alloc_output(%[[CTX]], %[[M]], %[[N]]) {out_idx = 0 : i64} : memref<?x?xf16, #hipsr.mem<device>>
// CHECK: return %[[OUT]]
func.func @dynamic_output(
    %ctx: !hipsr.context, %M: index, %N: index)
    -> memref<?x?xf16, #hipsr.mem<device>> {
  %out = memref.alloc(%M, %N)
      : memref<?x?xf16, #hipsr.mem<device>>
  return %out : memref<?x?xf16, #hipsr.mem<device>>
}


// ---

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

    %shape1 = scf.execute_region -> !shape.shape {
      %s1 = shape.from_extents %m, %n : index, index
      scf.yield %s1 : !shape.shape
    }

    %shape2 = scf.execute_region -> !shape.shape {
      %s2 = shape.from_extents %m, %n : index, index
      scf.yield %s2 : !shape.shape
    }

    %shape3 = scf.execute_region -> !shape.shape {
      %s3 = shape.from_extents %flat_size : index
      scf.yield %s3 : !shape.shape
    }

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

    hipsr.preserve_shape %shape1, %init1
      : memref<?x256xf16, #hipsr.mem<device>>

    hipsr.preserve_shape %shape2, %init2
      : memref<?x256xf16, #hipsr.mem<device>>

    hipsr.preserve_shape %shape3, %flat
      : memref<?xf16, #hipsr.mem<device>>

    hipsr.pool_domain_yield %flat
      : memref<?xf16, #hipsr.mem<device>>
  } -> memref<?xf16, #hipsr.mem<device>> {
    domain_id = 0 : i64
  }

  return %out : memref<?xf16, #hipsr.mem<device>>
}