// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -split-input-file -verify-diagnostics -hipsr-pool-alloc | FileCheck %s

// Cases are ordered by lifetime-interval topology (interference-graph chromatic
// number chi = group count), ascending:
//   chi=0  empty         : noalloc_noop
//   chi=1  single point   : align_up_rounding, dynamic_size
//   chi=1  independent    : coalesce_static, coalesce_dynamic, coalesce_mixed
//   chi=2  staggered edge : split_two_groups, split_two_groups_dynamic
//   chi=2  one-over-many  : split_with_coalesced_group
//   chi=3  clique K3      : split_three_groups

// A domain with no memref.alloc (tensor-mode) is left untouched.
// CHECK-LABEL: func.func @noalloc_noop
// CHECK: hipsr.pool_domain
// CHECK: %[[BUF:.+]] = tensor.empty() : tensor<2x4xi64>
// CHECK-NEXT: hipsr.pool_domain_yield %[[BUF]] : tensor<2x4xi64>
// CHECK-NOT: hipsr.get_pool
// CHECK-NOT: memref.view
func.func @noalloc_noop(%ctx: !hipsr.context,
                        %in: tensor<3x4xf32>) -> tensor<2x4xi64> {
  %0 = hipsr.pool_domain(%ctx, %in : !hipsr.context, tensor<3x4xf32>) {
  ^bb0(%dctx: !hipsr.context, %din: tensor<3x4xf32>):
    %buf = tensor.empty() : tensor<2x4xi64>
    hipsr.pool_domain_yield %buf : tensor<2x4xi64>
  } -> tensor<2x4xi64>
  return %0 : tensor<2x4xi64>
}

// -----

// 3xf16 = 6 B is not a multiple of 256, so the alignUp chain must round up.
// CHECK-LABEL: func.func @align_up_rounding
// CHECK: %[[C6:.+]] = arith.constant 6 : index
// CHECK-NEXT: %[[C256:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[NUM:.+]] = arith.addi %[[C6]], %[[C255]] : index
// CHECK-NEXT: %[[DIV:.+]] = arith.divui %[[NUM]], %[[C256]] : index
// CHECK-NEXT: %[[SZ:.+]] = arith.muli %[[DIV]], %[[C256]] : index
// CHECK-NEXT: %[[POOL:.+]] = hipsr.get_pool(%{{.+}}, %[[SZ]]) : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT: %[[OFF:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[V0:.+]] = memref.view %[[POOL]][%[[OFF]]][] : memref<?xi8, #hipsr.mem<device>> to memref<3xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} : memref<3xf16, #hipsr.mem<device>>, memref<3xf16, #hipsr.mem<device>>) outs(%[[V0]] : memref<3xf16, #hipsr.mem<device>>)
// CHECK-NOT: arith.maxui
// CHECK-NOT: memref.alloc
func.func @align_up_rounding(%ctx: !hipsr.context,
                        %in: memref<3xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %in : !hipsr.context, memref<3xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %d: memref<3xf16, #hipsr.mem<device>>):
    %a1 = memref.alloc() : memref<3xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%d, %d : memref<3xf16, #hipsr.mem<device>>, memref<3xf16, #hipsr.mem<device>>)
               outs(%a1 : memref<3xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  }
  return
}

// -----

// Size must multiply the static factor by the dynamic-size operand (%dim), not
// by the erased buffer value; guards the cyclic-SSA bug from shape_of %alloc.
// CHECK-LABEL: func.func @dynamic_size
// CHECK: %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[DIM:.+]] = memref.dim %{{.+}}, %[[C0]] : memref<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[C1024:.+]] = arith.constant 1024 : index
// CHECK-NEXT: %[[BYTES:.+]] = arith.muli %[[C1024]], %[[DIM]] : index
// CHECK-NEXT: %[[C256:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[NUM:.+]] = arith.addi %[[BYTES]], %[[C255]] : index
// CHECK-NEXT: %[[DIV:.+]] = arith.divui %[[NUM]], %[[C256]] : index
// CHECK-NEXT: %[[SZ:.+]] = arith.muli %[[DIV]], %[[C256]] : index
// CHECK-NEXT: %[[POOL:.+]] = hipsr.get_pool(%{{.+}}, %[[SZ]]) : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT: %[[OFF:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[V0:.+]] = memref.view %[[POOL]][%[[OFF]]][%[[DIM]]] : memref<?xi8, #hipsr.mem<device>> to memref<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} : memref<?x512xf16, #hipsr.mem<device>>, memref<?x512xf16, #hipsr.mem<device>>) outs(%[[V0]] : memref<?x512xf16, #hipsr.mem<device>>)
// CHECK-NOT: memref.alloc
func.func @dynamic_size(%ctx: !hipsr.context,
                   %in: memref<?x512xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %in : !hipsr.context, memref<?x512xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<?x512xf16, #hipsr.mem<device>>):
    %c0 = arith.constant 0 : index
    %d = memref.dim %din, %c0 : memref<?x512xf16, #hipsr.mem<device>>
    %a1 = memref.alloc(%d) : memref<?x512xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<?x512xf16, #hipsr.mem<device>>, memref<?x512xf16, #hipsr.mem<device>>)
               outs(%a1 : memref<?x512xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  }
  return
}

// -----

// A static f32 alloc and a dynamic f16 alloc share one pool: exercises the
// static/dynamic and different-element-type paths together (max over both).
// CHECK-LABEL: func.func @coalesce_mixed
// CHECK: %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[DIM:.+]] = memref.dim %{{.+}}, %[[C0]] : memref<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[C4096:.+]] = arith.constant 4096 : index
// CHECK-NEXT: %[[C1024:.+]] = arith.constant 1024 : index
// CHECK-NEXT: %[[BYTES:.+]] = arith.muli %[[C1024]], %[[DIM]] : index
// CHECK-NEXT: %[[MAX:.+]] = arith.maxui %[[C4096]], %[[BYTES]] : index
// CHECK-NEXT: %[[C256:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[NUM:.+]] = arith.addi %[[MAX]], %[[C255]] : index
// CHECK-NEXT: %[[DIV:.+]] = arith.divui %[[NUM]], %[[C256]] : index
// CHECK-NEXT: %[[SZ:.+]] = arith.muli %[[DIV]], %[[C256]] : index
// CHECK-NEXT: %[[POOL:.+]] = hipsr.get_pool(%{{.+}}, %[[SZ]]) : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT: %[[OFF:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[V0:.+]] = memref.view %[[POOL]][%[[OFF]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x256xf32, #hipsr.mem<device>>
// CHECK-NEXT: %[[V1:.+]] = memref.view %[[POOL]][%[[OFF]]][%[[DIM]]] : memref<?xi8, #hipsr.mem<device>> to memref<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} : memref<4x256xf32, #hipsr.mem<device>>, memref<4x256xf32, #hipsr.mem<device>>) outs(%[[V0]] : memref<4x256xf32, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} : memref<?x512xf16, #hipsr.mem<device>>, memref<?x512xf16, #hipsr.mem<device>>) outs(%[[V1]] : memref<?x512xf16, #hipsr.mem<device>>)
// CHECK-NOT: memref.alloc
func.func @coalesce_mixed(%ctx: !hipsr.context,
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
  }
  return
}

// -----

// Four disjoint allocs of differing sizes coalesce into one group: three
// arith.maxui fold the sizes, poolSize is the lone aligned muli (no addi),
// there is a single constant-0 offset (no addi), and all four views reuse it.
// CHECK-LABEL: func.func @coalesce_static
// CHECK: %[[C16384:.+]] = arith.constant 16384 : index
// CHECK-NEXT: %[[C8192A:.+]] = arith.constant 8192 : index
// CHECK-NEXT: %[[C4096:.+]] = arith.constant 4096 : index
// CHECK-NEXT: %[[C8192B:.+]] = arith.constant 8192 : index
// CHECK-NEXT: %[[M0:.+]] = arith.maxui %[[C16384]], %[[C8192A]] : index
// CHECK-NEXT: %[[M1:.+]] = arith.maxui %[[M0]], %[[C4096]] : index
// CHECK-NEXT: %[[M2:.+]] = arith.maxui %[[M1]], %[[C8192B]] : index
// CHECK-NEXT: %[[C256:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[NUM:.+]] = arith.addi %[[M2]], %[[C255]] : index
// CHECK-NEXT: %[[DIV:.+]] = arith.divui %[[NUM]], %[[C256]] : index
// CHECK-NEXT: %[[SZ:.+]] = arith.muli %[[DIV]], %[[C256]] : index
// CHECK-NEXT: %[[POOL:.+]] = hipsr.get_pool(%{{.+}}, %[[SZ]]) : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT: %[[OFF:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[V0:.+]] = memref.view %[[POOL]][%[[OFF]]][] : memref<?xi8, #hipsr.mem<device>> to memref<8x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[V1:.+]] = memref.view %[[POOL]][%[[OFF]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[V2:.+]] = memref.view %[[POOL]][%[[OFF]]][] : memref<?xi8, #hipsr.mem<device>> to memref<2x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[V3:.+]] = memref.view %[[POOL]][%[[OFF]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} : memref<8x1024xf16, #hipsr.mem<device>>, memref<8x1024xf16, #hipsr.mem<device>>) outs(%[[V0]] : memref<8x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[V0]], %{{.+}} : memref<8x1024xf16, #hipsr.mem<device>>, memref<8x1024xf16, #hipsr.mem<device>>) outs(%{{.+}} : memref<8x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%[[V1]] : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[V1]], %{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} : memref<2x1024xf16, #hipsr.mem<device>>, memref<2x1024xf16, #hipsr.mem<device>>) outs(%[[V2]] : memref<2x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[V2]], %{{.+}} : memref<2x1024xf16, #hipsr.mem<device>>, memref<2x1024xf16, #hipsr.mem<device>>) outs(%{{.+}} : memref<2x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%[[V3]] : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[V3]], %{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NOT: memref.alloc
// CHECK-NOT: shape.
func.func @coalesce_static(%ctx: !hipsr.context,
                               %in8: memref<8x1024xf16, #hipsr.mem<device>>,
                               %in4a: memref<4x1024xf16, #hipsr.mem<device>>,
                               %in2: memref<2x1024xf16, #hipsr.mem<device>>,
                               %in4b: memref<4x1024xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %in8, %in4a, %in2, %in4b :
      !hipsr.context,
      memref<8x1024xf16, #hipsr.mem<device>>,
      memref<4x1024xf16, #hipsr.mem<device>>,
      memref<2x1024xf16, #hipsr.mem<device>>,
      memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context,
       %d8: memref<8x1024xf16, #hipsr.mem<device>>,
       %d4a: memref<4x1024xf16, #hipsr.mem<device>>,
       %d2: memref<2x1024xf16, #hipsr.mem<device>>,
       %d4b: memref<4x1024xf16, #hipsr.mem<device>>):
    %a1 = memref.alloc() : memref<8x1024xf16, #hipsr.mem<device>>
    %a2 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    %a3 = memref.alloc() : memref<2x1024xf16, #hipsr.mem<device>>
    %a4 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%d8, %d8 : memref<8x1024xf16, #hipsr.mem<device>>, memref<8x1024xf16, #hipsr.mem<device>>) outs(%a1 : memref<8x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a1, %a1 : memref<8x1024xf16, #hipsr.mem<device>>, memref<8x1024xf16, #hipsr.mem<device>>) outs(%d8 : memref<8x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%d4a, %d4a : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a2 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a2, %a2 : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%d4a : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%d2, %d2 : memref<2x1024xf16, #hipsr.mem<device>>, memref<2x1024xf16, #hipsr.mem<device>>) outs(%a3 : memref<2x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a3, %a3 : memref<2x1024xf16, #hipsr.mem<device>>, memref<2x1024xf16, #hipsr.mem<device>>) outs(%d2 : memref<2x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%d4b, %d4b : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a4 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a4, %a4 : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%d4b : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  }
  return
}

// -----

// Two same-shape dynamic disjoint allocs coalesce into one group (no size-
// bucketing): both dynamic byte sizes (memref.dim * static factor) are folded by
// one arith.maxui, poolSize is the lone aligned size, off is a single constant 0,
// and both views share it. The CHECK-NEXT chain (muli feeding get_pool, constant-0
// offset immediately followed by the views) proves there is no second group.
// CHECK-LABEL: func.func @coalesce_dynamic
// CHECK: %[[DIM:.+]] = memref.dim %{{.+}}, %{{.+}} : memref<?x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[C2048A:.+]] = arith.constant 2048 : index
// CHECK-NEXT: %[[S0:.+]] = arith.muli %[[C2048A]], %[[DIM]] : index
// CHECK-NEXT: %[[C2048B:.+]] = arith.constant 2048 : index
// CHECK-NEXT: %[[S1:.+]] = arith.muli %[[C2048B]], %[[DIM]] : index
// CHECK-NEXT: %[[MAX:.+]] = arith.maxui %[[S0]], %[[S1]] : index
// CHECK-NEXT: %[[C256:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[NUM:.+]] = arith.addi %[[MAX]], %[[C255]] : index
// CHECK-NEXT: %[[DIV:.+]] = arith.divui %[[NUM]], %[[C256]] : index
// CHECK-NEXT: %[[SZ:.+]] = arith.muli %[[DIV]], %[[C256]] : index
// CHECK-NEXT: %[[POOL:.+]] = hipsr.get_pool(%{{.+}}, %[[SZ]]) : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT: %[[OFF:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[V0:.+]] = memref.view %[[POOL]][%[[OFF]]][%[[DIM]]] : memref<?xi8, #hipsr.mem<device>> to memref<?x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[V1:.+]] = memref.view %[[POOL]][%[[OFF]]][%[[DIM]]] : memref<?xi8, #hipsr.mem<device>> to memref<?x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} : memref<?x1024xf16, #hipsr.mem<device>>, memref<?x1024xf16, #hipsr.mem<device>>) outs(%[[V0]] : memref<?x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[V0]], %{{.+}} : memref<?x1024xf16, #hipsr.mem<device>>, memref<?x1024xf16, #hipsr.mem<device>>) outs(%{{.+}} : memref<?x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} : memref<?x1024xf16, #hipsr.mem<device>>, memref<?x1024xf16, #hipsr.mem<device>>) outs(%[[V1]] : memref<?x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[V1]], %{{.+}} : memref<?x1024xf16, #hipsr.mem<device>>, memref<?x1024xf16, #hipsr.mem<device>>) outs(%{{.+}} : memref<?x1024xf16, #hipsr.mem<device>>)
// CHECK-NOT: memref.alloc
func.func @coalesce_dynamic(%ctx: !hipsr.context, %in: memref<?x1024xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %in : !hipsr.context, memref<?x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<?x1024xf16, #hipsr.mem<device>>):
    %c0 = arith.constant 0 : index
    %d = memref.dim %din, %c0 : memref<?x1024xf16, #hipsr.mem<device>>
    %a1 = memref.alloc(%d) : memref<?x1024xf16, #hipsr.mem<device>>
    %a2 = memref.alloc(%d) : memref<?x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<?x1024xf16, #hipsr.mem<device>>, memref<?x1024xf16, #hipsr.mem<device>>) outs(%a1 : memref<?x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a1, %din : memref<?x1024xf16, #hipsr.mem<device>>, memref<?x1024xf16, #hipsr.mem<device>>) outs(%din : memref<?x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%din, %din : memref<?x1024xf16, #hipsr.mem<device>>, memref<?x1024xf16, #hipsr.mem<device>>) outs(%a2 : memref<?x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a2, %din : memref<?x1024xf16, #hipsr.mem<device>>, memref<?x1024xf16, #hipsr.mem<device>>) outs(%din : memref<?x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  }
  return
}

// -----

// a1 is read while writing a2, so their lifetimes overlap and land in separate
// groups: two aligned sizes, poolSize = g0 + g1, and off1 = off0 + g0.
// CHECK-LABEL: func.func @split_two_groups
// CHECK: %[[C8192A:.+]] = arith.constant 8192 : index
// CHECK-NEXT: %[[C256A:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255A:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[N0:.+]] = arith.addi %[[C8192A]], %[[C255A]] : index
// CHECK-NEXT: %[[D0:.+]] = arith.divui %[[N0]], %[[C256A]] : index
// CHECK-NEXT: %[[G0:.+]] = arith.muli %[[D0]], %[[C256A]] : index
// CHECK-NEXT: %[[C8192B:.+]] = arith.constant 8192 : index
// CHECK-NEXT: %[[C256B:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255B:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[N1:.+]] = arith.addi %[[C8192B]], %[[C255B]] : index
// CHECK-NEXT: %[[D1:.+]] = arith.divui %[[N1]], %[[C256B]] : index
// CHECK-NEXT: %[[G1:.+]] = arith.muli %[[D1]], %[[C256B]] : index
// CHECK-NEXT: %[[POOLSZ:.+]] = arith.addi %[[G0]], %[[G1]] : index
// CHECK-NEXT: %[[POOL:.+]] = hipsr.get_pool(%{{.+}}, %[[POOLSZ]]) : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT: %[[OFF0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[OFF1:.+]] = arith.addi %[[OFF0]], %[[G0]] : index
// CHECK-NEXT: %[[V0:.+]] = memref.view %[[POOL]][%[[OFF0]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[V1:.+]] = memref.view %[[POOL]][%[[OFF1]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%[[V0]] : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[V0]], %{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%[[V1]] : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NOT: memref.alloc
func.func @split_two_groups(%ctx: !hipsr.context, %in: memref<4x1024xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %in : !hipsr.context, memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<4x1024xf16, #hipsr.mem<device>>):
    %a1 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    %a2 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a1 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a1, %din : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a2 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  }
  return
}

// -----

// A dynamic f16 alloc and a static f16 alloc overlap, so each takes its own
// group: the dynamic group size flows memref.dim into the addi(g0,g1) poolSize
// and the addi(off0,g0) offset chain alongside a static group.
// CHECK-LABEL: func.func @split_two_groups_dynamic
// CHECK: %[[DIM:.+]] = memref.dim %{{.+}}, %{{.+}} : memref<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[C1024:.+]] = arith.constant 1024 : index
// CHECK-NEXT: %[[BYTES:.+]] = arith.muli %[[C1024]], %[[DIM]] : index
// CHECK-NEXT: %[[C256A:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255A:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[N0:.+]] = arith.addi %[[BYTES]], %[[C255A]] : index
// CHECK-NEXT: %[[D0:.+]] = arith.divui %[[N0]], %[[C256A]] : index
// CHECK-NEXT: %[[G0:.+]] = arith.muli %[[D0]], %[[C256A]] : index
// CHECK-NEXT: %[[C8192:.+]] = arith.constant 8192 : index
// CHECK-NEXT: %[[C256B:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255B:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[N1:.+]] = arith.addi %[[C8192]], %[[C255B]] : index
// CHECK-NEXT: %[[D1:.+]] = arith.divui %[[N1]], %[[C256B]] : index
// CHECK-NEXT: %[[G1:.+]] = arith.muli %[[D1]], %[[C256B]] : index
// CHECK-NEXT: %[[POOLSZ:.+]] = arith.addi %[[G0]], %[[G1]] : index
// CHECK-NEXT: %[[POOL:.+]] = hipsr.get_pool(%{{.+}}, %[[POOLSZ]]) : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT: %[[OFF0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[OFF1:.+]] = arith.addi %[[OFF0]], %[[G0]] : index
// CHECK-NEXT: %[[V0:.+]] = memref.view %[[POOL]][%[[OFF0]]][%[[DIM]]] : memref<?xi8, #hipsr.mem<device>> to memref<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[V1:.+]] = memref.view %[[POOL]][%[[OFF1]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} : memref<?x512xf16, #hipsr.mem<device>>, memref<?x512xf16, #hipsr.mem<device>>) outs(%[[V0]] : memref<?x512xf16, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%[[V1]] : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NOT: memref.alloc
func.func @split_two_groups_dynamic(%ctx: !hipsr.context,
                         %inf16: memref<?x512xf16, #hipsr.mem<device>>,
                         %sin: memref<4x1024xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %inf16, %sin :
      !hipsr.context,
      memref<?x512xf16, #hipsr.mem<device>>,
      memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context,
       %din: memref<?x512xf16, #hipsr.mem<device>>,
       %dsin: memref<4x1024xf16, #hipsr.mem<device>>):
    %c0 = arith.constant 0 : index
    %d = memref.dim %din, %c0 : memref<?x512xf16, #hipsr.mem<device>>
    %a_dyn = memref.alloc(%d) : memref<?x512xf16, #hipsr.mem<device>>
    %a_static = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<?x512xf16, #hipsr.mem<device>>, memref<?x512xf16, #hipsr.mem<device>>) outs(%a_dyn : memref<?x512xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%dsin, %dsin : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a_static : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a_dyn, %a_dyn : memref<?x512xf16, #hipsr.mem<device>>, memref<?x512xf16, #hipsr.mem<device>>) outs(%din : memref<?x512xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a_static, %a_static : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%dsin : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  }
  return
}

// -----

// Three disjoint allocs of mixed element/shape (static f16, static f32, dynamic
// f16) coalesce into one group folded by two arith.maxui (the last over a dynamic
// memref.dim size) and sharing off0, while a fourth alloc that overlaps all three
// takes its own group at off1.
// CHECK-LABEL: func.func @split_with_coalesced_group
// CHECK: %[[DIM:.+]] = memref.dim %{{.+}}, %{{.+}} : memref<?x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[C16384:.+]] = arith.constant 16384 : index
// CHECK-NEXT: %[[C4096:.+]] = arith.constant 4096 : index
// CHECK-NEXT: %[[C2048:.+]] = arith.constant 2048 : index
// CHECK-NEXT: %[[S2:.+]] = arith.muli %[[C2048]], %[[DIM]] : index
// CHECK-NEXT: %[[M0:.+]] = arith.maxui %[[C16384]], %[[C4096]] : index
// CHECK-NEXT: %[[M1:.+]] = arith.maxui %[[M0]], %[[S2]] : index
// CHECK-NEXT: %[[C256A:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255A:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[N0:.+]] = arith.addi %[[M1]], %[[C255A]] : index
// CHECK-NEXT: %[[D0:.+]] = arith.divui %[[N0]], %[[C256A]] : index
// CHECK-NEXT: %[[G0:.+]] = arith.muli %[[D0]], %[[C256A]] : index
// CHECK-NEXT: %[[C8192:.+]] = arith.constant 8192 : index
// CHECK-NEXT: %[[C256B:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255B:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[N1:.+]] = arith.addi %[[C8192]], %[[C255B]] : index
// CHECK-NEXT: %[[D1:.+]] = arith.divui %[[N1]], %[[C256B]] : index
// CHECK-NEXT: %[[G1:.+]] = arith.muli %[[D1]], %[[C256B]] : index
// CHECK-NEXT: %[[POOLSZ:.+]] = arith.addi %[[G0]], %[[G1]] : index
// CHECK-NEXT: %[[POOL:.+]] = hipsr.get_pool(%{{.+}}, %[[POOLSZ]]) : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT: %[[OFF0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[OFF1:.+]] = arith.addi %[[OFF0]], %[[G0]] : index
// CHECK-NEXT: %[[V0:.+]] = memref.view %[[POOL]][%[[OFF0]]][] : memref<?xi8, #hipsr.mem<device>> to memref<8x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[V1:.+]] = memref.view %[[POOL]][%[[OFF0]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x256xf32, #hipsr.mem<device>>
// CHECK-NEXT: %[[V2:.+]] = memref.view %[[POOL]][%[[OFF0]]][%[[DIM]]] : memref<?xi8, #hipsr.mem<device>> to memref<?x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[V3:.+]] = memref.view %[[POOL]][%[[OFF1]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} : memref<8x1024xf16, #hipsr.mem<device>>, memref<8x1024xf16, #hipsr.mem<device>>) outs(%[[V0]] : memref<8x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%[[V3]] : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[V0]], %{{.+}} : memref<8x1024xf16, #hipsr.mem<device>>, memref<8x1024xf16, #hipsr.mem<device>>) outs(%{{.+}} : memref<8x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} : memref<4x256xf32, #hipsr.mem<device>>, memref<4x256xf32, #hipsr.mem<device>>) outs(%[[V1]] : memref<4x256xf32, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[V1]], %{{.+}} : memref<4x256xf32, #hipsr.mem<device>>, memref<4x256xf32, #hipsr.mem<device>>) outs(%{{.+}} : memref<4x256xf32, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} : memref<?x1024xf16, #hipsr.mem<device>>, memref<?x1024xf16, #hipsr.mem<device>>) outs(%[[V2]] : memref<?x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[V2]], %{{.+}} : memref<?x1024xf16, #hipsr.mem<device>>, memref<?x1024xf16, #hipsr.mem<device>>) outs(%{{.+}} : memref<?x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[V3]], %{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NOT: memref.alloc
func.func @split_with_coalesced_group(%ctx: !hipsr.context,
                               %in16: memref<8x1024xf16, #hipsr.mem<device>>,
                               %inf32: memref<4x256xf32, #hipsr.mem<device>>,
                               %in2: memref<?x1024xf16, #hipsr.mem<device>>,
                               %in4: memref<4x1024xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %in16, %inf32, %in2, %in4 :
      !hipsr.context,
      memref<8x1024xf16, #hipsr.mem<device>>,
      memref<4x256xf32, #hipsr.mem<device>>,
      memref<?x1024xf16, #hipsr.mem<device>>,
      memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context,
       %d16: memref<8x1024xf16, #hipsr.mem<device>>,
       %df32: memref<4x256xf32, #hipsr.mem<device>>,
       %d2: memref<?x1024xf16, #hipsr.mem<device>>,
       %d4: memref<4x1024xf16, #hipsr.mem<device>>):
    %c0 = arith.constant 0 : index
    %d = memref.dim %d2, %c0 : memref<?x1024xf16, #hipsr.mem<device>>
    %a1 = memref.alloc() : memref<8x1024xf16, #hipsr.mem<device>>
    %a2 = memref.alloc() : memref<4x256xf32, #hipsr.mem<device>>
    %a3 = memref.alloc(%d) : memref<?x1024xf16, #hipsr.mem<device>>
    %a4 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%d16, %d16 : memref<8x1024xf16, #hipsr.mem<device>>, memref<8x1024xf16, #hipsr.mem<device>>) outs(%a1 : memref<8x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%d4, %d4 : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a4 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a1, %a1 : memref<8x1024xf16, #hipsr.mem<device>>, memref<8x1024xf16, #hipsr.mem<device>>) outs(%d16 : memref<8x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%df32, %df32 : memref<4x256xf32, #hipsr.mem<device>>, memref<4x256xf32, #hipsr.mem<device>>) outs(%a2 : memref<4x256xf32, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a2, %a2 : memref<4x256xf32, #hipsr.mem<device>>, memref<4x256xf32, #hipsr.mem<device>>) outs(%df32 : memref<4x256xf32, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%d2, %d2 : memref<?x1024xf16, #hipsr.mem<device>>, memref<?x1024xf16, #hipsr.mem<device>>) outs(%a3 : memref<?x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a3, %a3 : memref<?x1024xf16, #hipsr.mem<device>>, memref<?x1024xf16, #hipsr.mem<device>>) outs(%d2 : memref<?x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a4, %a4 : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%d4 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  }
  return
}

// -----

// a1, a2, a3 overlap pairwise, forcing three distinct groups: poolSize chains two
// arith.addi (g0+g1+g2) and offsets chain off0 / off1 / off2.
// CHECK-LABEL: func.func @split_three_groups
// CHECK: %[[C8192A:.+]] = arith.constant 8192 : index
// CHECK-NEXT: %[[C256A:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255A:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[N0:.+]] = arith.addi %[[C8192A]], %[[C255A]] : index
// CHECK-NEXT: %[[D0:.+]] = arith.divui %[[N0]], %[[C256A]] : index
// CHECK-NEXT: %[[G0:.+]] = arith.muli %[[D0]], %[[C256A]] : index
// CHECK-NEXT: %[[C8192B:.+]] = arith.constant 8192 : index
// CHECK-NEXT: %[[C256B:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255B:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[N1:.+]] = arith.addi %[[C8192B]], %[[C255B]] : index
// CHECK-NEXT: %[[D1:.+]] = arith.divui %[[N1]], %[[C256B]] : index
// CHECK-NEXT: %[[G1:.+]] = arith.muli %[[D1]], %[[C256B]] : index
// CHECK-NEXT: %[[C8192C:.+]] = arith.constant 8192 : index
// CHECK-NEXT: %[[C256C:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255C:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[N2:.+]] = arith.addi %[[C8192C]], %[[C255C]] : index
// CHECK-NEXT: %[[D2:.+]] = arith.divui %[[N2]], %[[C256C]] : index
// CHECK-NEXT: %[[G2:.+]] = arith.muli %[[D2]], %[[C256C]] : index
// CHECK-NEXT: %[[SUM0:.+]] = arith.addi %[[G0]], %[[G1]] : index
// CHECK-NEXT: %[[POOLSZ:.+]] = arith.addi %[[SUM0]], %[[G2]] : index
// CHECK-NEXT: %[[POOL:.+]] = hipsr.get_pool(%{{.+}}, %[[POOLSZ]]) : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT: %[[OFF0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[OFF1:.+]] = arith.addi %[[OFF0]], %[[G0]] : index
// CHECK-NEXT: %[[OFF2:.+]] = arith.addi %[[OFF1]], %[[G1]] : index
// CHECK-NEXT: %[[V0:.+]] = memref.view %[[POOL]][%[[OFF0]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[V1:.+]] = memref.view %[[POOL]][%[[OFF1]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[V2:.+]] = memref.view %[[POOL]][%[[OFF2]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%[[V0]] : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%[[V1]] : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%[[V2]] : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[V0]], %[[V1]] : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[V1]], %[[V2]] : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[V0]], %[[V2]] : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NOT: memref.alloc
func.func @split_three_groups(%ctx: !hipsr.context, %in: memref<4x1024xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %in : !hipsr.context, memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<4x1024xf16, #hipsr.mem<device>>):
    %a1 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    %a2 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    %a3 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a1 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%din, %din : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a2 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%din, %din : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a3 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a1, %a2 : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%din : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a2, %a3 : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%din : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a1, %a3 : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%din : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  }
  return
}
