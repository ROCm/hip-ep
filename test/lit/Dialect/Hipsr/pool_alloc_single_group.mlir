// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -split-input-file -verify-diagnostics -hipsr-pool-alloc | FileCheck %s

// Allocs sit adjacent at the top: only first-write/last-use lifetimes keep them
// disjoint, so this guards against a regression to alloc-index liveness (which
// would mark all three overlapping and refuse to pool).
// CHECK-LABEL: func.func @pool_diff_sizes
// CHECK: %[[C16384:.+]] = arith.constant 16384 : index
// CHECK-NEXT: %[[C8192:.+]] = arith.constant 8192 : index
// CHECK-NEXT: %[[M0:.+]] = arith.maxui %[[C16384]], %[[C8192]] : index
// CHECK-NEXT: %[[C4096:.+]] = arith.constant 4096 : index
// CHECK-NEXT: %[[M1:.+]] = arith.maxui %[[M0]], %[[C4096]] : index
// CHECK-NEXT: %[[C256:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[NUM:.+]] = arith.addi %[[M1]], %[[C255]] : index
// CHECK-NEXT: %[[DIV:.+]] = arith.divui %[[NUM]], %[[C256]] : index
// CHECK-NEXT: %[[SZ:.+]] = arith.muli %[[DIV]], %[[C256]] : index
// CHECK-NEXT: %[[POOL:.+]] = hipsr.get_pool(%{{.+}}, %[[SZ]]) : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT: %[[OFF:.+]] = arith.constant 0 : index
// CHECK-NEXT: memref.view %[[POOL]][%[[OFF]]][] : memref<?xi8, #hipsr.mem<device>> to memref<8x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: memref.view %[[POOL]][%[[OFF]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: memref.view %[[POOL]][%[[OFF]]][] : memref<?xi8, #hipsr.mem<device>> to memref<2x1024xf16, #hipsr.mem<device>>
// CHECK-NOT: memref.alloc
// CHECK-NOT: shape.
func.func @pool_diff_sizes(
    %ctx: !hipsr.context,
    %in8: memref<8x1024xf16, #hipsr.mem<device>>,
    %in4: memref<4x1024xf16, #hipsr.mem<device>>,
    %in2: memref<2x1024xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %in8, %in4, %in2 :
      !hipsr.context,
      memref<8x1024xf16, #hipsr.mem<device>>,
      memref<4x1024xf16, #hipsr.mem<device>>,
      memref<2x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context,
       %d8: memref<8x1024xf16, #hipsr.mem<device>>,
       %d4: memref<4x1024xf16, #hipsr.mem<device>>,
       %d2: memref<2x1024xf16, #hipsr.mem<device>>):
    %a1 = memref.alloc() : memref<8x1024xf16, #hipsr.mem<device>>
    %a2 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    %a3 = memref.alloc() : memref<2x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%d8, %d8 : memref<8x1024xf16, #hipsr.mem<device>>, memref<8x1024xf16, #hipsr.mem<device>>)
               outs(%a1 : memref<8x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%d4, %d4 : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>)
               outs(%a2 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%d2, %d2 : memref<2x1024xf16, #hipsr.mem<device>>, memref<2x1024xf16, #hipsr.mem<device>>)
               outs(%a3 : memref<2x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  }
  return
}

// -----

// 3xf16 = 6 B is not a multiple of 256, so the alignUp chain must round up.
// CHECK-LABEL: func.func @single_align
// CHECK: %[[C6:.+]] = arith.constant 6 : index
// CHECK-NEXT: %[[C256:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[NUM:.+]] = arith.addi %[[C6]], %[[C255]] : index
// CHECK-NEXT: %[[DIV:.+]] = arith.divui %[[NUM]], %[[C256]] : index
// CHECK-NEXT: %[[SZ:.+]] = arith.muli %[[DIV]], %[[C256]] : index
// CHECK-NEXT: %[[POOL:.+]] = hipsr.get_pool(%{{.+}}, %[[SZ]]) : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT: %[[OFF:.+]] = arith.constant 0 : index
// CHECK-NEXT: memref.view %[[POOL]][%[[OFF]]][] : memref<?xi8, #hipsr.mem<device>> to memref<3xf16, #hipsr.mem<device>>
// CHECK-NOT: arith.maxui
// CHECK-NOT: memref.alloc
func.func @single_align(%ctx: !hipsr.context,
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
// CHECK-LABEL: func.func @dynamic
// CHECK: %[[DIM:.+]] = memref.dim
// CHECK-NEXT: %[[C1024:.+]] = arith.constant 1024 : index
// CHECK-NEXT: %[[BYTES:.+]] = arith.muli %[[C1024]], %[[DIM]] : index
// CHECK-NEXT: %[[C256:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[NUM:.+]] = arith.addi %[[BYTES]], %[[C255]] : index
// CHECK-NEXT: %[[DIV:.+]] = arith.divui %[[NUM]], %[[C256]] : index
// CHECK-NEXT: %[[SZ:.+]] = arith.muli %[[DIV]], %[[C256]] : index
// CHECK-NEXT: %[[POOL:.+]] = hipsr.get_pool(%{{.+}}, %[[SZ]]) : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT: %[[OFF:.+]] = arith.constant 0 : index
// CHECK-NEXT: memref.view %[[POOL]][%[[OFF]]][%[[DIM]]] : memref<?xi8, #hipsr.mem<device>> to memref<?x512xf16, #hipsr.mem<device>>
// CHECK-NOT: memref.alloc
func.func @dynamic(%ctx: !hipsr.context,
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
// CHECK-LABEL: func.func @mixed
// CHECK: %[[DIM:.+]] = memref.dim
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
// CHECK-NEXT: memref.view %[[POOL]][%[[OFF]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x256xf32, #hipsr.mem<device>>
// CHECK-NEXT: memref.view %[[POOL]][%[[OFF]]][%[[DIM]]] : memref<?xi8, #hipsr.mem<device>> to memref<?x512xf16, #hipsr.mem<device>>
// CHECK-NOT: memref.alloc
func.func @mixed(%ctx: !hipsr.context,
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

// Overlapping lifetimes yield two groups; 9a bails without touching the IR.
func.func @multigroup(%ctx: !hipsr.context,
                      %in: memref<4x1024xf16, #hipsr.mem<device>>) {
  // expected-error @+1 {{multi-group pooling not yet supported}}
  hipsr.pool_domain(%ctx, %in : !hipsr.context, memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<4x1024xf16, #hipsr.mem<device>>):
    %a1 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    %a2 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>)
               outs(%a1 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a1, %din : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>)
               outs(%a2 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  }
  return
}
