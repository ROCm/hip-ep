// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -split-input-file -hipsr-pool-alloc | FileCheck %s

// CHECK-LABEL: func.func @subview_extends_lifetime
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
// CHECK-NEXT: %[[OFF0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[POOLSZ:.+]] = arith.addi %[[G0]], %[[G1]] : index
// CHECK-NEXT: %[[POOL:.+]] = hipsr.get_pool(%{{.+}}, %[[POOLSZ]]) {domain_id = 0 : i64} : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT: %[[V0:.+]] = memref.view %[[POOL]][%[[OFF0]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[V1:.+]] = memref.view %[[POOL]][%[[G0]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[SUB:.+]] = memref.subview %[[V0]]
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} :{{.*}}) outs(%[[V0]] :{{.*}})
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} :{{.*}}) outs(%[[V1]] :{{.*}})
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[SUB]], %[[SUB]] :{{.*}}) outs(%{{.+}} :{{.*}})
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[V1]], %[[V1]] :{{.*}}) outs(%{{.+}} :{{.*}})
// CHECK-NOT: memref.alloc
func.func @subview_extends_lifetime(%ctx: !hipsr.context,
                                    %in: memref<4x1024xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %in : !hipsr.context, memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<4x1024xf16, #hipsr.mem<device>>):
    %a1 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    %a2 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    %v = memref.subview %a1[0, 0] [2, 1024] [1, 1] :
        memref<4x1024xf16, #hipsr.mem<device>> to
        memref<2x1024xf16, strided<[1024, 1]>, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a1 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%din, %din : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a2 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%v, %v : memref<2x1024xf16, strided<[1024, 1]>, #hipsr.mem<device>>, memref<2x1024xf16, strided<[1024, 1]>, #hipsr.mem<device>>) outs(%din : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a2, %a2 : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%din : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  } {domain_id = 0 : i64}
  return
}

// -----

// CHECK-LABEL: func.func @cast_extends_lifetime
// CHECK: %[[G0:.+]] = arith.muli %{{.+}}, %{{.+}} : index
// CHECK: %[[G1:.+]] = arith.muli %{{.+}}, %{{.+}} : index
// CHECK-NEXT: %[[OFF0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[POOLSZ:.+]] = arith.addi %[[G0]], %[[G1]] : index
// CHECK-NEXT: %[[POOL:.+]] = hipsr.get_pool(%{{.+}}, %[[POOLSZ]]) {domain_id = 0 : i64} : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT: %[[V0:.+]] = memref.view %[[POOL]][%[[OFF0]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[V1:.+]] = memref.view %[[POOL]][%[[G0]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[CAST:.+]] = memref.cast %[[V0]]
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} :{{.*}}) outs(%[[V0]] :{{.*}})
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} :{{.*}}) outs(%[[V1]] :{{.*}})
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[CAST]], %[[CAST]] :{{.*}}) outs(%{{.+}} :{{.*}})
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[V1]], %[[V1]] :{{.*}}) outs(%{{.+}} :{{.*}})
// CHECK-NOT: memref.alloc
func.func @cast_extends_lifetime(%ctx: !hipsr.context,
                                 %in: memref<4x1024xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %in : !hipsr.context, memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<4x1024xf16, #hipsr.mem<device>>):
    %a1 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    %a2 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    %c = memref.cast %a1 : memref<4x1024xf16, #hipsr.mem<device>> to memref<?x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a1 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%din, %din : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a2 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%c, %c : memref<?x1024xf16, #hipsr.mem<device>>, memref<?x1024xf16, #hipsr.mem<device>>) outs(%din : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a2, %a2 : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%din : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  } {domain_id = 0 : i64}
  return
}

// -----

// CHECK-LABEL: func.func @collapse_shape_extends_lifetime
// CHECK: %[[G0:.+]] = arith.muli %{{.+}}, %{{.+}} : index
// CHECK: %[[G1:.+]] = arith.muli %{{.+}}, %{{.+}} : index
// CHECK-NEXT: %[[OFF0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[POOLSZ:.+]] = arith.addi %[[G0]], %[[G1]] : index
// CHECK-NEXT: %[[POOL:.+]] = hipsr.get_pool(%{{.+}}, %[[POOLSZ]]) {domain_id = 0 : i64} : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT: %[[V0:.+]] = memref.view %[[POOL]][%[[OFF0]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[V1:.+]] = memref.view %[[POOL]][%[[G0]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[FLAT:.+]] = memref.collapse_shape %[[V0]]
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} :{{.*}}) outs(%[[V0]] :{{.*}})
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} :{{.*}}) outs(%[[V1]] :{{.*}})
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[FLAT]], %[[FLAT]] :{{.*}}) outs(%{{.+}} :{{.*}})
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[V1]], %[[V1]] :{{.*}}) outs(%{{.+}} :{{.*}})
// CHECK-NOT: memref.alloc
func.func @collapse_shape_extends_lifetime(%ctx: !hipsr.context,
                                           %in: memref<4x1024xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %in : !hipsr.context, memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<4x1024xf16, #hipsr.mem<device>>):
    %a1 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    %a2 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    %f = memref.collapse_shape %a1 [[0, 1]] :
        memref<4x1024xf16, #hipsr.mem<device>> into memref<4096xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a1 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%din, %din : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a2 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%f, %f : memref<4096xf16, #hipsr.mem<device>>, memref<4096xf16, #hipsr.mem<device>>) outs(%din : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a2, %a2 : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%din : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  } {domain_id = 0 : i64}
  return
}
