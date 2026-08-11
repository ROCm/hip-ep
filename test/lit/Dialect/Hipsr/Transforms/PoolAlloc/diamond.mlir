// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -hipsr-pool-alloc | FileCheck %s

// CHECK-LABEL: func.func @diamond
// CHECK: %[[DIM:.+]] = memref.dim
// CHECK-NEXT: %[[C2048:.+]] = arith.constant 2048 : index
// CHECK-NEXT: %[[SA:.+]] = arith.muli %[[C2048]], %[[DIM]] : index
// CHECK-NEXT: %[[C4096:.+]] = arith.constant 4096 : index
// CHECK-NEXT: %[[SD:.+]] = arith.muli %[[C4096]], %[[DIM]] : index
// CHECK-NEXT: %[[MAX:.+]] = arith.maxui %[[SA]], %[[SD]] : index
// CHECK-NEXT: %[[C256A:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255A:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[N0:.+]] = arith.addi %[[MAX]], %[[C255A]] : index
// CHECK-NEXT: %[[Q0:.+]] = arith.divui %[[N0]], %[[C256A]] : index
// CHECK-NEXT: %[[G0:.+]] = arith.muli %[[Q0]], %[[C256A]] : index
// CHECK-NEXT: %[[C1024:.+]] = arith.constant 1024 : index
// CHECK-NEXT: %[[SB:.+]] = arith.muli %[[C1024]], %[[DIM]] : index
// CHECK-NEXT: %[[C256B:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255B:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[N1:.+]] = arith.addi %[[SB]], %[[C255B]] : index
// CHECK-NEXT: %[[Q1:.+]] = arith.divui %[[N1]], %[[C256B]] : index
// CHECK-NEXT: %[[G1:.+]] = arith.muli %[[Q1]], %[[C256B]] : index
// CHECK-NEXT: %[[C512:.+]] = arith.constant 512 : index
// CHECK-NEXT: %[[SC:.+]] = arith.muli %[[C512]], %[[DIM]] : index
// CHECK-NEXT: %[[C256C:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255C:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[N2:.+]] = arith.addi %[[SC]], %[[C255C]] : index
// CHECK-NEXT: %[[Q2:.+]] = arith.divui %[[N2]], %[[C256C]] : index
// CHECK-NEXT: %[[G2:.+]] = arith.muli %[[Q2]], %[[C256C]] : index
// CHECK-NEXT: %[[OFF0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[OFF2:.+]] = arith.addi %[[G0]], %[[G1]] : index
// CHECK-NEXT: %[[POOLSZ:.+]] = arith.addi %[[OFF2]], %[[G2]] : index
// CHECK-NEXT: %[[POOL:.+]] = hipsr.get_pool(%{{.+}}, %[[POOLSZ]]) {domain_id = 0 : i64} : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT: %[[VA:.+]] = memref.view %[[POOL]][%[[OFF0]]][%[[DIM]]] : memref<?xi8, #hipsr.mem<device>> to memref<?x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[VD:.+]] = memref.view %[[POOL]][%[[OFF0]]][%[[DIM]]] : memref<?xi8, #hipsr.mem<device>> to memref<?x2048xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[VB:.+]] = memref.view %[[POOL]][%[[G0]]][%[[DIM]]] : memref<?xi8, #hipsr.mem<device>> to memref<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[VC:.+]] = memref.view %[[POOL]][%[[OFF2]]][%[[DIM]]] : memref<?xi8, #hipsr.mem<device>> to memref<?x256xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} :{{.*}}) outs(%[[VA]] :{{.*}})
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[VA]], %[[VA]] :{{.*}}) outs(%[[VB]] :{{.*}})
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[VA]], %[[VA]] :{{.*}}) outs(%[[VC]] :{{.*}})
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[VB]], %[[VC]] :{{.*}}) outs(%[[VD]] :{{.*}})
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[VD]], %[[VD]] :{{.*}}) outs(%{{.+}} :{{.*}})
// CHECK-NOT: memref.alloc
func.func @diamond(%ctx: !hipsr.context, %in: memref<?x1024xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %in : !hipsr.context, memref<?x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<?x1024xf16, #hipsr.mem<device>>):
    %c0 = arith.constant 0 : index
    %d = memref.dim %din, %c0 : memref<?x1024xf16, #hipsr.mem<device>>
    %a = memref.alloc(%d) : memref<?x1024xf16, #hipsr.mem<device>>
    %b = memref.alloc(%d) : memref<?x512xf16, #hipsr.mem<device>>
    %c = memref.alloc(%d) : memref<?x256xf16, #hipsr.mem<device>>
    %e = memref.alloc(%d) : memref<?x2048xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<?x1024xf16, #hipsr.mem<device>>, memref<?x1024xf16, #hipsr.mem<device>>) outs(%a : memref<?x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a, %a : memref<?x1024xf16, #hipsr.mem<device>>, memref<?x1024xf16, #hipsr.mem<device>>) outs(%b : memref<?x512xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a, %a : memref<?x1024xf16, #hipsr.mem<device>>, memref<?x1024xf16, #hipsr.mem<device>>) outs(%c : memref<?x256xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%b, %c : memref<?x512xf16, #hipsr.mem<device>>, memref<?x256xf16, #hipsr.mem<device>>) outs(%e : memref<?x2048xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%e, %e : memref<?x2048xf16, #hipsr.mem<device>>, memref<?x2048xf16, #hipsr.mem<device>>) outs(%din : memref<?x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  } {domain_id = 0 : i64}
  return
}
