// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -split-input-file -hipsr-pool-alloc | FileCheck %s

// CHECK-LABEL: func.func @line
// CHECK: %[[DIM:.+]] = memref.dim
// CHECK-NEXT: %[[C2048A:.+]] = arith.constant 2048 : index
// CHECK-NEXT: %[[S0:.+]] = arith.muli %[[C2048A]], %[[DIM]] : index
// CHECK-NEXT: %[[C2048B:.+]] = arith.constant 2048 : index
// CHECK-NEXT: %[[S2:.+]] = arith.muli %[[C2048B]], %[[DIM]] : index
// CHECK-NEXT: %[[C2048C:.+]] = arith.constant 2048 : index
// CHECK-NEXT: %[[S4:.+]] = arith.muli %[[C2048C]], %[[DIM]] : index
// CHECK-NEXT: %[[M0:.+]] = arith.maxui %[[S0]], %[[S2]] : index
// CHECK-NEXT: %[[M1:.+]] = arith.maxui %[[M0]], %[[S4]] : index
// CHECK-NEXT: %[[C256A:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255A:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[N0:.+]] = arith.addi %[[M1]], %[[C255A]] : index
// CHECK-NEXT: %[[D0:.+]] = arith.divui %[[N0]], %[[C256A]] : index
// CHECK-NEXT: %[[G0:.+]] = arith.muli %[[D0]], %[[C256A]] : index
// CHECK-NEXT: %[[C2048D:.+]] = arith.constant 2048 : index
// CHECK-NEXT: %[[S1:.+]] = arith.muli %[[C2048D]], %[[DIM]] : index
// CHECK-NEXT: %[[C2048E:.+]] = arith.constant 2048 : index
// CHECK-NEXT: %[[S3:.+]] = arith.muli %[[C2048E]], %[[DIM]] : index
// CHECK-NEXT: %[[M2:.+]] = arith.maxui %[[S1]], %[[S3]] : index
// CHECK-NEXT: %[[C256B:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255B:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[N1:.+]] = arith.addi %[[M2]], %[[C255B]] : index
// CHECK-NEXT: %[[D1:.+]] = arith.divui %[[N1]], %[[C256B]] : index
// CHECK-NEXT: %[[G1:.+]] = arith.muli %[[D1]], %[[C256B]] : index
// CHECK-NEXT: %[[OFF0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[POOLSZ:.+]] = arith.addi %[[G0]], %[[G1]] : index
// CHECK-NEXT: %[[POOL:.+]] = hipsr.get_pool(%{{.+}}, %[[POOLSZ]]) {domain_id = 0 : i64} : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT: %[[V0:.+]] = memref.view %[[POOL]][%[[OFF0]]][%[[DIM]]] : memref<?xi8, #hipsr.mem<device>> to memref<?x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[V2:.+]] = memref.view %[[POOL]][%[[OFF0]]][%[[DIM]]] : memref<?xi8, #hipsr.mem<device>> to memref<?x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[V4:.+]] = memref.view %[[POOL]][%[[OFF0]]][%[[DIM]]] : memref<?xi8, #hipsr.mem<device>> to memref<?x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[V1:.+]] = memref.view %[[POOL]][%[[G0]]][%[[DIM]]] : memref<?xi8, #hipsr.mem<device>> to memref<?x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[V3:.+]] = memref.view %[[POOL]][%[[G0]]][%[[DIM]]] : memref<?xi8, #hipsr.mem<device>> to memref<?x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} :{{.*}}) outs(%[[V0]] :{{.*}})
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[V0]], %[[V0]] :{{.*}}) outs(%[[V1]] :{{.*}})
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[V1]], %[[V1]] :{{.*}}) outs(%[[V2]] :{{.*}})
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[V2]], %[[V2]] :{{.*}}) outs(%[[V3]] :{{.*}})
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[V3]], %[[V3]] :{{.*}}) outs(%[[V4]] :{{.*}})
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[V4]], %[[V4]] :{{.*}}) outs(%{{.+}} :{{.*}})
// CHECK-NOT: memref.alloc
func.func @line(%ctx: !hipsr.context, %in: memref<?x1024xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %in : !hipsr.context, memref<?x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<?x1024xf16, #hipsr.mem<device>>):
    %c0 = arith.constant 0 : index
    %d = memref.dim %din, %c0 : memref<?x1024xf16, #hipsr.mem<device>>
    %a0 = memref.alloc(%d) : memref<?x1024xf16, #hipsr.mem<device>>
    %a1 = memref.alloc(%d) : memref<?x1024xf16, #hipsr.mem<device>>
    %a2 = memref.alloc(%d) : memref<?x1024xf16, #hipsr.mem<device>>
    %a3 = memref.alloc(%d) : memref<?x1024xf16, #hipsr.mem<device>>
    %a4 = memref.alloc(%d) : memref<?x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<?x1024xf16, #hipsr.mem<device>>, memref<?x1024xf16, #hipsr.mem<device>>) outs(%a0 : memref<?x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a0, %a0 : memref<?x1024xf16, #hipsr.mem<device>>, memref<?x1024xf16, #hipsr.mem<device>>) outs(%a1 : memref<?x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a1, %a1 : memref<?x1024xf16, #hipsr.mem<device>>, memref<?x1024xf16, #hipsr.mem<device>>) outs(%a2 : memref<?x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a2, %a2 : memref<?x1024xf16, #hipsr.mem<device>>, memref<?x1024xf16, #hipsr.mem<device>>) outs(%a3 : memref<?x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a3, %a3 : memref<?x1024xf16, #hipsr.mem<device>>, memref<?x1024xf16, #hipsr.mem<device>>) outs(%a4 : memref<?x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a4, %a4 : memref<?x1024xf16, #hipsr.mem<device>>, memref<?x1024xf16, #hipsr.mem<device>>) outs(%din : memref<?x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  } {domain_id = 0 : i64}
  return
}

// -----

// CHECK-LABEL: func.func @static_line
// CHECK: %[[C16384A:.+]] = arith.constant 16384 : index
// CHECK-NEXT: %[[C4096:.+]] = arith.constant 4096 : index
// CHECK-NEXT: %[[C16384B:.+]] = arith.constant 16384 : index
// CHECK-NEXT: %[[M0:.+]] = arith.maxui %[[C16384A]], %[[C4096]] : index
// CHECK-NEXT: %[[M1:.+]] = arith.maxui %[[M0]], %[[C16384B]] : index
// CHECK-NEXT: %[[C256A:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255A:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[N0:.+]] = arith.addi %[[M1]], %[[C255A]] : index
// CHECK-NEXT: %[[D0:.+]] = arith.divui %[[N0]], %[[C256A]] : index
// CHECK-NEXT: %[[G0:.+]] = arith.muli %[[D0]], %[[C256A]] : index
// CHECK-NEXT: %[[C8192A:.+]] = arith.constant 8192 : index
// CHECK-NEXT: %[[C8192B:.+]] = arith.constant 8192 : index
// CHECK-NEXT: %[[M2:.+]] = arith.maxui %[[C8192A]], %[[C8192B]] : index
// CHECK-NEXT: %[[C256B:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255B:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[N1:.+]] = arith.addi %[[M2]], %[[C255B]] : index
// CHECK-NEXT: %[[D1:.+]] = arith.divui %[[N1]], %[[C256B]] : index
// CHECK-NEXT: %[[G1:.+]] = arith.muli %[[D1]], %[[C256B]] : index
// CHECK-NEXT: %[[OFF0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[POOLSZ:.+]] = arith.addi %[[G0]], %[[G1]] : index
// CHECK-NEXT: %[[POOL:.+]] = hipsr.get_pool(%{{.+}}, %[[POOLSZ]]) {domain_id = 0 : i64} : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT: %[[V0:.+]] = memref.view %[[POOL]][%[[OFF0]]][] : memref<?xi8, #hipsr.mem<device>> to memref<8x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[V2:.+]] = memref.view %[[POOL]][%[[OFF0]]][] : memref<?xi8, #hipsr.mem<device>> to memref<2x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[V4:.+]] = memref.view %[[POOL]][%[[OFF0]]][] : memref<?xi8, #hipsr.mem<device>> to memref<8x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[V1:.+]] = memref.view %[[POOL]][%[[G0]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[V3:.+]] = memref.view %[[POOL]][%[[G0]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} :{{.*}}) outs(%[[V0]] :{{.*}})
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[V0]], %[[V0]] :{{.*}}) outs(%[[V1]] :{{.*}})
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[V1]], %[[V1]] :{{.*}}) outs(%[[V2]] :{{.*}})
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[V2]], %[[V2]] :{{.*}}) outs(%[[V3]] :{{.*}})
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[V3]], %[[V3]] :{{.*}}) outs(%[[V4]] :{{.*}})
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[V4]], %[[V4]] :{{.*}}) outs(%{{.+}} :{{.*}})
// CHECK-NOT: memref.alloc
func.func @static_line(%ctx: !hipsr.context, %in: memref<4x1024xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %in : !hipsr.context, memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<4x1024xf16, #hipsr.mem<device>>):
    %a0 = memref.alloc() : memref<8x1024xf16, #hipsr.mem<device>>
    %a1 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    %a2 = memref.alloc() : memref<2x1024xf16, #hipsr.mem<device>>
    %a3 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    %a4 = memref.alloc() : memref<8x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a0 : memref<8x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a0, %a0 : memref<8x1024xf16, #hipsr.mem<device>>, memref<8x1024xf16, #hipsr.mem<device>>) outs(%a1 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a1, %a1 : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a2 : memref<2x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a2, %a2 : memref<2x1024xf16, #hipsr.mem<device>>, memref<2x1024xf16, #hipsr.mem<device>>) outs(%a3 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a3, %a3 : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a4 : memref<8x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a4, %a4 : memref<8x1024xf16, #hipsr.mem<device>>, memref<8x1024xf16, #hipsr.mem<device>>) outs(%din : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  } {domain_id = 0 : i64}
  return
}
