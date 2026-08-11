// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -split-input-file -hipsr-pool-alloc | FileCheck %s

// 3xf16 = 6 B is not a multiple of 256, so the alignUp chain must round up.
// CHECK-LABEL: func.func @align_up_rounding
// CHECK: %[[C6:.+]] = arith.constant 6 : index
// CHECK-NEXT: %[[C256:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[NUM:.+]] = arith.addi %[[C6]], %[[C255]] : index
// CHECK-NEXT: %[[DIV:.+]] = arith.divui %[[NUM]], %[[C256]] : index
// CHECK-NEXT: %[[G0:.+]] = arith.muli %[[DIV]], %[[C256]] : index
// CHECK-NEXT: %[[OFF:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[POOL:.+]] = hipsr.get_pool(%{{.+}}, %[[G0]]) {domain_id = 0 : i64} : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT: %[[V0:.+]] = memref.view %[[POOL]][%[[OFF]]][] : memref<?xi8, #hipsr.mem<device>> to memref<3xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} :{{.*}}) outs(%[[V0]] :{{.*}})
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
  } {domain_id = 0 : i64}
  return
}

// -----

// CHECK-LABEL: func.func @dead_alloc_skipped
// CHECK: memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[C8192:.+]] = arith.constant 8192 : index
// CHECK-NEXT: %[[C256:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[NUM:.+]] = arith.addi %[[C8192]], %[[C255]] : index
// CHECK-NEXT: %[[DIV:.+]] = arith.divui %[[NUM]], %[[C256]] : index
// CHECK-NEXT: %[[G0:.+]] = arith.muli %[[DIV]], %[[C256]] : index
// CHECK-NEXT: %[[OFF:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[POOL:.+]] = hipsr.get_pool(%{{.+}}, %[[G0]]) {domain_id = 0 : i64} : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT: %[[V0:.+]] = memref.view %[[POOL]][%[[OFF]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} :{{.*}}) outs(%[[V0]] :{{.*}})
// CHECK-NOT: arith.maxui
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

// CHECK-LABEL: func.func @mixed_dtypes
// CHECK: %[[DIM:.+]] = memref.dim %{{.+}}, %{{.+}} : memref<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[C4096:.+]] = arith.constant 4096 : index
// CHECK-NEXT: %[[C1024:.+]] = arith.constant 1024 : index
// CHECK-NEXT: %[[BYTES:.+]] = arith.muli %[[C1024]], %[[DIM]] : index
// CHECK-NEXT: %[[MAX:.+]] = arith.maxui %[[C4096]], %[[BYTES]] : index
// CHECK-NEXT: %[[C256:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[NUM:.+]] = arith.addi %[[MAX]], %[[C255]] : index
// CHECK-NEXT: %[[DIV:.+]] = arith.divui %[[NUM]], %[[C256]] : index
// CHECK-NEXT: %[[G0:.+]] = arith.muli %[[DIV]], %[[C256]] : index
// CHECK-NEXT: %[[OFF:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[POOL:.+]] = hipsr.get_pool(%{{.+}}, %[[G0]]) {domain_id = 0 : i64} : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT: %[[V0:.+]] = memref.view %[[POOL]][%[[OFF]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x256xf32, #hipsr.mem<device>>
// CHECK-NEXT: %[[V1:.+]] = memref.view %[[POOL]][%[[OFF]]][%[[DIM]]] : memref<?xi8, #hipsr.mem<device>> to memref<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} :{{.*}}) outs(%[[V0]] :{{.*}})
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} :{{.*}}) outs(%[[V1]] :{{.*}})
// CHECK-NOT: memref.alloc
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

// CHECK-LABEL: func.func @mixed_dynamic_dims
// CHECK: %[[DIM0:.+]] = memref.dim %{{.+}}, %{{.+}} : memref<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[DIM1:.+]] = memref.dim %{{.+}}, %{{.+}} : memref<?x256xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[C1024:.+]] = arith.constant 1024 : index
// CHECK-NEXT: %[[S0:.+]] = arith.muli %[[C1024]], %[[DIM0]] : index
// CHECK-NEXT: %[[C512:.+]] = arith.constant 512 : index
// CHECK-NEXT: %[[S1:.+]] = arith.muli %[[C512]], %[[DIM1]] : index
// CHECK-NEXT: %[[MAX:.+]] = arith.maxui %[[S0]], %[[S1]] : index
// CHECK-NEXT: %[[C256:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[NUM:.+]] = arith.addi %[[MAX]], %[[C255]] : index
// CHECK-NEXT: %[[DIV:.+]] = arith.divui %[[NUM]], %[[C256]] : index
// CHECK-NEXT: %[[G0:.+]] = arith.muli %[[DIV]], %[[C256]] : index
// CHECK-NEXT: %[[OFF:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[POOL:.+]] = hipsr.get_pool(%{{.+}}, %[[G0]]) {domain_id = 0 : i64} : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT: %[[V0:.+]] = memref.view %[[POOL]][%[[OFF]]][%[[DIM0]]] : memref<?xi8, #hipsr.mem<device>> to memref<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[V1:.+]] = memref.view %[[POOL]][%[[OFF]]][%[[DIM1]]] : memref<?xi8, #hipsr.mem<device>> to memref<?x256xf16, #hipsr.mem<device>>
// CHECK-NOT: arith.maxui
// CHECK-NOT: memref.alloc
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

// CHECK-LABEL: func.func @two_domains
// CHECK: %[[G0:.+]] = arith.muli %{{.+}}, %{{.+}} : index
// CHECK-NEXT: %[[OFF0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[POOL0:.+]] = hipsr.get_pool(%{{.+}}, %[[G0]]) {domain_id = 0 : i64} : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT: %[[V0:.+]] = memref.view %[[POOL0]][%[[OFF0]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} :{{.*}}) outs(%[[V0]] :{{.*}})
// CHECK: %[[G1:.+]] = arith.muli %{{.+}}, %{{.+}} : index
// CHECK-NEXT: %[[OFF1:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[POOL1:.+]] = hipsr.get_pool(%{{.+}}, %[[G1]]) {domain_id = 7 : i64} : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT: %[[V1:.+]] = memref.view %[[POOL1]][%[[OFF1]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} :{{.*}}) outs(%[[V1]] :{{.*}})
// CHECK-NOT: memref.alloc
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
