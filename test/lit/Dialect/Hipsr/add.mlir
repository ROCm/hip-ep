// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// UNSUPPORTED: true

// RUN: hip-mlir-opt --split-input-file %s | FileCheck %s
// RUN: hip-mlir-opt --split-input-file -hipsr-populate-shape-region %s | FileCheck %s --check-prefix=POPULATE

// CHECK-LABEL: func.func @add_tensor
// CHECK:      hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} : tensor<?x1024xf16>, tensor<1024xf16>)
// CHECK-SAME:   outs(%{{.+}} : tensor<?x1024xf16>) : tensor<?x1024xf16>
// CHECK-NEXT: return
func.func @add_tensor(%ctx: !hipsr.context, %lhs: tensor<?x1024xf16>,
                      %rhs: tensor<1024xf16>,
                      %init: tensor<?x1024xf16>) -> tensor<?x1024xf16> {
  %0 = hipsr.add(%ctx) ins(%lhs, %rhs : tensor<?x1024xf16>, tensor<1024xf16>)
                  outs(%init : tensor<?x1024xf16>) : tensor<?x1024xf16>
  return %0 : tensor<?x1024xf16>
}

// -----

// CHECK-LABEL: func.func @add_memref
// CHECK: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>, memref<1024xf16, #hipsr.mem<device>>)
// CHECK-SAME: outs(%{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>)
func.func @add_memref(%ctx: !hipsr.context,
                      %lhs: memref<4x1024xf16, #hipsr.mem<device>>,
                      %rhs: memref<1024xf16, #hipsr.mem<device>>,
                      %init: memref<4x1024xf16, #hipsr.mem<device>>) {
  hipsr.add(%ctx) ins(%lhs, %rhs : memref<4x1024xf16, #hipsr.mem<device>>, memref<1024xf16, #hipsr.mem<device>>)
             outs(%init : memref<4x1024xf16, #hipsr.mem<device>>)
  return
}

// -----

// POPULATE-LABEL: func.func @add_broadcast
// POPULATE: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} : tensor<?x1024xf16>, tensor<1024xf16>)
// POPULATE-SAME: outs(%{{.+}} : tensor<?x1024xf16>) : tensor<?x1024xf16> shape_region {
// POPULATE:   ^bb0(%[[LHS:.+]]: tensor<?x1024xf16>, %[[RHS:.+]]: tensor<1024xf16>):
// POPULATE:     %[[SHL:.+]] = shape.shape_of %[[LHS]] : tensor<?x1024xf16> -> tensor<2xindex>
// POPULATE:     %[[SHR:.+]] = shape.shape_of %[[RHS]] : tensor<1024xf16> -> tensor<1xindex>
// POPULATE:     %[[BC:.+]] = shape.broadcast %[[SHL]], %[[SHR]] : tensor<2xindex>, tensor<1xindex> -> tensor<?xindex>
// POPULATE:     %[[C0:.+]] = arith.constant 0 : index
// POPULATE:     %[[D0:.+]] = shape.get_extent %[[BC]], %[[C0]] : tensor<?xindex>, index -> index
// POPULATE:     %[[C1:.+]] = arith.constant 1 : index
// POPULATE:     %[[D1:.+]] = shape.get_extent %[[BC]], %[[C1]] : tensor<?xindex>, index -> index
// POPULATE:     hipsr.shape_yield (%[[D0]], %[[D1]]) : [f16]
// POPULATE:   }
func.func @add_broadcast(%ctx: !hipsr.context, %lhs: tensor<?x1024xf16>,
                         %rhs: tensor<1024xf16>,
                         %init: tensor<?x1024xf16>) -> tensor<?x1024xf16> {
  %0 = hipsr.add(%ctx) ins(%lhs, %rhs : tensor<?x1024xf16>, tensor<1024xf16>)
                  outs(%init : tensor<?x1024xf16>) : tensor<?x1024xf16>
  return %0 : tensor<?x1024xf16>
}
