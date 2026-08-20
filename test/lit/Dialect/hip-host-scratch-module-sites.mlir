// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// RUN: hip-mlir-opt --hip-materialize-host-scalars %s | FileCheck %s
// RUN: hip-mlir-opt --hip-materialize-host-scalars --convert-hip-to-llvm %s \
// RUN:   | FileCheck %s --check-prefix=LLVM
//
// Both functions use local scratch offset zero. Their module-site identities
// must differ so an outlined helper cannot overwrite its caller's live scalar.

// CHECK-LABEL: func.func @caller
// CHECK: %[[CALLER:.*]] = hip.get_host_scratch({{.*}}) : memref<?xi8>
// CHECK: memref.view %[[CALLER]][%{{.*}}][] : memref<?xi8> to memref<i64>
// LLVM-LABEL: llvm.func @caller
// LLVM: %[[SITE0:.*]] = llvm.mlir.constant(0 : i32) : i32
// LLVM: llvm.call @hipdnn_ep_get_host_scratch_base({{.*}}, %[[SITE0]], {{.*}}) : (!llvm.ptr, i32, i64) -> !llvm.ptr
func.func @caller(%ctx: !hip.context, %x: i64) -> i64 {
  %tmp = memref.alloc() : memref<i64>
  memref.store %x, %tmp[] : memref<i64>
  %value = memref.load %tmp[] : memref<i64>
  return %value : i64
}

// CHECK-LABEL: func.func @outlined_helper
// CHECK: %[[HELPER:.*]] = hip.get_host_scratch({{.*}}) {site_id = 1 : i64} : memref<?xi8>
// CHECK: memref.view %[[HELPER]][%{{.*}}][] : memref<?xi8> to memref<i64>
// LLVM-LABEL: llvm.func @outlined_helper
// LLVM: %[[SITE1:.*]] = llvm.mlir.constant(1 : i32) : i32
// LLVM: llvm.call @hipdnn_ep_get_host_scratch_base({{.*}}, %[[SITE1]], {{.*}}) : (!llvm.ptr, i32, i64) -> !llvm.ptr
func.func @outlined_helper(%ctx: !hip.context, %x: i64) -> i64 {
  %tmp = memref.alloc() : memref<i64>
  memref.store %x, %tmp[] : memref<i64>
  %value = memref.load %tmp[] : memref<i64>
  return %value : i64
}
