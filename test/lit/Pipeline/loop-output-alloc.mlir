// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// A loop result copied out after the loop, then pooled and lowered.
//
// hip-pool-allocs only sees allocations in the function entry block, so the
// buffer allocated inside scf.while is left for hip-lower-allocs. That pass
// frees it in the block that holds it. The graph output is one
// hip.alloc_output after the loop.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-use-output-allocator --hip-pool-allocs --hip-lower-allocs %s | FileCheck %s

// CHECK-LABEL: func.func @loop_carried_output
// CHECK:         scf.while
// CHECK-NOT:     hip.alloc_output
// CHECK:         hip.alloc(%{{.*}}, %{{.*}}) : memref<?x4xf32>
// CHECK:         memref.copy
// CHECK:         hip.free
// CHECK:         scf.condition
// CHECK:       } do {
// CHECK:       }
// CHECK:         hip.alloc_output(%{{.*}}, %{{.*}}) {out_idx = 0 : i64} : memref<?x4xf32>
// CHECK:         memref.copy
// CHECK:         hip.free
// CHECK:         return
func.func @loop_carried_output(%ctx: !hip.context, %init: memref<?x4xf32>,
                               %n: index) -> memref<?x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r:2 = scf.while (%i = %c0, %acc = %init)
      : (index, memref<?x4xf32>) -> (index, memref<?x4xf32>) {
    %cond = arith.cmpi slt, %i, %n : index
    %d = memref.dim %acc, %c0 : memref<?x4xf32>
    %copy = memref.alloc(%d) : memref<?x4xf32>
    memref.copy %acc, %copy : memref<?x4xf32> to memref<?x4xf32>
    scf.condition(%cond) %i, %copy : index, memref<?x4xf32>
  } do {
  ^bb0(%j: index, %body: memref<?x4xf32>):
    %next = arith.addi %j, %c1 : index
    scf.yield %next, %body : index, memref<?x4xf32>
  }
  return %r#1 : memref<?x4xf32>
}
