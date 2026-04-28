// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// FileCheck tests for --hip-insert-tensor-dump pass.
//
// Verifies that hip.dump_tensor ops are inserted after HIP compute ops.
// After bufferization, DPS ops have no SSA results — the output is the
// DPS init (outs) operand, which is what gets dumped.
//
// Non-compute ops (alloc, free, get_constant, get_pool, yield) should
// not get dump instrumentation.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-insert-tensor-dump="dump-tensors-dir=/tmp/dump" %s 2>&1 | FileCheck %s

// CHECK-LABEL: func.func @test_dump_inserted
// CHECK-SAME:    (%[[CTX:.*]]: !hip.context,
// CHECK:         hip.matmul
// CHECK-NEXT:    hip.dump_tensor(%[[CTX]]) %[[OUT1:.*]] {dump_tensors_dir = "/tmp/dump", name = "hip_matmul_0"} : memref<2x64x64xf32>
// CHECK:         hip.miopen.softmax
// CHECK-NEXT:    hip.dump_tensor(%[[CTX]]) %[[OUT2:.*]] {dump_tensors_dir = "/tmp/dump", name = "hip_miopen_softmax_1"} : memref<2x64x64xf32>
// CHECK:         return
func.func @test_dump_inserted(
    %ctx: !hip.context,
    %a: memref<2x64x64xf32, strided<[?, ?, ?], offset: ?>>,
    %b: memref<64x64xf32, strided<[?, ?], offset: ?>>) -> memref<2x64x64xf32> {
  %out1 = memref.alloc() {alignment = 64 : i64} : memref<2x64x64xf32>
  hip.matmul(%ctx) ins(%a, %b : memref<2x64x64xf32, strided<[?, ?, ?], offset: ?>>, memref<64x64xf32, strided<[?, ?], offset: ?>>) outs(%out1 : memref<2x64x64xf32>)
  %out2 = memref.alloc() {alignment = 64 : i64} : memref<2x64x64xf32>
  hip.miopen.softmax(%ctx) ins(%out1 : memref<2x64x64xf32>) outs(%out2 : memref<2x64x64xf32>)
  return %out2 : memref<2x64x64xf32>
}

// Verify hip.alloc and hip.free do NOT get dump instrumentation.
// CHECK-LABEL: func.func @test_no_dump_for_infra_ops
// CHECK:         hip.alloc
// CHECK-NOT:     hip.dump_tensor
// CHECK:         hip.free
// CHECK-NOT:     hip.dump_tensor
// CHECK:         return
func.func @test_no_dump_for_infra_ops(
    %ctx: !hip.context,
    %buf: memref<64xf32>) {
  %alloc = hip.alloc(%ctx) : memref<64xf32>
  hip.free(%ctx, %alloc) : memref<64xf32>
  return
}
