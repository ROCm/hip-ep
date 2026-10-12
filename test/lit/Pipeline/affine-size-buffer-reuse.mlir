// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// RUN: hip-mlir-opt %s --onnx-to-hip-pipeline --verify-each \
// RUN:   | FileCheck %s --implicit-check-not=affine.apply \
// RUN:       --implicit-check-not=arith.divsi --implicit-check-not=arith.select \
// RUN:       --implicit-check-not=memref.alloc
//
// Lowering the affine product lets canonicalization recover the input extent.
// Buffer reuse must see that equality before it assigns the two disjoint
// temporaries. The external output keeps both computations observable.

// CHECK-LABEL: func.func @reuse_normalized_extent(
// CHECK-SAME: %[[CTX:.*]]: !hip.context, %[[INPUT:.*]]: memref<?xf32>, %[[OUT:.*]]: memref<?xf32>
// CHECK: %[[POOL:.*]] = hip.get_pool
// CHECK: %[[BUFFER:.*]] = memref.view %[[POOL]]
// CHECK: hip.miopen.softmax(%[[CTX]]) ins(%[[INPUT]] : memref<?xf32>) outs(%[[BUFFER]] : memref<?xf32>)
// CHECK: hip.miopen.softmax(%[[CTX]]) ins(%[[BUFFER]] : memref<?xf32>) outs(%[[OUT]] : memref<?xf32>)
// CHECK-NOT: hip.get_pool
// CHECK-NOT: memref.view
// CHECK: hip.miopen.softmax(%[[CTX]]) ins(%[[INPUT]] : memref<?xf32>) outs(%[[BUFFER]] : memref<?xf32>)
// CHECK: hip.miopen.softmax(%[[CTX]]) ins(%[[BUFFER]] : memref<?xf32>) outs(%[[OUT]] : memref<?xf32>)
// CHECK: return

// The entry point supplies tensor metadata; the helper isolates the
// post-bufferization allocation pattern exercised by the production pipeline.
func.func @main_graph(%input: tensor<1xf32>) {
  return
}

func.func @reuse_normalized_extent(
    %ctx: !hip.context, %input: memref<?xf32>, %out: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %n = memref.dim %input, %c0 : memref<?xf32>
  %flat = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%n]
  %q = arith.divsi %flat, %c16 : index
  %is_one = arith.cmpi eq, %q, %c1 : index
  %extent = arith.select %is_one, %n, %q : index

  %a = memref.alloc(%n) : memref<?xf32>
  hip.miopen.softmax(%ctx) ins(%input : memref<?xf32>)
                           outs(%a : memref<?xf32>)
  hip.miopen.softmax(%ctx) ins(%a : memref<?xf32>)
                           outs(%out : memref<?xf32>)
  %b = memref.alloc(%extent) : memref<?xf32>
  hip.miopen.softmax(%ctx) ins(%input : memref<?xf32>)
                           outs(%b : memref<?xf32>)
  hip.miopen.softmax(%ctx) ins(%b : memref<?xf32>)
                           outs(%out : memref<?xf32>)
  return
}
