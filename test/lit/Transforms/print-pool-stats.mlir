// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// print-pool-stats.mlir
//
// Verify `--hip-print-pool-stats` emits the expected stderr block on
// a module that carries the slot-buffer-coalesce metadata. The pass
// is analysis-only and prints to stderr; we redirect stderr to stdout
// with 2>&1 and FileCheck against both streams.

// RUN: hip-mlir-opt %s --hip-print-pool-stats 2>&1 | FileCheck %s

// CHECK: [pool-stats] func=main_graph
// CHECK-NEXT: static_pool_bytes = 512
// CHECK-NEXT: dyn_dim_slots_count = 2
// CHECK-NEXT: next_dyn_slot_id = 2
// CHECK-NEXT: output_dim_specs.count = 4
// CHECK-NEXT: identity_propagator_predicate_hits = 1
// CHECK-NEXT: slot_publisher_count = 2

module attributes {
  hipdnn.pool_size = 512 : i64,
  hipdnn.dyn_dim_slots_count = 2 : i32,
  hipdnn.next_dyn_slot_id = 2 : i32,
  hipdnn.output_dim_specs = [
    [[array<i64: 0, 2, 0, 0, 0, -1, 0, 0>], [array<i64: 3, 0, 0, 0, 0, 0, 0, 0>]],
    [[array<i64: 0, 2, 0, 0, 0, -1, 0, 0>], [array<i64: 3, 0, 0, 0, 0, 0, 1, 0>]]
  ]
} {
  func.func @main_graph(%ctx: !hip.context, %x: tensor<3x4xi1>) -> tensor<3x4xf32> {
    %ub = arith.constant 12 : index
    %nz_init = tensor.empty(%ub) : tensor<2x?xi64>
    %nz = hip.nonzero(%ctx) ins(%x : tensor<3x4xi1>) outs(%nz_init : tensor<2x?xi64>) {
      input_data_type = 5 : i64,
      slot_id = 0 : i32
    } : tensor<2x?xi64>
    %nz2_init = tensor.empty(%ub) : tensor<2x?xi64>
    %nz2 = hip.nonzero(%ctx) ins(%x : tensor<3x4xi1>) outs(%nz2_init : tensor<2x?xi64>) {
      input_data_type = 5 : i64,
      slot_id = 1 : i32
    } : tensor<2x?xi64>
    // An identity transpose (perm = [0, 1]) so identity hits = 1.
    %y = arith.constant dense<0.0> : tensor<3x4xf32>
    %t_init = tensor.empty() : tensor<3x4xf32>
    %t = hip.transpose(%ctx) ins(%y : tensor<3x4xf32>) outs(%t_init : tensor<3x4xf32>) {
      perm = [0, 1]
    } : tensor<3x4xf32>
    return %t : tensor<3x4xf32>
  }
}
