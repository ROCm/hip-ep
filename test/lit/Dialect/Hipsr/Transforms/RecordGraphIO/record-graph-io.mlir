// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -split-input-file -hipsr-record-graph-io -verify-diagnostics | FileCheck %s

// Dynamic extents use -1. Element sizes round up to a whole byte.
// CHECK-LABEL: module attributes {
// CHECK-SAME: hipdnn.input_count = 1 : i64
// CHECK-SAME: hipdnn.input_element_sizes = array<i64: 2>
// CHECK-SAME: hipdnn.input_shapes = [array<i64: -1, 4>]
// CHECK-SAME: hipdnn.output_count = 1 : i64
// CHECK-SAME: hipdnn.output_element_sizes = array<i64: 1>
// CHECK-SAME: hipdnn.output_shapes = [array<i64: -1, 8>]
// CHECK-NEXT:  func.func private @main_graph(!hipsr.context, memref<?x4xf16, #hipsr.mem<device>>) -> memref<?x8xi1, #hipsr.mem<device>>
// CHECK-NEXT:}
module {
  func.func private @main_graph(
      !hipsr.context, memref<?x4xf16, #hipsr.mem<device>>)
      -> memref<?x8xi1, #hipsr.mem<device>>
}

// -----

module {
  // expected-error@+1 {{expected ranked input type in @main_graph, got 'i32'}}
  func.func private @main_graph(!hipsr.context, i32)
}

// -----

module {
  // expected-error@+1 {{unsupported element type in @main_graph input: 'complex<f32>'}}
  func.func private @main_graph(!hipsr.context, memref<4xcomplex<f32>>)
}
