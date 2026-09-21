// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -split-input-file -hipsr-main-graph-abi -verify-diagnostics | FileCheck %s

// The wrapper loads both descriptors and expands their fields in LLVM's memref
// argument order. The second descriptor has a different rank.
// CHECK-LABEL: llvm.func private @main_graph(
// CHECK-SAME:      %[[STATE:.*]]: !llvm.ptr,
// CHECK-SAME:      %[[INPUTS:.*]]: !llvm.ptr) -> i32 attributes {passthrough = ["noinline"]} {
// CHECK-NEXT:    %[[INDEX0:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[SLOT0:.*]] = llvm.getelementptr %[[INPUTS]][%[[INDEX0]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %[[PTR0:.*]] = llvm.load %[[SLOT0]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %[[DESC0:.*]] = llvm.load %[[PTR0]] : !llvm.ptr -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[ALLOC0:.*]] = llvm.extractvalue %[[DESC0]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[ALIGNED0:.*]] = llvm.extractvalue %[[DESC0]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[OFFSET0:.*]] = llvm.extractvalue %[[DESC0]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[SIZE00:.*]] = llvm.extractvalue %[[DESC0]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[SIZE01:.*]] = llvm.extractvalue %[[DESC0]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[STRIDE00:.*]] = llvm.extractvalue %[[DESC0]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[STRIDE01:.*]] = llvm.extractvalue %[[DESC0]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[INDEX1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[SLOT1:.*]] = llvm.getelementptr %[[INPUTS]][%[[INDEX1]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %[[PTR1:.*]] = llvm.load %[[SLOT1]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %[[DESC1:.*]] = llvm.load %[[PTR1]] : !llvm.ptr -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[ALLOC1:.*]] = llvm.extractvalue %[[DESC1]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[ALIGNED1:.*]] = llvm.extractvalue %[[DESC1]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[OFFSET1:.*]] = llvm.extractvalue %[[DESC1]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[SIZE10:.*]] = llvm.extractvalue %[[DESC1]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[STRIDE10:.*]] = llvm.extractvalue %[[DESC1]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %{{.*}} = llvm.call @main_graph_internal(%[[STATE]], %[[ALLOC0]], %[[ALIGNED0]], %[[OFFSET0]], %[[SIZE00]], %[[SIZE01]], %[[STRIDE00]], %[[STRIDE01]], %[[ALLOC1]], %[[ALIGNED1]], %[[OFFSET1]], %[[SIZE10]], %[[STRIDE10]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64) -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[SUCCESS:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return %[[SUCCESS]] : i32
// CHECK-NEXT:  }
// CHECK-LABEL: llvm.func private @main_graph_internal(
// CHECK-SAME:      %{{[^,]*}}: !llvm.ptr, %{{[^,]*}}: !llvm.ptr<1>, %{{[^,]*}}: !llvm.ptr<1>, %{{[^,]*}}: i64, %{{[^,]*}}: i64, %{{[^,]*}}: i64, %{{[^,]*}}: i64, %{{[^,]*}}: i64, %{{[^,]*}}: !llvm.ptr<1>, %{{[^,]*}}: !llvm.ptr<1>, %{{[^,]*}}: i64, %{{[^,]*}}: i64, %{{[^,]*}}: i64) -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> {
// CHECK-NEXT:    %[[RESULT:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    llvm.return %[[RESULT]] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:  }
module attributes {
  hipdnn.input_count = 2 : i64,
  hipdnn.input_shapes = [array<i64: 2, 3>, array<i64: 4>],
  hipdnn.output_count = 1 : i64,
  hipdnn.output_shapes = [array<i64: 2, 3>]
} {
  llvm.func @main_graph(
      %state: !llvm.ptr,
      %a_alloc: !llvm.ptr<1>, %a_aligned: !llvm.ptr<1>, %a_off: i64,
      %a_d0: i64, %a_d1: i64, %a_s0: i64, %a_s1: i64,
      %b_alloc: !llvm.ptr<1>, %b_aligned: !llvm.ptr<1>, %b_off: i64,
      %b_d0: i64, %b_s0: i64)
      -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> {
    %desc = llvm.mlir.poison
        : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.return %desc
        : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
  }
}

// -----

module attributes {hipdnn.input_shapes = [array<i64: 2, 3>]} {
  // expected-error@+1 {{@main_graph takes 6 parameters, but hipdnn.input_shapes describes 8}}
  llvm.func @main_graph(%state: !llvm.ptr, %alloc: !llvm.ptr<1>,
                        %aligned: !llvm.ptr<1>, %offset: i64, %size: i64,
                        %stride: i64) {
    llvm.return
  }
}

// -----

module {
  // expected-error@+1 {{hipdnn.input_shapes is required to wrap @main_graph}}
  llvm.func @main_graph(%state: !llvm.ptr) {
    llvm.return
  }
}
