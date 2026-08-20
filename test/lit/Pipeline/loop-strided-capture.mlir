// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --onnx-to-hip-pipeline \
// RUN:   --mlir-print-ir-after=hip-promote-strided-operands %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=PROMOTE
// RUN: hip-mlir-opt --hipdnn-pipeline %s \
// RUN:   | FileCheck %s --check-prefix=LLVM

// Bufferization turns the outer tensor.extract_slice capture into a strided
// memref.subview while selecting an identity-layout function-boundary type for
// the outlined body argument. The promotion pass repairs that transient ABI
// mismatch with one copy live exactly across hip.loop. The identity-layout
// carrier seed remains the loop operand, preserving its zero-trip may-alias
// lifetime.
//
// PROMOTE-LABEL: func.func @main_graph
// PROMOTE-SAME:    %[[PARENT:[^ ,]+]]: memref<4x8xf32>
// PROMOTE-SAME:    %[[SEED:[^ ,]+]]: memref<4x4xf32>
// PROMOTE:         %[[CAPTURE:.*]] = memref.subview %[[PARENT]]
// PROMOTE:         %[[TMP:.*]] = memref.alloc() : memref<4x4xf32>
// PROMOTE-NEXT:    memref.copy %[[CAPTURE]], %[[TMP]]
// PROMOTE-NEXT:    %[[LOOP:.*]]:2 = hip.loop
// PROMOTE-SAME:      iter_args(%[[SEED]] : memref<4x4xf32>)
// PROMOTE-SAME:      captures(%[[TMP]] : memref<4x4xf32>)
// PROMOTE-NEXT:    memref.dealloc %[[TMP]]
// PROMOTE-NOT:     memref.dealloc %[[SEED]]
// PROMOTE-LABEL: func.func private @main_graph_loop_body_n0
// PROMOTE-SAME:    , %{{[^,]+}}: memref<4x4xf32>,
// PROMOTE-SAME:    %{{[^,]+}}: !hip.loop_frame)
// LLVM-LABEL: llvm.func private @main_graph
// LLVM: llvm.call @hipdnn_ep_run_counted_loop
module {
  func.func @main_graph(%parent: tensor<4x8xf32>, %seed: tensor<4x4xf32>)
      -> tensor<4x4xf32> {
    %capture = tensor.extract_slice %parent[0, 2] [4, 4] [1, 1]
        : tensor<4x8xf32> to tensor<4x4xf32>
    %m = "onnx.Constant"() {value = dense<0> : tensor<i64>}
        : () -> tensor<i64>
    %cond = "onnx.Constant"() {value = dense<true> : tensor<i1>}
        : () -> tensor<i1>
    %result = "onnx.Loop"(%m, %cond, %seed) ({
    ^bb0(%iter: tensor<i64>, %cond_in: tensor<i1>,
         %current: tensor<4x4xf32>):
      %next = "onnx.Add"(%current, %capture)
          : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
      "onnx.Yield"(%cond_in, %next)
          : (tensor<i1>, tensor<4x4xf32>) -> ()
    }) : (tensor<i64>, tensor<i1>, tensor<4x4xf32>)
         -> tensor<4x4xf32>
    return %result : tensor<4x4xf32>
  }
}
