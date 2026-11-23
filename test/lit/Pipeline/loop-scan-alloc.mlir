// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// A scan result used after the loop. The concatenation buffer is allocated
// inside scf.while, so pooling does not take it. Allocation lowering frees
// iteration allocations in the loop, and frees the carried accumulator after
// its last use. That use is the copy into the caller's hip.alloc_output, so
// the accumulator is freed after the copy. The output allocation is returned
// and is not freed.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --onnx-to-hip-pipeline %s | FileCheck %s

module {
  func.func @main_graph(%A: tensor<4xf32>) -> tensor<?x4xf32> {
    %M = "onnx.Constant"() {value = dense<2> : tensor<i64>} : () -> tensor<i64>
    %cond_init = "onnx.Constant"() {value = dense<1> : tensor<i1>} : () -> tensor<i1>
    %scan = "onnx.Loop"(%M, %cond_init) ({
    ^bb0(%iter: tensor<i64>, %cond_in: tensor<i1>):
      %cond_out = "onnx.Identity"(%cond_in) : (tensor<i1>) -> tensor<i1>
      %chunk = "onnx.Identity"(%A) : (tensor<4xf32>) -> tensor<4xf32>
      "onnx.Yield"(%cond_out, %chunk) : (tensor<i1>, tensor<4xf32>) -> ()
    }) : (tensor<i64>, tensor<i1>) -> tensor<?x4xf32>
    %used = "onnx.Identity"(%scan) : (tensor<?x4xf32>) -> tensor<?x4xf32>
    return %used : tensor<?x4xf32>
  }
}

// CHECK-LABEL: func.func @main_graph
// CHECK:         scf.while
// CHECK-NOT:     hip.alloc_output
// CHECK:         hip.alloc
// CHECK:         hip.free
// CHECK:         scf.condition
// CHECK:         hip.free
// CHECK:         scf.yield
// CHECK:         hip.alloc_output
// CHECK:         memref.copy
// CHECK:         hip.free
// CHECK-NOT:     hip.free
// CHECK:         return
