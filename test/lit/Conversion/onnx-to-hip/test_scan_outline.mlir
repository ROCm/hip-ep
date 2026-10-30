// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --onnx-scan-outline %s | FileCheck %s

module {
  func.func @main_graph(%state: tensor<4xf32>,
                        %sequence: tensor<?x4xf32>,
                        %bias: tensor<4xf32>)
      -> (tensor<4xf32>, tensor<?x4xf32>) {
    %state_final, %scan_output =
        "onnx.Scan"(%state, %sequence) ({
        ^bb0(%state_in: tensor<4xf32>, %slice: tensor<4xf32>):
          %sum = "onnx.Add"(%state_in, %slice)
              : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
          %biased = "onnx.Add"(%sum, %bias)
              : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
          "onnx.Yield"(%sum, %biased)
              : (tensor<4xf32>, tensor<4xf32>) -> ()
        }) {num_scan_inputs = 1 : si64}
        : (tensor<4xf32>, tensor<?x4xf32>)
          -> (tensor<4xf32>, tensor<?x4xf32>)
    return %state_final, %scan_output
        : tensor<4xf32>, tensor<?x4xf32>
  }

  func.func @two_scan_inputs(%lhs: tensor<?x4xf32>,
                             %rhs: tensor<?x4xf32>)
      -> tensor<?x4xf32> {
    %scan_output = "onnx.Scan"(%lhs, %rhs) ({
      ^bb0(%lhs_slice: tensor<4xf32>, %rhs_slice: tensor<4xf32>):
        %sum = "onnx.Add"(%lhs_slice, %rhs_slice)
            : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        "onnx.Yield"(%sum) : (tensor<4xf32>) -> ()
      }) {num_scan_inputs = 2 : si64}
      : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
    return %scan_output : tensor<?x4xf32>
  }

  func.func @statically_empty(%sequence: tensor<0x4xf32>)
      -> tensor<0x4xf32> {
    %scan_output = "onnx.Scan"(%sequence) ({
      ^bb0(%slice: tensor<4xf32>):
        "onnx.Yield"(%slice) : (tensor<4xf32>) -> ()
      }) {num_scan_inputs = 1 : si64}
      : (tensor<0x4xf32>) -> tensor<0x4xf32>
    return %scan_output : tensor<0x4xf32>
  }

  // Iteration zero executes once in the non-empty branch. hip.scan consumes
  // only a non-empty tail. The empty branch returns initial states and empty
  // body outputs without touching the body.
  // CHECK-LABEL: func.func @main_graph(
  // CHECK: %[[NONEMPTY:.*]] = arith.cmpi ugt
  // CHECK: %[[IF:.*]]:2 = scf.if %[[NONEMPTY]]
  // CHECK: tensor.extract_slice
  // CHECK: "onnx.Add"
  // CHECK: } else {
  // CHECK: scf.yield
  // CHECK: %[[HAS_TAIL:.*]] = arith.cmpi ugt
  // CHECK: scf.if %[[HAS_TAIL]]
  // CHECK: %[[SCAN:[^ ]+]]:2 = hip.scan(%{{[^)]+}})
  // CHECK-SAME: scan_inputs(%{{.*}} : tensor<?x4xf32>)
  // CHECK-SAME: captures(%{{.*}} : tensor<4xf32>)
  // CHECK-SAME: outs(%{{.*}}, %{{.*}} : tensor<4xf32>, tensor<?x4xf32>)
  // CHECK-SAME: body @main_graph_scan_body_n0
  // CHECK-SAME: num_scan_inputs = 1 : i32
  // CHECK-SAME: num_scan_outputs = 1 : i32
  // CHECK-SAME: num_state_variables = 1 : i32
  // CHECK: tensor.insert_slice
  // CHECK: return

  // Every scan input contributes the iteration count. Dynamic mismatches are
  // rejected before any body execution.
  // CHECK-LABEL: func.func @two_scan_inputs(
  // CHECK: %[[LHS_LEN:.*]] = tensor.dim
  // CHECK: %[[RHS_LEN:.*]] = tensor.dim
  // CHECK: %[[SAME_LEN:.*]] = arith.cmpi eq, %[[LHS_LEN]], %[[RHS_LEN]]
  // CHECK: cf.assert %[[SAME_LEN]], "Scan input sequence lengths must match"

  // CHECK-LABEL: func.func @statically_empty(
  // CHECK-NOT: hip.scan
  // CHECK: bufferization.alloc_tensor()
  // CHECK: return

  // CHECK-LABEL: func.func private @main_graph_scan_body_n0
  // CHECK-SAME: (%{{[^:]+}}: !hip.context,
  // CHECK-SAME: %{{[^:]+}}: tensor<4xf32>,
  // CHECK-SAME: %{{[^:]+}}: tensor<4xf32>,
  // CHECK-SAME: %{{[^:]+}}: tensor<4xf32>)
  // CHECK-SAME: -> (tensor<4xf32>, tensor<4xf32>)
  // CHECK-SAME: attributes {hip.scan_num_state_variables = 1 : i32}
  // CHECK: "onnx.Add"
  // CHECK: return
}
