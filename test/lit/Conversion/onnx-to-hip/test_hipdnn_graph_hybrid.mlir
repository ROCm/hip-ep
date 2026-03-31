// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify that hip.hipdnn_graph ops (produced by OutlineOnnxToHipDNN +
// CompileHipDNNGraphs) pass through ConvertOnnxToHip unchanged, while
// remaining ONNX ops are lowered to their HIP dialect equivalents (hybrid
// execution model).
//
// This test validates:
// - hip.hipdnn_graph survives ConvertOnnxToHip (not an onnx.* op)
// - onnx.MatMul → hip.matmul with DPS (tensor.empty init)
// - onnx.Sigmoid → hip.sigmoid with DPS (tensor.empty init)
// - graph_id, input_uids, output_uids attributes are preserved
// - Proper !hip.context threading through all operations
//
// Model: Conv(hipDNN) + MatMul + Sigmoid hybrid pipeline
// ============================================================================

// RUN: hip-mlir-opt %s --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%ctx: !hip.context,
                        %x: tensor<1x1x8x8xf32>,
                        %w_conv: tensor<1x1x3x3xf32>,
                        %w_fc: tensor<8x8xf32>) -> tensor<1x1x8x8xf32> {
    %empty = tensor.empty() : tensor<1x1x8x8xf32>

    // Conv was already compiled by CompileHipDNNGraphs → hip.hipdnn_graph
    %conv_out = hip.hipdnn_graph(%ctx) graph_id(0)
        ins(%x, %w_conv : tensor<1x1x8x8xf32>, tensor<1x1x3x3xf32>)
        outs(%empty : tensor<1x1x8x8xf32>)
        {input_uids = [0, 1], output_uids = [2]} : tensor<1x1x8x8xf32>

    // Remaining ONNX ops lowered by ConvertOnnxToHip
    %matmul_out = "onnx.MatMul"(%conv_out, %w_fc) : (tensor<1x1x8x8xf32>, tensor<8x8xf32>) -> tensor<1x1x8x8xf32>
    %sigmoid_out = "onnx.Sigmoid"(%matmul_out) : (tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xf32>

    return %sigmoid_out : tensor<1x1x8x8xf32>
  }
}

// CHECK-LABEL: func.func @main_graph
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context,
// CHECK-SAME:   %[[X:.*]]: tensor<1x1x8x8xf32>,
// CHECK-SAME:   %[[W_CONV:.*]]: tensor<1x1x3x3xf32>,
// CHECK-SAME:   %[[W_FC:.*]]: tensor<8x8xf32>)
// CHECK-SAME:  -> tensor<1x1x8x8xf32>

// hip.hipdnn_graph preserved with all attributes
// CHECK:       %[[EMPTY0:.*]] = tensor.empty() : tensor<1x1x8x8xf32>
// CHECK:       %[[GRAPH:.*]] = hip.hipdnn_graph(%[[CTX]]) graph_id(0)
// CHECK-SAME:      ins(%[[X]], %[[W_CONV]] : tensor<1x1x8x8xf32>, tensor<1x1x3x3xf32>)
// CHECK-SAME:      outs(%[[EMPTY0]] : tensor<1x1x8x8xf32>)
// CHECK-SAME:      {input_uids = [0, 1], output_uids = [2]}
// CHECK-SAME:      : tensor<1x1x8x8xf32>

// onnx.MatMul → hip.matmul (DPS: tensor.empty init)
// CHECK:       %[[EMPTY1:.*]] = tensor.empty() : tensor<1x1x8x8xf32>
// CHECK:       %[[MATMUL:.*]] = hip.matmul(%[[CTX]])
// CHECK-SAME:      ins(%[[GRAPH]], %[[W_FC]] : tensor<1x1x8x8xf32>, tensor<8x8xf32>)
// CHECK-SAME:      outs(%[[EMPTY1]] : tensor<1x1x8x8xf32>)

// onnx.Sigmoid → hip.sigmoid (DPS: tensor.empty init)
// CHECK:       %[[EMPTY2:.*]] = tensor.empty() : tensor<1x1x8x8xf32>
// CHECK:       %[[SIG:.*]] = hip.sigmoid(%[[CTX]])
// CHECK-SAME:      ins(%[[MATMUL]] : tensor<1x1x8x8xf32>)
// CHECK-SAME:      outs(%[[EMPTY2]] : tensor<1x1x8x8xf32>)

// CHECK:       return %[[SIG]] : tensor<1x1x8x8xf32>

// No onnx ops should remain
// CHECK-NOT:   onnx.MatMul
// CHECK-NOT:   onnx.Sigmoid
