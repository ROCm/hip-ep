// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // Required by the OnnxToHip pass (which always reads `@main_graph`'s
  // signature when generating module metadata). Kept as a trivial passthrough
  // so that the actual NonZero patterns are exercised by the test functions
  // below.
  func.func @main_graph(%arg0: tensor<3x4xi1>) -> tensor<3x4xi1> {
    return %arg0 : tensor<3x4xi1>
  }

  // --- Case 1: bool input (i1), rank 2 ---
  func.func @nonzero_bool(%input: tensor<3x4xi1>) -> tensor<2x?xi64> {
    %result = "onnx.NonZero"(%input) : (tensor<3x4xi1>) -> tensor<2x?xi64>
    return %result : tensor<2x?xi64>
  }

  // CHECK-LABEL: func.func @nonzero_bool
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<3x4xi1>)
  // CHECK: %[[UB:.*]] = arith.constant 12 : index
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[UB]]) : tensor<2x?xi64>
  // CHECK: %[[CNT:.*]] = tensor.empty() : tensor<1xi32>
  // CHECK: hip.nonzero(%[[CTX]]) ins(%[[IN]] : tensor<3x4xi1>) outs(%[[INIT]], %[[CNT]] : tensor<2x?xi64>, tensor<1xi32>) {input_data_type = 5 : i64}

  // --- Case 2: f32 input, rank 3 ---
  func.func @nonzero_f32(%input: tensor<2x3x4xf32>) -> tensor<3x?xi64> {
    %result = "onnx.NonZero"(%input) : (tensor<2x3x4xf32>) -> tensor<3x?xi64>
    return %result : tensor<3x?xi64>
  }

  // CHECK-LABEL: func.func @nonzero_f32
  // CHECK-SAME: (%[[CTX2:.*]]: !hip.context, %[[IN2:.*]]: tensor<2x3x4xf32>)
  // CHECK: %[[UB2:.*]] = arith.constant 24 : index
  // CHECK: %[[INIT2:.*]] = tensor.empty(%[[UB2]]) : tensor<3x?xi64>
  // CHECK: %[[CNT2:.*]] = tensor.empty() : tensor<1xi32>
  // CHECK: hip.nonzero(%[[CTX2]]) ins(%[[IN2]] : tensor<2x3x4xf32>) outs(%[[INIT2]], %[[CNT2]] : tensor<3x?xi64>, tensor<1xi32>) {input_data_type = 0 : i64}

  // --- Case 3: i64 input, rank 1 ---
  func.func @nonzero_i64(%input: tensor<8xi64>) -> tensor<1x?xi64> {
    %result = "onnx.NonZero"(%input) : (tensor<8xi64>) -> tensor<1x?xi64>
    return %result : tensor<1x?xi64>
  }

  // CHECK-LABEL: func.func @nonzero_i64
  // CHECK-SAME: (%[[CTX3:.*]]: !hip.context, %[[IN3:.*]]: tensor<8xi64>)
  // CHECK: %[[UB3:.*]] = arith.constant 8 : index
  // CHECK: %[[INIT3:.*]] = tensor.empty(%[[UB3]]) : tensor<1x?xi64>
  // CHECK: %[[CNT3:.*]] = tensor.empty() : tensor<1xi32>
  // CHECK: hip.nonzero(%[[CTX3]]) ins(%[[IN3]] : tensor<8xi64>) outs(%[[INIT3]], %[[CNT3]] : tensor<1x?xi64>, tensor<1xi32>) {input_data_type = 4 : i64}

  // --- Case 4: dynamic input shape, rank 2 ---
  // Per-dim chain: upper bound = 1 * dim(input,0) * dim(input,1). The
  // running product feeds the dynsize of the rank-2 NonZero output.
  // Sequential CHECKs (NOT CHECK-DAG). Note: the running-product init
  // constant `1` and the per-dim index constant `1` (used by
  // `tensor.dim %input, 1`) collapse to a single SSA value because
  // `tensor::DimOp::build(..., int64_t)` materialises its index via
  // `createOrFold<arith::ConstantIndexOp>` and folds against the existing
  // index-1 constant. So we only see one `arith.constant 1 : index`.
  func.func @nonzero_dynamic_input(%input: tensor<?x?xf32>) -> tensor<2x?xi64> {
    %result = "onnx.NonZero"(%input) : (tensor<?x?xf32>) -> tensor<2x?xi64>
    return %result : tensor<2x?xi64>
  }

  // CHECK-LABEL: func.func @nonzero_dynamic_input
  // CHECK-SAME: (%[[CTX4:.*]]: !hip.context, %[[IN4:.*]]: tensor<?x?xf32>)
  // CHECK: arith.constant 1 : index
  // CHECK: arith.constant 0 : index
  // CHECK: tensor.dim %[[IN4]], %{{.*}} : tensor<?x?xf32>
  // CHECK: arith.muli %{{.*}}, %{{.*}} : index
  // CHECK: tensor.dim %[[IN4]], %{{.*}} : tensor<?x?xf32>
  // CHECK: arith.muli %{{.*}}, %{{.*}} : index
  // CHECK: tensor.empty(%{{.*}}) : tensor<2x?xi64>
  // CHECK: tensor.empty() : tensor<1xi32>
  // CHECK: hip.nonzero(%[[CTX4]]) ins(%[[IN4]] : tensor<?x?xf32>) outs(%{{.*}}, %{{.*}} : tensor<2x?xi64>, tensor<1xi32>) {input_data_type = 0 : i64}

  // --- Case 5: partially dynamic input (mix of static + dynamic dims).
  // Static dims contribute a compile-time arith.constant; dynamic dims
  // contribute tensor.dim. Both feed the same muli chain. Same
  // constant-folding caveat as Case 4: the running-product init `1` and
  // the dim-1 index `1` share one SSA value.
  func.func @nonzero_partial_dynamic(%input: tensor<4x?xi32>) -> tensor<2x?xi64> {
    %result = "onnx.NonZero"(%input) : (tensor<4x?xi32>) -> tensor<2x?xi64>
    return %result : tensor<2x?xi64>
  }

  // CHECK-LABEL: func.func @nonzero_partial_dynamic
  // CHECK-SAME: (%[[CTX5:.*]]: !hip.context, %[[IN5:.*]]: tensor<4x?xi32>)
  // CHECK: arith.constant 1 : index
  // CHECK: arith.constant 4 : index
  // CHECK: arith.muli %{{.*}}, %{{.*}} : index
  // CHECK: tensor.dim %[[IN5]], %{{.*}} : tensor<4x?xi32>
  // CHECK: arith.muli %{{.*}}, %{{.*}} : index
  // CHECK: tensor.empty(%{{.*}}) : tensor<2x?xi64>
  // CHECK: hip.nonzero

  // --- Case 6: ui8 input (ORT-imported bool convention) ---
  // ORT imports ONNX bool as `ui8`, not signless `i1`. MLIR's `isInteger(W)`
  // only matches signless integers, so without explicit unsigned-integer
  // checks the pattern silently fails and `onnx.NonZero` leaks past
  // `convert-onnx-to-hip`, breaking one-shot-bufferize downstream with
  // `op was not bufferized`. Keep this case forever — it's the only thing
  // a real ORT-imported model exercises (the other cases use synthetic i1).
  func.func @nonzero_ui8(%input: tensor<2x4x8xui8>) -> tensor<3x?xi64> {
    %result = "onnx.NonZero"(%input) : (tensor<2x4x8xui8>) -> tensor<3x?xi64>
    return %result : tensor<3x?xi64>
  }

  // CHECK-LABEL: func.func @nonzero_ui8
  // CHECK-SAME: (%[[CTX6:.*]]: !hip.context, %[[IN6:.*]]: tensor<2x4x8xui8>)
  // CHECK: %[[UB6:.*]] = arith.constant 64 : index
  // CHECK: %[[INIT6:.*]] = tensor.empty(%[[UB6]]) : tensor<3x?xi64>
  // CHECK: %[[CNT6:.*]] = tensor.empty() : tensor<1xi32>
  // ui8 maps to the dedicated UINT8 slot (7), NOT INT8 (5) — the runtime
  // enum keeps signed/unsigned distinct so any future ordered-comparison
  // or arithmetic backend can dispatch correctly.
  // CHECK: hip.nonzero(%[[CTX6]]) ins(%[[IN6]] : tensor<2x4x8xui8>) outs(%[[INIT6]], %[[CNT6]] : tensor<3x?xi64>, tensor<1xi32>) {input_data_type = 7 : i64}
}
