// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s
//
// D32 grouped prefill has a growing present cache. Its dynamic sequence
// dimension must fit total_seq_len even though the past has only 13 rows.
module {
  // CHECK-LABEL: func.func @main_graph
  // CHECK-SAME: %[[TOTAL:[a-zA-Z0-9_]+]]: tensor<i32>
  func.func @main_graph(%q: tensor<2x19x128xf16>,
      %k: tensor<2x19x64xf16>, %v: tensor<2x19x64xf16>,
      %pk: tensor<2x2x13x32xf16>, %pv: tensor<2x2x13x32xf16>,
      %lengths: tensor<2xi32>, %total: tensor<i32>)
      -> (tensor<2x19x128xf16>, tensor<2x2x?x32xf16>, tensor<2x2x?x32xf16>) {
    %none = "onnx.NoValue"() {value} : () -> none
    // CHECK: %[[CAP:.*]] = hip.readback_dim(%{{.*}}, %[[TOTAL]] : tensor<i32>) -> index
    // CHECK: %[[PK:.*]] = arith.maxsi %{{.*}}, %[[CAP]] : index
    // CHECK: tensor.empty(%[[PK]]) : tensor<2x2x?x32xf16>
    // CHECK: %[[PV:.*]] = arith.maxsi %{{.*}}, %[[CAP]] : index
    // CHECK: tensor.empty(%[[PV]]) : tensor<2x2x?x32xf16>
    // CHECK: hip.gqa
    %r:3 = "onnx.Custom"(%q, %k, %v, %pk, %pv, %lengths, %total, %none, %none)
        <{function_name = "GroupQueryAttention"}>
        {domain_name = "com.microsoft", num_heads = 4 : si64,
         kv_num_heads = 2 : si64, do_rotary = 0 : si64,
         scale = 0.17677669 : f32}
        : (tensor<2x19x128xf16>, tensor<2x19x64xf16>, tensor<2x19x64xf16>,
           tensor<2x2x13x32xf16>, tensor<2x2x13x32xf16>, tensor<2xi32>,
           tensor<i32>, none, none)
        -> (tensor<2x19x128xf16>, tensor<2x2x?x32xf16>, tensor<2x2x?x32xf16>)
    return %r#0, %r#1, %r#2 : tensor<2x19x128xf16>, tensor<2x2x?x32xf16>, tensor<2x2x?x32xf16>
  }
}
