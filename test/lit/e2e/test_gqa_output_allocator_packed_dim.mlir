// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
// Packed cache capacity and width must follow their matching past buffers.
// Logical D=256 is derived from Q, independently of opaque cache row width.
// RUN: hip-mlir-opt %s --onnx-to-hip-pipeline 2>&1 | FileCheck %s
// CHECK-LABEL: func.func @main_graph
// CHECK-SAME: %[[PK:[a-zA-Z0-9_]+]]: memref<1x4x?x?xf16> {onnx.name = "past_key_values.3.key"}
// CHECK-SAME: %[[PV:[a-zA-Z0-9_]+]]: memref<1x4x?x?xf16> {onnx.name = "past_key_values.3.value"}
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[KC:.*]] = memref.dim %[[PK]], %[[C2]] : memref<1x4x?x?xf16>
// CHECK: %[[KW:.*]] = memref.dim %[[PK]], %[[C3]] : memref<1x4x?x?xf16>
// CHECK: hip.alloc_output(%{{.*}}, %[[KC]], %[[KW]]) {out_idx = 0 : i64} : memref<1x4x?x?xf16>
// CHECK: %[[VC:.*]] = memref.dim %[[PV]], %[[C2]] : memref<1x4x?x?xf16>
// CHECK: %[[VW:.*]] = memref.dim %[[PV]], %[[C3]] : memref<1x4x?x?xf16>
// CHECK: hip.alloc_output(%{{.*}}, %[[VC]], %[[VW]]) {out_idx = 1 : i64} : memref<1x4x?x?xf16>

module {
  func.func @main_graph(
      %q: tensor<1x1x6144xf16> {onnx.name = "q"},
      %k: tensor<1x1x1024xf16> {onnx.name = "k"},
      %v: tensor<1x1x1024xf16> {onnx.name = "v"},
      %past_key: tensor<1x4x?x?xf16> {onnx.name = "past_key_values.3.key"},
      %past_value: tensor<1x4x?x?xf16> {onnx.name = "past_key_values.3.value"},
      %seqlens: tensor<1x1xi32> {onnx.name = "seqlens_k_in"},
      %total: tensor<i32> {onnx.name = "total_seq"})
      -> (tensor<1x4x?x?xf16> {onnx.name = "present.3.key"},
          tensor<1x4x?x?xf16> {onnx.name = "present.3.value"}) {
    %none0 = "onnx.NoValue"() {value} : () -> none
    %none1 = "onnx.NoValue"() {value} : () -> none
    %r:3 = "onnx.Custom"(%q, %k, %v, %past_key, %past_value, %seqlens, %total, %none0, %none1)
      {function_name = "GroupQueryAttention", do_rotary = 0 : si64, domain_name = "com.microsoft",
       kv_num_heads = 4 : si64, num_heads = 24 : si64, rotary_interleaved = 0 : si64,
       scale = 0.0625 : f32, softcap = 0.0 : f32}
      : (tensor<1x1x6144xf16>, tensor<1x1x1024xf16>, tensor<1x1x1024xf16>,
         tensor<1x4x?x?xf16>, tensor<1x4x?x?xf16>, tensor<1x1xi32>, tensor<i32>, none, none)
      -> (tensor<1x1x6144xf16>, tensor<1x4x?x?xf16>, tensor<1x4x?x?xf16>)
    "onnx.Return"(%r#1, %r#2) : (tensor<1x4x?x?xf16>, tensor<1x4x?x?xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
