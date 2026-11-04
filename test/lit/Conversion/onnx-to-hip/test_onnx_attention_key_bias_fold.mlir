// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// The onnx.Attention key-padding fold (AttentionWindowFold.cpp) drops the
// causal conjunct of Where(And(pad, causal), 0, fill) and keeps the padding
// leg as a hip.gqa key_bias row. Each case below changes one part of the
// folding control and must keep the whole mask as an attention_bias:
//   - causal_offset: the causal sides read one CumSum with different offsets
//     (strictly causal `q >= k + 1`, or shifted `q + 3 >= k`);
//   - causal_unproven: the sides read two different CumSums;
//   - unrecognized_conjunct: an extra query-dependent conjunct;
//   - select_arms_not_0_neg: the keep value is not 0;
//   - keep_too_deep: the And tree is deeper than the matcher's cap.
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%x: tensor<1xf16>) -> tensor<1xf16> {
    return %x : tensor<1xf16>
  }

  // Control: the plain mask folds into a key_bias row.
  func.func @fold_control(
      %query: tensor<?x?x2048xf16>, %key: tensor<?x?x1024xf16>,
      %value: tensor<?x?x1024xf16>, %attnmask: tensor<?x?xi64>,
      %past_key: tensor<?x8x?x128xf16>, %past_value: tensor<?x8x?x128xf16>)
      -> tensor<?x?x2048xf16> {
    %c1 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
    %c2 = "onnx.Constant"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>
    %keepv = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %dropv = "onnx.Constant"() {value = dense<-6.550400e+04> : tensor<f32>} : () -> tensor<f32>
    %ax1 = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %kidx = "onnx.CumSum"(%attnmask, %ax1) : (tensor<?x?xi64>, tensor<i64>) -> tensor<?x?xi64>
    %sbeg = "onnx.Constant"() {value = dense<-8> : tensor<1xi64>} : () -> tensor<1xi64>
    %send = "onnx.Constant"() {value = dense<9223372036854775807> : tensor<1xi64>} : () -> tensor<1xi64>
    %qidx = "onnx.Slice"(%kidx, %sbeg, %send, %c1)
        : (tensor<?x?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?x?xi64>
    %qpos0 = "onnx.Unsqueeze"(%qidx, %c2) : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x?x1xi64>
    %kpos0 = "onnx.Unsqueeze"(%kidx, %c1) : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x1x?xi64>
    %qpos = "onnx.Identity"(%qpos0) : (tensor<?x?x1xi64>) -> tensor<?x?x1xi64>
    %kpos = "onnx.Identity"(%kpos0) : (tensor<?x1x?xi64>) -> tensor<?x1x?xi64>
    %causal = "onnx.GreaterOrEqual"(%qpos, %kpos) : (tensor<?x?x1xi64>, tensor<?x1x?xi64>) -> tensor<?x?x?xi1>
    %padu = "onnx.Unsqueeze"(%attnmask, %c1) : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x1x?xi64>
    %padb = "onnx.Cast"(%padu) {to = 9 : si64} : (tensor<?x1x?xi64>) -> tensor<?x1x?xi1>
    %keep = "onnx.And"(%padb, %causal) : (tensor<?x1x?xi1>, tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
    %bias32 = "onnx.Where"(%keep, %keepv, %dropv)
        : (tensor<?x?x?xi1>, tensor<f32>, tensor<f32>) -> tensor<?x?x?xf32>
    %bias16 = "onnx.Cast"(%bias32) {to = 10 : si64} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf16>
    %bias = "onnx.Unsqueeze"(%bias16, %c1) : (tensor<?x?x?xf16>, tensor<1xi64>) -> tensor<?x1x?x?xf16>
    %out:3 = "onnx.Attention"(%query, %key, %value, %bias, %past_key, %past_value)
        {q_num_heads = 16 : si64, kv_num_heads = 8 : si64, is_causal = 1 : si64,
         scale = 8.838835e-02 : f32, softcap = 0.000000e+00 : f32}
        : (tensor<?x?x2048xf16>, tensor<?x?x1024xf16>, tensor<?x?x1024xf16>,
           tensor<?x1x?x?xf16>, tensor<?x8x?x128xf16>, tensor<?x8x?x128xf16>)
        -> (tensor<?x?x2048xf16>, tensor<?x8x?x128xf16>, tensor<?x8x?x128xf16>)
    return %out#0 : tensor<?x?x2048xf16>
  }

  // CHECK-LABEL: func.func @fold_control
  // CHECK: hip.gqa
  // CHECK-SAME: key_bias(%{{.*}} : tensor<?x?xf16>)

  // Both sides add 2 to the same positions, so `q + 2 >= k + 2` is still
  // the plain causal compare and folds.
  func.func @fold_equal_offsets(
      %query: tensor<?x?x2048xf16>, %key: tensor<?x?x1024xf16>,
      %value: tensor<?x?x1024xf16>, %attnmask: tensor<?x?xi64>,
      %past_key: tensor<?x8x?x128xf16>, %past_value: tensor<?x8x?x128xf16>)
      -> tensor<?x?x2048xf16> {
    %c1 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
    %c2 = "onnx.Constant"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>
    %keepv = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %dropv = "onnx.Constant"() {value = dense<-6.550400e+04> : tensor<f32>} : () -> tensor<f32>
    %ax1 = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %kidx = "onnx.CumSum"(%attnmask, %ax1) : (tensor<?x?xi64>, tensor<i64>) -> tensor<?x?xi64>
    %sbeg = "onnx.Constant"() {value = dense<-8> : tensor<1xi64>} : () -> tensor<1xi64>
    %send = "onnx.Constant"() {value = dense<9223372036854775807> : tensor<1xi64>} : () -> tensor<1xi64>
    %qidx = "onnx.Slice"(%kidx, %sbeg, %send, %c1)
        : (tensor<?x?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?x?xi64>
    %qpos0 = "onnx.Unsqueeze"(%qidx, %c2) : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x?x1xi64>
    %kpos0 = "onnx.Unsqueeze"(%kidx, %c1) : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x1x?xi64>
    %qpos_c = "onnx.Constant"() {value = dense<2> : tensor<i64>} : () -> tensor<i64>
    %qpos = "onnx.Add"(%qpos0, %qpos_c) : (tensor<?x?x1xi64>, tensor<i64>) -> tensor<?x?x1xi64>
    %kpos_c = "onnx.Constant"() {value = dense<2> : tensor<i64>} : () -> tensor<i64>
    %kpos = "onnx.Add"(%kpos0, %kpos_c) : (tensor<?x1x?xi64>, tensor<i64>) -> tensor<?x1x?xi64>
    %causal = "onnx.GreaterOrEqual"(%qpos, %kpos) : (tensor<?x?x1xi64>, tensor<?x1x?xi64>) -> tensor<?x?x?xi1>
    %padu = "onnx.Unsqueeze"(%attnmask, %c1) : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x1x?xi64>
    %padb = "onnx.Cast"(%padu) {to = 9 : si64} : (tensor<?x1x?xi64>) -> tensor<?x1x?xi1>
    %keep = "onnx.And"(%padb, %causal) : (tensor<?x1x?xi1>, tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
    %bias32 = "onnx.Where"(%keep, %keepv, %dropv)
        : (tensor<?x?x?xi1>, tensor<f32>, tensor<f32>) -> tensor<?x?x?xf32>
    %bias16 = "onnx.Cast"(%bias32) {to = 10 : si64} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf16>
    %bias = "onnx.Unsqueeze"(%bias16, %c1) : (tensor<?x?x?xf16>, tensor<1xi64>) -> tensor<?x1x?x?xf16>
    %out:3 = "onnx.Attention"(%query, %key, %value, %bias, %past_key, %past_value)
        {q_num_heads = 16 : si64, kv_num_heads = 8 : si64, is_causal = 1 : si64,
         scale = 8.838835e-02 : f32, softcap = 0.000000e+00 : f32}
        : (tensor<?x?x2048xf16>, tensor<?x?x1024xf16>, tensor<?x?x1024xf16>,
           tensor<?x1x?x?xf16>, tensor<?x8x?x128xf16>, tensor<?x8x?x128xf16>)
        -> (tensor<?x?x2048xf16>, tensor<?x8x?x128xf16>, tensor<?x8x?x128xf16>)
    return %out#0 : tensor<?x?x2048xf16>
  }

  // CHECK-LABEL: func.func @fold_equal_offsets
  // CHECK: hip.gqa
  // CHECK-SAME: key_bias(%{{.*}} : tensor<?x?xf16>)

  // `q >= k + 1` excludes the diagonal, which the runtime's mask keeps.
  func.func @causal_offset_strict(
      %query: tensor<?x?x2048xf16>, %key: tensor<?x?x1024xf16>,
      %value: tensor<?x?x1024xf16>, %attnmask: tensor<?x?xi64>,
      %past_key: tensor<?x8x?x128xf16>, %past_value: tensor<?x8x?x128xf16>)
      -> tensor<?x?x2048xf16> {
    %c1 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
    %c2 = "onnx.Constant"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>
    %keepv = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %dropv = "onnx.Constant"() {value = dense<-6.550400e+04> : tensor<f32>} : () -> tensor<f32>
    %ax1 = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %kidx = "onnx.CumSum"(%attnmask, %ax1) : (tensor<?x?xi64>, tensor<i64>) -> tensor<?x?xi64>
    %sbeg = "onnx.Constant"() {value = dense<-8> : tensor<1xi64>} : () -> tensor<1xi64>
    %send = "onnx.Constant"() {value = dense<9223372036854775807> : tensor<1xi64>} : () -> tensor<1xi64>
    %qidx = "onnx.Slice"(%kidx, %sbeg, %send, %c1)
        : (tensor<?x?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?x?xi64>
    %qpos0 = "onnx.Unsqueeze"(%qidx, %c2) : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x?x1xi64>
    %kpos0 = "onnx.Unsqueeze"(%kidx, %c1) : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x1x?xi64>
    %qpos = "onnx.Identity"(%qpos0) : (tensor<?x?x1xi64>) -> tensor<?x?x1xi64>
    %kpos_c = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %kpos = "onnx.Add"(%kpos0, %kpos_c) : (tensor<?x1x?xi64>, tensor<i64>) -> tensor<?x1x?xi64>
    %causal = "onnx.GreaterOrEqual"(%qpos, %kpos) : (tensor<?x?x1xi64>, tensor<?x1x?xi64>) -> tensor<?x?x?xi1>
    %padu = "onnx.Unsqueeze"(%attnmask, %c1) : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x1x?xi64>
    %padb = "onnx.Cast"(%padu) {to = 9 : si64} : (tensor<?x1x?xi64>) -> tensor<?x1x?xi1>
    %keep = "onnx.And"(%padb, %causal) : (tensor<?x1x?xi1>, tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
    %bias32 = "onnx.Where"(%keep, %keepv, %dropv)
        : (tensor<?x?x?xi1>, tensor<f32>, tensor<f32>) -> tensor<?x?x?xf32>
    %bias16 = "onnx.Cast"(%bias32) {to = 10 : si64} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf16>
    %bias = "onnx.Unsqueeze"(%bias16, %c1) : (tensor<?x?x?xf16>, tensor<1xi64>) -> tensor<?x1x?x?xf16>
    %out:3 = "onnx.Attention"(%query, %key, %value, %bias, %past_key, %past_value)
        {q_num_heads = 16 : si64, kv_num_heads = 8 : si64, is_causal = 1 : si64,
         scale = 8.838835e-02 : f32, softcap = 0.000000e+00 : f32}
        : (tensor<?x?x2048xf16>, tensor<?x?x1024xf16>, tensor<?x?x1024xf16>,
           tensor<?x1x?x?xf16>, tensor<?x8x?x128xf16>, tensor<?x8x?x128xf16>)
        -> (tensor<?x?x2048xf16>, tensor<?x8x?x128xf16>, tensor<?x8x?x128xf16>)
    return %out#0 : tensor<?x?x2048xf16>
  }

  // CHECK-LABEL: func.func @causal_offset_strict
  // CHECK: hip.gqa
  // CHECK-SAME: tensor<?x1x?x?xf16>)
  // CHECK-NOT: key_bias
  // CHECK-SAME: outs(

  // `q + 3 >= k` admits three future keys.
  func.func @causal_offset_shifted(
      %query: tensor<?x?x2048xf16>, %key: tensor<?x?x1024xf16>,
      %value: tensor<?x?x1024xf16>, %attnmask: tensor<?x?xi64>,
      %past_key: tensor<?x8x?x128xf16>, %past_value: tensor<?x8x?x128xf16>)
      -> tensor<?x?x2048xf16> {
    %c1 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
    %c2 = "onnx.Constant"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>
    %keepv = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %dropv = "onnx.Constant"() {value = dense<-6.550400e+04> : tensor<f32>} : () -> tensor<f32>
    %ax1 = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %kidx = "onnx.CumSum"(%attnmask, %ax1) : (tensor<?x?xi64>, tensor<i64>) -> tensor<?x?xi64>
    %sbeg = "onnx.Constant"() {value = dense<-8> : tensor<1xi64>} : () -> tensor<1xi64>
    %send = "onnx.Constant"() {value = dense<9223372036854775807> : tensor<1xi64>} : () -> tensor<1xi64>
    %qidx = "onnx.Slice"(%kidx, %sbeg, %send, %c1)
        : (tensor<?x?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?x?xi64>
    %qpos0 = "onnx.Unsqueeze"(%qidx, %c2) : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x?x1xi64>
    %kpos0 = "onnx.Unsqueeze"(%kidx, %c1) : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x1x?xi64>
    %qpos_c = "onnx.Constant"() {value = dense<3> : tensor<i64>} : () -> tensor<i64>
    %qpos = "onnx.Add"(%qpos0, %qpos_c) : (tensor<?x?x1xi64>, tensor<i64>) -> tensor<?x?x1xi64>
    %kpos = "onnx.Identity"(%kpos0) : (tensor<?x1x?xi64>) -> tensor<?x1x?xi64>
    %causal = "onnx.GreaterOrEqual"(%qpos, %kpos) : (tensor<?x?x1xi64>, tensor<?x1x?xi64>) -> tensor<?x?x?xi1>
    %padu = "onnx.Unsqueeze"(%attnmask, %c1) : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x1x?xi64>
    %padb = "onnx.Cast"(%padu) {to = 9 : si64} : (tensor<?x1x?xi64>) -> tensor<?x1x?xi1>
    %keep = "onnx.And"(%padb, %causal) : (tensor<?x1x?xi1>, tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
    %bias32 = "onnx.Where"(%keep, %keepv, %dropv)
        : (tensor<?x?x?xi1>, tensor<f32>, tensor<f32>) -> tensor<?x?x?xf32>
    %bias16 = "onnx.Cast"(%bias32) {to = 10 : si64} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf16>
    %bias = "onnx.Unsqueeze"(%bias16, %c1) : (tensor<?x?x?xf16>, tensor<1xi64>) -> tensor<?x1x?x?xf16>
    %out:3 = "onnx.Attention"(%query, %key, %value, %bias, %past_key, %past_value)
        {q_num_heads = 16 : si64, kv_num_heads = 8 : si64, is_causal = 1 : si64,
         scale = 8.838835e-02 : f32, softcap = 0.000000e+00 : f32}
        : (tensor<?x?x2048xf16>, tensor<?x?x1024xf16>, tensor<?x?x1024xf16>,
           tensor<?x1x?x?xf16>, tensor<?x8x?x128xf16>, tensor<?x8x?x128xf16>)
        -> (tensor<?x?x2048xf16>, tensor<?x8x?x128xf16>, tensor<?x8x?x128xf16>)
    return %out#0 : tensor<?x?x2048xf16>
  }

  // CHECK-LABEL: func.func @causal_offset_shifted
  // CHECK: hip.gqa
  // CHECK-SAME: tensor<?x1x?x?xf16>)
  // CHECK-NOT: key_bias
  // CHECK-SAME: outs(

  // The query side reads a second CumSum, so the compare is not proven to
  // relate positions on one scale.
  func.func @causal_unproven(
      %query: tensor<?x?x2048xf16>, %key: tensor<?x?x1024xf16>,
      %value: tensor<?x?x1024xf16>, %attnmask: tensor<?x?xi64>,
      %past_key: tensor<?x8x?x128xf16>, %past_value: tensor<?x8x?x128xf16>)
      -> tensor<?x?x2048xf16> {
    %c1 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
    %c2 = "onnx.Constant"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>
    %keepv = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %dropv = "onnx.Constant"() {value = dense<-6.550400e+04> : tensor<f32>} : () -> tensor<f32>
    %ax1 = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %kidx = "onnx.CumSum"(%attnmask, %ax1) : (tensor<?x?xi64>, tensor<i64>) -> tensor<?x?xi64>
    %kidx2 = "onnx.CumSum"(%attnmask, %ax1) {exclusive = 1 : si64} : (tensor<?x?xi64>, tensor<i64>) -> tensor<?x?xi64>
    %sbeg = "onnx.Constant"() {value = dense<-8> : tensor<1xi64>} : () -> tensor<1xi64>
    %send = "onnx.Constant"() {value = dense<9223372036854775807> : tensor<1xi64>} : () -> tensor<1xi64>
    %qidx = "onnx.Slice"(%kidx2, %sbeg, %send, %c1)
        : (tensor<?x?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?x?xi64>
    %qpos0 = "onnx.Unsqueeze"(%qidx, %c2) : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x?x1xi64>
    %kpos0 = "onnx.Unsqueeze"(%kidx, %c1) : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x1x?xi64>
    %qpos = "onnx.Identity"(%qpos0) : (tensor<?x?x1xi64>) -> tensor<?x?x1xi64>
    %kpos = "onnx.Identity"(%kpos0) : (tensor<?x1x?xi64>) -> tensor<?x1x?xi64>
    %causal = "onnx.GreaterOrEqual"(%qpos, %kpos) : (tensor<?x?x1xi64>, tensor<?x1x?xi64>) -> tensor<?x?x?xi1>
    %padu = "onnx.Unsqueeze"(%attnmask, %c1) : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x1x?xi64>
    %padb = "onnx.Cast"(%padu) {to = 9 : si64} : (tensor<?x1x?xi64>) -> tensor<?x1x?xi1>
    %keep = "onnx.And"(%padb, %causal) : (tensor<?x1x?xi1>, tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
    %bias32 = "onnx.Where"(%keep, %keepv, %dropv)
        : (tensor<?x?x?xi1>, tensor<f32>, tensor<f32>) -> tensor<?x?x?xf32>
    %bias16 = "onnx.Cast"(%bias32) {to = 10 : si64} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf16>
    %bias = "onnx.Unsqueeze"(%bias16, %c1) : (tensor<?x?x?xf16>, tensor<1xi64>) -> tensor<?x1x?x?xf16>
    %out:3 = "onnx.Attention"(%query, %key, %value, %bias, %past_key, %past_value)
        {q_num_heads = 16 : si64, kv_num_heads = 8 : si64, is_causal = 1 : si64,
         scale = 8.838835e-02 : f32, softcap = 0.000000e+00 : f32}
        : (tensor<?x?x2048xf16>, tensor<?x?x1024xf16>, tensor<?x?x1024xf16>,
           tensor<?x1x?x?xf16>, tensor<?x8x?x128xf16>, tensor<?x8x?x128xf16>)
        -> (tensor<?x?x2048xf16>, tensor<?x8x?x128xf16>, tensor<?x8x?x128xf16>)
    return %out#0 : tensor<?x?x2048xf16>
  }

  // CHECK-LABEL: func.func @causal_unproven
  // CHECK: hip.gqa
  // CHECK-SAME: tensor<?x1x?x?xf16>)
  // CHECK-NOT: key_bias
  // CHECK-SAME: outs(

  // An extra query-dependent conjunct, `q == k`, cannot be dropped.
  func.func @unrecognized_conjunct(
      %query: tensor<?x?x2048xf16>, %key: tensor<?x?x1024xf16>,
      %value: tensor<?x?x1024xf16>, %attnmask: tensor<?x?xi64>,
      %past_key: tensor<?x8x?x128xf16>, %past_value: tensor<?x8x?x128xf16>)
      -> tensor<?x?x2048xf16> {
    %c1 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
    %c2 = "onnx.Constant"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>
    %keepv = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %dropv = "onnx.Constant"() {value = dense<-6.550400e+04> : tensor<f32>} : () -> tensor<f32>
    %ax1 = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %kidx = "onnx.CumSum"(%attnmask, %ax1) : (tensor<?x?xi64>, tensor<i64>) -> tensor<?x?xi64>
    %sbeg = "onnx.Constant"() {value = dense<-8> : tensor<1xi64>} : () -> tensor<1xi64>
    %send = "onnx.Constant"() {value = dense<9223372036854775807> : tensor<1xi64>} : () -> tensor<1xi64>
    %qidx = "onnx.Slice"(%kidx, %sbeg, %send, %c1)
        : (tensor<?x?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?x?xi64>
    %qpos0 = "onnx.Unsqueeze"(%qidx, %c2) : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x?x1xi64>
    %kpos0 = "onnx.Unsqueeze"(%kidx, %c1) : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x1x?xi64>
    %qpos = "onnx.Identity"(%qpos0) : (tensor<?x?x1xi64>) -> tensor<?x?x1xi64>
    %kpos = "onnx.Identity"(%kpos0) : (tensor<?x1x?xi64>) -> tensor<?x1x?xi64>
    %causal = "onnx.GreaterOrEqual"(%qpos, %kpos) : (tensor<?x?x1xi64>, tensor<?x1x?xi64>) -> tensor<?x?x?xi1>
    %eq = "onnx.Equal"(%qpos, %kpos) : (tensor<?x?x1xi64>, tensor<?x1x?xi64>) -> tensor<?x?x?xi1>
    %both = "onnx.And"(%causal, %eq) : (tensor<?x?x?xi1>, tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
    %padu = "onnx.Unsqueeze"(%attnmask, %c1) : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x1x?xi64>
    %padb = "onnx.Cast"(%padu) {to = 9 : si64} : (tensor<?x1x?xi64>) -> tensor<?x1x?xi1>
    %keep = "onnx.And"(%padb, %both) : (tensor<?x1x?xi1>, tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
    %bias32 = "onnx.Where"(%keep, %keepv, %dropv)
        : (tensor<?x?x?xi1>, tensor<f32>, tensor<f32>) -> tensor<?x?x?xf32>
    %bias16 = "onnx.Cast"(%bias32) {to = 10 : si64} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf16>
    %bias = "onnx.Unsqueeze"(%bias16, %c1) : (tensor<?x?x?xf16>, tensor<1xi64>) -> tensor<?x1x?x?xf16>
    %out:3 = "onnx.Attention"(%query, %key, %value, %bias, %past_key, %past_value)
        {q_num_heads = 16 : si64, kv_num_heads = 8 : si64, is_causal = 1 : si64,
         scale = 8.838835e-02 : f32, softcap = 0.000000e+00 : f32}
        : (tensor<?x?x2048xf16>, tensor<?x?x1024xf16>, tensor<?x?x1024xf16>,
           tensor<?x1x?x?xf16>, tensor<?x8x?x128xf16>, tensor<?x8x?x128xf16>)
        -> (tensor<?x?x2048xf16>, tensor<?x8x?x128xf16>, tensor<?x8x?x128xf16>)
    return %out#0 : tensor<?x?x2048xf16>
  }

  // CHECK-LABEL: func.func @unrecognized_conjunct
  // CHECK: hip.gqa
  // CHECK-SAME: tensor<?x1x?x?xf16>)
  // CHECK-NOT: key_bias
  // CHECK-SAME: outs(

  // A keep value of 1 is a bias, not a mask.
  func.func @select_arms_not_0_neg(
      %query: tensor<?x?x2048xf16>, %key: tensor<?x?x1024xf16>,
      %value: tensor<?x?x1024xf16>, %attnmask: tensor<?x?xi64>,
      %past_key: tensor<?x8x?x128xf16>, %past_value: tensor<?x8x?x128xf16>)
      -> tensor<?x?x2048xf16> {
    %c1 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
    %c2 = "onnx.Constant"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>
    %keepv = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %dropv = "onnx.Constant"() {value = dense<-6.550400e+04> : tensor<f32>} : () -> tensor<f32>
    %ax1 = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %kidx = "onnx.CumSum"(%attnmask, %ax1) : (tensor<?x?xi64>, tensor<i64>) -> tensor<?x?xi64>
    %sbeg = "onnx.Constant"() {value = dense<-8> : tensor<1xi64>} : () -> tensor<1xi64>
    %send = "onnx.Constant"() {value = dense<9223372036854775807> : tensor<1xi64>} : () -> tensor<1xi64>
    %qidx = "onnx.Slice"(%kidx, %sbeg, %send, %c1)
        : (tensor<?x?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?x?xi64>
    %qpos0 = "onnx.Unsqueeze"(%qidx, %c2) : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x?x1xi64>
    %kpos0 = "onnx.Unsqueeze"(%kidx, %c1) : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x1x?xi64>
    %qpos = "onnx.Identity"(%qpos0) : (tensor<?x?x1xi64>) -> tensor<?x?x1xi64>
    %kpos = "onnx.Identity"(%kpos0) : (tensor<?x1x?xi64>) -> tensor<?x1x?xi64>
    %causal = "onnx.GreaterOrEqual"(%qpos, %kpos) : (tensor<?x?x1xi64>, tensor<?x1x?xi64>) -> tensor<?x?x?xi1>
    %padu = "onnx.Unsqueeze"(%attnmask, %c1) : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x1x?xi64>
    %padb = "onnx.Cast"(%padu) {to = 9 : si64} : (tensor<?x1x?xi64>) -> tensor<?x1x?xi1>
    %keep = "onnx.And"(%padb, %causal) : (tensor<?x1x?xi1>, tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
    %bias32 = "onnx.Where"(%keep, %keepv, %dropv)
        : (tensor<?x?x?xi1>, tensor<f32>, tensor<f32>) -> tensor<?x?x?xf32>
    %bias16 = "onnx.Cast"(%bias32) {to = 10 : si64} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf16>
    %bias = "onnx.Unsqueeze"(%bias16, %c1) : (tensor<?x?x?xf16>, tensor<1xi64>) -> tensor<?x1x?x?xf16>
    %out:3 = "onnx.Attention"(%query, %key, %value, %bias, %past_key, %past_value)
        {q_num_heads = 16 : si64, kv_num_heads = 8 : si64, is_causal = 1 : si64,
         scale = 8.838835e-02 : f32, softcap = 0.000000e+00 : f32}
        : (tensor<?x?x2048xf16>, tensor<?x?x1024xf16>, tensor<?x?x1024xf16>,
           tensor<?x1x?x?xf16>, tensor<?x8x?x128xf16>, tensor<?x8x?x128xf16>)
        -> (tensor<?x?x2048xf16>, tensor<?x8x?x128xf16>, tensor<?x8x?x128xf16>)
    return %out#0 : tensor<?x?x2048xf16>
  }

  // CHECK-LABEL: func.func @select_arms_not_0_neg
  // CHECK: hip.gqa
  // CHECK-SAME: tensor<?x1x?x?xf16>)
  // CHECK-NOT: key_bias
  // CHECK-SAME: outs(

  // Seventeen boolean Identity ops exceed the And-tree depth cap.
  func.func @keep_too_deep(
      %query: tensor<?x?x2048xf16>, %key: tensor<?x?x1024xf16>,
      %value: tensor<?x?x1024xf16>, %attnmask: tensor<?x?xi64>,
      %past_key: tensor<?x8x?x128xf16>, %past_value: tensor<?x8x?x128xf16>)
      -> tensor<?x?x2048xf16> {
    %c1 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
    %c2 = "onnx.Constant"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>
    %keepv = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %dropv = "onnx.Constant"() {value = dense<-6.550400e+04> : tensor<f32>} : () -> tensor<f32>
    %ax1 = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %kidx = "onnx.CumSum"(%attnmask, %ax1) : (tensor<?x?xi64>, tensor<i64>) -> tensor<?x?xi64>
    %sbeg = "onnx.Constant"() {value = dense<-8> : tensor<1xi64>} : () -> tensor<1xi64>
    %send = "onnx.Constant"() {value = dense<9223372036854775807> : tensor<1xi64>} : () -> tensor<1xi64>
    %qidx = "onnx.Slice"(%kidx, %sbeg, %send, %c1)
        : (tensor<?x?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?x?xi64>
    %qpos0 = "onnx.Unsqueeze"(%qidx, %c2) : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x?x1xi64>
    %kpos0 = "onnx.Unsqueeze"(%kidx, %c1) : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x1x?xi64>
    %qpos = "onnx.Identity"(%qpos0) : (tensor<?x?x1xi64>) -> tensor<?x?x1xi64>
    %kpos = "onnx.Identity"(%kpos0) : (tensor<?x1x?xi64>) -> tensor<?x1x?xi64>
    %causal = "onnx.GreaterOrEqual"(%qpos, %kpos) : (tensor<?x?x1xi64>, tensor<?x1x?xi64>) -> tensor<?x?x?xi1>
    %d0 = "onnx.Identity"(%causal) : (tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
    %d1 = "onnx.Identity"(%d0) : (tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
    %d2 = "onnx.Identity"(%d1) : (tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
    %d3 = "onnx.Identity"(%d2) : (tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
    %d4 = "onnx.Identity"(%d3) : (tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
    %d5 = "onnx.Identity"(%d4) : (tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
    %d6 = "onnx.Identity"(%d5) : (tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
    %d7 = "onnx.Identity"(%d6) : (tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
    %d8 = "onnx.Identity"(%d7) : (tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
    %d9 = "onnx.Identity"(%d8) : (tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
    %d10 = "onnx.Identity"(%d9) : (tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
    %d11 = "onnx.Identity"(%d10) : (tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
    %d12 = "onnx.Identity"(%d11) : (tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
    %d13 = "onnx.Identity"(%d12) : (tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
    %d14 = "onnx.Identity"(%d13) : (tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
    %d15 = "onnx.Identity"(%d14) : (tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
    %d16 = "onnx.Identity"(%d15) : (tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
    %d17 = "onnx.Identity"(%d16) : (tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
    %padu = "onnx.Unsqueeze"(%attnmask, %c1) : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x1x?xi64>
    %padb = "onnx.Cast"(%padu) {to = 9 : si64} : (tensor<?x1x?xi64>) -> tensor<?x1x?xi1>
    %keep = "onnx.And"(%padb, %d17) : (tensor<?x1x?xi1>, tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
    %bias32 = "onnx.Where"(%keep, %keepv, %dropv)
        : (tensor<?x?x?xi1>, tensor<f32>, tensor<f32>) -> tensor<?x?x?xf32>
    %bias16 = "onnx.Cast"(%bias32) {to = 10 : si64} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf16>
    %bias = "onnx.Unsqueeze"(%bias16, %c1) : (tensor<?x?x?xf16>, tensor<1xi64>) -> tensor<?x1x?x?xf16>
    %out:3 = "onnx.Attention"(%query, %key, %value, %bias, %past_key, %past_value)
        {q_num_heads = 16 : si64, kv_num_heads = 8 : si64, is_causal = 1 : si64,
         scale = 8.838835e-02 : f32, softcap = 0.000000e+00 : f32}
        : (tensor<?x?x2048xf16>, tensor<?x?x1024xf16>, tensor<?x?x1024xf16>,
           tensor<?x1x?x?xf16>, tensor<?x8x?x128xf16>, tensor<?x8x?x128xf16>)
        -> (tensor<?x?x2048xf16>, tensor<?x8x?x128xf16>, tensor<?x8x?x128xf16>)
    return %out#0 : tensor<?x?x2048xf16>
  }

  // CHECK-LABEL: func.func @keep_too_deep
  // CHECK: hip.gqa
  // CHECK-SAME: tensor<?x1x?x?xf16>)
  // CHECK-NOT: key_bias
  // CHECK-SAME: outs(
}
