// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test single Llama-3.1-8B decode layer E2E pipeline (full-size)
// This model covers the complete decode layer: Gather, LayerNorm, MatMul,
// RotaryEmbedding, GQA, SkipLayerNorm, MLP (SiLU gate), ReduceSum, Sub, Cast
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: All ONNX ops → HIP ops
// 2. canonicalize: Simplify redundant operations
// 3. memory-pooling: Pool output buffer into single allocation
// 4. convert-hip-to-llvm: HIP ops → LLVM runtime calls
// 5. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 5
// CHECK-SAME: hipdnn.output_count = 4
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
module {
  func.func @main_graph(%arg0: tensor<1x1xi64> {onnx.name = "input_ids"}, %arg1: tensor<1x128xi64> {onnx.name = "attention_mask"}, %arg2: tensor<1x1xi64> {onnx.name = "position_ids"}, %arg3: tensor<1x8x127x128xf16> {onnx.name = "past_key_values.0.key"}, %arg4: tensor<1x8x127x128xf16> {onnx.name = "past_key_values.0.value"}) -> (tensor<1x8x128x128xf16> {onnx.name = "present.0.key"}, tensor<1x8x128x128xf16> {onnx.name = "present.0.value"}, tensor<1x1x4096xf16> {onnx.name = "/model/layers.0/post_attention_layernorm/output_3"}, tensor<1x1x4096xf16> {onnx.name = "/model/layers.0/mlp/down_proj/MatMul/output_0"}) {
    %0 = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<128256x4096xf16>} : () -> tensor<128256x4096xf16>
    %1 = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<4096xf16>} : () -> tensor<4096xf16>
    %2 = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<4096x4096xf16>} : () -> tensor<4096x4096xf16>
    %3 = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<4096x1024xf16>} : () -> tensor<4096x1024xf16>
    %4 = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<4096x1024xf16>} : () -> tensor<4096x1024xf16>
    %5 = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<131072x64xf16>} : () -> tensor<131072x64xf16>
    %6 = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<131072x64xf16>} : () -> tensor<131072x64xf16>
    %7 = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<4096x4096xf16>} : () -> tensor<4096x4096xf16>
    %8 = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<4096xf16>} : () -> tensor<4096xf16>
    %9 = "onnx.Constant"() {value = dense<1.000000e-03> : tensor<4096x14336xf16>} : () -> tensor<4096x14336xf16>
    %10 = "onnx.Constant"() {value = dense<1.000000e-03> : tensor<4096x14336xf16>} : () -> tensor<4096x14336xf16>
    %11 = "onnx.Constant"() {value = dense<1.000000e-03> : tensor<14336x4096xf16>} : () -> tensor<14336x4096xf16>
    %12 = "onnx.Constant"() {value = dense<128> : tensor<i32>} : () -> tensor<i32>
    %13 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
    %14 = "onnx.Gather"(%0, %arg0) {axis = 0 : si64, onnx_node_name = "/model/embed_tokens/Gather"} : (tensor<128256x4096xf16>, tensor<1x1xi64>) -> tensor<1x1x4096xf16>
    %15 = "onnx.ReduceSum"(%arg1, %13) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64, onnx_node_name = "/model/attn_mask_reformat/attn_mask_subgraph/ReduceSum"} : (tensor<1x128xi64>, tensor<1xi64>) -> tensor<1x1xi64>
    %16 = "onnx.Custom"(%14, %1) {function_name = "SimplifiedLayerNormalization", axis = -1 : si64, domain_name = "", epsilon = 9.99999974E-6 : f32, onnx_node_name = "/model/layers.0/input_layernorm/LayerNorm", stash_type = 1 : si64} : (tensor<1x1x4096xf16>, tensor<4096xf16>) -> tensor<1x1x4096xf16>
    %17 = "onnx.Sub"(%15, %13) {onnx_node_name = "/model/attn_mask_reformat/attn_mask_subgraph/Sub"} : (tensor<1x1xi64>, tensor<1xi64>) -> tensor<1x1xi64>
    %18 = "onnx.MatMul"(%16, %2) {onnx_node_name = "/model/layers.0/attn/q_proj/MatMul"} : (tensor<1x1x4096xf16>, tensor<4096x4096xf16>) -> tensor<1x1x4096xf16>
    %19 = "onnx.MatMul"(%16, %3) {onnx_node_name = "/model/layers.0/attn/k_proj/MatMul"} : (tensor<1x1x4096xf16>, tensor<4096x1024xf16>) -> tensor<1x1x1024xf16>
    %20 = "onnx.MatMul"(%16, %4) {onnx_node_name = "/model/layers.0/attn/v_proj/MatMul"} : (tensor<1x1x4096xf16>, tensor<4096x1024xf16>) -> tensor<1x1x1024xf16>
    %21 = "onnx.Cast"(%17) {saturate = 1 : si64, to = i32, onnx_node_name = "/model/attn_mask_reformat/attn_mask_subgraph/Sub/Cast"} : (tensor<1x1xi64>) -> tensor<1x1xi32>
    %22 = "onnx.Custom"(%18, %arg2, %5, %6) {function_name = "RotaryEmbedding", domain_name = "com.microsoft", interleaved = 0 : si64, num_heads = 0 : si64, onnx_node_name = "/model/layers.0/attn/q_rotary/RotaryEmbedding", rotary_embedding_dim = 0 : si64} : (tensor<1x1x4096xf16>, tensor<1x1xi64>, tensor<131072x64xf16>, tensor<131072x64xf16>) -> tensor<1x1x4096xf16>
    %23 = "onnx.Custom"(%19, %arg2, %5, %6) {function_name = "RotaryEmbedding", domain_name = "com.microsoft", interleaved = 0 : si64, num_heads = 0 : si64, onnx_node_name = "/model/layers.0/attn/k_rotary/RotaryEmbedding", rotary_embedding_dim = 0 : si64} : (tensor<1x1x1024xf16>, tensor<1x1xi64>, tensor<131072x64xf16>, tensor<131072x64xf16>) -> tensor<1x1x1024xf16>
    %24 = "onnx.NoValue"() {value} : () -> none
    %25 = "onnx.NoValue"() {value} : () -> none
    %26:3 = "onnx.Custom"(%22, %23, %20, %arg3, %arg4, %21, %12, %24, %25) {function_name = "GroupQueryAttention", do_rotary = 0 : si64, domain_name = "com.microsoft", kv_num_heads = 8 : si64, num_heads = 32 : si64, onnx_node_name = "/model/layers.0/attn/GroupQueryAttention", rotary_interleaved = 0 : si64, scale = 0.0883883461 : f32, softcap = 0.000000e+00 : f32} : (tensor<1x1x4096xf16>, tensor<1x1x1024xf16>, tensor<1x1x1024xf16>, tensor<1x8x127x128xf16>, tensor<1x8x127x128xf16>, tensor<1x1xi32>, tensor<i32>, none, none) -> (tensor<1x1x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>)
    %27 = "onnx.MatMul"(%26#0, %7) {onnx_node_name = "/model/layers.0/attn/o_proj/MatMul"} : (tensor<1x1x4096xf16>, tensor<4096x4096xf16>) -> tensor<1x1x4096xf16>
    %28:4 = "onnx.Custom"(%14, %27, %8) {function_name = "SkipSimplifiedLayerNormalization", domain_name = "com.microsoft", epsilon = 9.99999974E-6 : f32, onnx_node_name = "/model/layers.0/post_attention_layernorm/SkipLayerNorm"} : (tensor<1x1x4096xf16>, tensor<1x1x4096xf16>, tensor<4096xf16>) -> (tensor<1x1x4096xf16>, none, none, tensor<1x1x4096xf16>)
    %29 = "onnx.MatMul"(%28#0, %9) {onnx_node_name = "/model/layers.0/mlp/gate_proj/MatMul"} : (tensor<1x1x4096xf16>, tensor<4096x14336xf16>) -> tensor<1x1x14336xf16>
    %30 = "onnx.MatMul"(%28#0, %10) {onnx_node_name = "/model/layers.0/mlp/up_proj/MatMul"} : (tensor<1x1x4096xf16>, tensor<4096x14336xf16>) -> tensor<1x1x14336xf16>
    %31 = "onnx.Sigmoid"(%29) {onnx_node_name = "/model/layers.0/mlp/act_fn/Sigmoid"} : (tensor<1x1x14336xf16>) -> tensor<1x1x14336xf16>
    %32 = "onnx.Mul"(%29, %31) {onnx_node_name = "/model/layers.0/mlp/act_fn/Mul"} : (tensor<1x1x14336xf16>, tensor<1x1x14336xf16>) -> tensor<1x1x14336xf16>
    %33 = "onnx.Mul"(%32, %30) {onnx_node_name = "/model/layers.0/mlp/Mul"} : (tensor<1x1x14336xf16>, tensor<1x1x14336xf16>) -> tensor<1x1x14336xf16>
    %34 = "onnx.MatMul"(%33, %11) {onnx_node_name = "/model/layers.0/mlp/down_proj/MatMul"} : (tensor<1x1x14336xf16>, tensor<14336x4096xf16>) -> tensor<1x1x4096xf16>
    "onnx.Return"(%26#1, %26#2, %28#3, %34) : (tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>, tensor<1x1x4096xf16>, tensor<1x1x4096xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
