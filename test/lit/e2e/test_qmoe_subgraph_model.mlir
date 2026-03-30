// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test QMoE subgraph E2E pipeline (router MatMul -> Add -> Reshape -> QMoE)
// Extracted from GPT-OSS-20B layer 0 MoE block with seq_len=128.
// All 4-bit weights use packed value 0x99 (two 4-bit 9s per byte),
// dequantized weight = (9 - 8) * 0.01 = 0.01.
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: All ONNX ops → HIP ops
// 2. bufferize / canonicalize
// 3. convert-hip-to-llvm: HIP ops → LLVM runtime calls
// 4. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
module {
  func.func @main_graph(%arg0: tensor<1x128x2880xf16> {onnx.name = "/model/layers.0/post_attention_layernorm/output_0"}) -> (tensor<1x128x2880xf16> {onnx.name = "/model/layers.0/moe/QMoE/output_0"}) attributes {onnx.graph.name = "moe_subgraph_seq128"} {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {node.outputs = ["model.layers.0.moe.experts.down_proj.bias"], value = dense<1.000000e-02> : tensor<32x2880xf16>} : () -> tensor<32x2880xf16>
    %2 = "onnx.Constant"() {node.outputs = ["model.layers.0.moe.experts.down_proj.qweight"], value = dense<153> : tensor<32x2880x1440xui8>} : () -> tensor<32x2880x1440xui8>
    %3 = "onnx.Constant"() {node.outputs = ["model.layers.0.moe.experts.down_proj.scales"], value = dense<1.000000e-02> : tensor<32x2880x90xf16>} : () -> tensor<32x2880x90xf16>
    %4 = "onnx.Constant"() {node.outputs = ["/model/constants/INT64/[-1, 32]"], value = dense<[-1, 32]> : tensor<2xi64>} : () -> tensor<2xi64>
    %5 = "onnx.Constant"() {node.outputs = ["model.layers.0.moe.router.Add.bias"], value = dense<1.000000e-02> : tensor<32xf16>} : () -> tensor<32xf16>
    %6 = "onnx.Constant"() {node.outputs = ["model.layers.0.moe.experts.gate_up_proj.qweight"], value = dense<153> : tensor<32x5760x1440xui8>} : () -> tensor<32x5760x1440xui8>
    %7 = "onnx.Constant"() {node.outputs = ["model.layers.0.moe.experts.gate_up_proj.bias"], value = dense<1.000000e-02> : tensor<32x5760xf16>} : () -> tensor<32x5760xf16>
    %8 = "onnx.Constant"() {node.outputs = ["model.layers.0.moe.experts.gate_up_proj.scales"], value = dense<1.000000e-02> : tensor<32x5760x90xf16>} : () -> tensor<32x5760x90xf16>
    %9 = "onnx.Constant"() {node.outputs = ["model.layers.0.moe.router.MatMul.weight_Q4"], value = dense<153> : tensor<32x90x16xui8>} : () -> tensor<32x90x16xui8>
    %10 = "onnx.Constant"() {node.outputs = ["model.layers.0.moe.router.MatMul.weight_scales"], value = dense<1.000000e-02> : tensor<32x90xf16>} : () -> tensor<32x90xf16>
    %11 = "onnx.Custom"(%arg0, %9, %10) {K = 2880 : si64, N = 32 : si64, accuracy_level = 4 : si64, bits = 4 : si64, block_size = 32 : si64, domain_name = "com.microsoft", function_name = "MatMulNBits", node.outputs = ["/model/layers.0/moe/router/MatMul/output_0"], onnx_node_name = "/model/layers.0/moe/router/MatMul_Q4"} : (tensor<1x128x2880xf16>, tensor<32x90x16xui8>, tensor<32x90xf16>) -> tensor<1x128x32xf16>
    %12 = "onnx.Add"(%11, %5) {node.outputs = ["/model/layers.0/moe/router/Add/output_0"], onnx_node_name = "/model/layers.0/moe/router/Add"} : (tensor<1x128x32xf16>, tensor<32xf16>) -> tensor<1x128x32xf16>
    %13 = "onnx.Reshape"(%12, %4) {allowzero = 0 : si64, node.outputs = ["/model/layers.0/moe/router/Reshape/output_0"], onnx_node_name = "/model/layers.0/moe/router/Reshape"} : (tensor<1x128x32xf16>, tensor<2xi64>) -> tensor<128x32xf16>
    %14 = "onnx.Custom"(%arg0, %13, %6, %8, %7, %2, %3, %1, %0, %0, %0) {activation_alpha = 1.702000e+00 : f32, activation_beta = 1.000000e+00 : f32, activation_type = "swiglu", block_size = 32 : si64, domain_name = "com.microsoft", expert_weight_bits = 4 : si64, function_name = "QMoE", k = 4 : si64, node.outputs = ["/model/layers.0/moe/QMoE/output_0"], normalize_routing_weights = 1 : si64, onnx_node_name = "/model/layers.0/moe/QMoE", swiglu_fusion = 1 : si64, swiglu_limit = 7.000000e+00 : f32, use_sparse_mixer = 0 : si64} : (tensor<1x128x2880xf16>, tensor<128x32xf16>, tensor<32x5760x1440xui8>, tensor<32x5760x90xf16>, tensor<32x5760xf16>, tensor<32x2880x1440xui8>, tensor<32x2880x90xf16>, tensor<32x2880xf16>, none, none, none) -> tensor<1x128x2880xf16>
    "onnx.Return"(%14) : (tensor<1x128x2880xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
