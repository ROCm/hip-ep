// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test QMoE E2E full pipeline
// Verifies the complete hipdnn-pipeline for quantized Mixture-of-Experts:
// 1. convert-onnx-to-hip: ONNX Custom op → hip.qmoe
// 2. convert-hip-to-llvm: HIP op → wrap_qmoe runtime call
// 3. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_qmoe
// CHECK: llvm.call @wrap_qmoe
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Custom
module {
  func.func @main_graph(%arg0: tensor<1x128x2880xf16> {onnx.name = "input"}, %arg1: tensor<128x32xf16> {onnx.name = "router_probs"}) -> (tensor<1x128x2880xf16> {onnx.name = "output"}) {
    %fc1_w = "onnx.Constant"() {value = dense<1> : tensor<32x5760x1440xui8>} : () -> tensor<32x5760x1440xui8>
    %fc1_s = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<32x5760x90xf16>} : () -> tensor<32x5760x90xf16>
    %fc1_b = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<32x5760xf16>} : () -> tensor<32x5760xf16>
    %fc2_w = "onnx.Constant"() {value = dense<1> : tensor<32x2880x1440xui8>} : () -> tensor<32x2880x1440xui8>
    %fc2_s = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<32x2880x90xf16>} : () -> tensor<32x2880x90xf16>
    %fc2_b = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<32x2880xf16>} : () -> tensor<32x2880xf16>
    %Y = "onnx.Custom"(%arg0, %arg1, %fc1_w, %fc1_s, %fc1_b, %fc2_w, %fc2_s, %fc2_b) {function_name = "QMoE", activation_alpha = 1.702000e+00 : f32, activation_beta = 1.000000e+00 : f32, activation_type = "swiglu", block_size = 32 : si64, domain_name = "com.microsoft", expert_weight_bits = 4 : si64, k = 4 : si64, normalize_routing_weights = 1 : si64, onnx_node_name = "QMoE_0", swiglu_fusion = 1 : si64, swiglu_limit = 7.000000e+00 : f32, use_sparse_mixer = 0 : si64} : (tensor<1x128x2880xf16>, tensor<128x32xf16>, tensor<32x5760x1440xui8>, tensor<32x5760x90xf16>, tensor<32x5760xf16>, tensor<32x2880x1440xui8>, tensor<32x2880x90xf16>, tensor<32x2880xf16>) -> tensor<1x128x2880xf16>
    "onnx.Return"(%Y) : (tensor<1x128x2880xf16>) -> ()
  }
  "onnx.EntryPoint"() <{func = @main_graph}> : () -> ()
}
