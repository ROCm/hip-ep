// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test com.amd::QMoE (Nemotron-H LatentMoE) E2E full pipeline.
// Verifies the complete hipdnn-pipeline, distinct from the com.microsoft::QMoE
// path exercised by test_qmoe_model.mlir:
// 1. convert-onnx-to-hip: ONNX Custom(function_name=QMoE, domain=com.amd) ->
//    hip.qmoe_amd
// 2. convert-hip-to-llvm: hip.qmoe_amd -> wrap_qmoe_amd runtime call
// 3. generate-interface: Create inference_init/compute/cleanup/metadata
//
// Dims mirror test/numeric/tests/test_qmoe_amd.py (hidden=32, latent=16,
// moe_intermediate=16, shared_intermediate=32, num_experts=4, k=2,
// block_size=32 -- matches the real Nemotron-H checkpoint, see that file's
// module docstring for why block_size=16 is deliberately avoided) so the
// same shapes are exercised for real GPU execution
// (E2E_Execute_test_qmoe_amd_model) as for the CPU-reference numeric check.

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_qmoe_amd
// CHECK: llvm.call @wrap_qmoe_amd
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Custom
module {
  func.func @main_graph(%arg0: tensor<1x4x32xf16> {onnx.name = "hidden_states"}) -> (tensor<1x4x32xf16> {onnx.name = "output"}) {
    %fc1e_w = "onnx.Constant"() {value = dense<1> : tensor<4x16x1x16xui8>} : () -> tensor<4x16x1x16xui8>
    %fc1e_s = "onnx.Constant"() {value = dense<1.000000e-02> : tensor<4x16x1xf16>} : () -> tensor<4x16x1xf16>
    %fc2e_w = "onnx.Constant"() {value = dense<1> : tensor<4x16x1x16xui8>} : () -> tensor<4x16x1x16xui8>
    %fc2e_s = "onnx.Constant"() {value = dense<1.000000e-02> : tensor<4x16x1xf16>} : () -> tensor<4x16x1xf16>
    %fc1l_w = "onnx.Constant"() {value = dense<1> : tensor<16x1x16xui8>} : () -> tensor<16x1x16xui8>
    %fc1l_s = "onnx.Constant"() {value = dense<1.000000e-02> : tensor<16x1xf16>} : () -> tensor<16x1xf16>
    %fc2l_w = "onnx.Constant"() {value = dense<1> : tensor<32x1x16xui8>} : () -> tensor<32x1x16xui8>
    %fc2l_s = "onnx.Constant"() {value = dense<1.000000e-02> : tensor<32x1xf16>} : () -> tensor<32x1xf16>
    %sh1_w = "onnx.Constant"() {value = dense<1> : tensor<32x1x16xui8>} : () -> tensor<32x1x16xui8>
    %sh1_s = "onnx.Constant"() {value = dense<1.000000e-02> : tensor<32x1xf16>} : () -> tensor<32x1xf16>
    %sh2_w = "onnx.Constant"() {value = dense<1> : tensor<32x1x16xui8>} : () -> tensor<32x1x16xui8>
    %sh2_s = "onnx.Constant"() {value = dense<1.000000e-02> : tensor<32x1xf16>} : () -> tensor<32x1xf16>
    %router_w = "onnx.Constant"() {value = dense<1.000000e-01> : tensor<32x4xf16>} : () -> tensor<32x4xf16>
    %corr_b = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<4xf16>} : () -> tensor<4xf16>
    %Y = "onnx.Custom"(%arg0, %fc1e_w, %fc1e_s, %fc2e_w, %fc2e_s,
                        %fc1l_w, %fc1l_s, %fc2l_w, %fc2l_s,
                        %sh1_w, %sh1_s, %sh2_w, %sh2_s,
                        %router_w, %corr_b) {
      function_name = "QMoE",
      domain_name = "com.amd",
      k = 2 : si64, expert_weight_bits = 4 : si64, block_size = 32 : si64,
      normalize_routing_weights = 1 : si64, use_correction_bias = 1 : si64,
      routed_scaling_factor = 2.000000e+00 : f32,
      activation_type = "relu2", routing_type = "sigmoid",
      onnx_node_name = "QMoE_amd_0"
    } : (tensor<1x4x32xf16>,
         tensor<4x16x1x16xui8>, tensor<4x16x1xf16>,
         tensor<4x16x1x16xui8>, tensor<4x16x1xf16>,
         tensor<16x1x16xui8>, tensor<16x1xf16>,
         tensor<32x1x16xui8>, tensor<32x1xf16>,
         tensor<32x1x16xui8>, tensor<32x1xf16>,
         tensor<32x1x16xui8>, tensor<32x1xf16>,
         tensor<32x4xf16>, tensor<4xf16>) -> tensor<1x4x32xf16>
    "onnx.Return"(%Y) : (tensor<1x4x32xf16>) -> ()
  }
  "onnx.EntryPoint"() <{func = @main_graph}> : () -> ()
}
