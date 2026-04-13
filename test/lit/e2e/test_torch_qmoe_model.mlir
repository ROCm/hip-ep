// RUN: hip-mlir-opt %s --torch-hipdnn-pipeline | FileCheck %s

// Test Torch QMoE (Quantized Mixture of Experts) E2E full pipeline
// This tests the fused QMoE op injected during model export.
//
// Input:       tensor<1x128x2880xf16>  (batch x seq x hidden)
// RouterProbs: tensor<128x32xf16>      (seq x num_experts)
// fc1_weights: tensor<32x5760x1440xui8>  (experts x out x packed_in)
// fc1_scales:  tensor<32x5760x90xf16>    (experts x out x num_blocks)
// fc2_weights: tensor<32x2880x1440xui8>
// fc2_scales:  tensor<32x2880x90xf16>
// Output:      tensor<1x128x2880xf16>
//
// Verifies:
// 1. torch.aten.qmoe -> hip.qmoe
// 2. Full pipeline lowering to LLVM wrap_qmoe runtime call

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 6
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_qmoe
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK-NOT: torch.aten.qmoe
module {
  func.func @main_graph(
    %input: tensor<1x128x2880xf16>,
    %router_probs: tensor<128x32xf16>,
    %fc1_w: tensor<32x5760x1440xui8>,
    %fc1_s: tensor<32x5760x90xf16>,
    %fc2_w: tensor<32x2880x1440xui8>,
    %fc2_s: tensor<32x2880x90xf16>
  ) -> tensor<1x128x2880xf16> {
    %none = "torch.constant.none"() : () -> !torch.none
    %Y = "torch.aten.qmoe"(%input, %router_probs,
        %fc1_w, %fc1_s, %none,
        %fc2_w, %fc2_s, %none)
      {activation_alpha = 1.702000e+00 : f32,
       activation_beta = 1.000000e+00 : f32,
       activation_type = "swiglu",
       block_size = 32 : si64,
       expert_weight_bits = 4 : si64,
       k = 4 : si64,
       normalize_routing_weights = 1 : si64,
       swiglu_fusion = 1 : si64,
       swiglu_limit = 7.000000e+00 : f32,
       use_sparse_mixer = 0 : si64}
      : (tensor<1x128x2880xf16>, tensor<128x32xf16>,
         tensor<32x5760x1440xui8>, tensor<32x5760x90xf16>, !torch.none,
         tensor<32x2880x1440xui8>, tensor<32x2880x90xf16>, !torch.none)
      -> tensor<1x128x2880xf16>
    return %Y : tensor<1x128x2880xf16>
  }
}
