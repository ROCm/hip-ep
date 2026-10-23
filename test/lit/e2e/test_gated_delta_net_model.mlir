// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 4
// CHECK-SAME: hipdnn.output_count = 2
// CHECK: llvm.func @wrap_gated_delta_net
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK-NOT: onnx.Custom
module {
  func.func @main_graph(
      %arg0: tensor<?x?x10240xf16> {onnx.name = "query"},
      %arg1: tensor<?x?x48xf32> {onnx.name = "decay"},
      %arg2: tensor<?x?x48xf32> {onnx.name = "beta"},
      %arg3: tensor<?x48x128x128xf32> {onnx.name = "initial_state"})
      -> (tensor<?x?x48x128xf16> {onnx.name = "output"},
          tensor<?x48x128x128xf32> {onnx.name = "final_state"}) {
    %a_log = "onnx.Constant"() {value = dense<0.5> : tensor<48xf32>} : () -> tensor<48xf32>
    %dt_bias = "onnx.Constant"() {value = dense<0.25> : tensor<48xf32>} : () -> tensor<48xf32>
    %none = "onnx.NoValue"() {value} : () -> none
    %0:2 = "onnx.Custom"(%arg0, %none, %none, %none, %arg1, %arg2, %arg3, %a_log, %dt_bias) {
      function_name = "GatedDeltaNet",
      domain_name = "com.microsoft",
      onnx_node_name = "gdn_node",
      update_rule = "gated_delta",
      gate_activation = "qwen",
      beta_activation = "sigmoid",
      qk_l2_norm = 1 : si64,
      scale = 0.0 : f32,
      chunk_size = 64 : si64
    } : (tensor<?x?x10240xf16>, none, none, none, tensor<?x?x48xf32>, tensor<?x?x48xf32>, tensor<?x48x128x128xf32>, tensor<48xf32>, tensor<48xf32>)
      -> (tensor<?x?x48x128xf16>, tensor<?x48x128x128xf32>)
    "onnx.Return"(%0#0, %0#1) : (tensor<?x?x48x128xf16>, tensor<?x48x128x128xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
