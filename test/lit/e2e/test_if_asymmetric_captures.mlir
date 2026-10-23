// RUN: hip-mlir-opt %s --hipdnn-pipeline --split-input-file | FileCheck %s

// The two branches capture different outer tensors. Both outlined bodies must
// take that union plus the out-param, or hip.if legalization reports
// "body arg count mismatch".

// CHECK: llvm.func @hipdnn_ep_run_if
// CHECK: llvm.func @inference_compute
// CHECK-NOT: onnx.If
module {
  func.func @main_graph(%cond: tensor<i1> {onnx.name = "cond"}, %a: tensor<4xf32> {onnx.name = "a"}, %b: tensor<4xf32> {onnx.name = "b"}) -> (tensor<4xf32> {onnx.name = "output"}) {
    %res = "onnx.If"(%cond) ({
    ^bb0():
      %then_out = "onnx.Identity"(%a) : (tensor<4xf32>) -> tensor<4xf32>
      "onnx.Yield"(%then_out) : (tensor<4xf32>) -> ()
    }, {
    ^bb0():
      %else_out = "onnx.Identity"(%b) : (tensor<4xf32>) -> tensor<4xf32>
      "onnx.Yield"(%else_out) : (tensor<4xf32>) -> ()
    }) : (tensor<i1>) -> tensor<4xf32>
    "onnx.Return"(%res) : (tensor<4xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// -----

// tensor<1xui8> is the Greater broadcast shape. The byte must become a
// signless i1 without an unrealized_conversion_cast left for LLVM.
// CHECK: llvm.func @hipdnn_ep_run_if
// CHECK-NOT: unrealized_conversion_cast
// CHECK-NOT: onnx.If
module {
  func.func @main_graph(%cond: tensor<1xui8> {onnx.name = "cond"}) -> (tensor<4xf32> {onnx.name = "output"}) {
    %res = "onnx.If"(%cond) ({
    ^bb0():
      %then_out = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<4xf32>} : () -> tensor<4xf32>
      "onnx.Yield"(%then_out) : (tensor<4xf32>) -> ()
    }, {
    ^bb0():
      %else_out = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<4xf32>} : () -> tensor<4xf32>
      "onnx.Yield"(%else_out) : (tensor<4xf32>) -> ()
    }) : (tensor<1xui8>) -> tensor<4xf32>
    "onnx.Return"(%res) : (tensor<4xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
