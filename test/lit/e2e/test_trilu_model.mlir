// RUN: hip-mlir-opt %s --split-input-file --hipdnn-pipeline | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x4x4xf32>) -> tensor<1x4x4xf32> {
    %trilu = "onnx.Trilu"(%arg0) {upper = 0 : si64} : (tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
    return %trilu : tensor<1x4x4xf32>
  }
}

// CHECK-NOT: onnx.Trilu
// CHECK-NOT: hip.trilu
// CHECK: llvm.call @wrap_trilu
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup

// -----

// ChatGLM causal mask: Trilu of a dynamic ConstantOfShape. The fill value is
// written by the host into host-mapped scratch, and wrap_trilu reads that
// one-element buffer instead of a full-size device tensor.
module {
  func.func @main_graph(%shape: tensor<3xi64>) -> tensor<?x?x?xf32> {
    %ones = "onnx.ConstantOfShape"(%shape) {value = dense<1.000000e+00> : tensor<1xf32>} : (tensor<3xi64>) -> tensor<?x?x?xf32>
    %k = "onnx.Constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
    %trilu = "onnx.Trilu"(%ones, %k) {upper = 0 : si64} : (tensor<?x?x?xf32>, tensor<i64>) -> tensor<?x?x?xf32>
    return %trilu : tensor<?x?x?xf32>
  }
}

// CHECK-NOT: onnx.ConstantOfShape
// CHECK-NOT: hip.trilu
// CHECK: llvm.func private @main_graph_internal
// CHECK: %[[ONE:.*]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
// CHECK: %[[SCRATCH:.*]] = llvm.call @hipdnn_ep_get_host_scratch_base(
// CHECK: llvm.insertvalue %[[SCRATCH]]
// CHECK: llvm.store %[[ONE]], %{{.*}} : f32, !llvm.ptr
// CHECK-NEXT: %[[FILL:.*]] = llvm.extractvalue
// CHECK: llvm.call @wrap_trilu(%{{.*}}, %[[FILL]],
// CHECK: llvm.func @inference_cleanup
