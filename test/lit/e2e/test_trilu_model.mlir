// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

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
