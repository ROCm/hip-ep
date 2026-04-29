// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

module {
  // Simple model: apply GELU to input
  func.func @main_graph(%arg0: tensor<1x128x768xf16>) -> tensor<1x128x768xf16> {
    %gelu = "onnx.Gelu"(%arg0) : (tensor<1x128x768xf16>) -> tensor<1x128x768xf16>
    return %gelu : tensor<1x128x768xf16>
  }
}

// Verify ONNX op is removed
// CHECK-NOT: onnx.Gelu

// Verify HIP op is removed (lowered to LLVM)
// CHECK-NOT: hip.gelu

// Verify runtime call is present
// CHECK: llvm.call @wrap_gelu

// Verify interface functions are generated
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
