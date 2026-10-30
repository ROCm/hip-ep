// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// RUN: hip-mlir-opt --onnx-to-hip-pipeline %s | FileCheck %s
//
// NonZero reserves 180 i64 elements. Shape creates a separate one-element
// buffer for Slice. Assign host-accessible storage before buffer reuse can
// merge that scalar buffer into the larger allocation.
//
// CHECK-LABEL: func.func @main_graph
// CHECK: %[[SCRATCH:.*]] = hip.get_host_scratch
// CHECK: hip.nonzero
// CHECK: hip.gather_nd
// CHECK: %[[ENDS:.*]] = memref.view %[[SCRATCH]]{{.*}} to memref<1xi64>
// CHECK: memref.store %{{.*}}, %[[ENDS]][%{{.*}}] : memref<1xi64>
// CHECK: hip.slice{{.*}}ins(%{{.*}}, %{{.*}}, %[[ENDS]] :
// CHECK: return
module {
  func.func @main_graph(%x: tensor<1x1x60xf32>) -> tensor<?xf32> {
    %zero = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
    %mask = "onnx.Less"(%x, %x) : (tensor<1x1x60xf32>, tensor<1x1x60xf32>) -> tensor<1x1x60xui8>
    %coords = "onnx.NonZero"(%mask) : (tensor<1x1x60xui8>) -> tensor<3x?xi64>
    %indices = "onnx.Transpose"(%coords) {perm = [1, 0]} : (tensor<3x?xi64>) -> tensor<?x3xi64>
    %gathered = "onnx.GatherND"(%x, %indices) {batch_dims = 0 : si64} : (tensor<1x1x60xf32>, tensor<?x3xi64>) -> tensor<?xf32>
    %ends = "onnx.Shape"(%gathered) : (tensor<?xf32>) -> tensor<1xi64>
    %y = "onnx.Slice"(%gathered, %zero, %ends, %zero) : (tensor<?xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf32>
    return %y : tensor<?xf32>
  }
}
