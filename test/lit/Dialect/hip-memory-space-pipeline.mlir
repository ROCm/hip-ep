// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// onnx.Pad through convert-onnx-to-hip + one-shot-bufferize: the op still
// verifies and reaches memref form, AND its OUTPUT bufferizes to
// #hip.mem<device>. PadConversion builds the DPS init as a device-space
// bufferization.alloc_tensor, so one-shot-bufferize materializes a device
// memref.alloc and DPS ties hip.pad's outs/result to it -- device-typed by
// construction, not via the transitional space-less acceptance.
//
// (bufferize-function-boundaries is omitted on purpose: it would type the
// returned value as a fully-dynamic-strided result the device alloc_tensor
// can't bare-return. The real pipeline never bare-returns a device buffer.)
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip \
// RUN:   --one-shot-bufferize | FileCheck %s

// convert-onnx-to-hip requires a @main_graph for metadata generation; the
// pilot ops live in their own functions but are still converted + bufferized.
func.func @main_graph(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  return %arg0 : tensor<1xf32>
}

// CHECK-LABEL: func.func @pad_pipe
// The DPS output init bufferizes to a device-space allocation, and hip.pad
// writes into that device buffer.
// CHECK:         memref.alloc() {{.*}} : memref<5x6xf32, #hip.mem<device>>
// CHECK:         hip.pad
// CHECK-SAME:    outs({{.*}} : memref<5x6xf32, #hip.mem<device>>)
func.func @pad_pipe(%data: tensor<3x4xf32>, %pads: tensor<4xi64>) -> tensor<5x6xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %r = "onnx.Pad"(%data, %pads, %none, %none) {mode = "constant"}
    : (tensor<3x4xf32>, tensor<4xi64>, none, none) -> tensor<5x6xf32>
  return %r : tensor<5x6xf32>
}
