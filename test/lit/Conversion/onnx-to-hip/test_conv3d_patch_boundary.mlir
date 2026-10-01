// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s --check-prefix=CONVERT
// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip --hip-fusion-transform %s | FileCheck %s --check-prefix=FUSE

// Conversion preserves the general rank-5 Conv. Patch embedding is a HIP
// fusion decision, after the shared shape contract has sized its destination.
// CONVERT-LABEL: func.func @main_graph
// CONVERT-NOT: hip.gemm
// CONVERT: hip.conv
// CONVERT-SAME: outs({{.*}} : tensor<?x1152x1x1x1xf16>)
// CONVERT-NOT: hip.gemm
// CONVERT: return
// FUSE-LABEL: func.func @main_graph
// FUSE-NOT: hip.conv
// FUSE-NOT: hip.transpose
// FUSE: hip.gemm
// FUSE-SAME: tensor<?x1536xf16>, tensor<1152x1536xf16>, tensor<1152xf16>
// FUSE-SAME: outs({{.*}} : tensor<?x1152xf16>)
// FUSE-SAME: transB = 1
// FUSE-NOT: hip.conv
// FUSE-NOT: hip.transpose
// FUSE: tensor.expand_shape {{.*}} into tensor<?x1152x1x1x1xf16>
// FUSE-NOT: hip.conv
// FUSE-NOT: hip.transpose
// FUSE: return
func.func @main_graph(
    %x: tensor<?x3x2x16x16xf16>,
    %w: tensor<1152x3x2x16x16xf16>,
    %b: tensor<1152xf16>) -> tensor<?x1152x1x1x1xf16> {
  %y = "onnx.Conv"(%x, %w, %b) {
    kernel_shape = [2, 16, 16], strides = [2, 16, 16],
    pads = [0, 0, 0, 0, 0, 0], dilations = [1, 1, 1], group = 1 : i64
  } : (tensor<?x3x2x16x16xf16>, tensor<1152x3x2x16x16xf16>,
       tensor<1152xf16>) -> tensor<?x1152x1x1x1xf16>
  return %y : tensor<?x1152x1x1x1xf16>
}
