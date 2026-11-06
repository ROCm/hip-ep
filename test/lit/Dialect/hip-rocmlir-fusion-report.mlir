// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// The report has one host record per function and one kernel record per
// dispatch. The two convolutions outline to equal kernels, which the
// duplicate-function elimination merges, so both dispatches name one kernel
// and carry the same body hash.

// RUN: rm -f %t.jsonl
// RUN: hip-mlir-opt --hip-rocmlir-fusion="report=%t.jsonl" %s -o /dev/null
// RUN: FileCheck %s --input-file=%t.jsonl

// CHECK: {"kind":"host","name":"main_graph","ops":{"func.return":1,"hip.rocmlir":2,"tensor.empty":2}}
// CHECK-NEXT: {"anchor":"hip.conv","hash":"[[HASH:[0-9a-f]+]]","inputs":["tensor<1x8x4x4xf16>","tensor<16x8x3x3xf16>","tensor<16xf16>"],"kind":"kernel","name":"[[KERNEL:rocMlir[0-9]+]]","ops":{"hip.conv":1,"tensor.empty":1,"ub.poison":1},"outputs":["tensor<1x16x4x4xf16>"]}
// CHECK-NEXT: {"anchor":"hip.conv","hash":"[[HASH]]",{{.*}}"name":"[[KERNEL]]",
// CHECK-NOT: kind

func.func @main_graph(%ctx: !hip.context, %in0: tensor<1x8x4x4xf16>,
                      %in1: tensor<1x8x4x4xf16>, %w: tensor<16x8x3x3xf16>,
                      %b: tensor<16xf16>)
    -> (tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16>) {
  %e0 = tensor.empty() : tensor<1x16x4x4xf16>
  %c0 = hip.conv(%ctx) ins(%in0, %w, %b : tensor<1x8x4x4xf16>,
                                          tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e0 : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  %e1 = tensor.empty() : tensor<1x16x4x4xf16>
  %c1 = hip.conv(%ctx) ins(%in1, %w, %b : tensor<1x8x4x4xf16>,
                                          tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e1 : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  return %c0, %c1 : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16>
}
