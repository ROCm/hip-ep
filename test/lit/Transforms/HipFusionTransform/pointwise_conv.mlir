// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// PointwiseConvToMatmul: a 1x1, stride-1, unpadded, ungrouped float hip.conv
// becomes a hip.matmul of the [Cout, Cin] filter against the [N, Cin, P] input
// (P = the pixels, collapsed), plus a broadcast hip.add of the bias on that
// [N, Cout, P] result, expanded back to the conv's extents. Every other conv is
// left to hip.conv or, for a single pixel of a rank-4/5 conv, to patch-embed's
// hip.gemm.

// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// f16 with a bias: the reshapes, the matmul with the filter as its rank-2 left
// operand, and the bias add writing into the conv's own destination, regrouped
// like the matmul result.
// CHECK-LABEL: func.func @pw_f16_bias
// CHECK-SAME:    (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x64x56x56xf16>, %[[W:.*]]: tensor<256x64x1x1xf16>, %[[B:.*]]: tensor<256xf16>)
// CHECK-NOT:     hip.conv
// CHECK-DAG:     %[[INIT:.*]] = tensor.empty() : tensor<1x256x56x56xf16>
// CHECK-DAG:     %[[XF:.*]] = tensor.collapse_shape %[[X]] {{\[\[}}0], [1], [2, 3]] : tensor<1x64x56x56xf16> into tensor<1x64x3136xf16>
// CHECK-DAG:     %[[WF:.*]] = tensor.collapse_shape %[[W]] {{\[\[}}0], [1, 2, 3]] : tensor<256x64x1x1xf16> into tensor<256x64xf16>
// CHECK-DAG:     %[[E:.*]] = tensor.empty() : tensor<1x256x3136xf16>
// CHECK:         %[[MM:.*]] = hip.matmul(%[[CTX]]) ins(%[[WF]], %[[XF]] : tensor<256x64xf16>, tensor<1x64x3136xf16>) outs(%[[E]] : tensor<1x256x3136xf16>)
// CHECK:         %[[B3:.*]] = tensor.expand_shape %[[B]] {{\[\[}}0, 1, 2]] output_shape [1, 256, 1] : tensor<256xf16> into tensor<1x256x1xf16>
// CHECK:         %[[IF:.*]] = tensor.collapse_shape %[[INIT]] {{\[\[}}0], [1], [2, 3]] : tensor<1x256x56x56xf16> into tensor<1x256x3136xf16>
// CHECK:         %[[S:.*]] = hip.add(%[[CTX]]) ins(%[[MM]], %[[B3]] : tensor<1x256x3136xf16>, tensor<1x256x1xf16>) outs(%[[IF]] : tensor<1x256x3136xf16>)
// CHECK:         %[[Y:.*]] = tensor.expand_shape %[[S]] {{\[\[}}0], [1], [2, 3]] output_shape [1, 256, 56, 56] : tensor<1x256x3136xf16> into tensor<1x256x56x56xf16>
// CHECK:         return %[[Y]]
func.func @pw_f16_bias(%ctx: !hip.context, %x: tensor<1x64x56x56xf16>,
                       %w: tensor<256x64x1x1xf16>,
                       %b: tensor<256xf16>) -> tensor<1x256x56x56xf16> {
  %init = tensor.empty() : tensor<1x256x56x56xf16>
  %y = hip.conv(%ctx)
      ins(%x, %w, %b : tensor<1x64x56x56xf16>, tensor<256x64x1x1xf16>, tensor<256xf16>)
      outs(%init : tensor<1x256x56x56xf16>)
      {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
       dilations = [1, 1], group = 1 : i64} : tensor<1x256x56x56xf16>
  return %y : tensor<1x256x56x56xf16>
}

// -----

// f32 without a bias: the expanded matmul result replaces the conv, no add.
// CHECK-LABEL: func.func @pw_f32_no_bias
// CHECK-NOT:     hip.conv
// CHECK:         %[[MM:.*]] = hip.matmul({{.*}}) ins({{.*}} : tensor<32x128xf32>, tensor<2x128x49xf32>) outs({{.*}} : tensor<2x32x49xf32>)
// CHECK:         %[[Y:.*]] = tensor.expand_shape %[[MM]] {{\[\[}}0], [1], [2, 3]] output_shape [2, 32, 7, 7] : tensor<2x32x49xf32> into tensor<2x32x7x7xf32>
// CHECK-NOT:     hip.add
// CHECK:         return %[[Y]]
func.func @pw_f32_no_bias(%ctx: !hip.context, %x: tensor<2x128x7x7xf32>,
                          %w: tensor<32x128x1x1xf32>) -> tensor<2x32x7x7xf32> {
  %init = tensor.empty() : tensor<2x32x7x7xf32>
  %y = hip.conv(%ctx)
      ins(%x, %w : tensor<2x128x7x7xf32>, tensor<32x128x1x1xf32>)
      outs(%init : tensor<2x32x7x7xf32>)
      {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
       dilations = [1, 1], group = 1 : i64} : tensor<2x32x7x7xf32>
  return %y : tensor<2x32x7x7xf32>
}

// -----

// A dynamic batch: the batch is the only dynamic extent, read off the collapsed
// input for the matmul's init and off the input for the expand. The bias add
// writes into the conv's dynamic destination, collapsed.
// CHECK-LABEL: func.func @pw_dynamic_batch
// CHECK-SAME:    (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<?x96x28x28xf16>, %[[W:.*]]: tensor<24x96x1x1xf16>, %[[B:.*]]: tensor<24xf16>, %[[INIT:.*]]: tensor<?x24x28x28xf16>)
// CHECK-NOT:     hip.conv
// CHECK-DAG:     %[[XF:.*]] = tensor.collapse_shape %[[X]] {{\[\[}}0], [1], [2, 3]] : tensor<?x96x28x28xf16> into tensor<?x96x784xf16>
// CHECK-DAG:     %[[N:.*]] = tensor.dim %[[XF]], %{{.*}} : tensor<?x96x784xf16>
// CHECK-DAG:     %[[E:.*]] = tensor.empty(%[[N]]) : tensor<?x24x784xf16>
// CHECK:         %[[MM:.*]] = hip.matmul(%[[CTX]]) ins(%{{.*}}, %[[XF]] : tensor<24x96xf16>, tensor<?x96x784xf16>) outs(%[[E]] : tensor<?x24x784xf16>)
// CHECK:         %[[IF:.*]] = tensor.collapse_shape %[[INIT]] {{\[\[}}0], [1], [2, 3]] : tensor<?x24x28x28xf16> into tensor<?x24x784xf16>
// CHECK:         %[[S:.*]] = hip.add(%[[CTX]]) ins(%[[MM]], %{{.*}} : tensor<?x24x784xf16>, tensor<1x24x1xf16>) outs(%[[IF]] : tensor<?x24x784xf16>)
// CHECK:         %[[N2:.*]] = tensor.dim %[[X]], %{{.*}} : tensor<?x96x28x28xf16>
// CHECK:         %[[Y:.*]] = tensor.expand_shape %[[S]] {{\[\[}}0], [1], [2, 3]] output_shape [%[[N2]], 24, 28, 28] : tensor<?x24x784xf16> into tensor<?x24x28x28xf16>
// CHECK:         return %[[Y]]
func.func @pw_dynamic_batch(%ctx: !hip.context, %x: tensor<?x96x28x28xf16>,
                            %w: tensor<24x96x1x1xf16>, %b: tensor<24xf16>,
                            %init: tensor<?x24x28x28xf16>) -> tensor<?x24x28x28xf16> {
  %y = hip.conv(%ctx)
      ins(%x, %w, %b : tensor<?x96x28x28xf16>, tensor<24x96x1x1xf16>, tensor<24xf16>)
      outs(%init : tensor<?x24x28x28xf16>)
      {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
       dilations = [1, 1], group = 1 : i64} : tensor<?x24x28x28xf16>
  return %y : tensor<?x24x28x28xf16>
}

// -----

// Dynamic batch and spatial extents: the collapsed pixel axis is dynamic too,
// and the expand restates every dynamic extent from the input.
// CHECK-LABEL: func.func @pw_dynamic_spatial
// CHECK-NOT:     hip.conv
// CHECK:         tensor.collapse_shape %{{.*}} {{\[\[}}0], [1], [2, 3]] : tensor<?x16x?x?xf32> into tensor<?x16x?xf32>
// CHECK:         tensor.empty(%{{.*}}, %{{.*}}) : tensor<?x8x?xf32>
// CHECK:         hip.matmul({{.*}}) ins({{.*}} : tensor<8x16xf32>, tensor<?x16x?xf32>) outs({{.*}} : tensor<?x8x?xf32>)
// CHECK:         %[[S:.*]] = hip.add({{.*}}) ins({{.*}} : tensor<?x8x?xf32>, tensor<1x8x1xf32>)
// CHECK:         tensor.expand_shape %[[S]] {{\[\[}}0], [1], [2, 3]] output_shape [%{{.*}}, 8, %{{.*}}, %{{.*}}] : tensor<?x8x?xf32> into tensor<?x8x?x?xf32>
func.func @pw_dynamic_spatial(%ctx: !hip.context, %x: tensor<?x16x?x?xf32>,
                              %w: tensor<8x16x1x1xf32>, %b: tensor<8xf32>,
                              %init: tensor<?x8x?x?xf32>) -> tensor<?x8x?x?xf32> {
  %y = hip.conv(%ctx)
      ins(%x, %w, %b : tensor<?x16x?x?xf32>, tensor<8x16x1x1xf32>, tensor<8xf32>)
      outs(%init : tensor<?x8x?x?xf32>)
      {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
       dilations = [1, 1], group = 1 : i64} : tensor<?x8x?x?xf32>
  return %y : tensor<?x8x?x?xf32>
}

// -----

// Rank 3 (NCL) already is [N, C, P]: no reshape of the input or the result,
// only of the filter and the bias.
// CHECK-LABEL: func.func @pw_rank3
// CHECK-SAME:    (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<4x32x100xf16>
// CHECK-NOT:     hip.conv
// CHECK:         %[[MM:.*]] = hip.matmul(%[[CTX]]) ins(%{{.*}}, %[[X]] : tensor<48x32xf16>, tensor<4x32x100xf16>) outs({{.*}} : tensor<4x48x100xf16>)
// CHECK:         %[[B3:.*]] = tensor.expand_shape %{{.*}} {{\[\[}}0, 1, 2]] output_shape [1, 48, 1] : tensor<48xf16> into tensor<1x48x1xf16>
// CHECK:         hip.add(%[[CTX]]) ins(%[[MM]], %[[B3]] : tensor<4x48x100xf16>, tensor<1x48x1xf16>)
// CHECK-NOT:     tensor.expand_shape
func.func @pw_rank3(%ctx: !hip.context, %x: tensor<4x32x100xf16>,
                    %w: tensor<48x32x1xf16>, %b: tensor<48xf16>) -> tensor<4x48x100xf16> {
  %init = tensor.empty() : tensor<4x48x100xf16>
  %y = hip.conv(%ctx)
      ins(%x, %w, %b : tensor<4x32x100xf16>, tensor<48x32x1xf16>, tensor<48xf16>)
      outs(%init : tensor<4x48x100xf16>)
      {kernel_shape = [1], strides = [1], pads = [0, 0],
       dilations = [1], group = 1 : i64} : tensor<4x48x100xf16>
  return %y : tensor<4x48x100xf16>
}

// -----

// bf16 is a hipBLASLt matmul type as well.
// CHECK-LABEL: func.func @pw_bf16
// CHECK-NOT:     hip.conv
// CHECK:         hip.matmul({{.*}}) ins({{.*}} : tensor<16x8xbf16>, tensor<1x8x64xbf16>)
func.func @pw_bf16(%ctx: !hip.context, %x: tensor<1x8x8x8xbf16>,
                   %w: tensor<16x8x1x1xbf16>) -> tensor<1x16x8x8xbf16> {
  %init = tensor.empty() : tensor<1x16x8x8xbf16>
  %y = hip.conv(%ctx)
      ins(%x, %w : tensor<1x8x8x8xbf16>, tensor<16x8x1x1xbf16>)
      outs(%init : tensor<1x16x8x8xbf16>)
      {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
       dilations = [1, 1], group = 1 : i64} : tensor<1x16x8x8xbf16>
  return %y : tensor<1x16x8x8xbf16>
}

// -----

// Stride 2: the input pixels the conv reads are not a matrix operand.
// CHECK-LABEL: func.func @pw_stride2_stays_conv
// CHECK-NOT:     hip.matmul
// CHECK:         hip.conv
func.func @pw_stride2_stays_conv(%ctx: !hip.context, %x: tensor<1x256x56x56xf16>,
                                 %w: tensor<512x256x1x1xf16>,
                                 %b: tensor<512xf16>) -> tensor<1x512x28x28xf16> {
  %init = tensor.empty() : tensor<1x512x28x28xf16>
  %y = hip.conv(%ctx)
      ins(%x, %w, %b : tensor<1x256x56x56xf16>, tensor<512x256x1x1xf16>, tensor<512xf16>)
      outs(%init : tensor<1x512x28x28xf16>)
      {kernel_shape = [1, 1], strides = [2, 2], pads = [0, 0, 0, 0],
       dilations = [1, 1], group = 1 : i64} : tensor<1x512x28x28xf16>
  return %y : tensor<1x512x28x28xf16>
}

// -----

// Padding adds zero pixels around the input, so the result is not the input's
// pixels contracted: stays a conv.
// CHECK-LABEL: func.func @pw_padded_stays_conv
// CHECK-NOT:     hip.matmul
// CHECK:         hip.conv
func.func @pw_padded_stays_conv(%ctx: !hip.context, %x: tensor<1x8x6x6xf32>,
                                %w: tensor<4x8x1x1xf32>) -> tensor<1x4x8x8xf32> {
  %init = tensor.empty() : tensor<1x4x8x8xf32>
  %y = hip.conv(%ctx)
      ins(%x, %w : tensor<1x8x6x6xf32>, tensor<4x8x1x1xf32>)
      outs(%init : tensor<1x4x8x8xf32>)
      {kernel_shape = [1, 1], strides = [1, 1], pads = [1, 1, 1, 1],
       dilations = [1, 1], group = 1 : i64} : tensor<1x4x8x8xf32>
  return %y : tensor<1x4x8x8xf32>
}

// -----

// A grouped 1x1 is a block-diagonal filter, not one GEMM.
// CHECK-LABEL: func.func @pw_grouped_stays_conv
// CHECK-NOT:     hip.matmul
// CHECK:         hip.conv
func.func @pw_grouped_stays_conv(%ctx: !hip.context, %x: tensor<1x128x32x32xf16>,
                                 %w: tensor<128x1x1x1xf16>) -> tensor<1x128x32x32xf16> {
  %init = tensor.empty() : tensor<1x128x32x32xf16>
  %y = hip.conv(%ctx)
      ins(%x, %w : tensor<1x128x32x32xf16>, tensor<128x1x1x1xf16>)
      outs(%init : tensor<1x128x32x32xf16>)
      {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
       dilations = [1, 1], group = 128 : i64} : tensor<1x128x32x32xf16>
  return %y : tensor<1x128x32x32xf16>
}

// -----

// A 3x3 kernel is not pointwise.
// CHECK-LABEL: func.func @conv3x3_stays_conv
// CHECK-NOT:     hip.matmul
// CHECK:         hip.conv
func.func @conv3x3_stays_conv(%ctx: !hip.context, %x: tensor<1x64x56x56xf16>,
                              %w: tensor<64x64x3x3xf16>) -> tensor<1x64x56x56xf16> {
  %init = tensor.empty() : tensor<1x64x56x56xf16>
  %y = hip.conv(%ctx)
      ins(%x, %w : tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>)
      outs(%init : tensor<1x64x56x56xf16>)
      {kernel_shape = [3, 3], strides = [1, 1], pads = [1, 1, 1, 1],
       dilations = [1, 1], group = 1 : i64} : tensor<1x64x56x56xf16>
  return %y : tensor<1x64x56x56xf16>
}

// -----

// An integer conv has no hipBLASLt matmul: stays on the integer conv kernel.
// CHECK-LABEL: func.func @pw_i32_stays_conv
// CHECK-NOT:     hip.matmul
// CHECK:         hip.conv
func.func @pw_i32_stays_conv(%ctx: !hip.context, %x: tensor<1x8x4x4xi32>,
                             %w: tensor<4x8x1x1xi32>) -> tensor<1x4x4x4xi32> {
  %init = tensor.empty() : tensor<1x4x4x4xi32>
  %y = hip.conv(%ctx)
      ins(%x, %w : tensor<1x8x4x4xi32>, tensor<4x8x1x1xi32>)
      outs(%init : tensor<1x4x4x4xi32>)
      {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
       dilations = [1, 1], group = 1 : i64} : tensor<1x4x4x4xi32>
  return %y : tensor<1x4x4x4xi32>
}

// -----

// A dynamic channel count would be the GEMM's K, which hip.matmul's caller
// must know: stays a conv.
// CHECK-LABEL: func.func @pw_dynamic_channels_stays_conv
// CHECK-NOT:     hip.matmul
// CHECK:         hip.conv
func.func @pw_dynamic_channels_stays_conv(%ctx: !hip.context, %x: tensor<1x?x4x4xf32>,
                                          %w: tensor<4x?x1x1xf32>) -> tensor<1x4x4x4xf32> {
  %init = tensor.empty() : tensor<1x4x4x4xf32>
  %y = hip.conv(%ctx)
      ins(%x, %w : tensor<1x?x4x4xf32>, tensor<4x?x1x1xf32>)
      outs(%init : tensor<1x4x4x4xf32>)
      {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
       dilations = [1, 1], group = 1 : i64} : tensor<1x4x4x4xf32>
  return %y : tensor<1x4x4x4xf32>
}

// -----

// A single pixel per image is patch-embed's single-patch case: a 2-D hip.gemm
// with the bias as its C operand, so this pattern leaves it alone.
// CHECK-LABEL: func.func @pw_single_pixel_goes_to_gemm
// CHECK-NOT:     hip.matmul
// CHECK-NOT:     hip.conv
// CHECK:         hip.gemm
// CHECK-NOT:     hip.add
func.func @pw_single_pixel_goes_to_gemm(%ctx: !hip.context, %x: tensor<1x64x1x1xf16>,
                                        %w: tensor<256x64x1x1xf16>,
                                        %b: tensor<256xf16>) -> tensor<1x256x1x1xf16> {
  %init = tensor.empty() : tensor<1x256x1x1xf16>
  %y = hip.conv(%ctx)
      ins(%x, %w, %b : tensor<1x64x1x1xf16>, tensor<256x64x1x1xf16>, tensor<256xf16>)
      outs(%init : tensor<1x256x1x1xf16>)
      {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
       dilations = [1, 1], group = 1 : i64} : tensor<1x256x1x1xf16>
  return %y : tensor<1x256x1x1xf16>
}

// -----

// Rank 5 (NCDHW) collapses three spatial axes; the bias add stays rank 3, which
// the elementwise lowering supports (it stops at rank 4).
// CHECK-LABEL: func.func @pw_rank5_bias
// CHECK-NOT:     hip.conv
// CHECK:         tensor.collapse_shape %{{.*}} {{\[\[}}0], [1], [2, 3, 4]] : tensor<2x16x4x8x8xf16> into tensor<2x16x256xf16>
// CHECK:         %[[MM:.*]] = hip.matmul({{.*}}) ins({{.*}} : tensor<32x16xf16>, tensor<2x16x256xf16>) outs({{.*}} : tensor<2x32x256xf16>)
// CHECK:         %[[S:.*]] = hip.add({{.*}}) ins(%[[MM]], %{{.*}} : tensor<2x32x256xf16>, tensor<1x32x1xf16>) outs({{.*}} : tensor<2x32x256xf16>)
// CHECK:         tensor.expand_shape %[[S]] {{\[\[}}0], [1], [2, 3, 4]] output_shape [2, 32, 4, 8, 8] : tensor<2x32x256xf16> into tensor<2x32x4x8x8xf16>
func.func @pw_rank5_bias(%ctx: !hip.context, %x: tensor<2x16x4x8x8xf16>,
                         %w: tensor<32x16x1x1x1xf16>, %b: tensor<32xf16>) -> tensor<2x32x4x8x8xf16> {
  %init = tensor.empty() : tensor<2x32x4x8x8xf16>
  %y = hip.conv(%ctx)
      ins(%x, %w, %b : tensor<2x16x4x8x8xf16>, tensor<32x16x1x1x1xf16>, tensor<32xf16>)
      outs(%init : tensor<2x32x4x8x8xf16>)
      {kernel_shape = [1, 1, 1], strides = [1, 1, 1], pads = [0, 0, 0, 0, 0, 0],
       dilations = [1, 1, 1], group = 1 : i64} : tensor<2x32x4x8x8xf16>
  return %y : tensor<2x32x4x8x8xf16>
}

// -----

// A rank-3 conv over a single position is still taken: patch-embed only
// handles rank 4 and up, and the matmul is exact at any extent.
// CHECK-LABEL: func.func @pw_rank3_single_position
// CHECK-NOT:     hip.conv
// CHECK:         hip.matmul({{.*}}) ins({{.*}} : tensor<8x4xf32>, tensor<3x4x1xf32>)
func.func @pw_rank3_single_position(%ctx: !hip.context, %x: tensor<3x4x1xf32>,
                                    %w: tensor<8x4x1xf32>) -> tensor<3x8x1xf32> {
  %init = tensor.empty() : tensor<3x8x1xf32>
  %y = hip.conv(%ctx)
      ins(%x, %w : tensor<3x4x1xf32>, tensor<8x4x1xf32>)
      outs(%init : tensor<3x8x1xf32>)
      {kernel_shape = [1], strides = [1], pads = [0, 0],
       dilations = [1], group = 1 : i64} : tensor<3x8x1xf32>
  return %y : tensor<3x8x1xf32>
}

// -----

// A dilation other than 1 is declined as stated, even though a unit kernel
// would ignore it: the guard follows the attributes, not a reinterpretation.
// CHECK-LABEL: func.func @pw_dilated_stays_conv
// CHECK-NOT:     hip.matmul
// CHECK:         hip.conv
func.func @pw_dilated_stays_conv(%ctx: !hip.context, %x: tensor<1x8x6x6xf32>,
                                 %w: tensor<4x8x1x1xf32>) -> tensor<1x4x6x6xf32> {
  %init = tensor.empty() : tensor<1x4x6x6xf32>
  %y = hip.conv(%ctx)
      ins(%x, %w : tensor<1x8x6x6xf32>, tensor<4x8x1x1xf32>)
      outs(%init : tensor<1x4x6x6xf32>)
      {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
       dilations = [2, 2], group = 1 : i64} : tensor<1x4x6x6xf32>
  return %y : tensor<1x4x6x6xf32>
}

// -----

// f64 has no hipBLASLt matmul in hip-ep's runtime: stays a conv.
// CHECK-LABEL: func.func @pw_f64_stays_conv
// CHECK-NOT:     hip.matmul
// CHECK:         hip.conv
func.func @pw_f64_stays_conv(%ctx: !hip.context, %x: tensor<1x8x4x4xf64>,
                             %w: tensor<4x8x1x1xf64>) -> tensor<1x4x4x4xf64> {
  %init = tensor.empty() : tensor<1x4x4x4xf64>
  %y = hip.conv(%ctx)
      ins(%x, %w : tensor<1x8x4x4xf64>, tensor<4x8x1x1xf64>)
      outs(%init : tensor<1x4x4x4xf64>)
      {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
       dilations = [1, 1], group = 1 : i64} : tensor<1x4x4x4xf64>
  return %y : tensor<1x4x4x4xf64>
}
