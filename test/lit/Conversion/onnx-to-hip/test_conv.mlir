// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX Conv is correctly lowered to hip.conv in tensor-first mode.
//
// Test cases:
// 1. conv_basic          — standard 2D conv with bias (7x7 kernel, stride 2)
// 2. conv_grouped        — grouped conv (group=2)
// 3. conv_depthwise      — depthwise conv (group=channels)
// 4. conv_stride2        — strided conv (stride=2)
// 5. conv_asymmetric_stride — asymmetric stride [2,3]
// 6. conv_dynamic_spatial — dynamic batch + dynamic spatial output dim
// 7. conv_3d               — rank-5 NCDHW, depth kernel 3, spatial kernel 1x1
// 8. conv_rank6            — left unconverted (no runtime path)
// 9. conv_inferred_kernel_shape — kernel_shape omitted, inferred from weights
// 10. conv_dynamic_weights_no_kernel_shape — nothing to infer, unconverted
// 11. conv_static_result_no_kernel_shape — same, with a static result
// 12. conv_result_outranks_input — result rank > input rank, unconverted
// 13. conv_same_upper_no_kernel_shape — SAME_UPPER resolved to explicit pads
// 14. conv_auto_pad_valid — auto_pad VALID zeroes the explicit pads
// 15. conv_same_lower_dynamic — SAME_* needs static extents, unconverted
// 16. conv_same_lower_odd_pad — odd budget, extra pad at the begin
// 17. conv_same_upper_odd_pad — same dims, extra pad at the end
// 18. conv_short_strides — strides arity under the spatial rank, unconverted
// 19. conv_short_pads — pads arity under twice the spatial rank, unconverted
//
// Note: conv without bias requires onnx.NoValue syntax which the current
// ConvToHipPattern does not guard against NoneType operands; tracked separately.
//
// All cases assert:
// - context argument prepended
// - tensor.empty() for output init (no hip.alloc)
// - all Conv attributes forwarded (kernel_shape, strides, pads, dilations, group)
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // Dummy entry point required by generateModuleMetadata.
  func.func @main_graph(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xf32> {
    return %arg0 : tensor<1x3x224x224xf32>
  }

  // --------------------------------------------------------------------------
  // 1. Basic conv with bias
  // --------------------------------------------------------------------------
  func.func @conv_basic(%input: tensor<1x3x224x224xf32>, %weights: tensor<64x3x7x7xf32>, %bias: tensor<64xf32>) -> tensor<1x64x112x112xf32> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [7, 7],
      strides = [2, 2],
      pads = [3, 3, 3, 3],
      dilations = [1, 1],
      group = 1 : i64
    } : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
    return %output : tensor<1x64x112x112xf32>
  }

  // CHECK-LABEL: func.func @conv_basic
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<1x3x224x224xf32>, %[[W:.*]]: tensor<64x3x7x7xf32>, %[[B:.*]]: tensor<64xf32>) -> tensor<1x64x112x112xf32>
  // CHECK: tensor.empty() : tensor<1x64x112x112xf32>
  // CHECK: hip.conv(%[[CTX]]) ins(%[[IN]], %[[W]], %[[B]] : tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>) outs({{.*}} : tensor<1x64x112x112xf32>) {dilations = [1, 1], group = 1 : i64, kernel_shape = [7, 7], pads = [3, 3, 3, 3], strides = [2, 2]}
  // CHECK-NOT: hip.alloc

  // --------------------------------------------------------------------------
  // 2. Grouped conv (group=2)
  // --------------------------------------------------------------------------
  func.func @conv_grouped(%input: tensor<1x64x56x56xf32>, %weights: tensor<128x32x3x3xf32>, %bias: tensor<128xf32>) -> tensor<1x128x56x56xf32> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [3, 3],
      strides = [1, 1],
      pads = [1, 1, 1, 1],
      dilations = [1, 1],
      group = 2 : i64
    } : (tensor<1x64x56x56xf32>, tensor<128x32x3x3xf32>, tensor<128xf32>) -> tensor<1x128x56x56xf32>
    return %output : tensor<1x128x56x56xf32>
  }

  // CHECK-LABEL: func.func @conv_grouped
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<1x64x56x56xf32>, %[[W:.*]]: tensor<128x32x3x3xf32>, %[[B:.*]]: tensor<128xf32>) -> tensor<1x128x56x56xf32>
  // CHECK: tensor.empty() : tensor<1x128x56x56xf32>
  // CHECK: hip.conv(%[[CTX]]) ins(%[[IN]], %[[W]], %[[B]] : tensor<1x64x56x56xf32>, tensor<128x32x3x3xf32>, tensor<128xf32>) outs({{.*}} : tensor<1x128x56x56xf32>) {dilations = [1, 1], group = 2 : i64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]}
  // CHECK-NOT: hip.alloc

  // --------------------------------------------------------------------------
  // 3. Depthwise conv (group = num_channels)
  // --------------------------------------------------------------------------
  func.func @conv_depthwise(%input: tensor<1x64x56x56xf32>, %weights: tensor<64x1x3x3xf32>, %bias: tensor<64xf32>) -> tensor<1x64x56x56xf32> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [3, 3],
      strides = [1, 1],
      pads = [1, 1, 1, 1],
      dilations = [1, 1],
      group = 64 : i64
    } : (tensor<1x64x56x56xf32>, tensor<64x1x3x3xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>
    return %output : tensor<1x64x56x56xf32>
  }

  // CHECK-LABEL: func.func @conv_depthwise
  // CHECK-SAME: !hip.context
  // CHECK: hip.conv({{.*}}) ins({{.*}}) outs({{.*}}) {dilations = [1, 1], group = 64 : i64, kernel_shape = [3, 3]
  // CHECK-NOT: hip.alloc

  // --------------------------------------------------------------------------
  // 4. Strided conv (stride=2)
  // --------------------------------------------------------------------------
  func.func @conv_stride2(%input: tensor<1x64x56x56xf32>, %weights: tensor<128x64x3x3xf32>, %bias: tensor<128xf32>) -> tensor<1x128x28x28xf32> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [3, 3],
      strides = [2, 2],
      pads = [1, 1, 1, 1],
      dilations = [1, 1],
      group = 1 : i64
    } : (tensor<1x64x56x56xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    return %output : tensor<1x128x28x28xf32>
  }

  // CHECK-LABEL: func.func @conv_stride2
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<1x64x56x56xf32>, %[[W:.*]]: tensor<128x64x3x3xf32>, %[[B:.*]]: tensor<128xf32>) -> tensor<1x128x28x28xf32>
  // CHECK: tensor.empty() : tensor<1x128x28x28xf32>
  // CHECK: hip.conv(%[[CTX]]) ins(%[[IN]], %[[W]], %[[B]] : tensor<1x64x56x56xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>) outs({{.*}} : tensor<1x128x28x28xf32>) {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2]}
  // CHECK-NOT: hip.alloc

  // --------------------------------------------------------------------------
  // 5. Asymmetric stride [2, 3]
  // --------------------------------------------------------------------------
  func.func @conv_asymmetric_stride(%input: tensor<1x3x224x224xf32>, %weights: tensor<64x3x7x3xf32>, %bias: tensor<64xf32>) -> tensor<1x64x112x74xf32> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [7, 3],
      strides = [2, 3],
      pads = [3, 1, 3, 1],
      dilations = [1, 1],
      group = 1 : i64
    } : (tensor<1x3x224x224xf32>, tensor<64x3x7x3xf32>, tensor<64xf32>) -> tensor<1x64x112x74xf32>
    return %output : tensor<1x64x112x74xf32>
  }

  // CHECK-LABEL: func.func @conv_asymmetric_stride
  // CHECK-SAME: !hip.context
  // CHECK: hip.conv({{.*}}) ins({{.*}}) outs({{.*}}) {dilations = [1, 1], group = 1 : i64, kernel_shape = [7, 3], pads = [3, 1, 3, 1], strides = [2, 3]}
  // CHECK-NOT: hip.alloc

  // --------------------------------------------------------------------------
  // 6. Dynamic batch + dynamic spatial output dim (down-sampling front-end).
  //    Output H is sized at runtime from the conv formula:
  //      H' = (H + pad_begin + pad_end - dilation*(kernel-1) - 1)/stride + 1
  //    The static W dim (64) stays in the result type; only N and H are dynamic.
  // --------------------------------------------------------------------------
  func.func @conv_dynamic_spatial(%input: tensor<?x1x?x128xf16>, %weights: tensor<128x1x3x3xf16>, %bias: tensor<128xf16>) -> tensor<?x128x?x64xf16> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [3, 3],
      strides = [2, 2],
      pads = [1, 1, 1, 1],
      dilations = [1, 1],
      group = 1 : i64
    } : (tensor<?x1x?x128xf16>, tensor<128x1x3x3xf16>, tensor<128xf16>) -> tensor<?x128x?x64xf16>
    return %output : tensor<?x128x?x64xf16>
  }

  // CHECK-LABEL: func.func @conv_dynamic_spatial
  // CHECK-SAME: !hip.context
  // Spatial output dim resolved from tensor.dim of the input + arith, NOT the
  // conv result (would be a use-before-def).
  // CHECK: arith.divsi
  // CHECK: tensor.empty(%{{.*}}, %{{.*}}) : tensor<?x128x?x64xf16>
  // CHECK: hip.conv({{.*}}) outs({{.*}} : tensor<?x128x?x64xf16>) {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2]}
  // CHECK-NOT: hip.alloc

  // --------------------------------------------------------------------------
  // 7. Rank-5 NCDHW. Depth kernel 3 with pad 1 keeps D; the 1x1 spatial kernel
  //    with zero spatial pad keeps H and W. This is an overlapping conv, so it
  //    must become hip.conv rather than the patch-embed GEMM.
  // --------------------------------------------------------------------------
  func.func @conv_3d(%input: tensor<1x4x6x4x4xf16>, %weights: tensor<4x4x3x1x1xf16>, %bias: tensor<4xf16>) -> tensor<1x4x6x4x4xf16> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [3, 1, 1],
      strides = [1, 1, 1],
      pads = [1, 0, 0, 1, 0, 0],
      dilations = [1, 1, 1],
      group = 1 : i64
    } : (tensor<1x4x6x4x4xf16>, tensor<4x4x3x1x1xf16>, tensor<4xf16>) -> tensor<1x4x6x4x4xf16>
    return %output : tensor<1x4x6x4x4xf16>
  }

  // CHECK-LABEL: func.func @conv_3d
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<1x4x6x4x4xf16>, %[[W:.*]]: tensor<4x4x3x1x1xf16>, %[[B:.*]]: tensor<4xf16>) -> tensor<1x4x6x4x4xf16>
  // CHECK-NOT: hip.gemm
  // CHECK: tensor.empty() : tensor<1x4x6x4x4xf16>
  // CHECK: hip.conv(%[[CTX]]) ins(%[[IN]], %[[W]], %[[B]] : tensor<1x4x6x4x4xf16>, tensor<4x4x3x1x1xf16>, tensor<4xf16>) outs({{.*}} : tensor<1x4x6x4x4xf16>) {dilations = [1, 1, 1], group = 1 : i64, kernel_shape = [3, 1, 1], pads = [1, 0, 0, 1, 0, 0], strides = [1, 1, 1]}
  // CHECK-NOT: hip.alloc

  // --------------------------------------------------------------------------
  // 8. Rank 6 has no kernel path and must stay onnx.Conv.
  // --------------------------------------------------------------------------
  func.func @conv_rank6(%input: tensor<1x1x2x2x2x2xf32>, %weights: tensor<1x1x1x1x1x1xf32>, %bias: tensor<1xf32>) -> tensor<1x1x2x2x2x2xf32> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [1, 1, 1, 1],
      strides = [1, 1, 1, 1],
      pads = [0, 0, 0, 0, 0, 0, 0, 0],
      dilations = [1, 1, 1, 1],
      group = 1 : i64
    } : (tensor<1x1x2x2x2x2xf32>, tensor<1x1x1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x2x2x2x2xf32>
    return %output : tensor<1x1x2x2x2x2xf32>
  }

  // CHECK-LABEL: func.func @conv_rank6
  // CHECK-SAME: !hip.context
  // CHECK: onnx.Conv
  // CHECK-NOT: hip.conv

  // --------------------------------------------------------------------------
  // 9. kernel_shape omitted, which ONNX permits, combined with dynamic spatial
  //    dims. Resolving the dynamic output extents needs the kernel extents, so
  //    they are inferred from the weights' trailing dims ([9, 9] here) and the
  //    emitted hip.conv carries them explicitly.
  // --------------------------------------------------------------------------
  func.func @conv_inferred_kernel_shape(%input: tensor<?x3x?x?xf32>, %weights: tensor<32x3x9x9xf32>, %bias: tensor<32xf32>) -> tensor<?x32x?x?xf32> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      auto_pad = "NOTSET",
      strides = [1, 1],
      pads = [0, 0, 0, 0],
      dilations = [1, 1],
      group = 1 : i64
    } : (tensor<?x3x?x?xf32>, tensor<32x3x9x9xf32>, tensor<32xf32>) -> tensor<?x32x?x?xf32>
    return %output : tensor<?x32x?x?xf32>
  }

  // CHECK-LABEL: func.func @conv_inferred_kernel_shape
  // CHECK-SAME: !hip.context
  // CHECK: arith.divsi
  // CHECK: tensor.empty(%{{.*}}, %{{.*}}, %{{.*}}) : tensor<?x32x?x?xf32>
  // CHECK: hip.conv({{.*}}) outs({{.*}} : tensor<?x32x?x?xf32>) {dilations = [1, 1], group = 1 : i64, kernel_shape = [9, 9], pads = [0, 0, 0, 0], strides = [1, 1]}
  // CHECK-NOT: onnx.Conv

  // --------------------------------------------------------------------------
  // 10. kernel_shape omitted and the weights' spatial dims are dynamic too, so
  //     there is nothing to infer from. The dynamic output extents cannot be
  //     computed, so the op stays unconverted rather than guessing.
  // --------------------------------------------------------------------------
  func.func @conv_dynamic_weights_no_kernel_shape(%input: tensor<?x3x?x?xf32>, %weights: tensor<32x3x?x?xf32>, %bias: tensor<32xf32>) -> tensor<?x32x?x?xf32> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      auto_pad = "NOTSET",
      strides = [1, 1],
      pads = [0, 0, 0, 0],
      dilations = [1, 1],
      group = 1 : i64
    } : (tensor<?x3x?x?xf32>, tensor<32x3x?x?xf32>, tensor<32xf32>) -> tensor<?x32x?x?xf32>
    return %output : tensor<?x32x?x?xf32>
  }

  // CHECK-LABEL: func.func @conv_dynamic_weights_no_kernel_shape
  // CHECK-SAME: !hip.context
  // CHECK: onnx.Conv
  // CHECK-NOT: hip.conv

  // --------------------------------------------------------------------------
  // 11. Same uninferable kernel_shape as case 10, but with a fully static
  //     result. Nothing needs a runtime extent here, so the refusal cannot
  //     come from the dynamic-sizing path; without an arity check of its own
  //     the op would reach hip.conv carrying an empty kernel_shape, and
  //     ConvLowering reports that with emitError rather than a match failure,
  //     turning a declinable op into a failed compile.
  // --------------------------------------------------------------------------
  func.func @conv_static_result_no_kernel_shape(%input: tensor<1x3x32x32xf32>, %weights: tensor<32x3x?x?xf32>, %bias: tensor<32xf32>) -> tensor<1x32x24x24xf32> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      auto_pad = "NOTSET",
      strides = [1, 1],
      pads = [0, 0, 0, 0],
      dilations = [1, 1],
      group = 1 : i64
    } : (tensor<1x3x32x32xf32>, tensor<32x3x?x?xf32>, tensor<32xf32>) -> tensor<1x32x24x24xf32>
    return %output : tensor<1x32x24x24xf32>
  }

  // CHECK-LABEL: func.func @conv_static_result_no_kernel_shape
  // CHECK-SAME: !hip.context
  // CHECK: onnx.Conv
  // CHECK-NOT: hip.conv

  // --------------------------------------------------------------------------
  // 12. A result that outranks its input. onnx.Conv does not verify rank
  //     agreement, so this parses, and the trailing dim being dynamic is what
  //     makes it dangerous: the sizing loop walks the RESULT rank and indexes
  //     the per-spatial-axis attributes by (dim - 2), which reads past them
  //     for dim 4 when the input only has two spatial axes. Must be refused
  //     before the loop runs.
  // --------------------------------------------------------------------------
  func.func @conv_result_outranks_input(%input: tensor<1x3x32x32xf32>, %weights: tensor<32x3x3x3xf32>, %bias: tensor<32xf32>) -> tensor<1x32x30x30x?xf32> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [3, 3],
      strides = [1, 1],
      pads = [0, 0, 0, 0],
      dilations = [1, 1],
      group = 1 : i64
    } : (tensor<1x3x32x32xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x32x30x30x?xf32>
    return %output : tensor<1x32x30x30x?xf32>
  }

  // CHECK-LABEL: func.func @conv_result_outranks_input
  // CHECK-SAME: !hip.context
  // CHECK: onnx.Conv
  // CHECK-NOT: hip.conv

  // --------------------------------------------------------------------------
  // 13. auto_pad = SAME_UPPER with kernel_shape omitted, so the inference
  //     above supplies [3, 3] and the padding still has to come from the
  //     extents. hip.conv carries explicit pads only, so an unresolved mode
  //     would convolve with zero padding and return wrong results. A 5x5
  //     input, 3x3 kernel and stride 2 onto a 3x3 output needs
  //     pad_total = (3-1)*2 + 3 - 5 = 2 per axis, split 1/1.
  // --------------------------------------------------------------------------
  func.func @conv_same_upper_no_kernel_shape(%input: tensor<1x3x5x5xf32>, %weights: tensor<32x3x3x3xf32>, %bias: tensor<32xf32>) -> tensor<1x32x3x3xf32> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      auto_pad = "SAME_UPPER",
      strides = [2, 2],
      dilations = [1, 1],
      group = 1 : i64
    } : (tensor<1x3x5x5xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x32x3x3xf32>
    return %output : tensor<1x32x3x3xf32>
  }

  // CHECK-LABEL: func.func @conv_same_upper_no_kernel_shape
  // CHECK-SAME: !hip.context
  // CHECK: hip.conv({{.*}}) outs({{.*}} : tensor<1x32x3x3xf32>) {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2]}
  // CHECK-NOT: onnx.Conv

  // --------------------------------------------------------------------------
  // 14. auto_pad = VALID means no padding, and it outranks any explicit pads.
  //     The nonzero pads here must not survive into hip.conv.
  // --------------------------------------------------------------------------
  func.func @conv_auto_pad_valid(%input: tensor<1x3x32x32xf32>, %weights: tensor<32x3x3x3xf32>, %bias: tensor<32xf32>) -> tensor<1x32x30x30xf32> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      auto_pad = "VALID",
      kernel_shape = [3, 3],
      strides = [1, 1],
      pads = [1, 1, 1, 1],
      dilations = [1, 1],
      group = 1 : i64
    } : (tensor<1x3x32x32xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x32x30x30xf32>
    return %output : tensor<1x32x30x30xf32>
  }

  // CHECK-LABEL: func.func @conv_auto_pad_valid
  // CHECK-SAME: !hip.context
  // CHECK: hip.conv({{.*}}) outs({{.*}} : tensor<1x32x30x30xf32>) {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]}
  // CHECK-NOT: onnx.Conv

  // --------------------------------------------------------------------------
  // 15. auto_pad = SAME_LOWER with dynamic spatial dims. The pad budget needs
  //     both the input and output extents, so there is nothing to split here
  //     and the op stays put rather than being given zero padding. Note the
  //     odd pad would go to the begin under SAME_LOWER, the mirror of case 13.
  // --------------------------------------------------------------------------
  func.func @conv_same_lower_dynamic(%input: tensor<1x3x?x?xf32>, %weights: tensor<32x3x3x3xf32>, %bias: tensor<32xf32>) -> tensor<1x32x?x?xf32> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      auto_pad = "SAME_LOWER",
      strides = [2, 2],
      dilations = [1, 1],
      group = 1 : i64
    } : (tensor<1x3x?x?xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x32x?x?xf32>
    return %output : tensor<1x32x?x?xf32>
  }

  // CHECK-LABEL: func.func @conv_same_lower_dynamic
  // CHECK-SAME: !hip.context
  // CHECK: onnx.Conv
  // CHECK-NOT: hip.conv

  // --------------------------------------------------------------------------
  // 16/17. The odd-budget mirror pair, which is the only thing that tells
  //     SAME_LOWER and SAME_UPPER apart. Input 4, kernel 3, stride 2 onto a
  //     2-wide output needs pad_total = (2-1)*2 + 3 - 4 = 1 per axis. An even
  //     budget splits evenly and leaves the two modes indistinguishable (see
  //     case 13), so only an odd one pins down which end gets the extra pad:
  //     SAME_LOWER puts it at the begin, SAME_UPPER at the end.
  //
  //     pads are [begin_h, begin_w, end_h, end_w], so 1/0 per axis reads as
  //     [1, 1, 0, 0] for SAME_LOWER and [0, 0, 1, 1] for SAME_UPPER.
  // --------------------------------------------------------------------------
  func.func @conv_same_lower_odd_pad(%input: tensor<1x3x4x4xf32>, %weights: tensor<32x3x3x3xf32>, %bias: tensor<32xf32>) -> tensor<1x32x2x2xf32> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      auto_pad = "SAME_LOWER",
      strides = [2, 2],
      dilations = [1, 1],
      group = 1 : i64
    } : (tensor<1x3x4x4xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x32x2x2xf32>
    return %output : tensor<1x32x2x2xf32>
  }

  // CHECK-LABEL: func.func @conv_same_lower_odd_pad
  // CHECK-SAME: !hip.context
  // CHECK: hip.conv({{.*}}) outs({{.*}} : tensor<1x32x2x2xf32>) {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3], pads = [1, 1, 0, 0], strides = [2, 2]}
  // CHECK-NOT: onnx.Conv

  func.func @conv_same_upper_odd_pad(%input: tensor<1x3x4x4xf32>, %weights: tensor<32x3x3x3xf32>, %bias: tensor<32xf32>) -> tensor<1x32x2x2xf32> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      auto_pad = "SAME_UPPER",
      strides = [2, 2],
      dilations = [1, 1],
      group = 1 : i64
    } : (tensor<1x3x4x4xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x32x2x2xf32>
    return %output : tensor<1x32x2x2xf32>
  }

  // CHECK-LABEL: func.func @conv_same_upper_odd_pad
  // CHECK-SAME: !hip.context
  // CHECK: hip.conv({{.*}}) outs({{.*}} : tensor<1x32x2x2xf32>) {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3], pads = [0, 0, 1, 1], strides = [2, 2]}
  // CHECK-NOT: onnx.Conv

  // --------------------------------------------------------------------------
  // 18. One stride for a two-axis conv. onnx.Conv does not verify attribute
  //     arity, so this parses, and everything else about the op is convertible
  //     — only the arity is wrong. ConvLowering reports the disagreement with
  //     emitError, so forwarding it would fail the compile outright instead of
  //     leaving the op for another EP; refused here, before any IR exists.
  // --------------------------------------------------------------------------
  func.func @conv_short_strides(%input: tensor<1x3x8x8xf32>, %weights: tensor<32x3x3x3xf32>, %bias: tensor<32xf32>) -> tensor<1x32x3x3xf32> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [3, 3],
      strides = [2],
      pads = [0, 0, 0, 0],
      dilations = [1, 1],
      group = 1 : i64
    } : (tensor<1x3x8x8xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x32x3x3xf32>
    return %output : tensor<1x32x3x3xf32>
  }

  // CHECK-LABEL: func.func @conv_short_strides
  // CHECK-SAME: !hip.context
  // CHECK: onnx.Conv
  // CHECK-NOT: hip.conv

  // --------------------------------------------------------------------------
  // 19. pads carries one entry per axis instead of a begin/end pair, so it is
  //     half the length hip.conv needs. auto_pad is absent (NOTSET), which is
  //     the only path where an explicit pads attribute survives to be checked
  //     — VALID and SAME_* overwrite it with a correctly sized vector.
  // --------------------------------------------------------------------------
  func.func @conv_short_pads(%input: tensor<1x3x8x8xf32>, %weights: tensor<32x3x3x3xf32>, %bias: tensor<32xf32>) -> tensor<1x32x8x8xf32> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [3, 3],
      strides = [1, 1],
      pads = [1, 1],
      dilations = [1, 1],
      group = 1 : i64
    } : (tensor<1x3x8x8xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x32x8x8xf32>
    return %output : tensor<1x32x8x8xf32>
  }

  // CHECK-LABEL: func.func @conv_short_pads
  // CHECK-SAME: !hip.context
  // CHECK: onnx.Conv
  // CHECK-NOT: hip.conv
}
