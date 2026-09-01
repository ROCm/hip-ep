// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// A Conv whose stride equals its kernel partitions the input into disjoint
// patches, which makes it a GEMM. PatchEmbedConvToGemm rewrites it into
// Reshape/Transpose/Gemm at the ONNX level so it reaches the existing
// hipBLASLt Gemm path and never needs a Conv runtime.
//
// Two shapes in the supported model set land here:
//   * Qwen3.5/3.6/3.8 vision.onnx, rank-5, kernel == input spatial, so every
//     output spatial dim is 1 -- one patch per batch element. Both
//     permutations are identities on the linear layout, so this stays on the
//     original two-reshape form with NO transposes. This is the regression
//     half of the test.
//   * gemma3-4b vision.onnx, rank-4, 14x14 stride 14 over 896x896, so 64x64
//     patches per image. 73% of all Conv FLOPs in the supported set. Needs a
//     gather transpose in and a channels-first transpose out.
//
// Also asserts the guards, since a Conv that overlaps, pads, dilates or groups
// is not a patch partition and must fall through to hip.conv.
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // Dummy entry point required by generateModuleMetadata.
  func.func @main_graph(%arg0: tensor<1x3x896x896xf16>) -> tensor<1x3x896x896xf16> {
    return %arg0 : tensor<1x3x896x896xf16>
  }

  // --------------------------------------------------------------------------
  // Qwen rank-5 patch embed: kernel == input spatial, one patch per batch.
  // Regression guard -- must stay on the plain two-reshape form.
  // --------------------------------------------------------------------------
  func.func @qwen_patch_embed(%x: tensor<?x3x2x16x16xf16>,
                              %w: tensor<1152x3x2x16x16xf16>,
                              %b: tensor<1152xf16>)
      -> tensor<?x1152x1x1x1xf16> {
    %y = "onnx.Conv"(%x, %w, %b) {
      kernel_shape = [2, 16, 16],
      strides = [2, 16, 16],
      pads = [0, 0, 0, 0, 0, 0],
      group = 1 : i64
    } : (tensor<?x3x2x16x16xf16>, tensor<1152x3x2x16x16xf16>, tensor<1152xf16>)
      -> tensor<?x1152x1x1x1xf16>
    return %y : tensor<?x1152x1x1x1xf16>
  }

  // CHECK-LABEL: func.func @qwen_patch_embed
  // CHECK-NOT:     hip.conv
  // CHECK-NOT:     hip.transpose
  // CHECK:         hip.gemm
  // CHECK-NOT:     hip.conv
  // CHECK:         return

  // --------------------------------------------------------------------------
  // gemma3-4b rank-4 patch embed: 14x14 stride 14 over 896x896 -> 64x64
  // patches. Needs the gather transpose in and the channels-first transpose
  // out; C*kh*kw = 3*14*14 = 588 is the contraction extent.
  // --------------------------------------------------------------------------
  func.func @gemma3_patch_embed(%x: tensor<1x3x896x896xf16>,
                                %w: tensor<1152x3x14x14xf16>,
                                %b: tensor<1152xf16>)
      -> tensor<1x1152x64x64xf16> {
    %y = "onnx.Conv"(%x, %w, %b) {
      kernel_shape = [14, 14],
      strides = [14, 14],
      pads = [0, 0, 0, 0],
      group = 1 : i64
    } : (tensor<1x3x896x896xf16>, tensor<1152x3x14x14xf16>, tensor<1152xf16>)
      -> tensor<1x1152x64x64xf16>
    return %y : tensor<1x1152x64x64xf16>
  }

  // CHECK-LABEL: func.func @gemma3_patch_embed
  // CHECK-NOT:     hip.conv
  // CHECK:         hip.gemm
  // CHECK-NOT:     hip.conv
  // CHECK:         return

  // --------------------------------------------------------------------------
  // Same shape without a bias: the pattern synthesizes a zero C operand
  // because the Gemm conversion needs three inputs.
  // --------------------------------------------------------------------------
  func.func @gemma3_patch_embed_no_bias(%x: tensor<1x3x896x896xf16>,
                                        %w: tensor<1152x3x14x14xf16>)
      -> tensor<1x1152x64x64xf16> {
    %y = "onnx.Conv"(%x, %w) {
      kernel_shape = [14, 14],
      strides = [14, 14],
      pads = [0, 0, 0, 0],
      group = 1 : i64
    } : (tensor<1x3x896x896xf16>, tensor<1152x3x14x14xf16>)
      -> tensor<1x1152x64x64xf16>
    return %y : tensor<1x1152x64x64xf16>
  }

  // CHECK-LABEL: func.func @gemma3_patch_embed_no_bias
  // CHECK-NOT:     hip.conv
  // CHECK:         hip.gemm
  // CHECK-NOT:     hip.conv
  // CHECK:         return

  // --------------------------------------------------------------------------
  // Dynamic batch on the multi-patch path: N flows through the reshapes as -1
  // while the patch grid stays static.
  // --------------------------------------------------------------------------
  func.func @gemma3_patch_embed_dyn_batch(%x: tensor<?x3x896x896xf16>,
                                          %w: tensor<1152x3x14x14xf16>,
                                          %b: tensor<1152xf16>)
      -> tensor<?x1152x64x64xf16> {
    %y = "onnx.Conv"(%x, %w, %b) {
      kernel_shape = [14, 14],
      strides = [14, 14],
      pads = [0, 0, 0, 0],
      group = 1 : i64
    } : (tensor<?x3x896x896xf16>, tensor<1152x3x14x14xf16>, tensor<1152xf16>)
      -> tensor<?x1152x64x64xf16>
    return %y : tensor<?x1152x64x64xf16>
  }

  // CHECK-LABEL: func.func @gemma3_patch_embed_dyn_batch
  // CHECK-NOT:     hip.conv
  // CHECK:         hip.gemm
  // CHECK-NOT:     hip.conv
  // CHECK:         return

  // --------------------------------------------------------------------------
  // auto_pad = VALID with no `pads` attribute at all. This is exactly how the
  // gemma3-4b patch embed is exported, and VALID means zero padding, so it has
  // to be accepted rather than treated as an unknown padding mode.
  // --------------------------------------------------------------------------
  func.func @gemma3_patch_embed_auto_pad_valid(%x: tensor<1x3x896x896xf16>,
                                               %w: tensor<1152x3x14x14xf16>,
                                               %b: tensor<1152xf16>)
      -> tensor<1x1152x64x64xf16> {
    %y = "onnx.Conv"(%x, %w, %b) {
      auto_pad = "VALID",
      dilations = [1, 1],
      kernel_shape = [14, 14],
      strides = [14, 14],
      group = 1 : i64
    } : (tensor<1x3x896x896xf16>, tensor<1152x3x14x14xf16>, tensor<1152xf16>)
      -> tensor<1x1152x64x64xf16>
    return %y : tensor<1x1152x64x64xf16>
  }

  // CHECK-LABEL: func.func @gemma3_patch_embed_auto_pad_valid
  // CHECK-NOT:     hip.conv
  // CHECK:         hip.gemm
  // CHECK-NOT:     hip.conv
  // CHECK:         return

  // --------------------------------------------------------------------------
  // GUARD: stride < kernel. Patches overlap, so no reshape expresses the
  // gather. Must stay on hip.conv.
  // --------------------------------------------------------------------------
  func.func @guard_overlapping(%x: tensor<1x3x896x896xf16>,
                               %w: tensor<1152x3x14x14xf16>,
                               %b: tensor<1152xf16>)
      -> tensor<1x1152x127x127xf16> {
    %y = "onnx.Conv"(%x, %w, %b) {
      kernel_shape = [14, 14],
      strides = [7, 7],
      pads = [0, 0, 0, 0],
      group = 1 : i64
    } : (tensor<1x3x896x896xf16>, tensor<1152x3x14x14xf16>, tensor<1152xf16>)
      -> tensor<1x1152x127x127xf16>
    return %y : tensor<1x1152x127x127xf16>
  }

  // CHECK-LABEL: func.func @guard_overlapping
  // CHECK:         hip.conv

  // --------------------------------------------------------------------------
  // GUARD: nonzero pads. A padded patch is not a slice of the input.
  // --------------------------------------------------------------------------
  func.func @guard_padded(%x: tensor<1x3x892x892xf16>,
                          %w: tensor<1152x3x14x14xf16>,
                          %b: tensor<1152xf16>)
      -> tensor<1x1152x64x64xf16> {
    %y = "onnx.Conv"(%x, %w, %b) {
      kernel_shape = [14, 14],
      strides = [14, 14],
      pads = [2, 2, 2, 2],
      group = 1 : i64
    } : (tensor<1x3x892x892xf16>, tensor<1152x3x14x14xf16>, tensor<1152xf16>)
      -> tensor<1x1152x64x64xf16>
    return %y : tensor<1x1152x64x64xf16>
  }

  // CHECK-LABEL: func.func @guard_padded
  // CHECK:         hip.conv

  // --------------------------------------------------------------------------
  // GUARD: the input is not an exact multiple of the kernel, so the trailing
  // partial patch would be dropped -- a Slice, not a Reshape.
  // --------------------------------------------------------------------------
  func.func @guard_ragged(%x: tensor<1x3x900x900xf16>,
                          %w: tensor<1152x3x14x14xf16>,
                          %b: tensor<1152xf16>)
      -> tensor<1x1152x64x64xf16> {
    %y = "onnx.Conv"(%x, %w, %b) {
      kernel_shape = [14, 14],
      strides = [14, 14],
      pads = [0, 0, 0, 0],
      group = 1 : i64
    } : (tensor<1x3x900x900xf16>, tensor<1152x3x14x14xf16>, tensor<1152xf16>)
      -> tensor<1x1152x64x64xf16>
    return %y : tensor<1x1152x64x64xf16>
  }

  // CHECK-LABEL: func.func @guard_ragged
  // CHECK:         hip.conv

  // --------------------------------------------------------------------------
  // GUARD: grouped. Each group contracts over C/group, not C, so a single
  // dense GEMM is the wrong arithmetic.
  // --------------------------------------------------------------------------
  func.func @guard_grouped(%x: tensor<1x6x896x896xf16>,
                           %w: tensor<1152x3x14x14xf16>,
                           %b: tensor<1152xf16>)
      -> tensor<1x1152x64x64xf16> {
    %y = "onnx.Conv"(%x, %w, %b) {
      kernel_shape = [14, 14],
      strides = [14, 14],
      pads = [0, 0, 0, 0],
      group = 2 : i64
    } : (tensor<1x6x896x896xf16>, tensor<1152x3x14x14xf16>, tensor<1152xf16>)
      -> tensor<1x1152x64x64xf16>
    return %y : tensor<1x1152x64x64xf16>
  }

  // CHECK-LABEL: func.func @guard_grouped
  // CHECK:         hip.conv

  // --------------------------------------------------------------------------
  // GUARD: dilated. The patch is strided within the input, not contiguous.
  // Previously this was rejected only implicitly, by output spatial == 1.
  // --------------------------------------------------------------------------
  func.func @guard_dilated(%x: tensor<1x3x64x64xf16>,
                           %w: tensor<1152x3x2x2xf16>,
                           %b: tensor<1152xf16>)
      -> tensor<1x1152x31x31xf16> {
    %y = "onnx.Conv"(%x, %w, %b) {
      kernel_shape = [2, 2],
      strides = [2, 2],
      dilations = [2, 2],
      pads = [0, 0, 0, 0],
      group = 1 : i64
    } : (tensor<1x3x64x64xf16>, tensor<1152x3x2x2xf16>, tensor<1152xf16>)
      -> tensor<1x1152x31x31xf16>
    return %y : tensor<1x1152x31x31xf16>
  }

  // CHECK-LABEL: func.func @guard_dilated
  // CHECK:         hip.conv

  // --------------------------------------------------------------------------
  // GUARD: auto_pad = SAME_UPPER. It overrides the `pads` the pattern reads,
  // and it means nonzero padding, so the patches would not be plain slices.
  // --------------------------------------------------------------------------
  func.func @guard_auto_pad(%x: tensor<1x3x896x896xf16>,
                            %w: tensor<1152x3x14x14xf16>,
                            %b: tensor<1152xf16>)
      -> tensor<1x1152x64x64xf16> {
    %y = "onnx.Conv"(%x, %w, %b) {
      auto_pad = "SAME_UPPER",
      kernel_shape = [14, 14],
      strides = [14, 14],
      group = 1 : i64
    } : (tensor<1x3x896x896xf16>, tensor<1152x3x14x14xf16>, tensor<1152xf16>)
      -> tensor<1x1152x64x64xf16>
    return %y : tensor<1x1152x64x64xf16>
  }

  // CHECK-LABEL: func.func @guard_auto_pad
  // CHECK:         hip.conv

  // --------------------------------------------------------------------------
  // GUARD: 1x1 kernel, stride 1. It satisfies `stride == kernel` and tiles the
  // input, but the patch is a single pixel: the gather path's two transposes
  // would be a plain NCHW <-> NHWC round trip around a contraction that is
  // already a GEMM over C, so the rewrite is pure added traffic. detr's
  // ResNet-50 backbone has 33 of these and converting them cost 36% of the
  // model's runtime. They belong on hip.conv, which contracts over K = C with
  // no data movement.
  // --------------------------------------------------------------------------
  func.func @guard_unit_kernel(%x: tensor<1x256x200x272xf16>,
                               %w: tensor<64x256x1x1xf16>,
                               %b: tensor<64xf16>)
      -> tensor<1x64x200x272xf16> {
    %y = "onnx.Conv"(%x, %w, %b) {
      kernel_shape = [1, 1],
      strides = [1, 1],
      pads = [0, 0, 0, 0],
      group = 1 : i64
    } : (tensor<1x256x200x272xf16>, tensor<64x256x1x1xf16>, tensor<64xf16>)
      -> tensor<1x64x200x272xf16>
    return %y : tensor<1x64x200x272xf16>
  }

  // CHECK-LABEL: func.func @guard_unit_kernel
  // CHECK-NOT:     hip.transpose
  // CHECK:         hip.conv

  // --------------------------------------------------------------------------
  // The unit-kernel guard is scoped to the gather path. A 1x1 kernel over a 1x1
  // input still has exactly one patch, so both permutations are identities that
  // are never emitted and the rewrite is a free reshape into a GEMM. Keep it.
  // --------------------------------------------------------------------------
  func.func @unit_kernel_single_patch(%x: tensor<1x1152x1x1xf16>,
                                      %w: tensor<512x1152x1x1xf16>,
                                      %b: tensor<512xf16>)
      -> tensor<1x512x1x1xf16> {
    %y = "onnx.Conv"(%x, %w, %b) {
      kernel_shape = [1, 1],
      strides = [1, 1],
      pads = [0, 0, 0, 0],
      group = 1 : i64
    } : (tensor<1x1152x1x1xf16>, tensor<512x1152x1x1xf16>, tensor<512xf16>)
      -> tensor<1x512x1x1xf16>
    return %y : tensor<1x512x1x1xf16>
  }

  // CHECK-LABEL: func.func @unit_kernel_single_patch
  // CHECK-NOT:     hip.conv
  // CHECK:         hip.gemm
  // CHECK-NOT:     hip.conv

  // --------------------------------------------------------------------------
  // The guard must not catch a real patch embed with a small patch: convnext's
  // downsample stages are 2x2 stride 2, which is four elements per patch and
  // still a genuine partition of the input.
  // --------------------------------------------------------------------------
  func.func @convnext_downsample(%x: tensor<1x192x56x56xf16>,
                                 %w: tensor<384x192x2x2xf16>,
                                 %b: tensor<384xf16>)
      -> tensor<1x384x28x28xf16> {
    %y = "onnx.Conv"(%x, %w, %b) {
      kernel_shape = [2, 2],
      strides = [2, 2],
      pads = [0, 0, 0, 0],
      group = 1 : i64
    } : (tensor<1x192x56x56xf16>, tensor<384x192x2x2xf16>, tensor<384xf16>)
      -> tensor<1x384x28x28xf16>
    return %y : tensor<1x384x28x28xf16>
  }

  // CHECK-LABEL: func.func @convnext_downsample
  // CHECK-NOT:     hip.conv
  // CHECK:         hip.gemm
  // CHECK-NOT:     hip.conv
}
