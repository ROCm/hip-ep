/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- patch_embed_conv_to_gemm.hpp - hip.conv that is really a GEMM ------===//
//
// A `hip.conv` whose stride equals its kernel on every spatial axis, with no
// padding, no dilation and no grouping, tiles its input into disjoint patches
// and contracts each against every filter -- which is a GEMM:
//
//   one patch per batch element (kernel covers the whole input)
//     hip.conv -> collapse_shape x2, hip.gemm, expand_shape
//   several patches, whose rows need gathering and result reordering
//     hip.conv -> expand_shape, hip.transpose, collapse_shape x2, hip.gemm,
//                 expand_shape, hip.transpose
//
// A 1x1 kernel is excluded: it meets `stride == kernel` without being a patch
// embed, and rewriting it only adds data movement. Declining is always safe,
// since `hip.conv` has a runtime for every rank it can hold.
//
// Before/After IR: patch_embed_conv_to_gemm.cpp.
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/IR/PatternMatch.h"

namespace hip {
namespace fusion_transform {

void populatePatchEmbedConvToGemmPattern(mlir::RewritePatternSet &patterns,
                                         mlir::PatternBenefit benefit);

} // namespace fusion_transform
} // namespace hip
