/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- linear_attention_group_fold.hpp - drop grouped q/k broadcasts
//-------===//
//
// Feed hip.linear_attention the grouped query/key instead of a copy that
// repeats each group across the heads sharing it:
//
//   %e = tensor.expand_shape %x [[0], [1], [2, 3, 4]] : [B,T,G*D] ->
//   [B,T,G,1,D] %r = hip.expand(%ctx) ins(%e, %shape) : -> [B,T,G,R,D] %c =
//   tensor.collapse_shape %r [[0], [1], [2, 3, 4]] : -> [B,T,G*R*D]
//
// with G*R == kv_num_heads. hip.linear_attention maps state head h to key
// head h*n_k/H_kv and query head h*H_q/H_kv, which is h/R for both, so the
// result is unchanged. Folding the query also sets q_num_heads to G.
//
// Before/After IR: linear_attention_group_fold.cpp.
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/IR/PatternMatch.h"

namespace hip {
namespace fusion_transform {

void populateLinearAttentionGroupFoldPattern(mlir::RewritePatternSet &patterns,
                                             mlir::PatternBenefit benefit);

} // namespace fusion_transform
} // namespace hip
