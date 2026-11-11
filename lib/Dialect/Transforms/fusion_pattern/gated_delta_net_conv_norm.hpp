/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- gated_delta_net_conv_norm.hpp - linear-attention core to one op ----===//
//
// Collapse the chain between the linear-attention input projections and the
// output projection
//
//   %x    = tensor.extract_slice %proj[.., 0] [.., C]
//   %c:2  = hip.causal_conv_with_state(%x, ...)     (SiLU, channels-last, or
//                                                    bracketed by transposes)
//   %d    = hip.cast(tensor.extract_slice %gates[.., dc] [.., Hv])  (to f32)
//   %b    = hip.cast(tensor.extract_slice %gates[.., bc] [.., Hv])  (to f32)
//   %o:2  = hip.gated_delta_net(%c#0, %d, %b, ...)
//   %z    = tensor.extract_slice %proj[.., zc] [.., Hv * dv]
//   %y    = hip.gated_rms_norm(collapse(%o#0), %w, %z)  (sigmoid)
//
// into one hip.gated_delta_net_conv_norm whose results replace %y, %c#1 and
// %o#1, so the runtime can run the decode case as one kernel.
//
// Before/After IR: gated_delta_net_conv_norm.cpp.
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/IR/PatternMatch.h"

namespace hip {
namespace fusion_transform {

void populateGatedDeltaNetConvNormPattern(mlir::RewritePatternSet &patterns,
                                          mlir::PatternBenefit benefit);

} // namespace fusion_transform
} // namespace hip
