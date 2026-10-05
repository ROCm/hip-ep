/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_CONVERSION_RESIZELAYOUT_H
#define HIP_CONVERSION_RESIZELAYOUT_H

#include "mlir/IR/BuiltinTypes.h"

#include <optional>

namespace mlir {
namespace hip {

// How a ranked resize is passed to the existing runtime kernel. That kernel
// is (N, C, s0, s1, s2): the prefix axes are copied, and a trailing window of
// `spatialRank` axes (1..3) is resampled. A copied axis may sit in the window;
// equal extents make that sample an identity, so rank-4 NHWC
// `1xHxWxC -> 1xOHxOWxC` is `N`, `C = 1`, `spatialRank = 3`, window `(H, W,
// C)`.
struct HipResizeLaunch {
  int64_t prefixCount;
  int64_t spatialRank;
};

// The window is the shortest suffix that contains every statically resized
// axis. Axes before it are the copied prefix and must number at most 2. A
// dynamic axis is legal only in that prefix, and only when both the input and
// the output axis are dynamic (the output copies the input extent).
inline std::optional<HipResizeLaunch>
planHipResizeLaunch(ShapedType inputType, ShapedType outputType) {
  if (!inputType.hasRank() || !outputType.hasRank())
    return std::nullopt;
  const int64_t rank = inputType.getRank();
  if (rank < 3 || rank > 5 || outputType.getRank() != rank)
    return std::nullopt;

  // true: copied. false: statically resized. nullopt: not expressible.
  auto axis = [&](int64_t i) -> std::optional<bool> {
    const bool inDyn = inputType.isDynamicDim(i);
    const bool outDyn = outputType.isDynamicDim(i);
    if (inDyn || outDyn)
      return (inDyn && outDyn) ? std::optional<bool>(true) : std::nullopt;
    return inputType.getDimSize(i) == outputType.getDimSize(i);
  };

  int64_t firstResized = rank;
  for (int64_t i = 0; i < rank; ++i) {
    std::optional<bool> copied = axis(i);
    if (!copied)
      return std::nullopt;
    if (!*copied) {
      firstResized = i;
      break;
    }
  }

  int64_t prefixCount = std::min<int64_t>(2, firstResized);
  int64_t spatialRank = rank - prefixCount;
  if (prefixCount > 2 || spatialRank < 1 || spatialRank > 3)
    return std::nullopt;

  for (int64_t i = prefixCount; i < rank; ++i) {
    if (inputType.isDynamicDim(i) || outputType.isDynamicDim(i))
      return std::nullopt;
  }
  return HipResizeLaunch{prefixCount, spatialRank};
}

} // namespace hip
} // namespace mlir

#endif // HIP_CONVERSION_RESIZELAYOUT_H
