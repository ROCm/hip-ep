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
// axis. Axes before it are the copied prefix and must number at most 2.
// A dynamic axis is legal when both sides are dynamic. In the prefix the
// output copies the input extent. In the window, conversion fills the output
// extent from a constant scale and lowering reads it from the memref
// descriptor. When nothing is statically resized, a trailing static channel
// after a dynamic axis is channels-last (prefix is N). Otherwise the copied
// prefix is the leading two axes.
inline std::optional<HipResizeLaunch>
planHipResizeLaunch(ShapedType inputType, ShapedType outputType) {
  if (!inputType.hasRank() || !outputType.hasRank())
    return std::nullopt;
  const int64_t rank = inputType.getRank();
  if (rank < 3 || rank > 5 || outputType.getRank() != rank)
    return std::nullopt;

  enum class AxisKind { Copied, Resized, Dynamic };
  auto axis = [&](int64_t i) -> std::optional<AxisKind> {
    const bool inDyn = inputType.isDynamicDim(i);
    const bool outDyn = outputType.isDynamicDim(i);
    if (inDyn || outDyn)
      return (inDyn && outDyn) ? std::optional<AxisKind>(AxisKind::Dynamic)
                               : std::nullopt;
    return inputType.getDimSize(i) == outputType.getDimSize(i)
               ? AxisKind::Copied
               : AxisKind::Resized;
  };

  AxisKind kinds[5];
  int64_t firstResized = rank;
  int64_t firstDynamic = rank;
  for (int64_t i = 0; i < rank; ++i) {
    std::optional<AxisKind> kind = axis(i);
    if (!kind)
      return std::nullopt;
    kinds[i] = *kind;
    if (*kind == AxisKind::Resized && firstResized == rank)
      firstResized = i;
    if (*kind == AxisKind::Dynamic && firstDynamic == rank)
      firstDynamic = i;
  }

  int64_t prefixCount;
  if (firstResized < rank) {
    prefixCount = std::min<int64_t>(2, firstResized);
  } else if (firstDynamic < rank) {
    bool trailingStaticChannel = false;
    if (kinds[rank - 1] == AxisKind::Copied) {
      for (int64_t i = 1; i < rank - 1; ++i) {
        if (kinds[i] == AxisKind::Dynamic) {
          trailingStaticChannel = true;
          break;
        }
      }
    }
    prefixCount = trailingStaticChannel ? std::min<int64_t>(2, firstDynamic)
                                        : std::min<int64_t>(2, rank - 1);
  } else {
    prefixCount = std::min<int64_t>(2, rank - 1);
  }

  const int64_t spatialRank = rank - prefixCount;
  if (prefixCount > 2 || spatialRank < 1 || spatialRank > 3)
    return std::nullopt;
  for (int64_t i = 0; i < prefixCount; ++i) {
    if (kinds[i] == AxisKind::Resized)
      return std::nullopt;
  }
  return HipResizeLaunch{prefixCount, spatialRank};
}

} // namespace hip
} // namespace mlir

#endif // HIP_CONVERSION_RESIZELAYOUT_H
