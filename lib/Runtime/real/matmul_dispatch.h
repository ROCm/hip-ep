/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_MATMUL_DISPATCH_H
#define HIPDNN_EP_MATMUL_DISPATCH_H

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <string_view>

namespace hipdnn_ep_real {

// hipBLASLt answers a problem its tuned tables do not cover with every
// applicable solution, unranked -- each reports a wavesCount of 1, where a
// ranked candidate carries the table's 0 -- and in an order that changes from
// process to process. Taking whichever came first made the same shape run a
// different kernel, and round differently, from one run to the next. Ordering
// unranked candidates by their serialized algorithm (which holds the solution
// index) keeps a shape on one kernel in every process; a ranked list keeps its
// order. `HeuristicResult` is hipblasLtMatmulHeuristicResult_t.
//
// Before: unranked {1286, 1284, 1287, 1285} in one process,
//                  {1285, 1286, 1284, 1287} in the next.
// After:  {1284, 1285, 1286, 1287} in both.
template <typename HeuristicResult>
inline void orderUnrankedCandidates(HeuristicResult *candidates, int count) {
  for (int i = 0; i < count; i++)
    if (candidates[i].wavesCount != 1.0f)
      return;
  std::stable_sort(candidates, candidates + count,
                   [](const HeuristicResult &a, const HeuristicResult &b) {
                     return std::lexicographical_compare(
                         std::begin(a.algo.data), std::end(a.algo.data),
                         std::begin(b.algo.data), std::end(b.algo.data));
                   });
}

inline bool isGfx1151(std::string_view architecture) {
  return architecture == "gfx1151" ||
         (architecture.size() >= 8 && architecture.substr(0, 8) == "gfx1151:");
}

inline constexpr bool
useF32TransposedBDecodeGemv(bool isValidatedTarget, int64_t M, int64_t N,
                            int64_t K, int64_t batchCount, int64_t elementSize,
                            int64_t transA, int64_t transB) {
  return isValidatedTarget && M == 1 && batchCount == 1 && elementSize == 4 &&
         transA == 0 && transB == 1 && N > 0 && N <= 256 && K >= 256 &&
         K <= std::numeric_limits<int>::max();
}

// The multi-wave kernel covers every decode GEMV shape the caller admits;
// K is bounded there.
inline constexpr bool useF32TransposedBMultiwaveGfx1151(int64_t N, int64_t K) {
  return N > 0 && N <= 256 && K > 0;
}

} // namespace hipdnn_ep_real

#endif // HIPDNN_EP_MATMUL_DISPATCH_H
