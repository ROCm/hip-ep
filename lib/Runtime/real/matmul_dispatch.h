/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_MATMUL_DISPATCH_H
#define HIPDNN_EP_MATMUL_DISPATCH_H

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>

namespace hipdnn_ep_real {

struct MacroTile {
  int m = 0;
  int n = 0;
  int depth = 0;
};

// matmul.cpp is compiled into the runtime bitcode, which the EP JIT-links. MSVC
// lowers std::lexicographical_compare and std::string_view::find(substring) to
// its vectorized helpers (__std_mismatch_1, __std_search_1), which that JIT
// cannot materialize (see chrome_trace.cpp), so the helpers below use plain
// loops.

// Whether serialized algorithm `a` sorts before `b`, byte by byte.
template <typename Algo>
inline bool algoBytesLess(const Algo &a, const Algo &b) {
  for (size_t i = 0; i < sizeof(a.data); i++)
    if (a.data[i] != b.data[i])
      return a.data[i] < b.data[i];
  return false;
}

// Reads the macro tile out of a Tensile kernel name ("..._MT16x16x8_SN_...").
inline bool parseMacroTile(std::string_view kernelName, MacroTile &tile) {
  const auto readInt = [&](size_t &pos, int &value) {
    const size_t start = pos;
    value = 0;
    while (pos < kernelName.size() && kernelName[pos] >= '0' &&
           kernelName[pos] <= '9' && pos - start < 6)
      value = value * 10 + (kernelName[pos++] - '0');
    return pos > start && value > 0;
  };
  for (size_t at = 0; at + 3 <= kernelName.size(); at++) {
    if (kernelName[at] != '_' || kernelName[at + 1] != 'M' ||
        kernelName[at + 2] != 'T')
      continue;
    size_t pos = at + 3;
    MacroTile parsed;
    if (readInt(pos, parsed.m) && pos < kernelName.size() &&
        kernelName[pos++] == 'x' && readInt(pos, parsed.n) &&
        pos < kernelName.size() && kernelName[pos++] == 'x' &&
        readInt(pos, parsed.depth)) {
      tile = parsed;
      return true;
    }
  }
  return false;
}

// hipBLASLt answers a problem its tuned tables do not cover with every
// applicable solution, unranked -- each reports a wavesCount of 1, where a
// ranked candidate carries the table's 0 -- and in an order that changes from
// process to process. Taking whichever came first made the same shape run a
// different kernel, and round differently, from one run to the next.
//
// Unranked candidates are therefore put in a fixed order before the first is
// taken: the largest output tile (m x n) first, then the smallest depth, then
// the serialized algorithm (which holds the solution index); a kernel name
// without a macro tile goes last. A larger output tile reuses each loaded
// value across more outputs, so this order leads with a fast kernel where the
// lowest solution index alone can pick the slowest one (on gfx1151 the
// generic fp32 kernels are indexed MT8x8x8, MT16x16x8, MT8x8x16, MT16x16x16).
// A ranked list keeps hipBLASLt's order.
// `HeuristicResult` is hipblasLtMatmulHeuristicResult_t; `kernelName` maps a
// candidate to its kernel name.
template <typename HeuristicResult, typename KernelName>
inline void orderUnrankedCandidates(HeuristicResult *candidates, int count,
                                    KernelName kernelName) {
  for (int i = 0; i < count; i++)
    if (candidates[i].wavesCount != 1.0f)
      return;
  struct Keyed {
    int64_t area;
    int depth;
    HeuristicResult result;
  };
  std::vector<Keyed> keyed;
  keyed.reserve(static_cast<size_t>(count));
  for (int i = 0; i < count; i++) {
    MacroTile tile;
    const bool known = parseMacroTile(kernelName(candidates[i]), tile);
    keyed.push_back({known ? int64_t{tile.m} * tile.n : 0,
                     known ? tile.depth : std::numeric_limits<int>::max(),
                     candidates[i]});
  }
  std::sort(keyed.begin(), keyed.end(), [](const Keyed &a, const Keyed &b) {
    if (a.area != b.area)
      return a.area > b.area;
    if (a.depth != b.depth)
      return a.depth < b.depth;
    return algoBytesLess(a.result.algo, b.result.algo);
  });
  for (int i = 0; i < count; i++)
    candidates[i] = keyed[static_cast<size_t>(i)].result;
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
