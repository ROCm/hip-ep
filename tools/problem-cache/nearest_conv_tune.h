/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Nearest rocMLIR conv tile from a problem-cache table.
//
// Same shape as the MatMulNBits LUT: a hard group, then one weighted log2
// distance. Arch, conv kind, -F, layouts, stride, dilation, and the fusion
// suffix (-inputFusions, -outputFusions, -supportsSplitK) must match. Inside
// that group the distance is over the implicit GEMM:
//
//   M = K_out
//   N = batch * out_h * out_w
//   K = (C / groups) * filter_h * filter_w
//   d = sqrt(wM*log2(M)^2 + wN*log2(N)^2 + wK*log2(K)^2)
//
// wM, wN, and wK are 1. The neighbor supplies only its gemm:/attn: perfConfig.
// An identical problem string is returned as stored, including a gemm key:
// that tile already compiled for this kernel. tileLegal applies only when
// borrowing a neighbor. `rejectedSolutions` are tiles that already failed to
// compile; the next nearest legal row is returned.

#ifndef HIP_PROBLEM_CACHE_NEAREST_CONV_TUNE_H
#define HIP_PROBLEM_CACHE_NEAREST_CONV_TUNE_H

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace mlir {
namespace hip {

struct ProblemCacheConvEntry {
  std::string problem;
  std::string solution;
};

struct NearestConvTuneHit {
  std::string solution;
  std::string matchedProblem;
  // Weighted log2 distance. Zero when M, N, and K all match.
  float distance = 0;
  bool exact = false;
};

// Rows whose solution is a gemm:/attn: perfConfig. A fused_reduce row is
// skipped. Gemm problem keys are kept so an exact string match can hit them.
bool loadProblemCacheConvEntries(const uint8_t *data, size_t size,
                                 std::vector<ProblemCacheConvEntry> &entries,
                                 std::string &error);

bool loadProblemCacheConvEntries(const std::string &path,
                                 std::vector<ProblemCacheConvEntry> &entries,
                                 std::string &error);

std::optional<NearestConvTuneHit>
findNearestConvTune(std::string_view problemKey,
                    const std::vector<ProblemCacheConvEntry> &entries,
                    const std::vector<std::string> &rejectedSolutions = {});

} // namespace hip
} // namespace mlir

#endif
