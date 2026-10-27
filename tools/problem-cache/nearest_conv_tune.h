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
// The distance weights live in the .fb (`weight_m/n/k`), defaulting to 1.
// The neighbor supplies only its gemm:/attn: perfConfig. An identical problem
// string is returned as stored, including a gemm key: that tile already
// compiled for this kernel. tileLegal applies only when borrowing a neighbor.
// `rejectedSolutions` are tiles that already failed to compile; the next
// nearest legal row is returned.

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

// One loaded .fb. `entries` are the gemm:/attn: points; other perfConfigs are
// dropped. Weights are the table's log-space metric.
struct ProblemCacheLut {
  float weightM = 1.0f;
  float weightN = 1.0f;
  float weightK = 1.0f;
  std::vector<ProblemCacheConvEntry> entries;
};

struct NearestConvTuneHit {
  std::string solution;
  std::string matchedProblem;
  // Weighted log2 distance. Zero when M, N, and K all match.
  float distance = 0;
  bool exact = false;
};

// Implicit GEMM of a conv problem string. False, and m/n/k left at 0, when
// `problem` is not a conv. The dump writes these into each LUT point.
bool implicitGemmDims(std::string_view problem, int64_t &m, int64_t &n,
                      int64_t &k);

// Rows whose perfConfig is gemm:/attn:. Anything else is skipped. Gemm
// problem keys are kept so an exact string match can hit them.
bool loadProblemCacheLut(const uint8_t *data, size_t size, ProblemCacheLut &lut,
                         std::string &error);

bool loadProblemCacheLut(const std::string &path, ProblemCacheLut &lut,
                         std::string &error);

std::optional<NearestConvTuneHit>
findNearestConvTune(std::string_view problemKey,
                    const std::vector<ProblemCacheConvEntry> &entries,
                    const std::vector<std::string> &rejectedSolutions = {},
                    float weightM = 1.0f, float weightN = 1.0f,
                    float weightK = 1.0f);

} // namespace hip
} // namespace mlir

#endif
