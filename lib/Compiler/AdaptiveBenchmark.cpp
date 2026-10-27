/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// See AdaptiveBenchmark.h. Kept free of HIP and MLIR includes so the GPU-free
// unit test compiles this file directly.

#include "hip/Compiler/AdaptiveBenchmark.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <thread>
#include <tuple>

namespace mlir {
namespace hip {

namespace {

// Floor for measured times when sizing run counts and bundles, so a kernel
// that times near zero cannot divide by zero.
constexpr double kMinLaunchMs = 1e-3;

constexpr double kInvalid = std::numeric_limits<double>::infinity();

double timeOrInvalid(const std::optional<double> &time) {
  if (!time || !std::isfinite(*time) || *time < 0)
    return kInvalid;
  return *time;
}

} // namespace

unsigned computeBenchmarkRuns(std::size_t budgetMs, double launchMs,
                              unsigned bundle, std::size_t maxRuns) {
  double runs = static_cast<double>(budgetMs) /
                (std::max(launchMs, kMinLaunchMs) * std::max(bundle, 1u));
  return static_cast<unsigned>(std::clamp(
      runs, 1.0, static_cast<double>(std::max<std::size_t>(maxRuns, 1))));
}

std::optional<BenchmarkSelection>
selectByAdaptiveBenchmark(std::size_t numCandidates,
                          const AdaptiveBenchmarkOptions &options,
                          const CandidateTimer &timer) {
  if (numCandidates == 0 || options.maxRuns == 0 || options.coarseMaxRuns == 0)
    return std::nullopt;

  BenchmarkSelection result;
  result.coarseMs.assign(numCandidates, kInvalid);
  result.preciseMs.assign(numCandidates, kInvalid);

  // A candidate too slow for a second coarse run is ranked by its estimate.
  for (std::size_t index = 0; index < numCandidates; ++index) {
    double estimate = timeOrInvalid(timer(index, 1, 1, 1));
    if (!std::isfinite(estimate))
      continue;
    unsigned runs = computeBenchmarkRuns(options.coarseMs, estimate, 1,
                                         options.coarseMaxRuns);
    result.coarseMs[index] =
        runs == 1 ? estimate : timeOrInvalid(timer(index, 0, 1, runs));
  }

  std::vector<std::size_t> selected;
  for (std::size_t index = 0; index < numCandidates; ++index)
    if (std::isfinite(result.coarseMs[index]))
      selected.push_back(index);
  if (selected.empty())
    return std::nullopt;
  const std::vector<double> &coarse = result.coarseMs;
  std::sort(selected.begin(), selected.end(),
            [&](std::size_t lhs, std::size_t rhs) {
              return std::tie(coarse[lhs], lhs) < std::tie(coarse[rhs], rhs);
            });
  if (options.topK > 0 && selected.size() > options.topK)
    selected.resize(options.topK);
  // The precise pass only separates close candidates; one this far behind the
  // best coarse time cannot win it.
  if (options.topK > 0 && options.coarseCutoffFactor > 0) {
    const double cutoff = static_cast<double>(options.coarseCutoffFactor) *
                          std::max(coarse[selected.front()], kMinLaunchMs);
    selected.erase(std::upper_bound(selected.begin(), selected.end(), cutoff,
                                    [&](double limit, std::size_t index) {
                                      return limit < coarse[index];
                                    }),
                   selected.end());
  }

  // One bundle for every precise measurement, sized so the fastest candidate
  // still fits maxRuns event pairs in its budget.
  const double referenceMs = coarse[selected.front()];
  const unsigned bundle = static_cast<unsigned>(
      std::max(static_cast<double>(options.preciseMs) /
                   (std::max(referenceMs, kMinLaunchMs) *
                    static_cast<double>(options.maxRuns)),
               static_cast<double>(options.preciseMinBundle)));

  if (options.cooldownMs > 0)
    std::this_thread::sleep_for(std::chrono::milliseconds(options.cooldownMs));

  // Every selected candidate already ran in the coarse pass, so no warmup.
  std::optional<std::size_t> fastest;
  for (std::size_t index : selected) {
    unsigned runs = computeBenchmarkRuns(options.preciseMs, coarse[index],
                                         bundle, options.maxRuns);
    double time = timeOrInvalid(timer(index, 0, bundle, runs));
    result.preciseMs[index] = time;
    if (std::isfinite(time) && (!fastest || time < result.preciseMs[*fastest]))
      fastest = index;
  }

  if (fastest) {
    result.winner = *fastest;
    result.winnerMs = result.preciseMs[*fastest];
  } else {
    result.winner = selected.front();
    result.winnerMs = coarse[selected.front()];
  }
  return result;
}

std::optional<BenchmarkSelection>
selectBySimpleBenchmark(std::size_t numCandidates, unsigned warmupLaunches,
                        unsigned bundle, unsigned nruns,
                        const CandidateTimer &timer) {
  if (numCandidates == 0 || bundle == 0 || nruns == 0)
    return std::nullopt;
  BenchmarkSelection result;
  result.preciseMs.assign(numCandidates, kInvalid);
  std::optional<std::size_t> fastest;
  for (std::size_t index = 0; index < numCandidates; ++index) {
    double time = timeOrInvalid(timer(index, warmupLaunches, bundle, nruns));
    result.preciseMs[index] = time;
    if (std::isfinite(time) && (!fastest || time < result.preciseMs[*fastest]))
      fastest = index;
  }
  if (!fastest)
    return std::nullopt;
  result.winner = *fastest;
  result.winnerMs = result.preciseMs[*fastest];
  return result;
}

} // namespace hip
} // namespace mlir
