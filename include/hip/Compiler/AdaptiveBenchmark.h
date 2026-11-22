/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Candidate selection for the rocMLIR perfConfig search: which candidates to
// time, how many launches each gets, and which one wins.
//
// Timing every candidate with a fixed launch count spends most of a search on
// candidates that cannot win: a perfConfig that takes 400 ms per launch costs
// the same 120 launches as the 1 ms winner. The adaptive selection is a port
// of MIGraphX's adaptive_topk_benchmark (src/targets/gpu/time_op.cpp), with
// the same defaults, so both stacks rank the same candidate list the same
// way:
//
//   1. Coarse pass. Every candidate gets one warmup launch and one timed
//      launch. Candidates fast enough get up to `coarseMaxRuns` more timed
//      launches within `coarseMs`.
//   2. Selection. Keep the `topK` fastest coarse times (ties go to the lower
//      index), then drop any slower than `coarseCutoffFactor` times the best.
//   3. Precise pass. Each event pair wraps `bundle` launches, so launch
//      overhead and timer resolution do not dominate small kernels. One
//      bundle size serves every selected candidate; the run count fits
//      `preciseMs`.
//
// The timer is injected, so this file has no HIP or MLIR dependency and the
// selection rules are tested without a GPU (test/unit/test_adaptive_benchmark).

#ifndef HIP_COMPILER_ADAPTIVEBENCHMARK_H
#define HIP_COMPILER_ADAPTIVEBENCHMARK_H

#include <cstddef>
#include <functional>
#include <optional>
#include <vector>

namespace mlir {
namespace hip {

struct AdaptiveBenchmarkOptions {
  // Number of candidates timed precisely. Zero times every candidate
  // precisely and disables the cutoff.
  std::size_t topK = 10;
  // Per-candidate time budgets, in milliseconds.
  std::size_t preciseMs = 20;
  std::size_t coarseMs = 5;
  std::size_t preciseMinBundle = 4;
  // Most event pairs in one precise or coarse measurement.
  std::size_t maxRuns = 20;
  std::size_t coarseMaxRuns = 4;
  // Zero keeps all topK candidates.
  std::size_t coarseCutoffFactor = 4;
  // Pause between the passes so the precise pass does not start throttled.
  unsigned cooldownMs = 10;
};

// Times candidate `index`: `warmupLaunches` untimed launches, then `nruns`
// event pairs that each wrap `bundle` launches. Returns the trimmed-mean
// milliseconds per launch, or nullopt when the candidate cannot run.
using CandidateTimer = std::function<std::optional<double>(
    std::size_t index, unsigned warmupLaunches, unsigned bundle,
    unsigned nruns)>;

struct BenchmarkSelection {
  std::size_t winner = 0;
  // The winner's precise time, or its coarse time when every precise
  // measurement failed.
  double winnerMs = 0;
  // Per candidate; infinity marks a candidate that failed or was not timed in
  // that pass.
  std::vector<double> coarseMs;
  std::vector<double> preciseMs;
};

// Returns nullopt when no candidate ran.
std::optional<BenchmarkSelection>
selectByAdaptiveBenchmark(std::size_t numCandidates,
                          const AdaptiveBenchmarkOptions &options,
                          const CandidateTimer &timer);

// Times every candidate once with the same warmup, bundle, and run count, and
// keeps the fastest. `preciseMs` holds the times; `coarseMs` stays empty.
std::optional<BenchmarkSelection>
selectBySimpleBenchmark(std::size_t numCandidates, unsigned warmupLaunches,
                        unsigned bundle, unsigned nruns,
                        const CandidateTimer &timer);

// The run count that fits `budgetMs` for a kernel of `launchMs` per launch
// timed in bundles of `bundle`, clamped to [1, maxRuns].
unsigned computeBenchmarkRuns(std::size_t budgetMs, double launchMs,
                              unsigned bundle, std::size_t maxRuns);

} // namespace hip
} // namespace mlir

#endif // HIP_COMPILER_ADAPTIVEBENCHMARK_H
