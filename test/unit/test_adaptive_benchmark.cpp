/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// GPU-free tests for the rocMLIR autotune candidate selection
// (lib/Compiler/AdaptiveBenchmark.cpp). A fake timer stands in for the GPU and
// records every measurement the selection asks for.

#include "hip/Compiler/AdaptiveBenchmark.h"

#include <cstdio>
#include <optional>
#include <vector>

using namespace mlir::hip;

namespace {

int g_failures = 0;

void check(bool cond, const char *what) {
  std::printf("%s %s\n", cond ? "[ OK ]" : "[FAIL]", what);
  if (!cond)
    ++g_failures;
}

struct Call {
  std::size_t index;
  unsigned warmup;
  unsigned bundle;
  unsigned nruns;
};

// Coarse measurements (bundle 1) return `coarse`; bundled ones return
// `precise`. A negative entry fails that measurement.
struct FakeGpu {
  std::vector<double> coarse;
  std::vector<double> precise;
  std::vector<Call> calls;

  CandidateTimer timer() {
    return [this](std::size_t index, unsigned warmup, unsigned bundle,
                  unsigned nruns) -> std::optional<double> {
      calls.push_back({index, warmup, bundle, nruns});
      double time = bundle == 1 ? coarse[index] : precise[index];
      if (time < 0)
        return std::nullopt;
      return time;
    };
  }

  std::vector<Call> callsFor(std::size_t index) const {
    std::vector<Call> out;
    for (const Call &call : calls)
      if (call.index == index)
        out.push_back(call);
    return out;
  }
};

AdaptiveBenchmarkOptions testOptions() {
  AdaptiveBenchmarkOptions options;
  options.cooldownMs = 0;
  return options;
}

void testRunCounts() {
  check(computeBenchmarkRuns(5, 1.066, 1, 4) == 4, "runs: fast kernel caps");
  check(computeBenchmarkRuns(5, 426.0, 1, 4) == 1, "runs: slow kernel gets 1");
  check(computeBenchmarkRuns(20, 1.066, 4, 20) == 4, "runs: bundled budget");
  check(computeBenchmarkRuns(20, 0.0, 4, 20) == 20, "runs: zero time floors");
}

void testSlowCandidatesSkipPrecisePass() {
  FakeGpu gpu;
  gpu.coarse = {1.07, 426.0, 1.20, 1.10, 50.0, 1.25};
  gpu.precise = {1.06, 426.0, 1.19, 1.00, 50.0, 1.24};
  auto selection =
      selectByAdaptiveBenchmark(gpu.coarse.size(), testOptions(), gpu.timer());
  check(selection.has_value(), "adaptive: returns a winner");
  if (!selection)
    return;
  check(selection->winner == 3, "adaptive: precise time decides the winner");
  check(selection->winnerMs == 1.00, "adaptive: reports the precise time");

  std::vector<Call> slow = gpu.callsFor(1);
  check(slow.size() == 1 && slow[0].warmup == 1 && slow[0].nruns == 1,
        "adaptive: slow candidate gets only warmup plus one estimate");
  check(gpu.callsFor(4).size() == 1,
        "adaptive: candidate past the cutoff is not timed precisely");

  std::vector<Call> fast = gpu.callsFor(0);
  check(fast.size() == 3, "adaptive: fast candidate gets estimate, coarse, "
                          "and precise measurements");
  if (fast.size() == 3) {
    check(fast[1].warmup == 0 && fast[1].bundle == 1 && fast[1].nruns == 4,
          "adaptive: coarse pass uses coarseMaxRuns without warmup");
    check(fast[2].warmup == 0 && fast[2].bundle == 4 && fast[2].nruns == 4,
          "adaptive: precise pass bundles 4 launches in 4 runs");
  }
}

void testTopK() {
  FakeGpu gpu;
  for (int i = 0; i < 15; ++i) {
    gpu.coarse.push_back(1.0 + 0.01 * (14 - i));
    gpu.precise.push_back(gpu.coarse.back());
  }
  auto selection =
      selectByAdaptiveBenchmark(gpu.coarse.size(), testOptions(), gpu.timer());
  unsigned preciseCalls = 0;
  bool onlyFastest = true;
  for (const Call &call : gpu.calls) {
    if (call.bundle == 1)
      continue;
    ++preciseCalls;
    onlyFastest = onlyFastest && call.index >= 5;
  }
  check(preciseCalls == 10 && onlyFastest,
        "adaptive: only the topK fastest are timed precisely");
  check(selection && selection->winner == 14, "adaptive: fastest wins");
}

void testTiesGoToLowerIndex() {
  FakeGpu gpu;
  gpu.coarse = {2.0, 1.0, 1.0};
  gpu.precise = {2.0, 1.0, 1.0};
  AdaptiveBenchmarkOptions options = testOptions();
  options.topK = 1;
  auto selection =
      selectByAdaptiveBenchmark(gpu.coarse.size(), options, gpu.timer());
  check(selection && selection->winner == 1,
        "adaptive: equal coarse times keep the lower index");
}

void testFailures() {
  FakeGpu gpu;
  gpu.coarse = {-1.0, 3.0, 2.0};
  gpu.precise = {-1.0, -1.0, -1.0};
  auto selection =
      selectByAdaptiveBenchmark(gpu.coarse.size(), testOptions(), gpu.timer());
  check(selection && selection->winner == 2 && selection->winnerMs == 2.0,
        "adaptive: falls back to the best coarse time when precise fails");
  check(gpu.callsFor(0).size() == 1,
        "adaptive: a failed estimate is not measured again");

  FakeGpu broken;
  broken.coarse = {-1.0, -1.0};
  broken.precise = {-1.0, -1.0};
  check(!selectByAdaptiveBenchmark(2, testOptions(), broken.timer()),
        "adaptive: no winner when nothing runs");
  check(!selectByAdaptiveBenchmark(0, testOptions(), broken.timer()),
        "adaptive: no winner without candidates");
}

void testNearZeroKernel() {
  FakeGpu gpu;
  gpu.coarse = {0.0, 0.0005};
  gpu.precise = {0.0, 0.0005};
  selectByAdaptiveBenchmark(gpu.coarse.size(), testOptions(), gpu.timer());
  bool sized = false;
  for (const Call &call : gpu.calls)
    if (call.bundle != 1)
      sized = call.bundle == 1000 && call.nruns == 20;
  check(sized, "adaptive: near-zero kernels bundle 1000 launches");
}

void testSimple() {
  FakeGpu gpu;
  gpu.coarse = {3.0, 1.0, -1.0, 2.0};
  gpu.precise = gpu.coarse;
  auto selection = selectBySimpleBenchmark(4, 20, 1, 100, gpu.timer());
  bool uniform = gpu.calls.size() == 4;
  for (const Call &call : gpu.calls)
    uniform =
        uniform && call.warmup == 20 && call.bundle == 1 && call.nruns == 100;
  check(uniform, "simple: every candidate gets the same measurement");
  check(selection && selection->winner == 1, "simple: fastest wins");
}

} // namespace

int main() {
  testRunCounts();
  testSlowCandidatesSkipPrecisePass();
  testTopK();
  testTiesGoToLowerIndex();
  testFailures();
  testNearZeroKernel();
  testSimple();
  if (g_failures) {
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
  }
  return 0;
}
