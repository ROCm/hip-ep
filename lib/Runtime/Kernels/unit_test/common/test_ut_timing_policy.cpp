/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ut_timing_policy.h"
#include "ut_timing_device_profiles.h"
#include "ut_timing_context.h"

#include <cassert>
#include <iostream>
#include <limits>

using namespace hipdnn_ep_test::timing;

namespace {

void checkTotals(const Result& r) {
  uint64_t group_calls = 0, group_batches = 0;
  double group_ms = 0;
  for (const auto& g : r.groups) {
    assert(g.calls >= 1);
    group_calls += g.calls; group_batches += g.batches; group_ms += g.elapsed_ms;
  }
  assert(group_calls == r.calls && group_batches == r.batches && group_ms == r.elapsed_ms);
  assert(r.pilot_calls >= 1 && r.pilot_batches >= 1 && r.pilot_ms > 0.0);
  assert(!r.early_stop || r.converged);
}

}  // namespace

int main() {
  assert(deviceProfile("gfx9999", 71626375, "matmul_nbits_fp16_wu4", false).batch_call_cap == 64);
  const char* shape = "M1_N2048_K2048_gs128";
  assert(deviceProfile("gfx1151", 71626375, "matmul_nbits_fp16_wu4", false, shape).batch_call_cap == 128);
  assert(deviceProfile("gfx1151", 71626375, "matmul_nbits_fp16_wu4", false).batch_call_cap == 64);
  assert(deviceProfile("gfx1151", 71626375, "matmul_nbits_fp16_wu4", false, "other_shape").batch_call_cap == 64);
  assert(deviceProfile("gfx1170", 71260601, "matmul_nbits_fp16_wu4", false, shape).screen_target_ms == 500.0);
  assert(deviceProfile("gfx1170", 71260602, "matmul_nbits_fp16_wu4", false, shape).screen_target_ms == 100.0);
  assert(deviceProfile("gfx1151", 71626375, "gqa", false).batch_call_cap == 64);
  assert(deviceProfile("gfx1151", 71626375, "matmul_nbits_fp16_wu4", true).batch_call_cap == 64);
  assert(deviceProfile("gfx1151", 71626376, "matmul_nbits_fp16_wu4", false).batch_call_cap == 64);
  assert(std::string(Policy::screen().name()) == "screen");
  assert(std::string(Policy::standard().name()) == "standard");
  assert(std::string(Policy{}.name()) == "confirm");

  // Stable calls stop at whichever of the call goal (after the GPU-time
  // floor) or the GPU-time goal comes first; no global iteration cap.
  struct Case { double call_ms; double gpu_ms; uint64_t calls; };
  for (const Case c : {Case{0.001, 100.0, 100000}, Case{0.1, 1000.0, 10000},
                       Case{10.0, 10000.0, 1000}, Case{3000.0, 15000.0, 5},
                       Case{25000.0, 125000.0, 5}}) {
    uint64_t actual_calls = 0;
    auto r = measure(Policy{}, [&](uint32_t n) {
      assert(n >= 1 && n <= 64);
      actual_calls += n;
      return c.call_ms * n;
    });
    checkTotals(r);
    assert(r.calls + r.pilot_calls == actual_calls);  // excluded warm-up
    assert(r.pilot_ms >= 25.0 && (r.pilot_batches == 1 || r.pilot_ms - c.call_ms * 64 < 25.0));
    assert(r.early_stop && r.converged && r.spread_fraction == 0.0);
    assert(std::abs(r.elapsed_ms / c.gpu_ms - 1.0) < 1e-3);
    assert(r.calls >= c.calls && r.calls <= c.calls + kGroups);
    assert(std::abs(r.mean_ms / c.call_ms - 1.0) < 1e-10);
    assert(r.exceeded_soft_limit == (c.gpu_ms > 20000.0));
  }
  auto screen = measure(Policy::screen(), [](uint32_t n) { return 0.01 * n; });
  assert(screen.early_stop && screen.calls >= 100 && screen.calls <= 105);
  assert(screen.elapsed_ms >= 1.0 && screen.elapsed_ms < 1.1);
  auto standard = measure(Policy::standard(), [](uint32_t n) { return 0.01 * n; });
  assert(standard.early_stop && standard.calls >= 1000 && standard.calls <= 1005);
  assert(standard.elapsed_ms >= 10.0 && standard.elapsed_ms < 10.1);
  auto standard_mid = measure(Policy::standard(), [](uint32_t n) { return 5.0 * n; });
  assert(standard_mid.early_stop && standard_mid.calls == 200);
  auto standard_long = measure(Policy::standard(), [](uint32_t n) { return 1500.0 * n; });
  assert(standard_long.early_stop && standard_long.calls == 5);
  assert(standard_long.exceeded_soft_limit);

  // An unstable first round continues every group to the full GPU-time goal
  // and two calls per group; the first-round batches are kept.
  int attempts = 0;
  auto noisy_long = measure(Policy::standard(), [&](uint32_t n) {
    return (++attempts % 2 ? 1500.0 : 1700.0) * n;
  });
  assert(!noisy_long.early_stop && noisy_long.calls == 10 && noisy_long.converged);
  for (const auto& g : noisy_long.groups) assert(g.calls == 2);
  // A measurement still unstable at the full goal is kept but not converged.
  attempts = 0;
  auto unstable = measure(Policy::standard(), [&](uint32_t n) {
    ++attempts;
    return (attempts == 2 || attempts == 7 ? 3000.0 : 1500.0) * n;
  });
  assert(!unstable.early_stop && !unstable.converged && unstable.calls == 10);
  assert(unstable.groups[0].meanMs() == 3000.0 && unstable.mean_ms == 1500.0);
  Policy one_call_warmup = Policy::standard();
  one_call_warmup.warmup_ms = 0.0;
  attempts = 0;
  auto noisy_small = measure(one_call_warmup, [&](uint32_t n) {
    return (++attempts < 6 ? 0.02 : 0.01) * n;
  });
  checkTotals(noisy_small);
  assert(!noisy_small.early_stop && noisy_small.elapsed_ms >= 1000.0);
  for (const auto& g : noisy_small.groups) assert(g.elapsed_ms >= 200.0 && g.calls >= 2);
  attempts = 0;
  auto steady_after_ramp = measure(Policy::standard(), [&](uint32_t n) {
    return (++attempts == 1 ? 0.05 : 0.01) * n;
  });
  assert(steady_after_ramp.early_stop);  // a slow first warm-up call is excluded

  int bad = 0;
  auto recovered = measure(Policy::screen(), [&](uint32_t n) {
    if (++bad <= 2) return std::numeric_limits<double>::quiet_NaN();
    return double(n);
  });
  assert(recovered.invalid_batches == 2 && recovered.invalid_calls == 2);
  bool failed = false;
  attempts = 0;
  try {
    (void)measure(Policy{}, [&](uint32_t) { ++attempts; return -1.0; });
  } catch (const std::runtime_error&) { failed = true; }
  assert(failed && attempts == 3);

  // Sporadic invalid batches in a long measurement are discarded: one in
  // every 997 attempts stays far below the 1% allowance.
  Policy long_confirm;
  long_confirm.call_goal = kNoCallGoal;
  attempts = 0;
  auto sporadic = measure(long_confirm, [&](uint32_t n) {
    return (++attempts % 997 == 0) ? 0.0 : 0.01 * n;
  });
  assert(sporadic.invalid_batches > 2 && sporadic.elapsed_ms >= 10000.0);
  assert(std::abs(sporadic.mean_ms / 0.01 - 1.0) < 1e-10);

  // Invalid retries cannot reset at every new batch. Valid slow samples stay.
  attempts = 0;
  try {
    (void)measure(Policy::screen(), [&](uint32_t n) {
      return (++attempts % 2 == 0) ? 0.0 : double(n);
    });
    assert(false);
  } catch (const std::runtime_error&) { assert(attempts == 6); }
  attempts = 0;
  auto slow = measure(Policy::screen(), [&](uint32_t n) {
    return (++attempts == 4 ? 300.0 : 1.0) * n;  // two warm-up batches first
  });
  assert(slow.pilot_batches == 2);
  assert(slow.exceeded_soft_limit && slow.spread_fraction > 0.0);
  assert(slow.groups[1].elapsed_ms >= 300.0);

  // API exceptions are not retried as though they were valid timing samples.
  attempts = 0;
  try {
    (void)measure(Policy{}, [&](uint32_t) -> double {
      ++attempts;
      throw std::runtime_error("HIP failure");
    });
    assert(false);
  } catch (const std::runtime_error&) { assert(attempts == 1); }

  // A slower later device clock changes N only between complete batches.
  attempts = 0;
  auto drift = measure(Policy::screen(), [&](uint32_t n) {
    return n * (++attempts < 8 ? 0.01 : 0.02);
  });
  assert(drift.calls >= 10 && drift.elapsed_ms >= 1.0);
  assert(drift.mean_ms > 0.0 && drift.spread_fraction >= 0.0);
  assert(nextCalls(1e30, 1e-30, 0, 128) == 128);
  assert(nextCalls(-1.0, 30000.0, 2, 128) == 2);

  Policy invalid = Policy::standard();
  invalid.call_goal = 4;
  try { invalid.validate(); assert(false); } catch (const std::invalid_argument&) {}
  invalid = Policy::standard();
  invalid.floor_ms = 2000.0;
  try { invalid.validate(); assert(false); } catch (const std::invalid_argument&) {}
  invalid = Policy::standard();
  invalid.max_spread = 0.0;
  try { invalid.validate(); assert(false); } catch (const std::invalid_argument&) {}
  invalid = Policy::standard();
  invalid.warmup_ms = -1.0;
  try { invalid.validate(); assert(false); } catch (const std::invalid_argument&) {}

  // Binding selects the CSV's returned result, including when it is not last.
  {
    CaseScope scope("test/leaf", "shape", "complete_entry");
    auto first = rememberResult(0.25);
    rememberResult(0.5);
    emitCsvBinding("shape", "cfg", "PASS", 0.0, 0.25);
    assert(completedRecords[0].id == first && completedRecords[0].bound);
    try { emitCsvBinding("foreign", "cfg", "PASS", 0.0, 0.5); assert(false); }
    catch (const std::runtime_error&) {}
    rememberResult(0.5);
    try { emitCsvBinding("shape", "cfg", "PASS", 0.0, 0.5); assert(false); }
    catch (const std::runtime_error&) {}
    assert(jsonString("\"a\\b\n") == "\"\\\"a\\\\b\\n\"");
  }
  assert(completedRecords.empty() && currentCase.leaf.empty());
  std::cout << "UT adaptive timing policy: PASS\n";
}
