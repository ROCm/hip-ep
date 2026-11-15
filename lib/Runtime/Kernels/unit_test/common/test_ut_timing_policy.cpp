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
  // Equal GPU-time goals, different devices/shapes, no global iteration cap.
  for (double call_ms : {0.001, 0.1, 10.0, 3000.0, 25000.0}) {
    uint64_t actual_calls = 0;
    auto r = measure(Policy{}, [&](uint32_t n) {
      assert(n >= 1 && n <= 64);
      actual_calls += n;
      return call_ms * n;
    });
    assert(r.calls + 1 == actual_calls);  // one excluded pilot
    assert(r.calls >= 10 && r.elapsed_ms >= 10000.0);
    assert(r.pilot_calls == 1 && r.pilot_batches == 1);
    uint64_t group_calls = 0, group_batches = 0;
    double group_ms = 0;
    for (const auto& g : r.groups) {
      assert(g.calls >= 2 && g.elapsed_ms >= 2000.0);
      group_calls += g.calls; group_batches += g.batches; group_ms += g.elapsed_ms;
    }
    assert(group_calls == r.calls && group_batches == r.batches && group_ms == r.elapsed_ms);
    assert(std::abs(r.mean_ms / call_ms - 1.0) < 1e-10);
    if (call_ms < 1.0) assert(r.calls > 100);
    if (call_ms > 2000.0) assert(r.exceeded_soft_limit && r.calls == 10);
  }
  auto screen = measure(Policy::screen(), [](uint32_t n) { return 0.01 * n; });
  assert(screen.elapsed_ms >= 100.0 && screen.elapsed_ms < 100.1);
  assert(std::string(Policy::screen().name()) == "screen");
  assert(std::string(Policy::standard().name()) == "standard");
  assert(std::string(Policy{}.name()) == "confirm");
  auto standard = measure(Policy::standard(), [](uint32_t n) { return 0.01 * n; });
  assert(standard.elapsed_ms >= 1000.0 && standard.elapsed_ms < 1000.1);
  auto standard_long = measure(Policy::standard(), [](uint32_t n) { return 1500.0 * n; });
  assert(standard_long.calls == 10 && standard_long.exceeded_soft_limit);

  int bad = 0;
  auto recovered = measure(Policy::screen(), [&](uint32_t n) {
    if (++bad <= 2) return std::numeric_limits<double>::quiet_NaN();
    return double(n);
  });
  assert(recovered.invalid_batches == 2 && recovered.invalid_calls == 2);
  bool failed = false;
  int attempts = 0;
  try {
    (void)measure(Policy{}, [&](uint32_t) { ++attempts; return -1.0; });
  } catch (const std::runtime_error&) { failed = true; }
  assert(failed && attempts == 3);

  // Sporadic invalid batches in a long measurement are discarded: one in
  // every 997 attempts stays far below the 1% allowance.
  attempts = 0;
  auto sporadic = measure(Policy{}, [&](uint32_t n) {
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
    return (++attempts == 3 ? 300.0 : 1.0) * n;
  });
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
  assert(drift.calls >= 10 && drift.elapsed_ms >= 100.0);
  assert(drift.mean_ms > 0.0 && drift.spread_fraction >= 0.0);
  assert(nextCalls(1e30, 1e-30, 0, 128) == 128);
  assert(nextCalls(-1.0, 30000.0, 2, 128) == 2);
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
