/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace hipdnn_ep_test {
namespace timing {

inline constexpr const char* kPolicyVersion = "ut-adaptive-v2";
inline constexpr size_t kGroups = 5;

enum class Preset { Screen, Standard, Confirm };

// These are cumulative GPU-time goals for ONE measurement, not per batch.
// The cap bounds the number of complete op calls between two HIP events; it
// is not a cap on total iterations. It must not be raised by a sweep driver.
struct Policy {
  Preset preset = Preset::Confirm;
  double target_ms = 10000.0;
  double soft_limit_ms = 20000.0;
  uint32_t min_calls_per_group = 2;
  uint32_t max_calls_per_batch = 64;
  // Invalid batches are discarded. A cell fails only when they exceed
  // max(invalid_retries, invalid_fraction * attempted batches).
  uint32_t invalid_retries = 2;
  double invalid_fraction = 0.01;

  static Policy screen() {
    Policy p;
    p.preset = Preset::Screen;
    p.target_ms = 100.0;
    p.soft_limit_ms = 200.0;
    return p;
  }

  static Policy standard() {
    Policy p;
    p.preset = Preset::Standard;
    p.target_ms = 1000.0;
    p.soft_limit_ms = 2000.0;
    return p;
  }

  const char* name() const {
    switch (preset) {
      case Preset::Screen: return "screen";
      case Preset::Standard: return "standard";
      case Preset::Confirm: return "confirm";
    }
    return "confirm";
  }

  void validate() const {
    if (!(target_ms > 0.0) || !std::isfinite(target_ms) ||
        !std::isfinite(soft_limit_ms) || soft_limit_ms < target_ms ||
        min_calls_per_group < 2 || max_calls_per_batch == 0 ||
        invalid_retries > 2 || !(invalid_fraction >= 0.0) ||
        !(invalid_fraction < 1.0))
      throw std::invalid_argument("invalid UT timing policy");
  }
};

struct Group {
  double elapsed_ms = 0.0;
  uint64_t calls = 0;
  uint64_t batches = 0;
  double meanMs() const { return elapsed_ms / static_cast<double>(calls); }
};

struct Result {
  std::array<Group, kGroups> groups{};
  double pilot_ms = 0.0;
  uint64_t pilot_calls = 0;
  uint64_t pilot_batches = 0;
  double elapsed_ms = 0.0;
  double mean_ms = 0.0;
  double spread_fraction = 0.0;
  uint64_t calls = 0;
  uint64_t batches = 0;
  uint64_t invalid_batches = 0;
  uint64_t invalid_calls = 0;
  uint32_t largest_batch_calls = 0;
  bool exceeded_soft_limit = false;
};

inline uint32_t nextCalls(double remaining_ms, double per_call_ms,
                          uint64_t minimum_calls, uint32_t cap) {
  if (!(per_call_ms > 0.0) || !std::isfinite(per_call_ms) || cap == 0)
    throw std::invalid_argument("invalid UT pilot duration or batch cap");
  const double predicted = std::ceil(std::max(0.0, remaining_ms) / per_call_ms);
  const double wanted = std::max(predicted, static_cast<double>(minimum_calls));
  return static_cast<uint32_t>(std::max(1.0, std::min(wanted, double(cap))));
}

// runner(N) returns GPU milliseconds for N COMPLETE calls. No per-iteration
// host polling is required. N is frozen during each call to runner. Exceptions
// (including HIP API errors) propagate. Invalid durations are discarded and
// repeated; only an invalid share above the policy allowance fails the cell.
template <class Runner>
Result measure(const Policy& policy, Runner&& runner) {
  policy.validate();
  Result result;
  uint64_t attempted = 0;
  auto validBatch = [&](uint32_t& count) {
    for (;;) {
      result.largest_batch_calls = std::max(result.largest_batch_calls, count);
      const double ms = runner(count);
      ++attempted;
      if (ms > 0.0 && std::isfinite(ms)) return ms;
      ++result.invalid_batches;
      result.invalid_calls += count;
      // The allowance is shared by the entire cell, including its pilot.
      const double allowed = std::max(double(policy.invalid_retries),
                                      std::ceil(policy.invalid_fraction * double(attempted)));
      if (double(result.invalid_batches) > allowed)
        throw std::runtime_error("UT timing: invalid event durations above the allowed share");
      count = std::max(uint32_t(1), count / 2);
    }
  };

  uint32_t pilot_calls = 1;
  result.pilot_ms = validBatch(pilot_calls);
  result.pilot_calls = pilot_calls;
  result.pilot_batches = 1;
  double per_call_ms = result.pilot_ms / pilot_calls;
  const double group_goal_ms = policy.target_ms / kGroups;
  for (auto& group : result.groups) {
    while (group.elapsed_ms < group_goal_ms ||
           group.calls < policy.min_calls_per_group) {
      const uint64_t minimum = group.calls < policy.min_calls_per_group
                                   ? policy.min_calls_per_group - group.calls
                                   : 0;
      uint32_t count = nextCalls(group_goal_ms - group.elapsed_ms, per_call_ms,
                                 minimum, policy.max_calls_per_batch);
      const double ms = validBatch(count);
      group.elapsed_ms += ms;
      group.calls += count;
      ++group.batches;
      per_call_ms = ms / count;
    }
    result.elapsed_ms += group.elapsed_ms;
    result.calls += group.calls;
    result.batches += group.batches;
  }

  std::array<double, kGroups> means{};
  for (size_t i = 0; i < kGroups; ++i) means[i] = result.groups[i].meanMs();
  std::sort(means.begin(), means.end());
  result.mean_ms = means[kGroups / 2];
  result.spread_fraction = (means.back() - means.front()) / result.mean_ms;
  result.exceeded_soft_limit = result.elapsed_ms > policy.soft_limit_ms;
  return result;
}

}  // namespace timing
}  // namespace hipdnn_ep_test
