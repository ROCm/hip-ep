/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace hipdnn_ep_test {
namespace timing {

inline constexpr const char* kPolicyVersion = "ut-adaptive-v3";
inline constexpr size_t kGroups = 5;
inline constexpr uint64_t kNoCallGoal = std::numeric_limits<uint64_t>::max();

enum class Preset { Screen, Standard, Confirm };

// These are cumulative GPU-time goals for ONE measurement, not per batch.
// The cap bounds the number of complete op calls between two HIP events; it
// is not a cap on total iterations. It must not be raised by a sweep driver.
//
// Excluded warm-up batches run until warmup_ms of GPU time (at least one
// call). A measurement then takes one round per group that stops at the call
// goal (after the GPU-time floor) or at the GPU-time goal, whichever comes
// first, with at least one call per group. It ends there when the spread of
// the five group means is within max_spread. Otherwise every group continues
// to the full GPU-time goal and min_calls_per_group, keeping earlier batches.
struct Policy {
  Preset preset = Preset::Confirm;
  double target_ms = 10000.0;
  double soft_limit_ms = 20000.0;
  double warmup_ms = 25.0;
  uint64_t call_goal = 10000;
  double floor_ms = 100.0;
  double max_spread = 0.01;
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
    p.warmup_ms = 5.0;
    p.call_goal = 100;
    p.floor_ms = 1.0;
    p.max_spread = std::numeric_limits<double>::infinity();
    return p;
  }

  static Policy standard() {
    Policy p;
    p.preset = Preset::Standard;
    p.target_ms = 1000.0;
    p.soft_limit_ms = 2000.0;
    p.warmup_ms = 10.0;
    p.call_goal = 1000;
    p.floor_ms = 10.0;
    p.max_spread = 0.03;
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
        !(warmup_ms >= 0.0) || !std::isfinite(warmup_ms) ||
        call_goal < kGroups || !(floor_ms >= 0.0) || floor_ms > target_ms ||
        !(max_spread > 0.0) ||
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
  bool early_stop = false;
  bool converged = false;
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

inline void summarize(Result& result, const Policy& policy) {
  result.elapsed_ms = 0.0;
  result.calls = 0;
  result.batches = 0;
  std::array<double, kGroups> means{};
  for (size_t i = 0; i < kGroups; ++i) {
    const Group& group = result.groups[i];
    result.elapsed_ms += group.elapsed_ms;
    result.calls += group.calls;
    result.batches += group.batches;
    means[i] = group.meanMs();
  }
  std::sort(means.begin(), means.end());
  result.mean_ms = means[kGroups / 2];
  result.spread_fraction = (means.back() - means.front()) / result.mean_ms;
  result.converged = result.spread_fraction <= policy.max_spread;
  result.exceeded_soft_limit = result.elapsed_ms > policy.soft_limit_ms;
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

  double per_call_ms = 0.0;
  uint32_t warmup_calls = 1;
  do {
    const double ms = validBatch(warmup_calls);
    result.pilot_ms += ms;
    result.pilot_calls += warmup_calls;
    ++result.pilot_batches;
    per_call_ms = ms / warmup_calls;
    warmup_calls = nextCalls(policy.warmup_ms - result.pilot_ms, per_call_ms, 0,
                             policy.max_calls_per_batch);
  } while (result.pilot_ms < policy.warmup_ms);
  const double group_goal_ms = policy.target_ms / kGroups;
  const double group_floor_ms = policy.floor_ms / kGroups;

  auto fill = [&](Group& group, uint64_t min_calls, uint64_t call_goal) {
    auto done = [&] {
      return group.calls >= min_calls &&
             (group.elapsed_ms >= group_goal_ms ||
              (group.calls >= call_goal && group.elapsed_ms >= group_floor_ms));
    };
    while (!done()) {
      double remaining_ms = group_goal_ms - group.elapsed_ms;
      if (call_goal != kNoCallGoal) {
        const double to_goal =
            double(call_goal - std::min(call_goal, group.calls)) * per_call_ms;
        remaining_ms = std::min(remaining_ms,
                                std::max(to_goal, group_floor_ms - group.elapsed_ms));
      }
      const uint64_t minimum = group.calls < min_calls ? min_calls - group.calls : 0;
      uint32_t count = nextCalls(remaining_ms, per_call_ms, minimum,
                                 policy.max_calls_per_batch);
      const double ms = validBatch(count);
      group.elapsed_ms += ms;
      group.calls += count;
      ++group.batches;
      per_call_ms = ms / count;
    }
  };

  const uint64_t group_call_goal =
      policy.call_goal / kGroups + (policy.call_goal % kGroups != 0);
  for (auto& group : result.groups) fill(group, 1, group_call_goal);
  summarize(result, policy);
  if (result.converged) {
    result.early_stop = true;
    return result;
  }
  for (auto& group : result.groups) fill(group, policy.min_calls_per_group, kNoCallGoal);
  summarize(result, policy);
  return result;
}

}  // namespace timing
}  // namespace hipdnn_ep_test
