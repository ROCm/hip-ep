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

// Invalid event durations above the policy allowance. A ranking run may repeat
// the affected measurement; an ordinary UT cell fails.
struct InvalidTimingShare : std::runtime_error {
  InvalidTimingShare()
      : std::runtime_error("UT timing: invalid event durations above the allowed share") {}
};

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

// The same complete-call policy serves single measurements and balanced
// candidate comparisons. runner(candidate, N) executes N complete calls.
// Every comparison round uses a common call count and group boundary, so all
// candidates cover the same drift history. Satisfied candidates keep contributing
// valid batches until the other candidates reach their unchanged policy goals.
struct NoMeasurementStart { void operator()() const {} };

template <class Runner, class Start = NoMeasurementStart>
std::vector<Result> measureInterleaved(const Policy& policy, size_t candidates,
                                       Runner&& runner, Start start = {}, bool grouped = false) {
  policy.validate();
  if (!candidates) throw std::invalid_argument("UT timing: empty candidate set");
  std::vector<Result> results(candidates);
  std::vector<uint64_t> attempted(candidates, 0);
  std::vector<double> per_call_ms(candidates, 0.0);
  size_t round = 0;
  auto validBatch = [&](size_t candidate, uint32_t& count) {
    Result& result = results[candidate];
    for (;;) {
      result.largest_batch_calls = std::max(result.largest_batch_calls, count);
      const double ms = runner(candidate, count);
      ++attempted[candidate];
      if (ms > 0.0 && std::isfinite(ms)) return ms;
      ++result.invalid_batches;
      result.invalid_calls += count;
      const double allowed = std::max(double(policy.invalid_retries),
          std::ceil(policy.invalid_fraction * double(attempted[candidate])));
      if (double(result.invalid_batches) > allowed)
        throw InvalidTimingShare();
      count = std::max(uint32_t(1), count / 2);
    }
  };
  auto runRound = [&](uint32_t count, size_t group, bool pilot) {
    for (size_t turn = 0; turn < candidates; ++turn) {
      const size_t candidate = (round + turn) % candidates;
      uint32_t actual = count;
      const double ms = validBatch(candidate, actual);
      Result& result = results[candidate];
      if (pilot) {
        result.pilot_ms += ms;
        result.pilot_calls += actual;
        ++result.pilot_batches;
      } else {
        result.groups[group].elapsed_ms += ms;
        result.groups[group].calls += actual;
        ++result.groups[group].batches;
      }
      per_call_ms[candidate] = ms / actual;
    }
    ++round;
  };
  // A ranking group can keep one candidate's batches contiguous to avoid
  // repeatedly switching its wrapper/cache state. Every candidate receives
  // the same complete-call block; events still bound at most the profile cap.
  auto runGroupBlock = [&](uint32_t total, size_t group) {
    for (size_t turn = 0; turn < candidates; ++turn) {
      const size_t candidate = (round + turn) % candidates;
      uint64_t done = 0;
      while (done < total) {
        uint32_t actual = static_cast<uint32_t>(std::min<uint64_t>(total - done, policy.max_calls_per_batch));
        const double ms = validBatch(candidate, actual);
        auto& result = results[candidate];
        result.groups[group].elapsed_ms += ms;
        result.groups[group].calls += actual;
        ++result.groups[group].batches;
        per_call_ms[candidate] = ms / actual;
        done += actual;
      }
    }
    ++round;
  };
  runRound(1, 0, true);
  for (;;) {
    bool done = true;
    uint32_t count = 1;
    for (size_t candidate = 0; candidate < candidates; ++candidate) {
      const double remaining = policy.warmup_ms - results[candidate].pilot_ms;
      if (remaining > 0.0) {
        done = false;
        count = std::max(count, nextCalls(remaining, per_call_ms[candidate], 0,
                                         policy.max_calls_per_batch));
      }
    }
    if (done) break;
    runRound(count, 0, true);
  }
  start();  // warmup has drained; optional observers start outside measured batches
  const double goal_ms = policy.target_ms / kGroups;
  const double floor_ms = policy.floor_ms / kGroups;
  auto fill = [&](size_t group_index, uint64_t min_calls, uint64_t call_goal) {
    for (;;) {
      bool done = true;
      uint32_t count = 1;
      for (size_t candidate = 0; candidate < candidates; ++candidate) {
        const Group& group = results[candidate].groups[group_index];
        if (group.calls >= min_calls &&
            (group.elapsed_ms >= goal_ms ||
             (group.calls >= call_goal && group.elapsed_ms >= floor_ms))) continue;
        done = false;
        double remaining = goal_ms - group.elapsed_ms;
        if (call_goal != kNoCallGoal) {
          const double to_goal = double(call_goal - std::min(call_goal, group.calls)) *
                                 per_call_ms[candidate];
          remaining = std::min(remaining, std::max(to_goal, floor_ms - group.elapsed_ms));
        }
        const uint64_t minimum = group.calls < min_calls ? min_calls - group.calls : 0;
        count = std::max(count, nextCalls(remaining, per_call_ms[candidate], minimum,
            grouped ? std::numeric_limits<uint32_t>::max() : policy.max_calls_per_batch));
      }
      if (done) break;
      if (grouped) runGroupBlock(count, group_index);
      else runRound(count, group_index, false);
    }
  };
  const uint64_t call_goal = policy.call_goal / kGroups + (policy.call_goal % kGroups != 0);
  for (size_t group = 0; group < kGroups; ++group) fill(group, 1, call_goal);
  bool converged = true;
  for (auto& result : results) {
    summarize(result, policy);
    converged = converged && result.converged;
  }
  if (converged) {
    for (auto& result : results) result.early_stop = true;
  } else {
    for (size_t group = 0; group < kGroups; ++group)
      fill(group, policy.min_calls_per_group, kNoCallGoal);
    for (auto& result : results) summarize(result, policy);
  }
  return results;
}

template <class Runner>
Result measure(const Policy& policy, Runner&& runner) {
  return measureInterleaved(policy, 1,
      [&](size_t, uint32_t count) { return runner(count); }).front();
}

struct Ratio {
  std::array<double, kGroups> groups{};
  double median = 1.0, low = 1.0, high = 1.0;
  bool crosses(double line) const { return low <= line && high >= line; }
};

inline Ratio pairedRatio(const Result& candidate, const Result& reference) {
  Ratio ratio;
  for (size_t group = 0; group < kGroups; ++group) {
    const double c = candidate.groups[group].meanMs();
    const double r = reference.groups[group].meanMs();
    if (!(c > 0.0) || !(r > 0.0) || !std::isfinite(c) || !std::isfinite(r))
      throw std::invalid_argument("UT rank: invalid paired group");
    ratio.groups[group] = c / r;
  }
  auto sorted = ratio.groups;
  std::sort(sorted.begin(), sorted.end());
  ratio.low = sorted.front(); ratio.median = sorted[kGroups / 2]; ratio.high = sorted.back();
  return ratio;
}

inline std::vector<size_t> finalists(const std::vector<Result>& screen, size_t default_index) {
  if (screen.empty() || default_index >= screen.size())
    throw std::invalid_argument("UT rank: missing default candidate");
  std::vector<size_t> order;
  for (size_t i = 0; i < screen.size(); ++i) {
    if (!(screen[i].mean_ms > 0.0) || !std::isfinite(screen[i].mean_ms))
      throw std::invalid_argument("UT rank: invalid screen result");
    order.push_back(i);
  }
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return screen[a].mean_ms < screen[b].mean_ms;
  });
  const double limit = screen[order.front()].mean_ms * 1.05;
  std::vector<size_t> selected;
  for (size_t i : order) {
    if (selected.size() == 4 || screen[i].mean_ms > limit) break;
    selected.push_back(i);
  }
  if (std::find(selected.begin(), selected.end(), default_index) == selected.end())
    selected.push_back(default_index);
  return selected;
}

struct RankDecision {
  size_t winner = 0, runner_up = 0, selected = 0;
  Ratio versus_default, versus_second;
  double score_versus_second = 1.0;
  bool insert_nondefault = false, obvious = false, nondefault_tie = false;
  bool needs_confirm = false, uncertain = false;
  bool winner_overlap = false, winner_nonconverged = false;
  bool entry_boundary = false, inconsistent_pair = false;
};

inline RankDecision decideRank(const std::vector<Result>& results, size_t default_index,
                               bool priority = false, bool absolute = false) {
  if (results.empty() || default_index >= results.size())
    throw std::invalid_argument("UT rank: missing default candidate");
  std::vector<size_t> order;
  std::vector<Ratio> ratios;
  for (size_t i = 0; i < results.size(); ++i) {
    order.push_back(i);
    ratios.push_back(pairedRatio(results[i], results[default_index]));
  }
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return absolute ? results[a].mean_ms < results[b].mean_ms : ratios[a].median < ratios[b].median;
  });
  RankDecision decision;
  decision.winner = order[0];
  decision.runner_up = order.size() > 1 ? order[1] : order[0];
  decision.versus_default = ratios[decision.winner];
  decision.versus_second = pairedRatio(results[decision.winner], results[decision.runner_up]);
  decision.score_versus_second = ratios[decision.winner].median / ratios[decision.runner_up].median;
  if (absolute) {
    decision.versus_default.median = results[decision.winner].mean_ms / results[default_index].mean_ms;
    decision.versus_second.median = results[decision.winner].mean_ms / results[decision.runner_up].mean_ms;
    decision.score_versus_second = decision.versus_second.median;
  }
  const bool nondefault = decision.winner != default_index;
  const bool multiple = order.size() > 1;
  decision.winner_overlap = multiple && decision.versus_second.crosses(1.0);
  decision.winner_nonconverged = multiple &&
      (!results[decision.winner].converged || !results[decision.runner_up].converged);
  decision.inconsistent_pair = multiple && !absolute && decision.versus_second.median > 1.0;
  decision.entry_boundary = nondefault &&
      ((decision.versus_default.median >= 0.97 && decision.versus_default.median <= 0.99) ||
       decision.versus_default.crosses(0.98));
  // The order against the runner-up may stay unresolved after confirm; only
  // the median against the default decides insertion.
  decision.uncertain = !absolute && (decision.winner_overlap || decision.inconsistent_pair);
  decision.insert_nondefault = nondefault && decision.versus_default.median < 0.98;
  decision.selected = decision.insert_nondefault ? decision.winner : default_index;
  decision.obvious = !decision.versus_default.crosses(0.98) &&
                    (order.size() == 1 || !decision.versus_second.crosses(1.0));
  decision.nondefault_tie = nondefault && decision.runner_up != default_index &&
                           decision.versus_second.median > 0.99;
  decision.needs_confirm = multiple && (priority || decision.entry_boundary ||
      decision.winner_overlap || decision.winner_nonconverged || decision.inconsistent_pair);
  return decision;
}

}  // namespace timing
}  // namespace hipdnn_ep_test
