/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ut_timing_policy.h"
#include "ut_timing_device_profiles.h"
#include "ut_timing_context.h"
#include "ut_rank.h"
#include "ut_rank_input.h"
#include "ut_gpu_frequency.h"

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

  // Balanced candidate rounds preserve paired ratios through a common 10%
  // clock drop. Absolute group spread is diagnostic; it cannot change order.
  for (bool inject_drift : {false, true}) {
    size_t batches = 0;
    const std::array<double, 3> costs{{0.01, 0.0092, 0.00925}};
    auto ranked = measureInterleaved(Policy::standard(), costs.size(),
        [&](size_t candidate, uint32_t n) {
          const double drift_factor = inject_drift && batches / costs.size() >= 20 ? 1.1 : 1.0;
          ++batches;
          return costs[candidate] * drift_factor * n;
        });
    for (const auto& result : ranked) checkTotals(result);
    for (size_t group = 0; group < kGroups; ++group)
      for (size_t candidate = 1; candidate < costs.size(); ++candidate)
        assert(ranked[candidate].groups[group].calls == ranked[0].groups[group].calls);
    const auto decision = decideRank(ranked, 0);
    assert(decision.winner == 1 && decision.runner_up == 2);
    assert(decision.insert_nondefault && decision.nondefault_tie);
    assert(decision.needs_confirm == decision.winner_nonconverged);
    assert(std::abs(decision.versus_default.median - 0.92) < 1e-10);
  }
  for (double ratio : {0.975, 0.98, 0.985}) {
    auto ranked = measureInterleaved(Policy::standard(), 2,
        [&](size_t candidate, uint32_t n) { return 0.01 * n * (candidate ? ratio : 1.0); });
    const auto decision = decideRank(ranked, 0);
    assert(decision.needs_confirm && decision.insert_nondefault == (ratio < 0.98));
  }
  auto equal = measureInterleaved(Policy::standard(), 2,
      [](size_t, uint32_t n) { return 0.01 * n; });
  const auto equal_decision = decideRank(equal, 0);
  assert(equal_decision.winner == 0 && !equal_decision.insert_nondefault);
  assert(equal_decision.needs_confirm && !equal_decision.obvious && equal_decision.uncertain);
  assert(decideRank(equal, 0, true).needs_confirm);
  auto singleton = measureInterleaved(Policy::standard(), 1,
      [](size_t, uint32_t n) { return 0.01 * n; });
  const auto single_decision = decideRank(singleton, 0, true);
  assert(single_decision.winner == 0 && !single_decision.needs_confirm);
  assert(finalists(singleton, 0) == std::vector<size_t>{0});

  // Winner uncertainty is distinct from ordinary absolute group spread.
  auto fixture = [](std::initializer_list<std::array<double, kGroups>> costs) {
    std::vector<Result> out;
    for (const auto& c : costs) {
      Result r;
      for (size_t g = 0; g < kGroups; ++g) r.groups[g] = Group{c[g], 1, 1};
      summarize(r, Policy::standard()); out.push_back(r);
    }
    return out;
  };
  auto overlap = fixture({{{1,1,1,1,1}}, {{.90,.91,.92,.93,.94}}, {{.91,.90,.93,.92,.95}}});
  auto ambiguous = decideRank(overlap, 0);
  assert(ambiguous.winner == 1 && ambiguous.winner_overlap && ambiguous.needs_confirm);
  assert(ambiguous.uncertain && ambiguous.selected == 1 && ambiguous.insert_nondefault);
  // One slow group cannot veto a winner whose median clearly beats the default.
  auto one_slow_group = fixture({{{1,1,1,1,1}}, {{.911,1.005,.982,.927,.938}}});
  auto veto = decideRank(one_slow_group, 0);
  assert(veto.entry_boundary && veto.needs_confirm);
  assert(veto.selected == 1 && veto.insert_nondefault);
  // A non-default winner that is not >2% faster keeps the default.
  auto marginal = decideRank(fixture({{{1,1,1,1,1}}, {{.97,.99,.985,.99,.98}}}), 0);
  assert(marginal.winner == 1 && !marginal.insert_nondefault && marginal.selected == 0);
  auto separated = fixture({{{1,1.1,1.2,1.3,1.4}}, {{.8,.88,.96,1.04,1.12}}, {{.9,.99,1.08,1.17,1.26}}});
  auto spread_only = decideRank(separated, 0);
  assert(spread_only.winner_nonconverged && spread_only.needs_confirm);
  assert(!spread_only.uncertain && spread_only.selected == 1 && spread_only.insert_nondefault);

  // Grouped ranking retains five groups and identical complete-call blocks,
  // while a candidate executes consecutive capped batches within each block.
  for (bool inject_invalid : {false, true}) {
    size_t calls_to_runner = 0, previous_candidate = 99, consecutive = 0, longest = 0;
    bool started = false, invalid_used = false;
    auto grouped_rank = measureInterleaved(Policy::standard(), 3,
        [&](size_t candidate, uint32_t n) {
          ++calls_to_runner; assert(n <= 64);
          if (started) {
            consecutive = candidate == previous_candidate ? consecutive + 1 : 1;
            previous_candidate = candidate; longest = std::max(longest, consecutive);
            if (inject_invalid && !invalid_used && candidate == 1) { invalid_used = true; return 0.0; }
          }
          return n * (candidate == 1 ? .009 : .01);
        }, [&] { started = true; }, true);
    assert(calls_to_runner > 10 && longest >= 3);
    for (const auto& r : grouped_rank) { checkTotals(r); assert(r.largest_batch_calls <= 64); }
    for (size_t g = 0; g < kGroups; ++g)
      assert(grouped_rank[0].groups[g].calls == grouped_rank[1].groups[g].calls &&
             grouped_rank[1].groups[g].calls == grouped_rank[2].groups[g].calls);
    assert(grouped_rank[1].invalid_batches == (inject_invalid ? 1u : 0u));
    assert(decideRank(grouped_rank, 0).winner == 1);
  }

  // A nontransitive pairwise triangle must never drive the sort comparator.
  auto triangle = fixture({{{1,1,1,1,1}}, {{.7,1.1,.8,1.1,1.2}}, {{.8,1.1,.9,1.2,.9}}});
  assert(pairedRatio(triangle[1], triangle[2]).median < 1.0);
  assert(pairedRatio(triangle[2], triangle[0]).median < 1.0);
  assert(pairedRatio(triangle[0], triangle[1]).median < 1.0);
  const auto triangle_decision = decideRank(triangle, 0);
  assert(triangle_decision.winner == 2); // same-default scores 1, 1.1, .9
  for (int repeat = 0; repeat < 10; ++repeat)
    assert(decideRank(triangle, 0).winner == triangle_decision.winner);

  // A noisy candidate forces balanced continuation of every candidate, while
  // valid slow groups remain in the result and a nonconverged cell is kept.
  size_t noise_batches = 0;
  auto noisy_rank = measureInterleaved(Policy::standard(), 3,
      [&](size_t candidate, uint32_t n) {
        const size_t round = noise_batches++ / 3;
        return n * (candidate == 2 && round % 2 ? 1800.0 : 1500.0);
      });
  assert(!noisy_rank[0].early_stop && !noisy_rank[1].early_stop);
  assert(noisy_rank[0].calls >= 10 && noisy_rank[2].calls >= 10);
  assert(noisy_rank[0].calls == noisy_rank[2].calls);

  size_t invalid_batches = 0;
  auto recovered_rank = measureInterleaved(Policy::screen(), 2,
      [&](size_t candidate, uint32_t n) {
        if (candidate == 1 && ++invalid_batches <= 2) return 0.0;
        return double(n) * (candidate ? 0.9 : 1.0);
      });
  assert(recovered_rank[1].invalid_batches == 2 && recovered_rank[0].invalid_batches == 0);
  assert(decideRank(recovered_rank, 0).winner == 1);
  try {
    (void)measureInterleaved(Policy::screen(), 2,
        [](size_t candidate, uint32_t n) { return candidate ? 0.0 : double(n); });
    assert(false);
  } catch (const InvalidTimingShare&) {}
  try { (void)finalists({}, 0); assert(false); } catch (const std::invalid_argument&) {}
  try { (void)decideRank(singleton, 1); assert(false); } catch (const std::invalid_argument&) {}
  std::vector<Result> many(7, singleton.front());
  for (size_t i = 0; i < many.size(); ++i) many[i].mean_ms = 0.01 * (1.0 + 0.001 * i);
  many[6].mean_ms = 0.02;  // default still participates outside the screen band
  assert(finalists(many, 6) == (std::vector<size_t>{0, 1, 2, 3, 6}));

  // Sequential references compare absolute medians: their unmatched group
  // indices do not share a drift history and must not be ranked by pair ratios.
  std::vector<Result> sequential(3);
  const double sequential_costs[3][5] = {{1, 1, 100, 100, 100},
                                        {1, 1, 80, 110, 110},
                                        {.98, .98, 95, 95, 95}};
  for (size_t candidate = 0; candidate < sequential.size(); ++candidate) {
    for (size_t group = 0; group < kGroups; ++group) {
      sequential[candidate].groups[group].calls = 1;
      sequential[candidate].groups[group].elapsed_ms = sequential_costs[candidate][group];
    }
    summarize(sequential[candidate], Policy::standard());
  }
  assert(decideRank(sequential, 0).winner == 2);
  const auto absolute = decideRank(sequential, 0, false, true);
  assert(absolute.winner == 1 && absolute.insert_nondefault);
  assert(std::abs(absolute.versus_default.median - .8) < 1e-10);
  bool measurement_started = false;
  size_t pilot_batches = 0, observed_batches = 0;
  (void)measureInterleaved(Policy::screen(), 2, [&](size_t, uint32_t n) {
    if (measurement_started) ++observed_batches; else ++pilot_batches;
    return n * .01;
  }, [&] { measurement_started = true; });
  assert(pilot_batches >= 2 && observed_batches >= 10);
  assert(rankConfigInt("default;id=12", "id") == 12);

  assert(rankCsvFields("leaf,shape,1,\"wmma=0;splits=8\"\r") ==
         (std::vector<std::string>{"leaf", "shape", "1", "wmma=0;splits=8"}));
  assert(rankConfigInt("wmma=0;splits=8;bkv=16", "splits") == 8);
  assert(rankShapeFields("B1_H32_G8_D128_len4096_win128_sink").at("sink") == 1);
  try { (void)rankConfigInt("id=3x", "id"); assert(false); } catch (const std::invalid_argument&) {}
  try { (void)rankConfigInt("id=3;id=4", "id"); assert(false); } catch (const std::invalid_argument&) {}
  try { (void)rankShapeFields("B1_B2"); assert(false); } catch (const std::invalid_argument&) {}
#ifndef _WIN32
  GpuFrequency frequency;
  frequency.sample(); assert(frequency.json() == "null");
#endif
  {
    CaseScope scope("test/rank", "shape", "complete_entry");
    RankOptions options;
    options.configs = {"default", "fast"};
    RankPhase phase;
    phase.name = "standard"; phase.candidates = {0, 1}; phase.results = equal;
    printRank(options, {0.0, 0.0}, {phase}, equal_decision, false, 1.0);
  }

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
