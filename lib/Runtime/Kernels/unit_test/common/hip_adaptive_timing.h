/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#ifdef _WIN32
#include <windows.h>
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#endif

#include "ut_timing_policy.h"
#include "ut_timing_device_profiles.h"
#include "ut_timing_context.h"
#include "ut_rank.h"
#include "ut_gpu_frequency.h"

namespace hipdnn_ep_test {
namespace timing {

inline void checkHip(hipError_t status, const char* operation) {
  if (status != hipSuccess)
    throw std::runtime_error(std::string("UT timing: ") + operation + ": " +
                             hipGetErrorString(status));
}

inline Policy environmentPolicy() {
  const char* value = std::getenv("HIPDNN_UT_TIMING");
  if (!value || !value[0] || std::strcmp(value, "standard") == 0)
    return Policy::standard();
  if (std::strcmp(value, "confirm") == 0) return Policy{};
  if (std::strcmp(value, "screen") == 0) return Policy::screen();
  throw std::invalid_argument("HIPDNN_UT_TIMING must be screen, standard or confirm");
}

inline const char* envOr(const char* name, const char* fallback = "") {
  const char* value = std::getenv(name);
  return value && value[0] ? value : fallback;
}

inline void printTimingMode() {
  std::printf("UT timing: %s (%s); adaptive complete-call counts are in UT_TIMING\n",
              environmentPolicy().name(), kPolicyVersion);
}

inline std::string runtimeModule() {
#ifdef _WIN32
  for (const char* name : {"amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"}) {
    if (HMODULE module = GetModuleHandleA(name)) {
      char path[32768]{};
      const DWORD size = GetModuleFileNameA(module, path, sizeof(path));
      if (size > 0 && size < sizeof(path)) return std::string(path, size);
    }
  }
#endif
  return "";
}

class Events {
 public:
  Events() {
    checkHip(hipEventCreate(&start), "create start event");
    try {
      checkHip(hipEventCreate(&stop), "create stop event");
    } catch (...) {
      (void)hipEventDestroy(start);
      throw;
    }
  }
  ~Events() {
    (void)hipEventDestroy(stop);
    (void)hipEventDestroy(start);
  }
  Events(const Events&) = delete;
  Events& operator=(const Events&) = delete;
  hipEvent_t start{}, stop{};
};

struct NoPreparation {
  void operator()(uint32_t) const {}
};

// Copies and launch are the existing leaf's operands and COMPLETE op entry.
// hold enqueues that leaf's stream gate. It must not synchronize the stream.
// All mandatory wrapper kernels remain between the same two timing events.
template <class Copies, class Launch, class Hold, class Prepare = NoPreparation>
double adaptiveRotatedMeanMs(hipStream_t stream, const Copies& copies,
                            Launch&& launch, int* release, Hold&& hold,
                            bool requested_warm, size_t settle_bytes,
                            const char* entry = "unqualified_entry",
                            const char* complete_entry = nullptr,
                            Prepare prepare = {}, const RankOptions* ranking = nullptr) {
  EntryScope entry_scope(complete_entry);
  if (ranking && (ranking->configs.empty() || !ranking->select || !ranking->validate))
    throw std::invalid_argument("UT rank: incomplete candidate request");
  std::vector<double> correctness;
  if (ranking) {
    for (size_t candidate = 0; candidate < ranking->configs.size(); ++candidate) {
      const double rel_l2 = ranking->validate(candidate);
      if (!(rel_l2 >= 0.0) || !std::isfinite(rel_l2))
        throw std::runtime_error("UT rank: invalid CPU correctness result");
      correctness.push_back(rel_l2);
    }
    ranking->select(0);
  }
  Policy policy = environmentPolicy();
  int device = 0, runtime = 0, driver = 0;
  hipDeviceProp_t props{};
  checkHip(hipGetDevice(&device), "get device");
  checkHip(hipGetDeviceProperties(&props, device), "get device properties");
  checkHip(hipRuntimeGetVersion(&runtime), "get runtime version");
  checkHip(hipDriverGetVersion(&driver), "get driver version");
  const int copies_count = copies.count();
  if (copies_count < 1 || !release)
    throw std::runtime_error("UT timing: operand rotation or coherent stream gate unavailable");
  const bool actual_warm = requested_warm || copies_count == 1;
  const DeviceProfile profile = deviceProfile(props.gcnArchName, runtime, entry,
                                               actual_warm, currentCase.shape);
  policy.max_calls_per_batch = profile.batch_call_cap;
  if (policy.preset == Preset::Screen) {
    policy.target_ms = profile.screen_target_ms;
    policy.soft_limit_ms = profile.screen_soft_limit_ms;
  }
  void* release_device = nullptr;
  checkHip(hipHostGetDevicePointer(&release_device, release, 0), "map stream gate");
  volatile int* flag = release;
  uint64_t turn = 0;
  auto call = [&] {
    const int copy = static_cast<int>(turn++ % copies_count);
    if constexpr (std::is_void_v<std::invoke_result_t<Launch&, int>>) {
      launch(copy);
    } else {
      const auto status = launch(copy);
      if (status != 0)
        throw std::runtime_error("UT timing: complete op returned status " +
                                 std::to_string(static_cast<int>(status)));
    }
    checkHip(hipGetLastError(), "complete op launch");
  };
  const auto wall_start = std::chrono::steady_clock::now();
  uint64_t preparation_batches = 0;
  double preparation_wall_ms = 0.0;
  auto prepareBatch = [&](uint32_t count) {
    if constexpr (!std::is_same_v<Prepare, NoPreparation>) {
      const auto start = std::chrono::steady_clock::now();
      prepare(count);
      checkHip(hipGetLastError(), "prepare batch scratch");
      checkHip(hipStreamSynchronize(stream), "drain batch preparation");
      ++preparation_batches;
      preparation_wall_ms += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start).count();
    }
  };
  if (copies_count > 1 && copies.bytes() > 0) {
    const size_t settle = std::min<size_t>(
        copies_count, settle_bytes / copies.bytes() + (settle_bytes % copies.bytes() != 0));
    for (size_t done = 0; done < settle;) {
      const uint32_t count = static_cast<uint32_t>(std::min<size_t>(
          settle - done, policy.max_calls_per_batch));
      prepareBatch(count);
      for (uint32_t i = 0; i < count; ++i) call();
      done += count;
    }
  }
  checkHip(hipStreamSynchronize(stream), "drain warmup/rotation");
  const uint64_t prep_calls = turn;
  const double prep_wall_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - wall_start).count();
  Events events;
  GpuFrequency frequency;
  bool observe_frequency = false;
  auto runner = [&](uint32_t count) {
    // Reuse events only after the previous stop has completed. Reading every
    // bounded batch avoids retaining a start timestamp across a long command
    // sequence on PAL runtimes. No synchronization behind an unreleased gate.
    prepareBatch(count);
    *flag = 0;
    try {
      hold(static_cast<const int*>(release_device));
      checkHip(hipGetLastError(), "enqueue stream gate");
      checkHip(hipEventRecord(events.start, stream), "record start");
      for (uint32_t i = 0; i < count; ++i) call();
      checkHip(hipEventRecord(events.stop, stream), "record stop");
      *flag = 1;
      checkHip(hipEventSynchronize(events.stop), "wait stop");
      float ms = 0.0f;
      checkHip(hipEventElapsedTime(&ms, events.start, events.stop), "elapsed time");
      if (observe_frequency) frequency.sample();
      return static_cast<double>(ms);
    } catch (...) {
      *flag = 1;
      (void)hipStreamSynchronize(stream);
      throw;
    }
  };
  auto emitTiming = [&](const Result& result, double wall_ms) {
  const uint64_t record_id = rememberResult(result.mean_ms);
  // Spread only decides whether the first round may end the measurement; it
  // never rejects data. Every valid batch contributes, including slow batches.
  // The soft limit never drops data.
  char max_spread[32] = "null";
  if (std::isfinite(policy.max_spread))
    std::snprintf(max_spread, sizeof(max_spread), "%.17g", policy.max_spread);
  std::printf("UT_TIMING {\"policy\":\"%s\",\"preset\":\"%s\","
              "\"record_id\":%llu,\"leaf\":%s,\"shape\":%s,\"op_entry\":%s,"
              "\"evidence_binding\":%s,\"host\":%s,\"device_name\":%s,"
              "\"runtime_module\":%s,"
              "\"pci_bus\":%d,\"pci_device\":%d,\"pci_domain\":%d,"
              "\"profile\":\"%s\",\"entry\":\"%s\","
              "\"device\":%d,\"arch\":\"%s\",\"runtime\":%d,\"driver\":%d,"
              "\"cache\":\"%s\",\"copies\":%d,\"bytes_per_copy\":%llu,"
              "\"rotation_bytes\":%llu,\"requested_warm\":%s,"
              "\"target_ms\":%.17g,\"soft_limit_ms\":%.17g,\"warmup_ms\":%.17g,"
              "\"min_calls_per_group\":%u,\"call_goal\":%llu,\"floor_ms\":%.17g,"
              "\"max_spread\":%s,\"invalid_retry_limit\":%u,"
              "\"invalid_batch_fraction\":%.17g,"
              "\"events_per_batch\":2,\"batch_call_cap\":%u,"
              "\"largest_batch_calls\":%u,\"pilot_ms\":%.17g,"
              "\"pilot_calls\":%llu,\"pilot_batches\":%llu,\"prep_calls\":%llu,"
              "\"prep_wall_ms\":%.17g,\"calls\":%llu,"
              "\"preparation_batches\":%llu,\"preparation_wall_ms\":%.17g,"
              "\"batches\":%llu,\"gpu_ms\":%.17g,\"wall_ms\":%.17g,"
              "\"median_ms\":%.17g,\"spread_fraction\":%.17g,"
              "\"invalid_batches\":%llu,\"invalid_calls\":%llu,"
              "\"soft_limit_exceeded\":%s,\"early_stop\":%s,\"converged\":%s,"
              "\"group_means_ms\":[",
              kPolicyVersion, policy.name(), static_cast<unsigned long long>(record_id),
              jsonString(currentCase.leaf).c_str(), jsonString(currentCase.shape).c_str(),
              jsonString(currentCase.op_entry).c_str(),
              jsonString(envOr("HIPDNN_UT_EVIDENCE_BINDING")).c_str(),
              jsonString(envOr("COMPUTERNAME", envOr("HOSTNAME"))).c_str(),
              jsonString(props.name).c_str(), jsonString(runtimeModule()).c_str(),
              props.pciBusID, props.pciDeviceID, props.pciDomainID,
              profile.name, entry,
              device, props.gcnArchName, runtime, driver,
              requested_warm ? "warm" : (copies_count > 1 ? "operand_rotation" : "allocation_fallback"),
              copies_count, static_cast<unsigned long long>(copies.bytes()),
              static_cast<unsigned long long>(copies.bytes()) * copies_count,
              requested_warm ? "true" : "false", policy.target_ms, policy.soft_limit_ms,
              policy.warmup_ms,
              policy.min_calls_per_group, static_cast<unsigned long long>(policy.call_goal),
              policy.floor_ms, max_spread, policy.invalid_retries, policy.invalid_fraction,
              policy.max_calls_per_batch,
              result.largest_batch_calls, result.pilot_ms,
              static_cast<unsigned long long>(result.pilot_calls),
              static_cast<unsigned long long>(result.pilot_batches),
              static_cast<unsigned long long>(prep_calls), prep_wall_ms,
              static_cast<unsigned long long>(result.calls),
              static_cast<unsigned long long>(preparation_batches), preparation_wall_ms,
              static_cast<unsigned long long>(result.batches), result.elapsed_ms, wall_ms,
              result.mean_ms, result.spread_fraction,
              static_cast<unsigned long long>(result.invalid_batches),
              static_cast<unsigned long long>(result.invalid_calls),
              result.exceeded_soft_limit ? "true" : "false",
              result.early_stop ? "true" : "false",
              result.converged ? "true" : "false");
  for (size_t i = 0; i < kGroups; ++i)
    std::printf("%s%.17g", i ? "," : "", result.groups[i].meanMs());
  std::printf("],\"group_calls\":[");
  for (size_t i = 0; i < kGroups; ++i)
    std::printf("%s%llu", i ? "," : "",
                static_cast<unsigned long long>(result.groups[i].calls));
  std::printf("],\"group_gpu_ms\":[");
  for (size_t i = 0; i < kGroups; ++i)
    std::printf("%s%.17g", i ? "," : "", result.groups[i].elapsed_ms);
  std::printf("],\"group_batches\":[");
  for (size_t i = 0; i < kGroups; ++i)
    std::printf("%s%llu", i ? "," : "",
                static_cast<unsigned long long>(result.groups[i].batches));
  std::printf("],\"frequency\":%s,\"frequency_status\":%s}\n",
              frequency.json().c_str(), jsonString(frequency.status()).c_str());
  std::fflush(stdout);
  };
  auto elapsed = [](const auto& start) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
  };
  if (!ranking) {
    const Result result = measureInterleaved(policy, 1,
        [&](size_t, uint32_t count) { return runner(count); },
        [&] { observe_frequency = true; }).front();
    emitTiming(result, elapsed(wall_start));
    return result.mean_ms;
  }
  const Policy base_policy = policy;
  std::vector<RankPhase> phases;
  const std::string diagnostic = envOr("HIPDNN_UT_RANK_DIAGNOSTIC");
  if (!diagnostic.empty() && diagnostic != "interleaved_standard" && diagnostic != "sequential_standard" &&
      diagnostic != "grouped_standard")
    throw std::invalid_argument("unknown UT rank diagnostic schedule");
  const bool switch_settle = std::strcmp(envOr("HIPDNN_UT_RANK_SETTLE"), "1") == 0;
  auto runPhase = [&](const char* name, Policy selected_policy,
                      const std::vector<size_t>& candidates, bool interleaved) {
    selected_policy.max_calls_per_batch = base_policy.max_calls_per_batch;
    if (selected_policy.preset == Preset::Screen) {
      selected_policy.target_ms = profile.screen_target_ms;
      selected_policy.soft_limit_ms = profile.screen_soft_limit_ms;
    }
    policy = selected_policy;
    frequency = GpuFrequency{};
    RankPhase phase;
    phase.name = name; phase.candidates = candidates;
    const bool grouped = diagnostic == "grouped_standard" ||
        (diagnostic.empty() && std::strcmp(envOr("HIPDNN_UT_RANK_GROUPED", "1"), "0") != 0);
    phase.schedule = !interleaved ? "sequential" : grouped ? "group_interleaved" : "batch_interleaved";
    const auto stage_start = std::chrono::steady_clock::now();
    std::vector<double> candidate_wall(candidates.size(), 0.0);
    size_t previous = std::numeric_limits<size_t>::max();
    auto candidateRunner = [&](size_t candidate, uint32_t count) {
      ranking->select(candidates[candidate]);
      const auto batch_start = std::chrono::steady_clock::now();
      if (switch_settle && candidate != previous && !requested_warm && copies_count > 1 && copies.bytes() > 0) {
        const auto settle_start = std::chrono::steady_clock::now();
        const size_t settle = std::min<size_t>(copies_count,
            settle_bytes / copies.bytes() + (settle_bytes % copies.bytes() != 0));
        for (size_t done = 0; done < settle;) {
          const uint32_t n = static_cast<uint32_t>(std::min<size_t>(settle - done, policy.max_calls_per_batch));
          prepareBatch(n);
          for (uint32_t i = 0; i < n; ++i) call();
          done += n;
        }
        checkHip(hipStreamSynchronize(stream), "drain candidate rotation settle");
        phase.switch_settle_calls += settle;
        phase.switch_settle_wall_ms += elapsed(settle_start);
      }
      previous = candidate;
      const double ms = runner(count);
      candidate_wall[candidate] += elapsed(batch_start);
      return ms;
    };
    // An invalid-timing failure is repeated once. A repeated failure of a
    // non-default sequential candidate drops it; the default must measure.
    std::vector<size_t> positions;
    if (interleaved) {
      for (int attempt = 0;; ++attempt) {
        try {
          observe_frequency = false;
          std::fill(candidate_wall.begin(), candidate_wall.end(), 0.0);
          phase.results = measureInterleaved(policy, candidates.size(), candidateRunner,
              [&] { observe_frequency = true; }, grouped);
          break;
        } catch (const InvalidTimingShare&) {
          if (attempt) throw;
        }
      }
      for (size_t i = 0; i < candidates.size(); ++i) positions.push_back(i);
    } else {
      phase.candidates.clear();
      for (size_t i = 0; i < candidates.size(); ++i) {
        for (int attempt = 0;; ++attempt) {
          try {
            observe_frequency = false;
            candidate_wall[i] = 0.0;
            phase.results.push_back(measureInterleaved(policy, 1, [&](size_t, uint32_t count) {
              return candidateRunner(i, count);
            }, [&] { observe_frequency = true; }).front());
            phase.candidates.push_back(candidates[i]);
            positions.push_back(i);
            break;
          } catch (const InvalidTimingShare&) {
            if (!attempt) continue;
            if (candidates[i] == 0) throw;
            phase.excluded.push_back(candidates[i]);
            break;
          }
        }
      }
    }
    phase.wall_ms = elapsed(stage_start);
    phase.frequency = frequency.json();
    for (size_t i = 0; i < phase.results.size(); ++i) {
      emitTiming(phase.results[i], candidate_wall[positions[i]]);
      emitCsvBinding(currentCase.shape, ranking->configs[phase.candidates[i]], "PASS",
                     correctness[phase.candidates[i]], phase.results[i].mean_ms);
    }
    phases.push_back(std::move(phase));
  };
  std::vector<size_t> all;
  for (size_t i = 0; i < ranking->configs.size(); ++i) all.push_back(i);
  std::vector<size_t> selected;
  bool retested = false;
  std::string trigger = "none";
  if (!diagnostic.empty()) {
    selected = all;
    runPhase(diagnostic.c_str(), Policy::standard(), selected, diagnostic != "sequential_standard");
  } else if (ranking->sequential_confirm) {
    selected = all;
    runPhase("sequential_confirm", Policy{}, selected, false);
  } else {
    runPhase("screen", Policy::screen(), all, false);
    for (size_t index : finalists(phases.back().results, 0))
      selected.push_back(phases.back().candidates[index]);
    runPhase("standard", Policy::standard(), selected, true);
    const size_t default_index = static_cast<size_t>(
        std::find(selected.begin(), selected.end(), 0) - selected.begin());
    const auto first_decision = decideRank(phases.back().results, default_index, ranking->priority);
    if (first_decision.needs_confirm) {
      trigger.clear();
      auto reason = [&](bool enabled, const char* name) {
        if (enabled) { if (!trigger.empty()) trigger += ";"; trigger += name; }
      };
      reason(ranking->priority, "priority");
      reason(first_decision.entry_boundary, "entry_boundary");
      reason(first_decision.winner_overlap, "winner_range_crosses_1");
      reason(first_decision.winner_nonconverged, "winner_or_second_nonconverged");
      reason(first_decision.inconsistent_pair, "pair_score_disagreement");
      runPhase("confirm", Policy{}, selected, true);
      retested = true;
    }
  }
  const size_t default_index = static_cast<size_t>(
      std::find(selected.begin(), selected.end(), 0) - selected.begin());
  const auto decision = decideRank(phases.back().results, default_index, false,
      ranking->sequential_confirm || diagnostic == "sequential_standard");
  printRank(*ranking, correctness, phases, decision, retested, elapsed(wall_start), trigger);
  return phases.back().results[decision.selected].mean_ms;
}

}  // namespace timing
}  // namespace hipdnn_ep_test
