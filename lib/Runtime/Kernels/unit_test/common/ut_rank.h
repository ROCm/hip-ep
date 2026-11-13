/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include "ut_timing_context.h"
#include "ut_timing_policy.h"

namespace hipdnn_ep_test {
namespace timing {

inline constexpr const char* kRankVersion = "ut-rank-v3";

// Config strings name final launches, after op-specific legality checks and
// clamp/implementation normalization. The default is always index zero.
struct RankOptions {
  std::vector<std::string> configs;
  std::function<void(size_t)> select;
  // Execute the selected candidate and check it against the owning CPU
  // reference before any timing. Return relL2; throw on failure.
  std::function<double(size_t)> validate;
  bool priority = false;
  bool sequential_confirm = false;  // independent validation reference only
};

struct RankPhase {
  std::string name;
  std::vector<size_t> candidates;
  std::vector<Result> results;
  double wall_ms = 0.0;
  std::string frequency = "null";
  uint64_t switch_settle_calls = 0;
  double switch_settle_wall_ms = 0.0;
  std::string schedule = "batch_interleaved";
  // Non-default candidates dropped after a repeated invalid-timing failure.
  std::vector<size_t> excluded;
};

inline void printRatio(const Ratio& ratio) {
  std::printf("{\"median\":%.17g,\"low\":%.17g,\"high\":%.17g,\"groups\":[",
              ratio.median, ratio.low, ratio.high);
  for (size_t group = 0; group < kGroups; ++group)
    std::printf("%s%.17g", group ? "," : "", ratio.groups[group]);
  std::printf("]}");
}

inline void printRank(const RankOptions& options, const std::vector<double>& rel_l2,
                      const std::vector<RankPhase>& phases, const RankDecision& decision,
                      bool retested, double wall_ms, const std::string& trigger = "none") {
  const auto& final = phases.back();
  const size_t winner = final.candidates[decision.winner];
  const size_t second = final.candidates[decision.runner_up];
  bool converged = true;
  for (const auto& result : final.results) converged = converged && result.converged;
  std::printf("UT_RANK {\"version\":\"%s\",\"policy\":\"%s\",\"leaf\":%s,"
              "\"shape\":%s,\"evidence_binding\":%s,\"default\":0,\"candidates\":[",
              kRankVersion, kPolicyVersion, jsonString(currentCase.leaf).c_str(),
              jsonString(currentCase.shape).c_str(),
              jsonString(std::getenv("HIPDNN_UT_EVIDENCE_BINDING") ?
                  std::getenv("HIPDNN_UT_EVIDENCE_BINDING") : "").c_str());
  for (size_t i = 0; i < options.configs.size(); ++i)
    std::printf("%s{\"config\":%s,\"cpu_correct\":true,\"rel_l2\":%.17g}",
                i ? "," : "", jsonString(options.configs[i]).c_str(), rel_l2[i]);
  std::printf("],\"phases\":[");
  for (size_t phase_index = 0; phase_index < phases.size(); ++phase_index) {
    const auto& phase = phases[phase_index];
    size_t default_index = 0;
    while (phase.candidates.at(default_index) != 0) ++default_index;
    const auto phase_decision = decideRank(phase.results, default_index, false,
        phase.name == "screen" || phase.name == "sequential_confirm" || phase.name == "sequential_standard");
    double gpu_ms = 0.0, pilot_ms = 0.0;
    for (const auto& result : phase.results) {
      gpu_ms += result.elapsed_ms; pilot_ms += result.pilot_ms;
    }
    std::printf("%s{\"name\":%s,\"wall_ms\":%.17g,\"gpu_ms\":%.17g,"
                "\"pilot_ms\":%.17g,\"frequency\":%s,\"measurements\":[", phase_index ? "," : "",
                jsonString(phase.name).c_str(), phase.wall_ms, gpu_ms, pilot_ms, phase.frequency.c_str());
    for (size_t i = 0; i < phase.results.size(); ++i) {
      const auto& result = phase.results[i];
      std::printf("%s{\"candidate\":%zu,\"median_ms\":%.17g,\"gpu_ms\":%.17g,"
                  "\"calls\":%llu,\"invalid_batches\":%llu,\"converged\":%s,"
                  "\"group_means_ms\":[", i ? "," : "", phase.candidates[i],
                  result.mean_ms, result.elapsed_ms,
                  static_cast<unsigned long long>(result.calls),
                  static_cast<unsigned long long>(result.invalid_batches),
                  result.converged ? "true" : "false");
      for (size_t group = 0; group < kGroups; ++group)
        std::printf("%s%.17g", group ? "," : "", result.groups[group].meanMs());
      std::printf("],\"group_gpu_ms\":[");
      for (size_t group = 0; group < kGroups; ++group)
        std::printf("%s%.17g", group ? "," : "", result.groups[group].elapsed_ms);
      std::printf("],\"group_calls\":[");
      for (size_t group = 0; group < kGroups; ++group)
        std::printf("%s%llu", group ? "," : "",
                    static_cast<unsigned long long>(result.groups[group].calls));
      std::printf("],\"versus_default\":");
      printRatio(pairedRatio(result, phase.results[default_index]));
      std::printf(",\"versus_best\":");
      printRatio(pairedRatio(result, phase.results[phase_decision.winner]));
      std::printf("}");
    }
    std::printf("],\"switch_settle_calls\":%llu,\"switch_settle_wall_ms\":%.17g,\"schedule\":%s,"
                "\"excluded\":[",
                static_cast<unsigned long long>(phase.switch_settle_calls), phase.switch_settle_wall_ms,
                jsonString(phase.schedule).c_str());
    for (size_t i = 0; i < phase.excluded.size(); ++i)
      std::printf("%s%zu", i ? "," : "", phase.excluded[i]);
    std::printf("]}");
  }
  std::printf("],\"confirm_trigger\":%s,\"winner\":%zu,\"runner_up\":%zu,\"versus_default\":", jsonString(trigger).c_str(), winner, second);
  printRatio(decision.versus_default);
  std::printf(",\"versus_second\":"); printRatio(decision.versus_second);
  std::printf(",\"selected\":%zu,\"uncertain\":%s,\"decision_status\":\"%s\",\"score_versus_second\":%.17g,"
              "\"diagnostic\":%s,\"switch_settle\":%s,\"insert_nondefault\":%s,\"obvious\":%s,\"nondefault_tie\":%s,"
              "\"retested\":%s,\"priority\":%s,\"sequential_reference\":%s,"
              "\"converged\":%s,\"wall_ms\":%.17g,\"frequency\":%s}\n",
              final.candidates[decision.selected], decision.uncertain ? "true" : "false",
              decision.uncertain ? "winner_unresolved" : "resolved", decision.score_versus_second,
              jsonString(std::getenv("HIPDNN_UT_RANK_DIAGNOSTIC") ? std::getenv("HIPDNN_UT_RANK_DIAGNOSTIC") : "").c_str(),
              std::getenv("HIPDNN_UT_RANK_SETTLE") && std::string(std::getenv("HIPDNN_UT_RANK_SETTLE")) == "1" ? "true" : "false",
              decision.insert_nondefault ? "true" : "false", decision.obvious ? "true" : "false",
              decision.nondefault_tie ? "true" : "false", retested ? "true" : "false",
              options.priority ? "true" : "false", options.sequential_confirm ? "true" : "false",
              converged ? "true" : "false", wall_ms, final.frequency.c_str());
  std::fflush(stdout);
}

}  // namespace timing
}  // namespace hipdnn_ep_test
