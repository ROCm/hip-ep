/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include <algorithm>
#include <map>
#include "ut_gemm_rank.h"
#include "ut_rank_input.h"

namespace hipdnn_ep_test {
namespace timing {
inline UtGemmShape gemmRankShape(const std::string& text, int dtype) {
  // GEMM's legacy _bias flag is op-specific. Keep the common parser strict for
  // other ops by translating only this flag before parsing.
  std::string normalized = text;
  const size_t flag = normalized.find("_bias");
  if (flag != std::string::npos && (flag + 5 == normalized.size() || normalized[flag + 5] == '_'))
    normalized.insert(flag + 5, "1");
  const auto f = rankShapeFields(normalized);
  for (const auto& field : f)
    if (field.first != "M" && field.first != "N" && field.first != "K" &&
        field.first != "ta" && field.first != "tb" && field.first != "bias" &&
        field.first != "crows" && field.first != "ccols")
      throw std::invalid_argument("UT rank: unknown GEMM shape field");
  auto required = [&](const char* name) {
    auto it = f.find(name);
    if (it == f.end()) throw std::invalid_argument("UT rank: missing GEMM shape field");
    return it->second;
  };
  const int m = required("M"), n = required("N"), k = required("K");
  const int ta = required("ta"), tb = required("tb");
  const bool bias = f.count("bias") && f.at("bias");
  const int crows = f.count("crows") ? f.at("crows") : 1;
  const int ccols = f.count("ccols") ? f.at("ccols") : n;
  if (m <= 0 || n <= 0 || k <= 0 || ta > 1 || tb > 1 ||
      (f.count("bias") && f.at("bias") != 1) ||
      (f.count("crows") != f.count("ccols")) ||
      (f.count("crows") && !bias) || (crows != 1 && crows != m) || (ccols != 1 && ccols != n))
    throw std::invalid_argument("UT rank: invalid GEMM shape or C broadcast");
  return {m, n, k, ta, tb, dtype, crows, ccols, 1.0f, bias ? 1.0f : 0.0f};
}

#ifdef HIPDNN_UT_RANK_GEMM_ADAPTER
inline UtGemmConfig gemmRequestedConfig(const UtGemmShape& shape, const std::string& request) {
  UtGemmConfig candidate;
  if (request.rfind("id=", 0) == 0) {
    if (!hipdnn_ut_gemm_config(&shape, rankConfigInt(request, "id"), &candidate)) return candidate;
  } else {
    for (int id = 0; id < hipdnn_ut_gemm_config_count(&shape); ++id)
      if (!hipdnn_ut_gemm_config(&shape, id, &candidate) && candidate.geometry == request) return candidate;
  }
  throw std::invalid_argument("UT rank: GEMM config is not a legal production descriptor");
}
inline void printGemmCatalog(const UtGemmShape& shape) {
  UtGemmConfig baseline;
  if (hipdnn_ut_gemm_config(&shape, -1, &baseline))
    throw std::runtime_error("UT rank: GEMM no-LUT default cannot be resolved");
  std::map<std::string, std::string> final{{baseline.launch, baseline.geometry}};
  std::printf("UT_CANDIDATES {\"leaf\":%s,\"shape\":%s,\"default\":%s,\"configs\":[",
      jsonString(currentCase.leaf).c_str(), jsonString(currentCase.shape).c_str(), jsonString(baseline.geometry).c_str());
  bool comma = false;
  for (int id = 0; id < hipdnn_ut_gemm_config_count(&shape); ++id) {
    UtGemmConfig config;
    if (hipdnn_ut_gemm_config(&shape, id, &config)) continue;
    auto canonical = final.emplace(config.launch, config.geometry).first;
    std::printf("%s{\"request\":\"id=%d\",\"geometry\":%s,\"final\":%s,\"final_launch\":%s}",
        comma ? "," : "", id, jsonString(config.geometry).c_str(),
        jsonString(canonical->second).c_str(), jsonString(config.launch).c_str());
    comma = true;
  }
  std::printf("]}\n"); std::fflush(stdout);
}
template <class Validate>
RankOptions gemmRankOptions(const UtGemmShape& shape, const RankRow& row, Validate validate) {
  UtGemmConfig baseline;
  if (hipdnn_ut_gemm_config(&shape, -1, &baseline))
    throw std::runtime_error("UT rank: GEMM requires lookup policy, empty LUT and no production pins");
  for (const auto& request : row.configs) {
    if (request.rfind("default;", 0) == 0) {
      if (!std::getenv("HIPDNN_UT_RANK_REFERENCE"))
        throw std::invalid_argument("UT rank: frozen default is for independent reference only");
      baseline = gemmRequestedConfig(shape, request.substr(8));
    }
  }
  std::vector<UtGemmConfig> configs{baseline};
  std::map<std::string, size_t> seen{{baseline.launch, 0}};
  RankOptions options;
  options.configs.push_back(baseline.geometry);
  options.requests.push_back({baseline.geometry});
  options.final_launches.push_back(baseline.launch);
  // --with-catalog may omit an old geometry that aliases the default. Keep
  // those aliases even when the input CSV contains only the default row.
  for (int id = 0; id < hipdnn_ut_gemm_config_count(&shape); ++id) {
    UtGemmConfig candidate;
    if (!hipdnn_ut_gemm_config(&shape, id, &candidate) && candidate.launch == baseline.launch &&
        std::find(options.requests[0].begin(), options.requests[0].end(), candidate.geometry) == options.requests[0].end())
      options.requests[0].push_back(candidate.geometry);
  }
  options.priority = row.priority;
  options.sequential_confirm = std::getenv("HIPDNN_UT_RANK_REFERENCE") &&
      std::string(std::getenv("HIPDNN_UT_RANK_REFERENCE")) == "1";
  bool default_requested = false;
  for (const auto& request : row.configs) {
    if (request == "default" || request.rfind("default;", 0) == 0) { default_requested = true; continue; }
    const UtGemmConfig candidate = gemmRequestedConfig(shape, request);
    auto insertion = seen.emplace(candidate.launch, configs.size());
    if (insertion.second) {
      configs.push_back(candidate);
      options.configs.push_back(candidate.geometry);
      options.requests.push_back({});
      options.final_launches.push_back(candidate.launch);
    }
    auto& aliases = options.requests[insertion.first->second];
    for (const auto& alias : {candidate.geometry, request})
      if (std::find(aliases.begin(), aliases.end(), alias) == aliases.end()) aliases.push_back(alias);
  }
  if (!default_requested) throw std::invalid_argument("UT rank: candidate CSV must include default");
  options.select = [shape, configs](size_t index) {
    if (hipdnn_ut_gemm_select(&shape, &configs.at(index)))
      throw std::runtime_error("UT rank: cannot select GEMM production launch");
  };
  options.validate = [select = options.select, validate](size_t index) { select(index); return validate(); };
  return options;
}
#endif
}  // namespace timing
}  // namespace hipdnn_ep_test
