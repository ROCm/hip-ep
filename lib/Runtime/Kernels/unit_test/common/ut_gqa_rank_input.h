/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include <set>
#include "ut_gqa_rank.h"
#include "ut_rank_input.h"

namespace hipdnn_ep_test {
namespace timing {
inline std::string gqaConfigString(const UtGqaConfig& config) {
  char text[128];
  std::snprintf(text, sizeof(text), "wmma=%d;splits=%d;bkv=%d;hb=%d;qw=%d;mt=%d;nd=%d",
                config.wmma, config.splits, config.bkv, config.hb, config.qw, config.mt, config.nd);
  return text;
}
#ifdef HIPDNN_UT_RANK_GQA_ADAPTER
inline UtGqaConfig gqaRequestedConfig(const UtGqaShape& shape, const std::string& request) {
  UtGqaConfig candidate;
  if (request.rfind("id=", 0) == 0) {
    if (!hipdnn_ut_gqa_config(&shape, rankConfigInt(request, "id"), &candidate)) return candidate;
  } else {
    const int count = hipdnn_ut_gqa_config_count(&shape);
    for (int id = 0; id < count; ++id) {
      if (hipdnn_ut_gqa_config(&shape, id, &candidate)) continue;
      if (shape.sq == 1) {
        // LUT splits are exact integers, including counts between online
        // catalog ladder values. The configured entry supports these counts.
        const int splits = rankConfigInt(request, "splits");
        if (splits < 1 || splits > 64 || splits > shape.max_splits) continue;
        candidate.splits = splits;
      }
      if (gqaConfigString(candidate) == request) return candidate;
    }
  }
  throw std::invalid_argument("UT rank: invalid GQA candidate geometry");
}

inline void printGqaCatalog(const UtGqaShape& shape) {
  UtGqaConfig baseline;
  if (hipdnn_ut_gqa_config(&shape, 0, &baseline)) throw std::runtime_error("UT rank: no GQA default");
  std::printf("UT_CANDIDATES {\"leaf\":%s,\"shape\":%s,\"default\":%s,\"configs\":[",
              jsonString(currentCase.leaf).c_str(), jsonString(currentCase.shape).c_str(),
              jsonString(gqaConfigString(baseline)).c_str());
  const int count = hipdnn_ut_gqa_config_count(&shape);
  for (int id = 1; id < count; ++id) {
    UtGqaConfig config;
    if (hipdnn_ut_gqa_config(&shape, id, &config)) throw std::runtime_error("UT rank: catalog failed");
    std::printf("%s{\"request\":\"id=%d\",\"final\":%s}", id > 1 ? "," : "", id,
                jsonString(gqaConfigString(config)).c_str());
  }
  std::printf("]}\n"); std::fflush(stdout);
}

template <class Select, class Validate>
RankOptions gqaRankOptions(const UtGqaShape& shape, const RankRow& row, Select select, Validate validate) {
  for (const char* name : {"HIPDNN_GQA_DECODE_SCALAR", "HIPDNN_GQA_DECODE_WMMA",
                          "HIPDNN_GQA_DECODE_SPLITS", "HIPDNN_GQA_DECODE_BKV", "HIPDNN_GQA_CFG"}) {
    const char* value = std::getenv(name);
    if (value && value[0]) throw std::invalid_argument("UT rank: conflicting GQA environment config pin");
  }
  UtGqaConfig baseline;
  if (hipdnn_ut_gqa_config(&shape, 0, &baseline)) throw std::runtime_error("UT rank: no GQA default");
  std::vector<UtGqaConfig> configs{baseline};
  std::set<std::string> seen{gqaConfigString(baseline)};
  RankOptions options;
  options.configs.push_back(gqaConfigString(baseline)); options.priority = row.priority;
  options.sequential_confirm = std::getenv("HIPDNN_UT_RANK_REFERENCE") &&
      std::string(std::getenv("HIPDNN_UT_RANK_REFERENCE")) == "1";
  bool default_requested = false;
  for (const auto& request : row.configs) {
    if (request == "default") { default_requested = true; continue; }
    const UtGqaConfig candidate = gqaRequestedConfig(shape, request);
    const auto identity = gqaConfigString(candidate);
    if (seen.insert(identity).second) { configs.push_back(candidate); options.configs.push_back(identity); }
  }
  if (!default_requested) throw std::invalid_argument("UT rank: candidate CSV must include default");
  options.select = [configs, select](size_t candidate) { select(configs.at(candidate)); };
  options.validate = [choose = options.select, validate](size_t candidate) {
    choose(candidate); return validate();
  };
  return options;
}
#endif
}  // namespace timing
}  // namespace hipdnn_ep_test
