/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include <set>
#include "ut_matmul_nbits_rank.h"
#include "ut_rank_input.h"

namespace hipdnn_ep_test {
namespace timing {
inline std::string nbitsConfigString(const UtNbitsConfig& config) {
  const char* route = config.route == 1 ? "wmma" : config.route == 2 ? "gemv" :
                      config.route == 3 ? "gemv_m" : "naive";
  char text[256];
  std::snprintf(text, sizeof(text),
      "route=%s;bm=%d;bn=%d;swizzle=%d;wt_m=%d;wt_n=%d;bk=%d;fused=%d;"
      "ext=%d;variant=%d;threads=%d;tile_n=%d;tile_m=%d", route,
      config.bm, config.bn, config.swizzle, config.wt_m, config.wt_n, config.bk,
      config.fused, config.ext, config.variant, config.threads, config.tile_n, config.tile_m);
  return text;
}
#ifdef HIPDNN_UT_RANK_NBITS_ADAPTER
inline void printNbitsCatalog(const UtNbitsShape& shape) {
  std::printf("UT_CANDIDATES {\"leaf\":%s,\"shape\":%s,\"configs\":[",
              jsonString(currentCase.leaf).c_str(), jsonString(currentCase.shape).c_str());
  const int count = hipdnn_ut_nbits_config_count(&shape);
  bool comma = false;
  for (int id = 0; id < count; ++id) {
    UtNbitsConfig config;
    if (hipdnn_ut_nbits_config(&shape, id, &config) != 0) continue;
    std::printf("%s{\"request\":\"id=%d\",\"final\":%s}", comma ? "," : "", id,
                jsonString(nbitsConfigString(config)).c_str());
    comma = true;
  }
  std::printf("]}\n"); std::fflush(stdout);
}

template <class Validate>
RankOptions nbitsRankOptions(const UtNbitsShape& shape, const RankRow& row, Validate validate) {
  for (const char* name : {"HIPDNN_MATMUL_NBITS_WMMA_CFG", "HIPDNN_EP_MATMUL_SMALLM_CFG",
                          "HIPDNN_EP_MATMUL_SMALLM_I8_CFG", "HIPDNN_EP_MATMUL_DP4A_CFG"}) {
    const char* value = std::getenv(name);
    if (value && value[0]) throw std::invalid_argument("UT rank: conflicting latched production config pin");
  }
  UtNbitsConfig baseline;
  if (hipdnn_ut_nbits_config(&shape, -1, &baseline))
    throw std::runtime_error("UT rank: no final no-LUT default launch captured");
  // Independent sequential reference freezes the first process default ID.
  for (const auto& request : row.configs) {
    if (request.rfind("default;", 0) == 0) {
      if ((!std::getenv("HIPDNN_UT_RANK_REFERENCE") && !std::getenv("HIPDNN_UT_RANK_DIAGNOSTIC")) ||
          hipdnn_ut_nbits_config(&shape, rankConfigInt(request, "id"), &baseline))
        throw std::invalid_argument("UT rank: invalid frozen default");
    }
  }
  std::vector<UtNbitsConfig> configs{baseline};
  std::set<std::string> seen{nbitsConfigString(baseline)};
  RankOptions options;
  options.configs.push_back(nbitsConfigString(baseline));
  options.priority = row.priority;
  options.sequential_confirm = std::getenv("HIPDNN_UT_RANK_REFERENCE") &&
      std::string(std::getenv("HIPDNN_UT_RANK_REFERENCE")) == "1";
  bool default_requested = false;
  for (const auto& request : row.configs) {
    if (request == "default" || request.rfind("default;", 0) == 0) { default_requested = true; continue; }
    UtNbitsConfig candidate;
    if (hipdnn_ut_nbits_config(&shape, rankConfigInt(request, "id"), &candidate))
      throw std::invalid_argument("UT rank: requested config is not a legal final launch");
    const auto identity = nbitsConfigString(candidate);
    if (seen.insert(identity).second) { configs.push_back(candidate); options.configs.push_back(identity); }
  }
  if (!default_requested) throw std::invalid_argument("UT rank: candidate CSV must include default");
  options.select = [shape, configs](size_t candidate) {
    if (hipdnn_ut_nbits_select(&shape, &configs.at(candidate)))
      throw std::runtime_error("UT rank: cannot select exact no-LUT launch; build RANK=1 and use lookup policy");
  };
  options.validate = [select = options.select, validate](size_t candidate) {
    select(candidate); return validate();
  };
  return options;
}
#endif
}  // namespace timing
}  // namespace hipdnn_ep_test
