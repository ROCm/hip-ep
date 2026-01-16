// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include <glog/logging.h>
#include <memory>
#include <vector>

#include "google/protobuf/util/json_util.h"
#include "morphizen/env_config.hpp"
#include "morphizen/vaip.hpp"
#include "rocm.pb.h"

DEF_ENV_PARAM(MORPHIZEN_DEBUG_ROCM, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_ROCM) >= n)

using namespace vaip_core;

/**
 * Level-1 Pass: ROCm Orchestrator
 *
 * This pass serves as the entry point for ROCm-based operations.
 * It creates and runs Level-2 sub-passes (Conv, Gemm) for pattern matching.
 *
 * The original graph is read-only, so we clone the model and run sub-passes
 * on the cloned graph.
 *
 * Configuration via vaip_config.json pass_generic_param:
 * {
 *   "sub_pass_names": ["vaip-pass_level2_rocm_conv", "vaip-pass_level2_rocm_gemm"]
 * }
 */
struct Level1Rocm {
  Level1Rocm(IPass& self) : self_{self} {}

  void process(IPass& self, Graph& graph) {
    MY_LOG(1) << "[ROCm EP Level-1] Starting ROCm pass";

    // Get pass configuration from pass_generic_param (JSON)
    auto json_param = self_.get_pass_generic_param();
    MY_LOG(1) << "[ROCm EP Level-1] pass_generic_param: " << json_param;

    // Parse the JSON to get sub-pass configuration
    rocm::PassRocmConfigProto config;
    auto status =
        google::protobuf::util::JsonStringToMessage(json_param, &config);
    if (!status.ok()) {
      MY_LOG(1) << "[ROCm EP Level-1] Failed to parse pass_generic_param: "
                << status.ToString();
      return;
    }

    // Clone the model so we can modify the graph
    // The original graph is read-only
    auto& model = VAIP_ORT_API(graph_get_model)(graph);
    auto cloned_model = vaip_core::model_clone(model, 64);
    auto& cloned_graph = VAIP_ORT_API(model_main_graph)(*cloned_model);

    MY_LOG(1) << "[ROCm EP Level-1] Cloned model for sub-pass processing";

    // Create PassProto for each sub-pass and run them on cloned graph
    for (const auto& sub_pass_name : config.sub_pass_names()) {
      MY_LOG(1) << "[ROCm EP Level-1] Creating sub-pass: " << sub_pass_name;

      PassProto sub_pass_proto;
      sub_pass_proto.set_plugin(sub_pass_name);
      sub_pass_proto.set_name(sub_pass_name);

      auto sub_pass = IPass::create_pass(self_.get_context(), sub_pass_proto);
      if (sub_pass) {
        MY_LOG(1) << "[ROCm EP Level-1] Running sub-pass: " << sub_pass_name;
        std::vector<std::shared_ptr<IPass>> passes;
        passes.push_back(std::move(sub_pass));
        IPass::run_passes(passes, cloned_graph);
      }
    }

    MY_LOG(1) << "[ROCm EP Level-1] Completed";
  }

  IPass& self_;
};

DEFINE_VAIP_PASS(Level1Rocm, vaip_pass_level1_rocm)
