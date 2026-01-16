// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include <hip/hip_runtime.h>
#include <glog/logging.h>
#include <vector>
#include <memory>

#include "morphizen/env_config.hpp"
#include "morphizen/vaip.hpp"
#include "vaip/vaip_ort_api.h"

DEF_ENV_PARAM(MORPHIZEN_DEBUG_ROCM, "0")
DEF_ENV_PARAM_2(XLNX_model_clone_external_data_threshold, "128", int64_t)
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_ROCM) >= n)

using namespace vaip_core;

/**
 * Level-1 Pass: ROCm Orchestrator
 *
 * This pass serves as the entry point for ROCm-based operations.
 * It:
 * 1. Clones the model to avoid modifying the original graph
 * 2. Creates and runs Level-2 sub-passes (Conv, Gemm) on the cloned graph
 * 3. Fuses matched patterns back to the original graph
 */
struct Level1Rocm {
  Level1Rocm(IPass& self) : self_{self} {}

  void process_run_subpasses(Graph& cloned_graph) {
    auto& pass_proto = self_.get_pass_proto();
    
    // Create Level-2 passes from config (subPass in passRocmParam)
    all_passes_ = IPass::create_passes(
        self_.get_context(),
        pass_proto.pass_rocm_param().sub_pass());
    
    MY_LOG(1) << "[HIP EP Level-1] Created " << all_passes_.size() << " sub-passes";
    
    // Run all Level-2 passes (Conv, Gemm pattern matching) on cloned graph
    IPass::run_passes(all_passes_, cloned_graph);
    
    MY_LOG(1) << "[HIP EP Level-1] Completed running sub-passes";
  }

  void process(IPass& self, Graph& graph) {
    // Clone the model to avoid modifying the original graph during pattern matching
    const auto& model = graph_get_model(graph);
    auto cloned_model = model_clone(
        model, VAIP_PROVIDER_OPTION(*self.get_context(),
                                    XLNX_model_clone_external_data_threshold));
    auto& cloned_graph = VAIP_ORT_API(model_main_graph)(*cloned_model);
    
    MY_LOG(1) << "[HIP EP Level-1] Cloned model for pattern matching";
    
    // Run Level-2 sub-passes on the cloned graph
    process_run_subpasses(cloned_graph);
  }

  IPass& self_;
  std::vector<std::shared_ptr<IPass>> all_passes_;
};

DEFINE_VAIP_PASS(Level1Rocm, vaip_pass_level1_rocm)
