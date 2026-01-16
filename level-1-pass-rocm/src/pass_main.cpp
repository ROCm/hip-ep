// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include <glog/logging.h>
#include <vector>
#include <memory>

#include "morphizen/env_config.hpp"
#include "morphizen/vaip.hpp"

DEF_ENV_PARAM(MORPHIZEN_DEBUG_ROCM, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_ROCM) >= n)

using namespace vaip_core;

/**
 * Level-1 Pass: ROCm Orchestrator
 *
 * This pass serves as the entry point for ROCm-based operations.
 * It creates and runs Level-2 sub-passes (Conv, Gemm) for pattern matching.
 * 
 * TODO: When morphizen adds pass_rocm_param support to PassProto,
 * this pass will use pass_proto.pass_rocm_param().sub_pass() to 
 * dynamically configure sub-passes from vaip_config.json.
 */
struct Level1Rocm {
  Level1Rocm(IPass& self) : self_{self} {}

  void process(IPass& self, Graph& graph) {
    MY_LOG(1) << "[HIP EP Level-1] Starting ROCm pass";
    
    // TODO: Sub-pass orchestration when pass_rocm_param proto support is added:
    // auto& pass_proto = self_.get_pass_proto();
    // all_passes_ = IPass::create_passes(self_.get_context(), 
    //                                    pass_proto.pass_rocm_param().sub_pass());
    // IPass::run_passes(all_passes_, graph);
    
    // For now, Level-2 passes run separately in the pipeline via vaip_config.json
    MY_LOG(1) << "[HIP EP Level-1] Completed";
  }

  IPass& self_;
  std::vector<std::shared_ptr<IPass>> all_passes_;
};

DEFINE_VAIP_PASS(Level1Rocm, vaip_pass_level1_rocm)
