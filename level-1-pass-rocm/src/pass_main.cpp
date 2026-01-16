// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include <hip/hip_runtime.h>
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
 * It:
 * 1. Checks AMD GPU availability
 * 2. Logs device information
 * 3. Creates and runs Level-2 sub-passes (Conv, Gemm)
 */
struct Level1Rocm {
  Level1Rocm(IPass& self) : self_{self} {}

  void process_run_subpasses(Graph& graph) {
    auto& pass_proto = self_.get_pass_proto();
    
    // Create Level-2 passes from config (subPass in passRocmParam)
    all_passes_ = IPass::create_passes(
        self_.get_context(),
        pass_proto.pass_rocm_param().sub_pass());
    
    MY_LOG(1) << "[HIP EP Level-1] Created " << all_passes_.size() << " sub-passes";
    
    // Run all Level-2 passes (Conv, Gemm pattern matching)
    IPass::run_passes(all_passes_, graph);
    
    MY_LOG(1) << "[HIP EP Level-1] Completed running sub-passes";
  }

  void process(IPass& self, Graph& graph) {
    // 1. Check AMD GPU availability
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    
    if (err != hipSuccess || device_count == 0) {
      MY_LOG(1) << "[HIP EP Level-1] No AMD GPU found, skipping ROCm passes";
      MY_LOG(1) << "[HIP EP Level-1] hipGetDeviceCount error: " 
                << hipGetErrorString(err);
      return;
    }

    // 2. Log device info
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, 0) == hipSuccess) {
      MY_LOG(1) << "[HIP EP Level-1] Using device: " << props.name;
      MY_LOG(1) << "[HIP EP Level-1] Compute capability: " 
                << props.major << "." << props.minor;
      MY_LOG(2) << "[HIP EP Level-1] Total memory: " 
                << (props.totalGlobalMem / (1024 * 1024)) << " MB";
    }

    MY_LOG(1) << "[HIP EP Level-1] AMD GPU available, ROCm acceleration enabled";
    
    // 3. Run Level-2 sub-passes
    process_run_subpasses(graph);
  }

  IPass& self_;
  std::vector<std::shared_ptr<IPass>> all_passes_;
};

DEFINE_VAIP_PASS(Level1Rocm, vaip_pass_level1_rocm)
