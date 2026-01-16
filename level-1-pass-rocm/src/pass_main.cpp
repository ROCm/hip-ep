// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include <hip/hip_runtime.h>
#include <glog/logging.h>

#include "morphizen/vaip.hpp"

using namespace vaip_core;

/**
 * Level-1 Pass: ROCm Orchestrator
 *
 * This pass serves as the entry point for ROCm-based operations.
 * It:
 * 1. Checks AMD GPU availability
 * 2. Logs device information
 * 3. Dynamically loads and runs Level-2 sub-passes (Conv, Gemm)
 */
struct Level1Rocm {
  Level1Rocm(IPass& self) : self_{self} {}

  void process_run_subpasses(Graph& graph) {
    auto& pass_proto = self_.get_pass_proto();

    // Get sub-passes from configuration
    // The vaip_config.json defines subPass array inside passRocmParam
    const auto& rocm_param = pass_proto.pass_rocm_param();
    
    // Create Level-2 passes from config
    all_passes_ = IPass::create_passes(
        self_.get_context(),
        rocm_param.sub_pass());

    // Run all Level-2 passes (Conv, Gemm) on the graph
    IPass::run_passes(all_passes_, graph);
  }

  void process(IPass& self, Graph& graph) {
    // 1. Check AMD GPU availability
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    
    if (err != hipSuccess || device_count == 0) {
      LOG(WARNING) << "[HIP EP Level-1] No AMD GPU found, skipping ROCm passes";
      LOG(WARNING) << "[HIP EP Level-1] hipGetDeviceCount error: " 
                   << hipGetErrorString(err);
      return;
    }

    // 2. Log device info
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, 0) == hipSuccess) {
      LOG(INFO) << "[HIP EP Level-1] Using device: " << props.name;
      LOG(INFO) << "[HIP EP Level-1] Compute capability: " 
                << props.major << "." << props.minor;
      LOG(INFO) << "[HIP EP Level-1] Total memory: " 
                << (props.totalGlobalMem / (1024 * 1024)) << " MB";
    }

    // 3. Run Level-2 sub-passes
    LOG(INFO) << "[HIP EP Level-1] Running sub-passes...";
    process_run_subpasses(graph);
    LOG(INFO) << "[HIP EP Level-1] Sub-passes completed";
  }

  IPass& self_;
  std::vector<std::shared_ptr<IPass>> all_passes_;
};

DEFINE_VAIP_PASS(Level1Rocm, vaip_pass_level1_rocm)
