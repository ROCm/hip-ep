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
 * 
 * Note: Level-2 sub-passes (Conv, Gemm) are registered separately
 * and will run as part of the normal pass pipeline.
 */
struct Level1Rocm {
  Level1Rocm(IPass& self) : self_{self} {}

  void process(IPass& self, Graph& graph) {
    // 1. Check AMD GPU availability
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    
    if (err != hipSuccess || device_count == 0) {
      LOG(WARNING) << "[ROCm EP Level-1] No AMD GPU found, skipping ROCm passes";
      LOG(WARNING) << "[ROCm EP Level-1] hipGetDeviceCount error: " 
                   << hipGetErrorString(err);
      return;
    }

    // 2. Log device info
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, 0) == hipSuccess) {
      LOG(INFO) << "[ROCm EP Level-1] Using device: " << props.name;
      LOG(INFO) << "[ROCm EP Level-1] Compute capability: " 
                << props.major << "." << props.minor;
      LOG(INFO) << "[ROCm EP Level-1] Total memory: " 
                << (props.totalGlobalMem / (1024 * 1024)) << " MB";
    }

    LOG(INFO) << "[ROCm EP Level-1] AMD GPU available, ROCm acceleration enabled";
  }

  IPass& self_;
};

DEFINE_VAIP_PASS(Level1Rocm, vaip_pass_level1_rocm)
