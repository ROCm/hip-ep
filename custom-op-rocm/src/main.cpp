// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include <glog/logging.h>
#include "morphizen/morphizen.hpp"
#include "custom_op.hpp"

/**
 * Custom Op Registration Entry Point
 * 
 * This file registers the ROCm custom op with VAIP.
 * The plugin name must match what's in morphizen_config.json execution_providers section.
 */

static morphizen::ExecutionProvider* create_execution_provider_imp(
    std::shared_ptr<const morphizen::PassContext>& context,
    const morphizen::MetaDefProto& meta_def) {
  return new morphizen::ExecutionProviderImp<hip_ep::RocmCustomOp>(
      context, meta_def);
}

namespace {
// Register the ROCm custom op execution provider
// The name must match: "morphizen_custom_op_<DEVICE_NAME>" where DEVICE_NAME
// is the device passed to try_fuse() in the pass
static ::morphizen::StaticPluginRegister
    __register("morphizen_custom_op_ROCm_EP", "create_execution_provider",
               (void*)&create_execution_provider_imp);
}  // namespace
