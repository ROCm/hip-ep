/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "HipCustomOp.h"
#include "morphizen/morphizen.hpp"

static morphizen::ExecutionProvider *create_execution_provider_imp(
    std::shared_ptr<const morphizen::PassContext> &context,
    const morphizen::MetaDefProto &meta_def) {
  return new morphizen::ExecutionProviderImp<hip_compilation::HipCustomOp>(
      context, meta_def);
}

namespace {
static ::morphizen::StaticPluginRegister
    __register("morphizen_custom_op_HIP", "create_execution_provider",
               reinterpret_cast<void *>(&create_execution_provider_imp));
}
