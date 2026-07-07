/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include <glog/logging.h>

#include "./custom_op_generic.hpp"
#include "morphizen/morphizen.hpp"

static morphizen::ExecutionProvider* create_execution_provider_imp(
    std::shared_ptr<const morphizen::PassContext>& context,
    const morphizen::MetaDefProto& meta_def) {
  return new morphizen::ExecutionProviderImp<
      morphizen::custom_op_generic::MyCustomOp>(context, meta_def);
}

namespace {
static ::morphizen::StaticPluginRegister
    __register("morphizen_custom_op_GENERIC", "create_execution_provider",
               (void*)&create_execution_provider_imp);
} // namespace
