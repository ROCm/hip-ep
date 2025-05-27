/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include <glog/logging.h>

#include "./custom_op_generic.hpp"
#include "morphizen/vaip.hpp"

static vaip_core::ExecutionProvider* create_execution_provider_imp(
    std::shared_ptr<const vaip_core::PassContext>& context,
    const vaip_core::MetaDefProto& meta_def) {
  return new vaip_core::ExecutionProviderImp<
      morphizen::custom_op_generic::MyCustomOp>(context, meta_def);
}

namespace {
// TODO: do we need to rename ::vaip_core to ::morphizen?
static ::vaip_core::StaticPluginRegister
    __register("vaip_custom_op_GENERIC", "create_execution_provider",
               (void*)&create_execution_provider_imp);
} // namespace
