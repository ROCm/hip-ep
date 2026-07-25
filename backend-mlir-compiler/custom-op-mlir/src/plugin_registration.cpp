/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "MlirCustomOp.h"
#include "morphizen/morphizen.hpp"

// Create execution provider implementation for MLIR custom op
static morphizen::ExecutionProvider *create_execution_provider_imp(
    std::shared_ptr<const morphizen::PassContext> &context,
    const morphizen::MetaDefProto &meta_def) {
  return new morphizen::ExecutionProviderImp<mlir_compilation::MlirCustomOp>(
      context, meta_def);
}

// Register custom op with Morphizen
namespace {
static ::morphizen::StaticPluginRegister
    __register("morphizen_custom_op_MLIR", "create_execution_provider",
               reinterpret_cast<void *>(&create_execution_provider_imp));
}
