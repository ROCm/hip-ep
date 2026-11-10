/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "CorelibModelCustomOp.h"
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

// A whole model the level-1 pass handed to RyzenAI CoreLib.
static morphizen::ExecutionProvider *create_corelib_model_provider(
    std::shared_ptr<const morphizen::PassContext> &context,
    const morphizen::MetaDefProto &meta_def) {
  return new morphizen::ExecutionProviderImp<
      mlir_compilation::CorelibModelCustomOp>(context, meta_def);
}

namespace {
static ::morphizen::StaticPluginRegister __register_corelib_model(
    "morphizen_custom_op_CORELIBMODEL", "create_execution_provider",
    reinterpret_cast<void *>(&create_corelib_model_provider));
}
