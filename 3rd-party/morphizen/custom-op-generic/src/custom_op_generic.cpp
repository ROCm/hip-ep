/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./custom_op_generic.hpp"
#include "morphizen/onnxruntime_api.hpp"
#include <glog/logging.h>
#include <morphizen/env_config.hpp>

DEF_ENV_PARAM(MORPHIZEN_DEBUG_CUSTOM_OP_GENERIC, "0")
#define MY_LOG(n)                                                              \
  LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_CUSTOM_OP_GENERIC) >= n)

namespace morphizen {
namespace custom_op_generic {
using namespace morphizen;
MyCustomOp::MyCustomOp(std::shared_ptr<const PassContext> context,
                       const std::shared_ptr<MetaDefProto>& meta_def,
                       onnxruntime::Model* model)
    : CustomOpImp(context, meta_def, model) {
  MY_LOG(1) << "MyCustomOp ctor: ";
}
MyCustomOp::~MyCustomOp() { MY_LOG(1) << "MyCustomOp dtor: "; }
void MyCustomOp::Compute(const OrtApi* api, OrtKernelContext* context) const {
  // this->ComputeCpu(api, context);
}
} // namespace custom_op_generic
} // namespace morphizen
