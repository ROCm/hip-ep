/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#define MORPHIZEN_CUSTOM_OP // TODO: remove it
#include "morphizen/morphizen.hpp"

namespace morphizen {
namespace custom_op_generic {
using namespace morphizen;

class MyCustomOp : public ::morphizen::CustomOpImp {
public:
  MyCustomOp(std::shared_ptr<const PassContext> context,
             const std::shared_ptr<MetaDefProto>& meta_def,
             onnxruntime::Model* model);

  virtual ~MyCustomOp();

private:
  virtual void Compute(const OrtApi* api,
                       OrtKernelContext* context) const override final;
};
} // namespace custom_op_generic
} // namespace morphizen
