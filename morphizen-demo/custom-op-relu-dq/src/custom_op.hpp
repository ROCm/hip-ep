/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include "relu_dq.pb.h"
#include <algorithm>
#include <future>
#include <mutex>

#include "morphizen/morphizen.hpp"
namespace relu_dq {
using namespace morphizen;
class MyCustomOp : public CustomOpImp {
public:
  MyCustomOp(std::shared_ptr<const PassContext> context,
             const std::shared_ptr<MetaDefProto>& meta_def,
             onnxruntime::Model* model);

  virtual ~MyCustomOp();

private:
  virtual void Compute(const OrtApi* api,
                       OrtKernelContext* context) const override final;

private:
  ReluDqParamProto relu_dq_proto_;
};
} // namespace relu_dq
