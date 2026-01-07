/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include "hipdnn.pb.h"
#include <algorithm>
#include <future>
#include <mutex>

#include "morphizen/vaip.hpp"
namespace hipdnn {
using namespace vaip_core;
class HipdnnCustomOp : public CustomOpImp {
public:
  HipdnnCustomOp(std::shared_ptr<const PassContext> context,
                 const std::shared_ptr<MetaDefProto>& meta_def,
                 onnxruntime::Model* model);

  virtual ~HipdnnCustomOp();

private:
  virtual void Compute(const OrtApi* api,
                       OrtKernelContext* context) const override final;

private:
  HipdnnParamProto hipdnn_proto_;
};
} // namespace hipdnn
