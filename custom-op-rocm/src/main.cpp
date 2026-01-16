// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "morphizen/vaip.hpp"
#include "custom_op.hpp"

using namespace vaip_core;

/**
 * Custom Op Registration Entry Point
 * 
 * This file registers the ROCm custom ops (Conv, Gemm) with VAIP.
 */
extern "C" std::unique_ptr<CustomOp> 
vaip_create_custom_op_rocm_conv(
    std::shared_ptr<const PassContext> context,
    const std::shared_ptr<MetaDefProto>& meta_def,
    onnxruntime::Model* model) {
  return std::make_unique<rocm_ep::RocmCustomOp>(context, meta_def, model);
}

extern "C" std::unique_ptr<CustomOp> 
vaip_create_custom_op_rocm_gemm(
    std::shared_ptr<const PassContext> context,
    const std::shared_ptr<MetaDefProto>& meta_def,
    onnxruntime::Model* model) {
  return std::make_unique<rocm_ep::RocmCustomOp>(context, meta_def, model);
}
