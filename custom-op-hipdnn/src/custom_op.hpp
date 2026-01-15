/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 * 
 * MIOpen-based implementation (migrated from hipDNN)
 */

#pragma once

#include "hipdnn.pb.h"
#include <algorithm>
#include <future>
#include <mutex>
#include <vector>
#include <unordered_map>
#include <memory>

// MIOpen includes (replacing hipDNN)
#include <miopen/miopen.h>
#include <hip/hip_runtime.h>

#include "morphizen/vaip.hpp"

namespace hipdnn {
using namespace vaip_core;

/// @brief MIOpen-based custom operation implementation
/// Migrated from hipDNN to direct MIOpen usage
class HipdnnCustomOp : public CustomOpImp {
public:
  HipdnnCustomOp(std::shared_ptr<const PassContext> context,
                 const std::shared_ptr<MetaDefProto>& meta_def,
                 onnxruntime::Model* model);

  virtual ~HipdnnCustomOp();

private:
  virtual void Compute(const OrtApi* api,
                       OrtKernelContext* context) const override final;

  // Build and compile using MIOpen
  void BuildAndCompileMIOpen();
  
  // Helper to load metadata
  void LoadGraphMetadata();

private:
  // Proto parameter
  HipdnnParamProto hipdnn_proto_;
  
  // MIOpen handle (per kernel)
  miopenHandle_t miopen_handle_;
  
  // MIOpen descriptors for convolution
  miopenTensorDescriptor_t x_desc_;  // Input
  miopenTensorDescriptor_t w_desc_;  // Weights
  miopenTensorDescriptor_t y_desc_;  // Output
  miopenTensorDescriptor_t b_desc_;  // Bias (optional)
  miopenConvolutionDescriptor_t conv_desc_;
  
  // Convolution algorithm and workspace
  miopenConvFwdAlgorithm_t conv_algo_;
  size_t workspace_size_;
  void* workspace_;
  
  // Tensor shapes (stored at compile time)
  std::vector<int64_t> x_shape_;
  std::vector<int64_t> w_shape_;
  std::vector<int64_t> y_shape_;
  std::vector<int64_t> b_shape_;
  
  // Graph I/O info
  size_t num_inputs_;
  size_t num_outputs_;
  std::vector<std::vector<int64_t>> output_shapes_;
  
  // Bias support
  bool has_bias_;
  
  // Data type
  miopenDataType_t data_type_;
  
  // Device ID
  int device_id_;
};

} // namespace hipdnn
