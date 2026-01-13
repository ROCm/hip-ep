/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include "hipdnn.pb.h"
#include <algorithm>
#include <future>
#include <mutex>
#include <vector>
#include <unordered_map>
#include <memory>

#include <hipdnn_frontend.hpp>
#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
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

  // Graph loading and compilation helpers
  void LoadAndCompileGraph();
  void InitializeHeuristicDescriptor();
  void InitializeEngineConfig();
  void ExtractUIDsFromSerializedGraph(const std::vector<uint8_t>& buffer);
  void LoadConstantData(onnxruntime::Model* model);

private:
  // Proto parameter
  HipdnnParamProto hipdnn_proto_;
  
  // hipDNN handle
  hipdnnHandle_t handle_;
  
  // Backend descriptors (mimics Graph class internals)
  std::unique_ptr<hipdnn_frontend::ScopedHipdnnBackendDescriptor> graphDesc_;
  std::unique_ptr<hipdnn_frontend::ScopedHipdnnBackendDescriptor> engineHeuristicDesc_;
  std::unique_ptr<hipdnn_frontend::ScopedHipdnnBackendDescriptor> engineConfigDesc_;
  std::unique_ptr<hipdnn_frontend::ScopedHipdnnBackendDescriptor> executionPlanDesc_;
  
  // Execution resources
  std::vector<char> workspace_;
  
  // UID mappings for variant pack construction
  std::vector<int64_t> input_uids_;        // All graph inputs (runtime + constants)
  std::vector<int64_t> output_uids_;
  std::vector<std::vector<int64_t>> output_shapes_;
  
  // Constant initializer info
  std::vector<std::string> constant_initializer_names_;
  std::vector<int64_t> constant_input_uids_;  // UIDs of constant inputs in graph
  std::vector<std::vector<char>> constant_data_;  // Actual constant data (one per constant)
};

} // namespace hipdnn
