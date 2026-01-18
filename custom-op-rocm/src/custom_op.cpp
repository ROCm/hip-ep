// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "custom_op.hpp"
#include <glog/logging.h>
#include <stdexcept>
#include <set>
#include "google/protobuf/util/json_util.h"
#include "morphizen/env_config.hpp"

DEF_ENV_PARAM(MORPHIZEN_DEBUG_ROCM, "0")
DEF_ENV_PARAM(MORPHIZEN_GPU_TIMEOUT_MS, "5000")
DEF_ENV_PARAM(MORPHIZEN_GPU_WATCHDOG_ENABLED, "1")

#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_ROCM) >= n)

namespace hip_ep {

// ============================================================================
// HipContext Implementation
// ============================================================================

void HipContext::ensure_initialized() {
  std::call_once(init_flag_, [this]() {
    MY_LOG(2) << "[HipContext] ensure_initialized() starting...";
    
    // Check if HIP runtime is available
    MY_LOG(2) << "[HipContext] Calling hipGetDeviceCount...";
    int device_count = 0;
    hipError_t hip_err = hipGetDeviceCount(&device_count);
    MY_LOG(2) << "[HipContext] hipGetDeviceCount returned: " << hip_err 
              << " (" << hipGetErrorString(hip_err) << "), device_count=" << device_count;
    
    if (hip_err != hipSuccess || device_count == 0) {
      LOG(ERROR) << "[HipContext] No AMD GPU detected! HIP error: " 
                 << hipGetErrorString(hip_err);
      initialized_ = false;
      return;
    }
    
    // Get device properties
    MY_LOG(2) << "[HipContext] Calling hipGetDeviceProperties...";
    hipDeviceProp_t props;
    hip_err = hipGetDeviceProperties(&props, 0);
    MY_LOG(2) << "[HipContext] hipGetDeviceProperties returned: " << hip_err;
    if (hip_err == hipSuccess) {
      MY_LOG(1) << "[HipContext] GPU name: " << props.name 
                << ", gcnArchName: " << props.gcnArchName;
    }
    
    // Create HIP stream
    MY_LOG(2) << "[HipContext] Creating HIP stream...";
    hip_err = hipStreamCreate(&stream_);
    MY_LOG(2) << "[HipContext] hipStreamCreate returned: " << hip_err 
              << ", stream=" << stream_;
    if (hip_err != hipSuccess) {
      LOG(ERROR) << "[HipContext] hipStreamCreate failed!";
      initialized_ = false;
      return;
    }
    
    // Create MIOpen handle
    MY_LOG(2) << "[HipContext] Creating MIOpen handle...";
    miopenStatus_t miopen_status = miopenCreate(&miopen_handle_);
    MY_LOG(2) << "[HipContext] miopenCreate returned: " << miopen_status 
              << ", handle=" << miopen_handle_;
    if (miopen_status != miopenStatusSuccess) {
      LOG(ERROR) << "[HipContext] miopenCreate failed!";
      hipStreamDestroy(stream_);
      stream_ = nullptr;
      initialized_ = false;
      return;
    }
    
    // Set MIOpen stream
    MY_LOG(2) << "[HipContext] Setting MIOpen stream...";
    miopen_status = miopenSetStream(miopen_handle_, stream_);
    MY_LOG(2) << "[HipContext] miopenSetStream returned: " << miopen_status;
    
    // Create hipBLASLt handle
    MY_LOG(2) << "[HipContext] Creating hipBLASLt handle...";
    hipblasStatus_t blaslt_status = hipblasLtCreate(&hipblaslt_handle_);
    MY_LOG(2) << "[HipContext] hipblasLtCreate returned: " << blaslt_status 
              << ", handle=" << hipblaslt_handle_;
    if (blaslt_status != HIPBLAS_STATUS_SUCCESS) {
      LOG(ERROR) << "[HipContext] hipblasLtCreate failed!";
      miopenDestroy(miopen_handle_);
      miopen_handle_ = nullptr;
      hipStreamDestroy(stream_);
      stream_ = nullptr;
      initialized_ = false;
      return;
    }
    
    initialized_ = true;
    MY_LOG(1) << "[HipContext] HIP context initialized successfully!";
  });
}

TimeoutStatus HipContext::sync_stream_with_timeout(int timeout_ms) {
  ensure_initialized();
  if (!initialized_) {
    return TimeoutStatus::ERROR;
  }
  
  // Use environment variable default if not specified
  if (timeout_ms <= 0) {
    timeout_ms = ENV_PARAM(MORPHIZEN_GPU_TIMEOUT_MS);
  }
  
  MY_LOG(2) << "[HipContext] Synchronizing stream with " << timeout_ms << "ms timeout...";
  return WaitStreamWithTimeout(stream_, timeout_ms);
}

// ============================================================================
// RocmCustomOp Implementation
// ============================================================================

RocmCustomOp::RocmCustomOp(
    std::shared_ptr<const vaip_core::PassContext> context,
    const std::shared_ptr<vaip_core::MetaDefProto>& meta_def,
    onnxruntime::Model* model)
    : CustomOpImp(context, meta_def, model) {
  
  MY_LOG(2) << "[ROCm CustomOp] Constructor called";
  
  // Get JSON params attached by the pass
  auto json_str = get_meta_def_param();
  MY_LOG(2) << "[ROCm CustomOp] Received JSON params: " << json_str;
  
  // Try to parse as RocmSubgraphProto first (new multi-node format)
  auto status = google::protobuf::util::JsonStringToMessage(json_str, &subgraph_);
  
  if (status.ok() && subgraph_.nodes_size() > 0) {
    // New subgraph format
    is_single_node_ = false;
    MY_LOG(1) << "[ROCm CustomOp] Parsed RocmSubgraphProto with " 
              << subgraph_.nodes_size() << " nodes";
    
    // Initialize per-node runtime data
    node_data_.resize(subgraph_.nodes_size());
    for (size_t i = 0; i < node_data_.size(); ++i) {
      node_data_[i] = std::make_unique<NodeRuntimeData>();
    }
    
    // Build output name to index mapping
    for (int i = 0; i < subgraph_.outputs_size(); ++i) {
      const auto& ext_output = subgraph_.outputs(i);
      output_name_to_index_[ext_output.name()] = i;
      MY_LOG(2) << "[ROCm CustomOp] Output mapping: " << ext_output.name() 
                << " -> index " << i << " (from node " << ext_output.producer_node_id() 
                << ", output " << ext_output.output_index() << ")";
    }
    
  } else {
    // Try to parse as RocmParamProto (legacy single-node format)
    status = google::protobuf::util::JsonStringToMessage(json_str, &single_node_proto_);
    if (!status.ok()) {
      LOG(ERROR) << "[ROCm CustomOp] Failed to parse JSON params: " << status.ToString();
      throw std::runtime_error("Failed to parse ROCm params: " + status.ToString());
    }
    
    is_single_node_ = true;
    MY_LOG(1) << "[ROCm CustomOp] Parsed single-node RocmParamProto, op_type: " 
              << single_node_proto_.op_type();
    
    // Convert to subgraph format for unified execution
    auto* node = subgraph_.add_nodes();
    node->set_node_id(0);
    *node->mutable_params() = single_node_proto_;
    
    // Add external input reference (input from ORT context index 0)
    auto* input_ref = node->add_inputs();
    if (single_node_proto_.op_type() == "conv") {
      const auto& conv_params = single_node_proto_.conv_params();
      if (conv_params.input_names_size() > 0) {
        input_ref->set_external_name(conv_params.input_names(0));
      }
    } else if (single_node_proto_.op_type() == "gemm") {
      const auto& gemm_params = single_node_proto_.gemm_params();
      if (gemm_params.input_names_size() > 0) {
        input_ref->set_external_name(gemm_params.input_names(0));
      }
    }
    
    // Add external output
    auto* ext_output = subgraph_.add_outputs();
    if (single_node_proto_.op_type() == "conv") {
      const auto& conv_params = single_node_proto_.conv_params();
      if (conv_params.output_names_size() > 0) {
        ext_output->set_name(conv_params.output_names(0));
      }
    } else if (single_node_proto_.op_type() == "gemm") {
      const auto& gemm_params = single_node_proto_.gemm_params();
      if (gemm_params.output_names_size() > 0) {
        ext_output->set_name(gemm_params.output_names(0));
      }
    }
    ext_output->set_producer_node_id(0);
    ext_output->set_output_index(0);
    
    // Initialize single node runtime data
    node_data_.resize(1);
    node_data_[0] = std::make_unique<NodeRuntimeData>();
    output_name_to_index_[ext_output->name()] = 0;
  }
  
  // Load all weights
  MY_LOG(2) << "[ROCm CustomOp] Loading cached weights...";
  LoadAllWeights();
  MY_LOG(2) << "[ROCm CustomOp] Cached weights loaded successfully";
}

RocmCustomOp::~RocmCustomOp() {
  // Cleanup external input buffers
  for (auto& [name, ptr] : external_input_buffers_) {
    if (ptr) hipFree(ptr);
  }
  external_input_buffers_.clear();
  
  // node_data_ cleanup is handled by unique_ptr destructors
  MY_LOG(2) << "[ROCm CustomOp] Destructor completed";
}

void RocmCustomOp::LoadAllWeights() {
  auto pass_context = get_context();
  
  for (int i = 0; i < subgraph_.nodes_size(); ++i) {
    const auto& node = subgraph_.nodes(i);
    LoadNodeWeights(i, node.params());
  }
  
  weights_loaded_ = true;
}

void RocmCustomOp::LoadNodeWeights(int32_t node_id, const rocm::RocmParamProto& params) {
  auto pass_context = get_context();
  auto& data = *node_data_[node_id];
  
  const auto& op_type = params.op_type();
  
  if (op_type == "conv") {
    const auto& conv_params = params.conv_params();
    
    // Load weight file
    const auto& weight_file = conv_params.weight_file_path();
    if (!weight_file.empty()) {
      int64_t weight_size = conv_params.weight_file_size();
      MY_LOG(2) << "[ROCm CustomOp] Node " << node_id << ": Loading conv weight from: " 
                << weight_file << " (" << weight_size << " bytes)";
      
      auto reader = pass_context->open_file_for_read(weight_file);
      if (reader) {
        data.host_weight.resize(weight_size / sizeof(float));
        reader->fread(data.host_weight.data(), weight_size);
        MY_LOG(1) << "[ROCm CustomOp] Node " << node_id << ": Loaded weight: " 
                  << data.host_weight.size() << " floats";
      } else {
        LOG(ERROR) << "[ROCm CustomOp] Node " << node_id 
                   << ": Failed to open weight file: " << weight_file;
      }
    }
    
    // Load bias file if present
    const auto& bias_file = conv_params.bias_file_path();
    if (!bias_file.empty() && conv_params.has_bias()) {
      int64_t bias_size = conv_params.bias_file_size();
      MY_LOG(2) << "[ROCm CustomOp] Node " << node_id << ": Loading conv bias from: " 
                << bias_file << " (" << bias_size << " bytes)";
      
      auto reader = pass_context->open_file_for_read(bias_file);
      if (reader) {
        data.host_bias.resize(bias_size / sizeof(float));
        reader->fread(data.host_bias.data(), bias_size);
        MY_LOG(1) << "[ROCm CustomOp] Node " << node_id << ": Loaded bias: " 
                  << data.host_bias.size() << " floats";
      } else {
        LOG(ERROR) << "[ROCm CustomOp] Node " << node_id 
                   << ": Failed to open bias file: " << bias_file;
      }
    }
  } else if (op_type == "gemm") {
    const auto& gemm_params = params.gemm_params();
    
    // Load weight file (B matrix)
    const auto& weight_file = gemm_params.weight_file_path();
    if (!weight_file.empty()) {
      int64_t weight_size = gemm_params.weight_file_size();
      MY_LOG(2) << "[ROCm CustomOp] Node " << node_id << ": Loading gemm weight from: " 
                << weight_file << " (" << weight_size << " bytes)";
      
      auto reader = pass_context->open_file_for_read(weight_file);
      if (reader) {
        data.host_weight.resize(weight_size / sizeof(float));
        reader->fread(data.host_weight.data(), weight_size);
        MY_LOG(1) << "[ROCm CustomOp] Node " << node_id << ": Loaded weight: " 
                  << data.host_weight.size() << " floats";
      } else {
        LOG(ERROR) << "[ROCm CustomOp] Node " << node_id 
                   << ": Failed to open weight file: " << weight_file;
      }
    }
    
    // Load bias file if present
    const auto& bias_file = gemm_params.bias_file_path();
    if (!bias_file.empty() && gemm_params.has_bias()) {
      int64_t bias_size = gemm_params.bias_file_size();
      MY_LOG(2) << "[ROCm CustomOp] Node " << node_id << ": Loading gemm bias from: " 
                << bias_file << " (" << bias_size << " bytes)";
      
      auto reader = pass_context->open_file_for_read(bias_file);
      if (reader) {
        data.host_bias.resize(bias_size / sizeof(float));
        reader->fread(data.host_bias.data(), bias_size);
        MY_LOG(1) << "[ROCm CustomOp] Node " << node_id << ": Loaded bias: " 
                  << data.host_bias.size() << " floats";
      } else {
        LOG(ERROR) << "[ROCm CustomOp] Node " << node_id 
                   << ": Failed to open bias file: " << bias_file;
      }
    }
  }
}

void RocmCustomOp::AllocateIntermediateBuffers() {
  if (buffers_allocated_) return;
  
  auto& hip_ctx = HipContext::instance();
  auto stream = hip_ctx.stream();
  
  // Allocate output buffers for each node
  for (int i = 0; i < subgraph_.nodes_size(); ++i) {
    const auto& node = subgraph_.nodes(i);
    auto& data = *node_data_[i];
    
    // Calculate output size based on operation parameters
    size_t output_size = GetOutputSize(i, 0);
    
    if (output_size > 0) {
      float* output_buf = nullptr;
      hipError_t err = hipMalloc(&output_buf, output_size);
      if (err != hipSuccess) {
        LOG(ERROR) << "[ROCm CustomOp] Failed to allocate output buffer for node " << i 
                   << ": " << hipGetErrorString(err);
        throw std::runtime_error("Failed to allocate GPU memory for node output");
      }
      data.output_buffers.push_back(output_buf);
      data.output_sizes.push_back(output_size);
      MY_LOG(2) << "[ROCm CustomOp] Node " << i << ": Allocated output buffer " 
                << output_size << " bytes";
    }
    
    // Upload weights to GPU if not already done
    if (!data.host_weight.empty() && data.d_weight == nullptr) {
      size_t weight_bytes = data.host_weight.size() * sizeof(float);
      hipError_t err = hipMalloc(&data.d_weight, weight_bytes);
      if (err != hipSuccess) {
        LOG(ERROR) << "[ROCm CustomOp] Failed to allocate weight buffer for node " << i;
        throw std::runtime_error("Failed to allocate GPU memory for weights");
      }
      err = hipMemcpyAsync(data.d_weight, data.host_weight.data(), weight_bytes, 
                           hipMemcpyHostToDevice, stream);
      if (err != hipSuccess) {
        LOG(ERROR) << "[ROCm CustomOp] Failed to upload weights for node " << i;
        throw std::runtime_error("Failed to copy weights to GPU");
      }
      MY_LOG(2) << "[ROCm CustomOp] Node " << i << ": Uploaded weights " 
                << weight_bytes << " bytes";
    }
    
    // Upload bias to GPU if present
    if (!data.host_bias.empty() && data.d_bias == nullptr) {
      size_t bias_bytes = data.host_bias.size() * sizeof(float);
      hipError_t err = hipMalloc(&data.d_bias, bias_bytes);
      if (err != hipSuccess) {
        LOG(ERROR) << "[ROCm CustomOp] Failed to allocate bias buffer for node " << i;
        throw std::runtime_error("Failed to allocate GPU memory for bias");
      }
      err = hipMemcpyAsync(data.d_bias, data.host_bias.data(), bias_bytes, 
                           hipMemcpyHostToDevice, stream);
      if (err != hipSuccess) {
        LOG(ERROR) << "[ROCm CustomOp] Failed to upload bias for node " << i;
        throw std::runtime_error("Failed to copy bias to GPU");
      }
      MY_LOG(2) << "[ROCm CustomOp] Node " << i << ": Uploaded bias " 
                << bias_bytes << " bytes";
    }
  }
  
  buffers_allocated_ = true;
}

size_t RocmCustomOp::GetOutputSize(int32_t node_id, int32_t output_index) const {
  const auto& node = subgraph_.nodes(node_id);
  const auto& params = node.params();
  
  if (params.op_type() == "conv") {
    const auto& conv_params = params.conv_params();
    size_t output_elements = static_cast<size_t>(conv_params.batch_size()) * 
                             conv_params.out_channels() * 
                             conv_params.out_height() * 
                             conv_params.out_width();
    return output_elements * sizeof(float);
  } else if (params.op_type() == "gemm") {
    const auto& gemm_params = params.gemm_params();
    size_t output_elements = static_cast<size_t>(gemm_params.m()) * gemm_params.n();
    if (gemm_params.batch_count() > 1) {
      output_elements *= gemm_params.batch_count();
    }
    return output_elements * sizeof(float);
  }
  
  return 0;
}

float* RocmCustomOp::ResolveTensorRef(const rocm::TensorRefProto& ref) const {
  if (ref.has_external_name()) {
    // External input - look up in external_input_buffers_
    auto it = external_input_buffers_.find(ref.external_name());
    if (it != external_input_buffers_.end()) {
      return it->second;
    }
    LOG(ERROR) << "[ROCm CustomOp] External input not found: " << ref.external_name();
    return nullptr;
  } else if (ref.has_internal()) {
    // Internal reference - look up in node_data_
    const auto& internal_ref = ref.internal();
    int32_t producer_node = internal_ref.producer_node_id();
    int32_t output_idx = internal_ref.output_index();
    
    if (producer_node >= 0 && producer_node < static_cast<int32_t>(node_data_.size())) {
      const auto& producer_data = *node_data_[producer_node];
      if (output_idx >= 0 && output_idx < static_cast<int32_t>(producer_data.output_buffers.size())) {
        return producer_data.output_buffers[output_idx];
      }
    }
    LOG(ERROR) << "[ROCm CustomOp] Internal tensor not found: node " << producer_node 
               << ", output " << output_idx;
    return nullptr;
  }
  
  return nullptr;
}

void RocmCustomOp::Compute(const OrtApi* api, OrtKernelContext* context) const {
  MY_LOG(1) << "[ROCm CustomOp] Compute() - " << subgraph_.nodes_size() << " nodes";

  // Check if GPU is available
  auto& hip_ctx = HipContext::instance();
  if (!hip_ctx.is_initialized()) {
    LOG(ERROR) << "[ROCm CustomOp] HIP context not initialized - no AMD GPU available!";
    throw std::runtime_error("ROCm CustomOp: No AMD GPU available");
  }

  // Lazy initialization of buffers
  const_cast<RocmCustomOp*>(this)->AllocateIntermediateBuffers();

  // Phase 1: Upload external inputs to GPU
  UploadExternalInputs(api, context);

  // Phase 2: Execute subgraph nodes in topological order
  ExecuteSubgraph(api, context);

  // Phase 3: Download external outputs from GPU
  DownloadExternalOutputs(api, context);

  // Phase 4: Synchronize with timeout
  MY_LOG(2) << "[ROCm CustomOp] Synchronizing stream...";
  auto timeout_status = hip_ctx.sync_stream_with_timeout();
  
  if (timeout_status == TimeoutStatus::TIMEOUT) {
    LOG(ERROR) << "[ROCm CustomOp] Subgraph execution TIMED OUT!";
    throw std::runtime_error("ROCm CustomOp: GPU operation timed out");
  } else if (timeout_status == TimeoutStatus::ERROR) {
    LOG(ERROR) << "[ROCm CustomOp] Stream synchronization ERROR!";
    throw std::runtime_error("ROCm CustomOp: Stream synchronization error");
  }
  
  MY_LOG(1) << "[ROCm CustomOp] Compute completed successfully";
}

void RocmCustomOp::UploadExternalInputs(const OrtApi* api, OrtKernelContext* context) const {
  auto& hip_ctx = HipContext::instance();
  auto stream = hip_ctx.stream();
  
  // Collect unique external input names from all nodes
  std::set<std::string> external_inputs;
  for (const auto& node : subgraph_.nodes()) {
    for (const auto& input : node.inputs()) {
      if (input.has_external_name()) {
        external_inputs.insert(input.external_name());
      }
    }
  }
  
  // Upload each external input
  size_t input_idx = 0;
  for (const auto& name : external_inputs) {
    const OrtValue* ort_input = nullptr;
    api->KernelContext_GetInput(context, input_idx, &ort_input);
    if (ort_input == nullptr) {
      LOG(ERROR) << "[ROCm CustomOp] External input " << name << " is NULL";
      throw std::runtime_error("External input is NULL: " + name);
    }
    
    // Get tensor info
    OrtTensorTypeAndShapeInfo* type_info = nullptr;
    api->GetTensorTypeAndShape(ort_input, &type_info);
    
    size_t elem_count = 0;
    api->GetTensorShapeElementCount(type_info, &elem_count);
    api->ReleaseTensorTypeAndShapeInfo(type_info);
    
    size_t bytes = elem_count * sizeof(float);
    
    // Get host pointer
    float* h_data = nullptr;
    api->GetTensorMutableData(const_cast<OrtValue*>(ort_input), (void**)&h_data);
    
    // Allocate or reuse device buffer
    auto it = external_input_buffers_.find(name);
    if (it == external_input_buffers_.end() || external_input_sizes_[name] < bytes) {
      if (it != external_input_buffers_.end()) {
        hipFree(it->second);
      }
      float* d_data = nullptr;
      hipError_t err = hipMalloc(&d_data, bytes);
      if (err != hipSuccess) {
        LOG(ERROR) << "[ROCm CustomOp] Failed to allocate device buffer for " << name;
        throw std::runtime_error("Failed to allocate GPU memory for external input");
      }
      external_input_buffers_[name] = d_data;
      external_input_sizes_[name] = bytes;
      MY_LOG(2) << "[ROCm CustomOp] Allocated external input buffer: " << name 
                << " (" << bytes << " bytes)";
    }
    
    // Async H2D copy
    hipError_t err = hipMemcpyAsync(external_input_buffers_[name], h_data, bytes,
                                    hipMemcpyHostToDevice, stream);
    if (err != hipSuccess) {
      LOG(ERROR) << "[ROCm CustomOp] Failed to upload external input " << name;
      throw std::runtime_error("Failed to copy external input to GPU");
    }
    MY_LOG(2) << "[ROCm CustomOp] Uploaded external input: " << name 
              << " (" << bytes << " bytes)";
    
    ++input_idx;
  }
}

void RocmCustomOp::ExecuteSubgraph(const OrtApi* api, OrtKernelContext* context) const {
  MY_LOG(2) << "[ROCm CustomOp] Executing " << subgraph_.nodes_size() << " nodes";
  
  for (const auto& node : subgraph_.nodes()) {
    ExecuteNode(node);
  }
}

void RocmCustomOp::ExecuteNode(const rocm::RocmNodeProto& node) const {
  const auto& params = node.params();
  int32_t node_id = node.node_id();
  
  MY_LOG(2) << "[ROCm CustomOp] Executing node " << node_id 
            << " (op_type: " << params.op_type() << ")";
  
  // Gather input pointers
  std::vector<float*> inputs;
  for (const auto& input_ref : node.inputs()) {
    float* ptr = ResolveTensorRef(input_ref);
    if (ptr == nullptr) {
      LOG(ERROR) << "[ROCm CustomOp] Failed to resolve input for node " << node_id;
      throw std::runtime_error("Failed to resolve tensor reference");
    }
    inputs.push_back(ptr);
  }
  
  // Get output buffer
  auto& data = *node_data_[node_id];
  float* output = data.output_buffers.empty() ? nullptr : data.output_buffers[0];
  
  if (output == nullptr) {
    LOG(ERROR) << "[ROCm CustomOp] No output buffer for node " << node_id;
    throw std::runtime_error("No output buffer allocated");
  }
  
  // Execute based on operation type
  if (params.op_type() == "conv") {
    ExecuteConvNode(params.conv_params(), inputs, output);
  } else if (params.op_type() == "gemm") {
    ExecuteGemmNode(params.gemm_params(), inputs, output);
  } else {
    LOG(ERROR) << "[ROCm CustomOp] Unknown op_type: " << params.op_type();
    throw std::runtime_error("Unknown operation type: " + params.op_type());
  }
}

void RocmCustomOp::ExecuteConvNode(const rocm::ConvParamProto& params, 
                                    const std::vector<float*>& inputs,
                                    float* output) const {
  MY_LOG(2) << "[ROCm CustomOp] ExecuteConvNode: batch=" << params.batch_size()
            << ", C_in=" << params.in_channels() << ", C_out=" << params.out_channels();
  
  auto& hip_ctx = HipContext::instance();
  auto miopen_handle = hip_ctx.miopen_handle();
  auto stream = hip_ctx.stream();
  
  float* d_input = inputs[0];
  
  // Find node data for weights (we need to identify which node this is)
  // For now, we'll find the matching node by conv params
  NodeRuntimeData* node_data = nullptr;
  for (int i = 0; i < subgraph_.nodes_size(); ++i) {
    const auto& node = subgraph_.nodes(i);
    if (node.params().op_type() == "conv") {
      const auto& np = node.params().conv_params();
      if (np.batch_size() == params.batch_size() &&
          np.in_channels() == params.in_channels() &&
          np.out_channels() == params.out_channels()) {
        node_data = node_data_[i].get();
        break;
      }
    }
  }
  
  if (!node_data || node_data->d_weight == nullptr) {
    LOG(ERROR) << "[ROCm CustomOp] Conv weights not found";
    throw std::runtime_error("Conv weights not loaded");
  }
  
  float* d_weight = node_data->d_weight;
  float* d_bias = node_data->d_bias;
  
  // Create MIOpen descriptors
  miopenTensorDescriptor_t input_desc, weight_desc, output_desc;
  miopenConvolutionDescriptor_t conv_desc;
  
  miopenCreateTensorDescriptor(&input_desc);
  miopenCreateTensorDescriptor(&weight_desc);
  miopenCreateTensorDescriptor(&output_desc);
  miopenCreateConvolutionDescriptor(&conv_desc);

  miopenSet4dTensorDescriptor(input_desc, miopenFloat,
                              params.batch_size(), params.in_channels(),
                              params.in_height(), params.in_width());
  
  miopenSet4dTensorDescriptor(weight_desc, miopenFloat,
                              params.out_channels(), params.in_channels(),
                              params.filter_height(), params.filter_width());
  
  miopenSet4dTensorDescriptor(output_desc, miopenFloat,
                              params.batch_size(), params.out_channels(),
                              params.out_height(), params.out_width());
  
  miopenInitConvolutionDescriptor(conv_desc, miopenConvolution,
                                  params.pad_h(), params.pad_w(),
                                  params.stride_h(), params.stride_w(),
                                  params.dilation_h(), params.dilation_w());

  // Get workspace size
  size_t workspace_size = 0;
  miopenConvolutionForwardGetWorkSpaceSize(miopen_handle, weight_desc, input_desc,
                                           conv_desc, output_desc, &workspace_size);
  
  // Allocate workspace if needed
  void* workspace_ptr = nullptr;
  if (workspace_size > 0) {
    if (node_data->workspace == nullptr || node_data->workspace_size < workspace_size) {
      if (node_data->workspace) hipFree(node_data->workspace);
      hipMalloc(&node_data->workspace, workspace_size);
      node_data->workspace_size = workspace_size;
    }
    workspace_ptr = node_data->workspace;
  }

  // Find best algorithm
  miopenConvAlgoPerf_t perf_results[4];
  int algo_count = 0;
  
  miopenStatus_t status = miopenFindConvolutionForwardAlgorithm(
      miopen_handle, input_desc, d_input,
      weight_desc, d_weight,
      conv_desc, output_desc, output,
      4, &algo_count, perf_results,
      workspace_ptr, workspace_size, false);
  
  if (status != miopenStatusSuccess || algo_count == 0) {
    miopenDestroyConvolutionDescriptor(conv_desc);
    miopenDestroyTensorDescriptor(output_desc);
    miopenDestroyTensorDescriptor(weight_desc);
    miopenDestroyTensorDescriptor(input_desc);
    LOG(ERROR) << "[ROCm CustomOp] Failed to find convolution algorithm";
    throw std::runtime_error("Failed to find convolution algorithm");
  }
  
  MY_LOG(2) << "[ROCm CustomOp] Found " << algo_count << " algorithms, best time: " 
            << perf_results[0].time << " ms";

  // Execute convolution
  float alpha = params.alpha();
  float beta = params.beta();
  
  status = miopenConvolutionForward(miopen_handle, &alpha,
                                    input_desc, d_input,
                                    weight_desc, d_weight,
                                    conv_desc, perf_results[0].fwd_algo, &beta,
                                    output_desc, output,
                                    workspace_ptr, workspace_size);
  
  // Cleanup descriptors
  miopenDestroyConvolutionDescriptor(conv_desc);
  miopenDestroyTensorDescriptor(output_desc);
  miopenDestroyTensorDescriptor(weight_desc);
  miopenDestroyTensorDescriptor(input_desc);
  
  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "[ROCm CustomOp] miopenConvolutionForward failed";
    throw std::runtime_error("miopenConvolutionForward failed");
  }
  
  MY_LOG(2) << "[ROCm CustomOp] Convolution executed successfully";
}

void RocmCustomOp::ExecuteGemmNode(const rocm::GemmParamProto& params,
                                    const std::vector<float*>& inputs,
                                    float* output) const {
  MY_LOG(2) << "[ROCm CustomOp] ExecuteGemmNode: M=" << params.m()
            << ", N=" << params.n() << ", K=" << params.k();
  
  auto& hip_ctx = HipContext::instance();
  auto blaslt_handle = hip_ctx.hipblaslt_handle();
  auto stream = hip_ctx.stream();
  
  float* d_a = inputs[0];
  
  // Find node data for weights
  NodeRuntimeData* node_data = nullptr;
  for (int i = 0; i < subgraph_.nodes_size(); ++i) {
    const auto& node = subgraph_.nodes(i);
    if (node.params().op_type() == "gemm") {
      const auto& np = node.params().gemm_params();
      if (np.m() == params.m() && np.n() == params.n() && np.k() == params.k()) {
        node_data = node_data_[i].get();
        break;
      }
    }
  }
  
  if (!node_data || node_data->d_weight == nullptr) {
    LOG(ERROR) << "[ROCm CustomOp] GEMM weights not found";
    throw std::runtime_error("GEMM weights not loaded");
  }
  
  float* d_b = node_data->d_weight;
  float* d_c = node_data->d_bias;  // May be nullptr
  
  // TODO: Full hipBLASLt implementation
  // For now, log that we need to implement this
  MY_LOG(1) << "[ROCm CustomOp] GEMM execution - hipBLASLt implementation pending";
  
  // Placeholder: zero output
  size_t output_bytes = static_cast<size_t>(params.m()) * params.n() * sizeof(float);
  hipMemsetAsync(output, 0, output_bytes, stream);
  
  MY_LOG(2) << "[ROCm CustomOp] GEMM executed (placeholder)";
}

void RocmCustomOp::DownloadExternalOutputs(const OrtApi* api, OrtKernelContext* context) const {
  auto& hip_ctx = HipContext::instance();
  auto stream = hip_ctx.stream();
  
  // For each external output, copy from GPU to ORT output tensor
  for (int i = 0; i < subgraph_.outputs_size(); ++i) {
    const auto& ext_output = subgraph_.outputs(i);
    int32_t producer_node = ext_output.producer_node_id();
    int32_t output_idx = ext_output.output_index();
    
    // Get device buffer
    const auto& producer_data = *node_data_[producer_node];
    if (output_idx >= static_cast<int32_t>(producer_data.output_buffers.size())) {
      LOG(ERROR) << "[ROCm CustomOp] Invalid output index for external output " << ext_output.name();
      throw std::runtime_error("Invalid output index");
    }
    float* d_data = producer_data.output_buffers[output_idx];
    size_t bytes = producer_data.output_sizes[output_idx];
    
    // Get output shape from node params
    const auto& node = subgraph_.nodes(producer_node);
    const auto& params = node.params();
    
    std::vector<int64_t> output_shape;
    if (params.op_type() == "conv") {
      const auto& conv_params = params.conv_params();
      output_shape = {conv_params.batch_size(), conv_params.out_channels(),
                      conv_params.out_height(), conv_params.out_width()};
    } else if (params.op_type() == "gemm") {
      const auto& gemm_params = params.gemm_params();
      output_shape = {gemm_params.m(), gemm_params.n()};
    }
    
    // Get ORT output tensor
    OrtValue* ort_output = nullptr;
    api->KernelContext_GetOutput(context, i, output_shape.data(), output_shape.size(), &ort_output);
    if (ort_output == nullptr) {
      LOG(ERROR) << "[ROCm CustomOp] Failed to get ORT output " << i;
      throw std::runtime_error("Failed to get ORT output tensor");
    }
    
    float* h_data = nullptr;
    api->GetTensorMutableData(ort_output, (void**)&h_data);
    
    // Async D2H copy
    hipError_t err = hipMemcpyAsync(h_data, d_data, bytes,
                                    hipMemcpyDeviceToHost, stream);
    if (err != hipSuccess) {
      LOG(ERROR) << "[ROCm CustomOp] Failed to download output " << ext_output.name();
      throw std::runtime_error("Failed to copy output from GPU");
    }
    
    MY_LOG(2) << "[ROCm CustomOp] Scheduled D2H for output: " << ext_output.name()
              << " (" << bytes << " bytes)";
  }
}

} // namespace hip_ep
