/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 * 
 * MIOpen-based implementation (migrated from hipDNN)
 */
// clang-format off
#include "morphizen/onnxruntime_api.hpp"

#include <glog/logging.h>
#include <sstream>
#include <fstream>
#include <morphizen/vaip.hpp>
#include <morphizen/env_config.hpp>
#include "./custom_op.hpp"
#include "google/protobuf/util/json_util.h"
#include <cstdlib>
#include <filesystem>
#include <thread>
#include <unordered_set>
#include <vaip/my_ort.h>
#include <nlohmann/json.hpp>

DEF_ENV_PARAM(MORPHIZEN_DEBUG_HIPDNN, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_HIPDNN) >= n)

namespace hipdnn {

// Helper macros for MIOpen error checking
#define MIOPEN_CHECK(call)                                                  \
  do {                                                                       \
    miopenStatus_t status = (call);                                         \
    if (status != miopenStatusSuccess) {                                    \
      LOG(ERROR) << "MIOpen error: " << status                             \
                 << " at " << __FILE__ << ":" << __LINE__;                  \
    }                                                                        \
  } while (0)

#define MIOPEN_THROW_IF_ERROR(call)                                         \
  do {                                                                       \
    miopenStatus_t status = (call);                                         \
    if (status != miopenStatusSuccess) {                                    \
      throw std::runtime_error(std::string("MIOpen error: ") +             \
                               std::to_string(status) +                     \
                               " at " + __FILE__ + ":" +                    \
                               std::to_string(__LINE__));                   \
    }                                                                        \
  } while (0)

// Helper to convert ONNX data type to MIOpen data type
static miopenDataType_t ToMIOpenDataType(int32_t onnx_dtype) {
  switch (onnx_dtype) {
    case 1:  // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
      return miopenFloat;
    case 10: // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
      return miopenHalf;
    default:
      return miopenFloat;
  }
}

// Helper to load constant data from file
static std::vector<char> LoadConstantFromFile(const std::string& filepath) {
  std::ifstream file(filepath, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("Failed to open constant data file: " + filepath);
  }
  
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  
  std::vector<char> data(size);
  if (!file.read(data.data(), size)) {
    throw std::runtime_error("Failed to read constant data file: " + filepath);
  }
  
  return data;
}

HipdnnCustomOp::HipdnnCustomOp(std::shared_ptr<const PassContext> context,
                               const std::shared_ptr<MetaDefProto>& meta_def,
                               onnxruntime::Model* model)
    : CustomOpImp(context, meta_def, model),
      miopen_handle_(nullptr),
      x_desc_(nullptr),
      w_desc_(nullptr),
      y_desc_(nullptr),
      b_desc_(nullptr),
      conv_desc_(nullptr),
      conv_algo_(miopenConvolutionFwdAlgoGEMM),
      workspace_size_(0),
      workspace_(nullptr),
      has_bias_(false),
      data_type_(miopenFloat),
      device_id_(0) {
  
  MY_LOG(1) << "HipdnnCustomOp constructor (MIOpen version)";
  
  // Parse proto
  auto hipdnn_json_str = get_meta_def_param();
  auto status = google::protobuf::util::JsonStringToMessage(hipdnn_json_str, &hipdnn_proto_);
  
  if (!status.ok()) {
    LOG(FATAL) << "Failed to parse hipdnn_json_str: " << status.ToString();
    return;
  }
  
  LOG(INFO) << "Metadata file: " << hipdnn_proto_.graph_file_name();
  
  // Get constant initializer names from meta_def
  auto constant_names = meta_def_->constant_initializers();
  LOG(INFO) << "Constant initializers count: " << constant_names.size();
  for (const auto& name : constant_names) {
    LOG(INFO) << "  Constant: " << name;
    constant_initializer_names_.push_back(name);
  }
  
  // Create MIOpen handle
  MIOPEN_THROW_IF_ERROR(miopenCreate(&miopen_handle_));
  
  // Get device ID
  hipError_t hip_err = hipGetDevice(&device_id_);
  if (hip_err != hipSuccess) {
    LOG(WARNING) << "Failed to get HIP device, using device 0";
    device_id_ = 0;
  }
  
  // Build and compile using MIOpen
  try {
    BuildAndCompileMIOpen();
    LOG(INFO) << "MIOpen kernel compiled successfully";
    
    // Load constant data from files
    if (!constant_initializer_names_.empty() && 
        hipdnn_proto_.constant_data_files_size() > 0) {
      LoadConstantData();
      LOG(INFO) << "Successfully loaded " << constant_data_.size() 
                << " constant initializers";
    }
  } catch (const std::exception& ex) {
    LOG(FATAL) << "Failed to compile MIOpen kernel: " << ex.what();
  }
}

HipdnnCustomOp::~HipdnnCustomOp() {
  // Free workspace
  if (workspace_ != nullptr) {
    hipFree(workspace_);
    workspace_ = nullptr;
  }
  
  // Destroy descriptors
  if (x_desc_) miopenDestroyTensorDescriptor(x_desc_);
  if (w_desc_) miopenDestroyTensorDescriptor(w_desc_);
  if (y_desc_) miopenDestroyTensorDescriptor(y_desc_);
  if (b_desc_) miopenDestroyTensorDescriptor(b_desc_);
  if (conv_desc_) miopenDestroyConvolutionDescriptor(conv_desc_);
  
  // Destroy MIOpen handle
  if (miopen_handle_) {
    miopenDestroy(miopen_handle_);
    miopen_handle_ = nullptr;
  }
}

void HipdnnCustomOp::BuildAndCompileMIOpen() {
  MY_LOG(1) << "=== Building MIOpen kernel ===";
  
  // Load JSON metadata from file
  std::string metadata_file = hipdnn_proto_.graph_file_name();
  std::ifstream file(metadata_file);
  if (!file) {
    throw std::runtime_error("Failed to open metadata file: " + metadata_file);
  }
  
  nlohmann::json j;
  try {
    file >> j;
  } catch (const std::exception& ex) {
    throw std::runtime_error("Failed to parse JSON metadata: " + 
                             std::string(ex.what()));
  }
  
  MY_LOG(1) << "Loaded metadata from: " << metadata_file;
  
  // Extract metadata
  std::string op_type = j["op_type"];
  if (op_type != "Conv") {
    throw std::runtime_error("Unsupported operation type: " + op_type);
  }
  
  // Extract shapes
  x_shape_ = j["input_shapes"][0].get<std::vector<int64_t>>();
  w_shape_ = j["input_shapes"][1].get<std::vector<int64_t>>();
  y_shape_ = j["output_shapes"][0].get<std::vector<int64_t>>();
  
  // Extract convolution parameters
  std::vector<int64_t> pads = j["pads"].get<std::vector<int64_t>>();
  std::vector<int64_t> strides = j["strides"].get<std::vector<int64_t>>();
  std::vector<int64_t> dilations = j["dilations"].get<std::vector<int64_t>>();
  has_bias_ = j["has_bias"];
  
  // Extract data types
  std::vector<int> input_dtypes = j["input_data_types"].get<std::vector<int>>();
  data_type_ = ToMIOpenDataType(input_dtypes[0]);
  
  MY_LOG(1) << "Op: " << op_type
            << ", X=" << x_shape_.size() << "D"
            << ", W=" << w_shape_.size() << "D"
            << ", Y=" << y_shape_.size() << "D"
            << ", has_bias=" << has_bias_;
  
  // Create MIOpen descriptors
  MIOPEN_THROW_IF_ERROR(miopenCreateTensorDescriptor(&x_desc_));
  MIOPEN_THROW_IF_ERROR(miopenCreateTensorDescriptor(&w_desc_));
  MIOPEN_THROW_IF_ERROR(miopenCreateTensorDescriptor(&y_desc_));
  MIOPEN_THROW_IF_ERROR(miopenCreateConvolutionDescriptor(&conv_desc_));
  
  // Set input tensor descriptor
  MIOPEN_THROW_IF_ERROR(miopenSet4dTensorDescriptor(
      x_desc_, data_type_,
      static_cast<int>(x_shape_[0]),
      static_cast<int>(x_shape_[1]),
      static_cast<int>(x_shape_[2]),
      static_cast<int>(x_shape_[3])));
  
  // Set weight tensor descriptor
  MIOPEN_THROW_IF_ERROR(miopenSet4dTensorDescriptor(
      w_desc_, data_type_,
      static_cast<int>(w_shape_[0]),
      static_cast<int>(w_shape_[1]),
      static_cast<int>(w_shape_[2]),
      static_cast<int>(w_shape_[3])));
  
  // Set output tensor descriptor
  MIOPEN_THROW_IF_ERROR(miopenSet4dTensorDescriptor(
      y_desc_, data_type_,
      static_cast<int>(y_shape_[0]),
      static_cast<int>(y_shape_[1]),
      static_cast<int>(y_shape_[2]),
      static_cast<int>(y_shape_[3])));
  
  // Set convolution descriptor
  MIOPEN_THROW_IF_ERROR(miopenInitConvolutionDescriptor(
      conv_desc_,
      miopenConvolution,
      static_cast<int>(pads[0]),
      static_cast<int>(pads[1]),
      static_cast<int>(strides[0]),
      static_cast<int>(strides[1]),
      static_cast<int>(dilations[0]),
      static_cast<int>(dilations[1])));
  
  // Get workspace size
  MIOPEN_THROW_IF_ERROR(miopenConvolutionForwardGetWorkSpaceSize(
      miopen_handle_, w_desc_, x_desc_, conv_desc_, y_desc_,
      &workspace_size_));
  
  MY_LOG(1) << "Workspace size: " << workspace_size_ << " bytes";
  
  // Allocate workspace
  if (workspace_size_ > 0) {
    hipError_t hip_err = hipMalloc(&workspace_, workspace_size_);
    if (hip_err != hipSuccess) {
      throw std::runtime_error("Failed to allocate workspace");
    }
  }
  
  // Allocate temporary buffers for algorithm finding
  void* temp_x = nullptr;
  void* temp_w = nullptr;
  void* temp_y = nullptr;
  
  size_t x_size = x_shape_[0] * x_shape_[1] * x_shape_[2] * x_shape_[3] * sizeof(float);
  size_t w_size = w_shape_[0] * w_shape_[1] * w_shape_[2] * w_shape_[3] * sizeof(float);
  size_t y_size = y_shape_[0] * y_shape_[1] * y_shape_[2] * y_shape_[3] * sizeof(float);
  
  hipError_t hip_err;
  hip_err = hipMalloc(&temp_x, x_size);
  if (hip_err != hipSuccess) {
    throw std::runtime_error("Failed to allocate temp input buffer");
  }
  
  hip_err = hipMalloc(&temp_w, w_size);
  if (hip_err != hipSuccess) {
    hipFree(temp_x);
    throw std::runtime_error("Failed to allocate temp weight buffer");
  }
  
  hip_err = hipMalloc(&temp_y, y_size);
  if (hip_err != hipSuccess) {
    hipFree(temp_x);
    hipFree(temp_w);
    throw std::runtime_error("Failed to allocate temp output buffer");
  }
  
  // Find the best algorithm using miopenFindConvolutionForwardAlgorithm
  // This is required by MIOpen - it registers the invoker for later use
  MY_LOG(1) << "Finding best convolution algorithm...";
  const int requestedAlgoCount = 1;
  int returnedAlgoCount = 0;
  miopenConvAlgoPerf_t perfResults;
  
  miopenStatus_t find_status = miopenFindConvolutionForwardAlgorithm(
      miopen_handle_,
      x_desc_, temp_x,
      w_desc_, temp_w,
      conv_desc_,
      y_desc_, temp_y,
      requestedAlgoCount,
      &returnedAlgoCount,
      &perfResults,
      workspace_, workspace_size_,
      false);  // exhaustiveSearch = false for faster compilation
  
  // Free temporary buffers
  hipFree(temp_x);
  hipFree(temp_w);
  hipFree(temp_y);
  
  // Check result
  MIOPEN_THROW_IF_ERROR(find_status);
  
  if (returnedAlgoCount > 0) {
    conv_algo_ = perfResults.fwd_algo;
    MY_LOG(1) << "Selected algorithm: " << conv_algo_ 
              << " (time: " << perfResults.time << " ms, "
              << "memory: " << perfResults.memory << " bytes)";
  } else {
    // Fallback to GEMM if find fails
    conv_algo_ = miopenConvolutionFwdAlgoGEMM;
    MY_LOG(1) << "Using fallback GEMM algorithm";
  }
  
  // If bias is present, create bias descriptor
  if (has_bias_) {
    MIOPEN_THROW_IF_ERROR(miopenCreateTensorDescriptor(&b_desc_));
    
    // Bias shape is typically [1, C, 1, 1]
    b_shape_ = {1, y_shape_[1], 1, 1};
    MIOPEN_THROW_IF_ERROR(miopenSet4dTensorDescriptor(
        b_desc_, data_type_,
        static_cast<int>(b_shape_[0]),
        static_cast<int>(b_shape_[1]),
        static_cast<int>(b_shape_[2]),
        static_cast<int>(b_shape_[3])));
  }
  
  // Setup output shapes for Compute
  output_shapes_.clear();
  output_shapes_.push_back(y_shape_);
  
  // Setup num_inputs and num_outputs
  num_inputs_ = has_bias_ ? 3 : 2;  // input + weight + optional bias
  num_outputs_ = 1;
  
  MY_LOG(1) << "=== MIOpen kernel build complete ===";
}

void HipdnnCustomOp::LoadGraphMetadata() {
  // This is now handled inside BuildAndCompileMIOpen
}

void HipdnnCustomOp::LoadConstantData() {
  MY_LOG(1) << "=== Loading constant data from files ===";
		
  // Load each constant initializer from file
  for (int i = 0; i < hipdnn_proto_.constant_data_files_size(); ++i) {
    const auto& data_file = hipdnn_proto_.constant_data_files(i);
    const auto& const_name = hipdnn_proto_.constant_names(i);
    
    MY_LOG(1) << "Loading constant from file: " << data_file;
    
    std::vector<char> data = LoadConstantFromFile(data_file);
    constant_data_.push_back(std::move(data));
    
    MY_LOG(1) << "  Loaded constant " << const_name 
              << " from " << data_file
              << " (" << constant_data_.back().size() << " bytes)";
  }
  
  MY_LOG(1) << "=== Constant data loading complete ===";
}

static std::string shape_to_string(const std::vector<int64_t>& shape) {
  std::ostringstream str;
  str << "[";
  int c = 0;
  for (auto s : shape) {
    if (c != 0) {
      str << ",";
    }
    str << s;
    c = c + 1;
  }
  str << "]";
  return str.str();
}

void HipdnnCustomOp::Compute(const OrtApi* api, OrtKernelContext* context) const {
  Ort::KernelContext ctx(context);
  auto num_runtime_inputs = ctx.GetInputCount();
  auto num_outputs = ctx.GetOutputCount();
  auto num_constants = constant_initializer_names_.size();
  
  MY_LOG(1) << "=== HipdnnCustomOp::Compute START (MIOpen) ===";
  MY_LOG(1) << "Runtime inputs: " << num_runtime_inputs 
            << ", outputs: " << num_outputs
            << ", constants: " << num_constants;
  
  // Set device
  hipError_t hip_err = hipSetDevice(device_id_);
  if (hip_err != hipSuccess) {
    LOG(ERROR) << "Failed to set HIP device: " << device_id_;
    return;
  }
  
  // Get input tensor (X)
  Ort::ConstValue input_tensor = ctx.GetInput(0);
  const void* x_data = input_tensor.GetTensorRawData();
  
  MY_LOG(1) << "Input X: " << shape_to_string(x_shape_);
  
  // Get weight tensor (W) - from constant data
  if (constant_data_.empty()) {
    LOG(ERROR) << "No constant data available for weights";
    return;
  }
  const void* w_data = constant_data_[0].data();
  
  MY_LOG(1) << "Weight W: " << shape_to_string(w_shape_)
            << " (" << constant_data_[0].size() << " bytes)";
  
  // Allocate output tensor (Y)
  Ort::UnownedValue output_tensor = ctx.GetOutput(0, output_shapes_[0]);
  void* y_data = output_tensor.GetTensorMutableRawData();
  
  MY_LOG(1) << "Output Y: " << shape_to_string(y_shape_);
  
  // Set scaling factors
  float alpha = 1.0f;
  float beta = 0.0f;
  
  // Execute convolution
  MIOPEN_CHECK(miopenConvolutionForward(
      miopen_handle_,
      &alpha,
      x_desc_, x_data,
      w_desc_, w_data,
      conv_desc_,
      conv_algo_,
      &beta,
      y_desc_, y_data,
      workspace_, workspace_size_));
  
  MY_LOG(1) << "Convolution forward completed";
  
  // If bias is present, add it
  if (has_bias_ && constant_data_.size() > 1) {
    const void* b_data = constant_data_[1].data();
    
    MY_LOG(1) << "Adding bias: " << shape_to_string(b_shape_)
              << " (" << constant_data_[1].size() << " bytes)";
    
    float alpha_bias = 1.0f;
    float beta_bias = 1.0f;  // Accumulate with existing output
    
    MIOPEN_CHECK(miopenOpTensor(
        miopen_handle_,
        miopenTensorOpAdd,
        &alpha_bias, y_desc_, y_data,  // Y (from conv)
        &alpha_bias, b_desc_, b_data,  // Bias
        &beta_bias, y_desc_, y_data));  // Y (output)
    
    MY_LOG(1) << "Bias addition completed";
  }
  
  // Synchronize to ensure completion (prevent driver timeout)
  hip_err = hipDeviceSynchronize();
  if (hip_err != hipSuccess) {
    LOG(WARNING) << "hipDeviceSynchronize failed: " << hip_err;
  }
  
  MY_LOG(1) << "=== HipdnnCustomOp::Compute END ===";
}

} // namespace hipdnn
