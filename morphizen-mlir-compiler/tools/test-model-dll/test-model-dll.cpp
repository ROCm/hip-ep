/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// test-model-dll - E2E testing tool for compiled MLIR models
//===----------------------------------------------------------------------===//
// Loads a compiled DLL, discovers metadata, generates test data, runs
// inference, and validates outputs. Supports shape overrides and performance
// measurement.
//===----------------------------------------------------------------------===//

#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

#include "metadata.pb.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <google/protobuf/util/json_util.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// Interface function types
typedef int (*InferenceInitFunc)(void** out_state);
typedef int (*InferenceComputeFunc)(void* state, void* inputs, void* outputs);
typedef int (*InferenceCleanupFunc)(void* state);
typedef const char* (*InferenceGetMetadataJsonFunc)(void);

// Tensor structures matching hipdnn_ep_runtime.h
typedef struct {
  void* data;     // Host data pointer
  int64_t* shape; // Array of dimension sizes
  size_t rank;    // Number of dimensions
} tensor_t;

typedef struct {
  tensor_t* data; // Array of tensors
  size_t count;   // Number of tensors
} span_t;

// Platform-specific DLL loading
class DllLoader {
public:
  DllLoader(const std::string& path) : handle_(nullptr) {
#ifdef _WIN32
    handle_ = LoadLibraryA(path.c_str());
    if (!handle_) {
      std::cerr << "Failed to load DLL: " << path
                << " (error code: " << GetLastError() << ")\n";
    }
#else
    handle_ = dlopen(path.c_str(), RTLD_LAZY);
    if (!handle_) {
      std::cerr << "Failed to load DLL: " << path << " - " << dlerror() << "\n";
    }
#endif
  }

  ~DllLoader() {
    if (handle_) {
#ifdef _WIN32
      FreeLibrary((HMODULE)handle_);
#else
      dlclose(handle_);
#endif
    }
  }

  void* getSymbol(const char* name) {
    if (!handle_)
      return nullptr;
#ifdef _WIN32
    return (void*)GetProcAddress((HMODULE)handle_, name);
#else
    return dlsym(handle_, name);
#endif
  }

  bool isValid() const { return handle_ != nullptr; }

private:
  void* handle_;
};

// Parse --input-shape arguments: "0=8,3,224,224;1=8,512"
std::map<int, std::vector<int64_t>>
parseShapeOverrides(const std::string& arg) {
  std::map<int, std::vector<int64_t>> result;
  std::istringstream iss(arg);
  std::string token;

  while (std::getline(iss, token, ';')) {
    size_t eq_pos = token.find('=');
    if (eq_pos == std::string::npos)
      continue;

    int index = std::stoi(token.substr(0, eq_pos));
    std::string dims_str = token.substr(eq_pos + 1);
    std::vector<int64_t> shape;

    std::istringstream dims_iss(dims_str);
    std::string dim;
    while (std::getline(dims_iss, dim, ',')) {
      shape.push_back(std::stoll(dim));
    }

    result[index] = shape;
  }

  return result;
}

// Resolve dynamic dimensions (-1) to concrete values
void resolveShape(std::vector<int64_t>& shape, int64_t default_batch = 1) {
  for (auto& dim : shape) {
    if (dim == -1) {
      dim = default_batch; // Replace -1 with batch size
    }
  }
}

// Calculate total element count
size_t elementCount(const std::vector<int64_t>& shape) {
  size_t count = 1;
  for (auto dim : shape) {
    count *= dim;
  }
  return count;
}

// Generate test data (sequential 0-255 pattern, normalized to float32)
void generateTestData(float* data, size_t count) {
  for (size_t i = 0; i < count; i++) {
    data[i] = static_cast<float>((i % 256)) / 255.0f;
  }
}

// Validate output (check for NaN/Inf)
bool validateOutput(const float* data, size_t count, bool verbose) {
  size_t nan_count = 0;
  size_t inf_count = 0;

  for (size_t i = 0; i < count; i++) {
    if (std::isnan(data[i]))
      nan_count++;
    else if (std::isinf(data[i]))
      inf_count++;
  }

  if (verbose && count > 0) {
    std::cout << "  First 10 values: ";
    for (size_t i = 0; i < std::min(count, size_t(10)); i++) {
      std::cout << data[i] << " ";
    }
    std::cout << "\n";
  }

  if (nan_count > 0 || inf_count > 0) {
    std::cerr << "VALIDATION FAILED: " << nan_count << " NaN, " << inf_count
              << " Inf values\n";
    return false;
  }

  return true;
}

int main(int argc, char** argv) {
  // Parse arguments
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " <model.dll> [--input-shape INDEX=DIMS;...] [--iterations N] "
                 "[--verbose] [--validate]\n";
    std::cerr << "Example: " << argv[0]
              << " model.dll --input-shape 0=8,3,224,224 --iterations 10 "
                 "--verbose --validate\n";
    return 1;
  }

  std::string dll_path = argv[1];
  std::map<int, std::vector<int64_t>> shape_overrides;
  int iterations = 1;
  bool verbose = false;
  bool validate = false;

  for (int i = 2; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--input-shape" && i + 1 < argc) {
      shape_overrides = parseShapeOverrides(argv[++i]);
    } else if (arg == "--iterations" && i + 1 < argc) {
      iterations = std::stoi(argv[++i]);
    } else if (arg == "--verbose") {
      verbose = true;
    } else if (arg == "--validate") {
      validate = true;
    }
  }

  // Load DLL
  if (verbose)
    std::cout << "Loading DLL: " << dll_path << "\n";
  DllLoader dll(dll_path);
  if (!dll.isValid()) {
    return 1;
  }

  // Load interface functions
  auto init_func = (InferenceInitFunc)dll.getSymbol("inference_init");
  auto compute_func = (InferenceComputeFunc)dll.getSymbol("inference_compute");
  auto cleanup_func = (InferenceCleanupFunc)dll.getSymbol("inference_cleanup");
  auto get_metadata_func = (InferenceGetMetadataJsonFunc)dll.getSymbol(
      "inference_get_metadata_json");

  if (!init_func || !compute_func || !cleanup_func || !get_metadata_func) {
    std::cerr << "ERROR: Missing required interface functions\n";
    std::cerr << "  inference_init: " << (init_func ? "OK" : "MISSING") << "\n";
    std::cerr << "  inference_compute: " << (compute_func ? "OK" : "MISSING")
              << "\n";
    std::cerr << "  inference_cleanup: " << (cleanup_func ? "OK" : "MISSING")
              << "\n";
    std::cerr << "  inference_get_metadata_json: "
              << (get_metadata_func ? "OK" : "MISSING") << "\n";
    return 1;
  }

  // Get and parse metadata
  const char* json_str = get_metadata_func();
  if (verbose)
    std::cout << "Metadata JSON: " << json_str << "\n";

  morphizen::ModelMetadata metadata;
  if (!google::protobuf::util::JsonStringToMessage(json_str, &metadata).ok()) {
    std::cerr << "ERROR: Failed to parse metadata JSON\n";
    return 1;
  }

  if (verbose) {
    std::cout << "Model metadata:\n";
    std::cout << "  Inputs: " << metadata.inputs_size() << "\n";
    std::cout << "  Outputs: " << metadata.outputs_size() << "\n";
    std::cout << "  Version: " << metadata.version() << "\n";
  }

  // Prepare input shapes (apply overrides and resolve dynamics)
  std::vector<std::vector<int64_t>> input_shapes;
  for (int i = 0; i < metadata.inputs_size(); i++) {
    std::vector<int64_t> shape;
    for (auto dim : metadata.inputs(i).shape()) {
      shape.push_back(dim);
    }

    // Apply override if provided
    if (shape_overrides.count(i)) {
      shape = shape_overrides[i];
    }

    // Resolve dynamic dimensions
    resolveShape(shape);
    input_shapes.push_back(shape);

    if (verbose) {
      std::cout << "  Input " << i << " shape: [";
      for (size_t j = 0; j < shape.size(); j++) {
        std::cout << shape[j];
        if (j + 1 < shape.size())
          std::cout << ", ";
      }
      std::cout << "]\n";
    }
  }

  // Prepare output shapes
  std::vector<std::vector<int64_t>> output_shapes;
  for (int i = 0; i < metadata.outputs_size(); i++) {
    std::vector<int64_t> shape;
    for (auto dim : metadata.outputs(i).shape()) {
      shape.push_back(dim);
    }
    resolveShape(shape);
    output_shapes.push_back(shape);

    if (verbose) {
      std::cout << "  Output " << i << " shape: [";
      for (size_t j = 0; j < shape.size(); j++) {
        std::cout << shape[j];
        if (j + 1 < shape.size())
          std::cout << ", ";
      }
      std::cout << "]\n";
    }
  }

  // Allocate input buffers and generate test data
  std::vector<std::vector<float>> input_buffers;
  std::vector<std::vector<int64_t>>
      input_shape_storage; // Store shapes persistently
  std::vector<tensor_t> input_tensors;
  for (size_t i = 0; i < input_shapes.size(); i++) {
    size_t count = elementCount(input_shapes[i]);
    std::vector<float> buffer(count);
    generateTestData(buffer.data(), count);
    input_buffers.push_back(std::move(buffer));
    input_shape_storage.push_back(input_shapes[i]);

    tensor_t tensor;
    tensor.data = input_buffers.back().data();
    tensor.shape = input_shape_storage.back().data();
    tensor.rank = input_shape_storage.back().size();
    input_tensors.push_back(tensor);
  }

  span_t inputs_span;
  inputs_span.data = input_tensors.data();
  inputs_span.count = input_tensors.size();

  // Allocate output buffers
  std::vector<std::vector<float>> output_buffers;
  std::vector<std::vector<int64_t>>
      output_shape_storage; // Store shapes persistently
  std::vector<tensor_t> output_tensors;
  for (size_t i = 0; i < output_shapes.size(); i++) {
    size_t count = elementCount(output_shapes[i]);
    std::vector<float> buffer(count);
    output_buffers.push_back(std::move(buffer));
    output_shape_storage.push_back(output_shapes[i]);

    tensor_t tensor;
    tensor.data = output_buffers.back().data();
    tensor.shape = output_shape_storage.back().data();
    tensor.rank = output_shape_storage.back().size();
    output_tensors.push_back(tensor);
  }

  span_t outputs_span;
  outputs_span.data = output_tensors.data();
  outputs_span.count = output_tensors.size();

  // Initialize inference context
  if (verbose)
    std::cout << "\nInitializing inference context...\n";
  void* state = nullptr;
  int ret = init_func(&state);
  if (ret != 0) {
    std::cerr << "ERROR: inference_init failed with code " << ret << "\n";
    return 1;
  }

  // Run inference
  if (verbose)
    std::cout << "Running inference (" << iterations << " iteration(s))...\n";

  for (int iter = 0; iter < iterations; iter++) {
    ret = compute_func(state, &inputs_span, &outputs_span);
    if (ret != 0) {
      std::cerr << "ERROR: inference_compute failed with code " << ret
                << " at iteration " << iter << "\n";
      cleanup_func(state);
      return 1;
    }

    if (verbose && iterations > 1)
      std::cout << "  Iteration " << (iter + 1) << "/" << iterations << "\n";
  }

  // Validate outputs
  if (validate) {
    if (verbose)
      std::cout << "\nValidating outputs...\n";
    bool all_valid = true;
    for (size_t i = 0; i < output_buffers.size(); i++) {
      if (verbose)
        std::cout << "Output " << i << ":\n";
      if (!validateOutput(output_buffers[i].data(), output_buffers[i].size(),
                          verbose)) {
        all_valid = false;
      }
    }

    if (!all_valid) {
      cleanup_func(state);
      return 1;
    }

    if (verbose)
      std::cout << "All outputs valid!\n";
  }

  // Cleanup
  if (verbose)
    std::cout << "\nCleaning up...\n";
  ret = cleanup_func(state);
  if (ret != 0) {
    std::cerr << "WARNING: inference_cleanup failed with code " << ret << "\n";
  }

  std::cout << "SUCCESS: Model executed successfully\n";
  return 0;
}
