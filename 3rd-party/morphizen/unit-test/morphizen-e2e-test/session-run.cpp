/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./session-run.hpp"

#include <chrono>
#include <iostream>
#include <random>

namespace morphizen_e2e_test {
std::mt19937 rng;
static int calculate_product(const std::vector<int64_t>& v) {
  int total = 1;
  for (auto& i : v)
    total *= (int)i;
  return total;
}

size_t get_data_type_size(ONNXTensorElementDataType type) {
  switch (type) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    return sizeof(float);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    return sizeof(uint8_t);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    return sizeof(int8_t);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    return 2;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    return sizeof(int64_t);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    return sizeof(int16_t);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    return sizeof(uint16_t);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    return sizeof(int32_t);
  default:
    std::cout << "unsupported data type " << type << std::endl;
    exit(1);
  }
}

void run_session(Ort::Session& session,
                 const E2ETestSessionRunProto& run_proto) {
  auto batch_number =
      run_proto.has_batch_number() ? run_proto.batch_number() : 1;
  Ort::AllocatorWithDefaultOptions allocator;
  auto input_count = session.GetInputCount();
  auto input_shapes = std::vector<std::vector<int64_t>>();
  auto input_names_ptr = std::vector<Ort::AllocatedStringPtr>();
  auto input_names = std::vector<const char*>();
  input_shapes.reserve(input_count);
  input_names_ptr.reserve(input_count);
  input_names.reserve(input_count);
  for (size_t i = 0; i < input_count; i++) {
    input_shapes.push_back(
        session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    auto name = session.GetInputNameAllocated(i, allocator);
    input_names.push_back(name.get());
    input_names_ptr.push_back(std::move(name));
  }

  // print name/shape of outputs
  auto output_count = session.GetOutputCount();
  auto output_shapes = std::vector<std::vector<int64_t>>();
  auto output_names_ptr = std::vector<Ort::AllocatedStringPtr>();
  auto output_names = std::vector<const char*>();
  output_shapes.reserve(output_count);
  output_names_ptr.reserve(output_count);
  output_names.reserve(output_count);

  for (size_t i = 0; i < output_count; i++) {
    auto shape =
        session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    output_shapes.push_back(shape);
    auto name = session.GetOutputNameAllocated(i, allocator);
    output_names.push_back(name.get());
    output_names_ptr.push_back(std::move(name));
  }

  // Create a single Ort tensor from input_file or random numbers
  std::vector<Ort::Value> input_tensors;
  input_tensors.reserve(input_count);
  auto input_tensor_values = std::vector<std::vector<char>>(input_count);
  int64_t batch = 1u;
  for (auto i = 0u; i < input_count; i++) {
    auto input_shape = input_shapes[i];
    if (input_shape[0] == -1) {
      input_shape[0] = batch_number;
    }
    auto input_type = session.GetInputTypeInfo(i)
                          .GetTensorTypeAndShapeInfo()
                          .GetElementType();
    int total_number_elements = calculate_product(input_shape);
    auto element_size = get_data_type_size(input_type);
    input_tensor_values[i].resize(total_number_elements * element_size);

    // preserve the golden
    auto floats =
        std::vector<float>(input_tensor_values[i].size() / sizeof(float));
    std::uniform_real_distribution<float> dist(-256.0f, 255.0f);
    std::generate(floats.begin(), floats.end(), [&] {
      return dist(rng); // generate same sequence in different platform
    });
    std::memcpy(input_tensor_values[i].data(), floats.data(),
                floats.size() * element_size);

    Ort::MemoryInfo info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    input_tensors.push_back(Ort::Value::CreateTensor(
        info, input_tensor_values[i].data(), input_tensor_values[i].size(),
        input_shape.data(), input_shape.size(), input_type));

    // double-check the dimensions of the input tensor
    auto input_tensor_shape =
        input_tensors[i].GetTensorTypeAndShapeInfo().GetShape();

    assert(input_tensors[i].IsTensor() && input_tensor_shape == input_shape);
  }

  // pass data through model
  std::cout << "Running model..." << std::endl;

  try {
    auto start_time = std::chrono::steady_clock::now();
    auto output_tensors =
        session.Run(Ort::RunOptions(), input_names.data(), input_tensors.data(),
                    input_count, output_names.data(), output_count);
    auto end_time = std::chrono::steady_clock::now();
    std::cout << " ONNXRuntime session run :  "
              << std::chrono::duration_cast<std::chrono::microseconds>(
                     end_time - start_time)
                     .count()
              << std::endl;
    std::cout << "done" << std::endl;

    // double-check the dimensions of the output tensors
    // NOTE: the number of output tensors is equal to the number of output
    // nodes specifed in the Run() call
    assert(output_tensors.size() == output_count);
    for (auto i = 0u; i < output_tensors.size(); ++i) {
      // CHECK(output_tensors[i].IsTensor(), output_names[i]);
      auto output_tensor_shape =
          output_tensors[i].GetTensorTypeAndShapeInfo().GetShape();
    }
  } catch (const Ort::Exception& exception) {
    std::cout << "ERROR running model inference: " << exception.what()
              << std::endl;
    exit(-1);
  }
}
} // namespace morphizen_e2e_test
