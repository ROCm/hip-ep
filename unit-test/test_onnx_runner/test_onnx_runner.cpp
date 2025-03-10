/*
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 **/
#include <assert.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wpedantic"
#  pragma GCC diagnostic ignored "-Wconversion"
#  pragma GCC diagnostic ignored "-Wsign-compare"
#  pragma GCC diagnostic ignored "-Wunused-variable"
#  pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#include <onnxruntime_cxx_api.h>
#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif
#include "./unit_test_env_params.hpp"
#include <algorithm> // std::generate
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

static int calculate_product(const std::vector<int64_t>& v) {
  int total = 1;
  for (auto& i : v)
    total *= (int)i;
  return total;
}

std::string replace(const std::string& s) {
  const std::string pat = ".";
  std::ostringstream str;
  for (auto c : s) {
    if (pat.find(c) != std::string::npos) {
      str << "_";
    } else {
      str << c;
    }
  }
  return str.str();
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

std::vector<uint8_t> ReadBinaryFile(const std::string& file_path) {
  std::ifstream file(file_path, std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<uint8_t> buffer(size);
  if (file.read(reinterpret_cast<char*>(buffer.data()), size)) {
    return buffer;
  } else {
    throw std::runtime_error("Failed to read model file");
  }
}

void save_input_output_json(const std::vector<std::string>& in,
                            const std::vector<std::string>& out) {
  if (in.size() + out.size() == 0) {
    return;
  }
  std::string json_str = "{\n";
  json_str += "\t\"input\": [";
  int count = 0;
  for (const auto& f : in) {
    if (count != 0) {
      json_str += ", ";
    }
    json_str += "\"" + f + "\"";
    count++;
  }
  count = 0;
  json_str += "],\n";
  json_str += "\t\"output\": [";
  for (const auto& f : out) {
    if (count != 0) {
      json_str += ", ";
    }
    json_str += "\"" + f + "\"";
    count++;
  }
  json_str += "]\n";
  json_str += "}\n";
  std::ofstream file("io.json");
  file << json_str;
}
std::mt19937 rng;
using namespace std;
class TesOnnxRunner : public ::testing::Test {
  void SetUp() override {}
  void TearDown() override { //
  }
};
void run(std::string& model_name, bool enable_cache_context) {
  LOG(INFO) << "start to test " << ENV_PARAM(INPUT_MODEL);
  int64_t batch_number = 1;
  bool enable_ep = true;
  int ort_opt_level = -1;
  std::string encryption_key = "";
  std::string opt_target_name = "";
  std::vector<std::string> customops;

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test_onnx_runner");
  auto model_path_in = std::filesystem::path(model_name);
  auto model_path_out = model_path_in;

  auto session_options = Ort::SessionOptions();

  // ORT_DISABLE_ALL = 0,
  // ORT_ENABLE_BASIC = 1,
  // ORT_ENABLE_EXTENDED = 2,
  // ORT_ENABLE_ALL = 99
  if (ort_opt_level != -1) {
    std::cout << "Setting graph_optimization_level to " << ort_opt_level;
    session_options.SetGraphOptimizationLevel(
        static_cast<GraphOptimizationLevel>(ort_opt_level));
  }
  for (auto& customop : customops) {
#if _WIN32
    session_options.RegisterCustomOpsLibrary(
        std::wstring(customop.begin(), customop.end()).c_str());
#else
    session_options.RegisterCustomOpsLibrary(customop.c_str());
#endif
  }

  if (enable_ep) {
    std::string config_file = ENV_PARAM(VITISAI_EP_JSON_CONFIG);
    auto options = std::unordered_map<std::string, std::string>{};
    if (!config_file.empty()) {
      options["config_file"] = config_file;
    }
    if (!ENV_PARAM(XLNX_USE_CACHE_KEY).empty()) {
      options["cacheKey"] = ENV_PARAM(XLNX_USE_CACHE_KEY);
    }
    if (!ENV_PARAM(XLNX_USE_CACHE_DIR).empty()) {
      options["cacheDir"] = ENV_PARAM(XLNX_USE_CACHE_DIR);
    }
    if (encryption_key != "") {
      options["encryptionKey"] = encryption_key;
    }
    if (opt_target_name != "") {
      options["target"] = opt_target_name;
    }
    if (enable_cache_context) {
      session_options.AddConfigEntry("ep.context_enable", "1");
      session_options.AddConfigEntry(
          "ep.context_embed_mode",
          ENV_PARAM(CACHE_CONTEXT_EMBEDED_MODE).c_str());
      session_options.AddConfigEntry(
          "ep.context_file_path", ENV_PARAM(CACHE_CONTEXT_FILE_PATH).c_str());
    }
    session_options.AppendExecutionProvider_VitisAI(options);
  }

  auto model_path =
#if _WIN32
      model_path_out.wstring()
#else
      model_path_out.u8string()
#endif
      ;
  Ort::AllocatorWithDefaultOptions allocator;
  std::unique_ptr<Ort::Session> p_session = nullptr;
  if (ENV_PARAM(XLNX_USE_MEMORY_MODEL)) {
    auto model_data = ReadBinaryFile(model_path_out.string());
    p_session = std::make_unique<Ort::Session>(
        env, model_data.data(), model_data.size(), session_options);
  } else {
    p_session =
        std::make_unique<Ort::Session>(env, model_path.data(), session_options);
  }
  auto& session = *p_session;
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
  auto input_file_list = std::vector<string>();
  auto output_file_list = std::vector<string>();
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
    std::uniform_real_distribution<float> dist(0.0f, 255.0f);
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
    if (ENV_PARAM(XLNX_ENABLE_DUMP)) {
      auto input_tensor_ptr = input_tensors[i].GetTensorData<char>();
      if (ENV_PARAM(XLNX_ENABLE_BATCH)) {
        batch = input_tensor_shape[0];
      }
      auto total_number_elements =
          input_tensors[i].GetTensorTypeAndShapeInfo().GetElementCount();
      auto size = total_number_elements / batch * element_size;
      for (int index = 0; index < batch; ++index) {
        auto filename = "onnx_input_" + std::to_string(i) + "_batch_" +
                        std::to_string(index) + ".bin";
        if (enable_cache_context) {
          filename = "ctx_" + filename;
        }
        input_file_list.push_back(filename);
        auto mode =
            std::ios_base::out | std::ios_base::binary | std::ios_base::trunc;
        CHECK(std::ofstream(filename, mode)
                  .write(input_tensor_ptr + size * index, size)
                  .good())
            << std::string(" faild to write to ") << filename;
      }
    }
  }

  // pass data through model
  cout << "Running model..." << endl;

  try {
    auto start_time = std::chrono::steady_clock::now();
    auto output_tensors =
        session.Run(Ort::RunOptions(), input_names.data(), input_tensors.data(),
                    input_count, output_names.data(), output_count);
    auto end_time = std::chrono::steady_clock::now();
    cout << " ONNXRuntime session run :  "
         << std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                                  start_time)
                .count()
         << endl;
    cout << "done" << endl;

    // double-check the dimensions of the output tensors
    // NOTE: the number of output tensors is equal to the number of output nodes
    // specifed in the Run() call
    assert(output_tensors.size() == output_count);
    for (auto i = 0u; i < output_tensors.size(); ++i) {
      CHECK(output_tensors[i].IsTensor()) << output_names[i];
      auto output_tensor_shape =
          output_tensors[i].GetTensorTypeAndShapeInfo().GetShape();

      if (ENV_PARAM(XLNX_ENABLE_DUMP)) {
        auto output_tensor_ptr =
            output_tensors[i].GetTensorMutableData<float>();
        if (ENV_PARAM(XLNX_ENABLE_BATCH)) {
          batch = output_tensor_shape[0];
        }
        auto total_number_elements =
            output_tensors[i].GetTensorTypeAndShapeInfo().GetElementCount();
        auto output_type = session.GetOutputTypeInfo(i)
                               .GetTensorTypeAndShapeInfo()
                               .GetElementType();
        auto element_size = get_data_type_size(output_type);
        auto size = total_number_elements / batch * element_size;
        for (int index = 0; index < batch; ++index) {
          auto filename = "onnx_output_" + std::to_string(i) + "_batch_" +
                          std::to_string(index) + ".bin";
          if (enable_cache_context) {
            filename = "ctx_" + filename;
          }
          cout << "output tensor: " << output_names[i]
               << ", file name:" << filename << endl;
          output_file_list.push_back(filename);
          auto mode =
              std::ios_base::out | std::ios_base::binary | std::ios_base::trunc;
          CHECK(std::ofstream(filename, mode)
                    .write((char*)output_tensor_ptr + size * index, size)
                    .good())
              << " faild to write to " << filename;
        }
      }
    }
  } catch (const Ort::Exception& exception) {
    cout << "ERROR running model inference: " << exception.what() << endl;
    exit(-1);
  }
  save_input_output_json(input_file_list, output_file_list);
}

TEST_F(TesOnnxRunner, Main) {
  auto model_name = ENV_PARAM(INPUT_MODEL);
  bool with_cache_context = true;
  std::filesystem::remove(ENV_PARAM(CACHE_CONTEXT_FILE_PATH));
  run(model_name, with_cache_context);

  with_cache_context = false;
  auto ctx_model_name = ENV_PARAM(CACHE_CONTEXT_FILE_PATH);
  run(ctx_model_name, with_cache_context);
}
