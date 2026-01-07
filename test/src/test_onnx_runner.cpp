/*
 *  Copyright 2022 Xilinx Inc.
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
#include <onnxruntime_cxx_api.h>

#include <algorithm> // std::generate
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>
#if _WIN32
extern "C" {
#  include "util/getopt.h"
}
#  include <codecvt>
#  include <locale>
using convert_t = std::codecvt_utf8<wchar_t>;
std::wstring_convert<convert_t, wchar_t> strconverter;
#else
#  include <unistd.h>
#endif
std::mt19937 rng;
#include "util/env_config.hpp"
DEF_ENV_PARAM_2(XLNX_EXTERNAL_SESSION_CONFIG, "", std::string);
DEF_ENV_PARAM_2(XLNX_EXTERNAL_PROVIDER_OPTION, "", std::string);
DEF_ENV_PARAM_2(XLNX_AIE_TOOL_PATH, "", std::string);
DEF_ENV_PARAM_2(XLNX_TARGET_NAME, "", std::string); // used by xcompiler
DEF_ENV_PARAM(XLNX_ENABLE_BATCH, "0");
DEF_ENV_PARAM(XLNX_ENABLE_CACHE_CONTEXT, "0");
DEF_ENV_PARAM_2(XLNX_ENABLE_EP_SHARED_CONTEXT, "", std::string);
DEF_ENV_PARAM_2(CACHE_CONTEXT_EMBEDED_MODE, "1", std::string);
DEF_ENV_PARAM_2(WORKLOAD_TYPE, "Default", std::string);
DEF_ENV_PARAM_2(CACHE_CONTEXT_FILE_PATH, "", std::string);
DEF_ENV_PARAM_2(ENABLE_CACHE_FILE_IO_IN_MEM, "", std::string);
DEF_ENV_PARAM_2(XLNX_CONFIG_TARGET_NAME, "", std::string); // used by vaip
DEF_ENV_PARAM(XLNX_USE_MEMORY_MODEL, "0");
DEF_ENV_PARAM_2(XLNX_USE_CACHE_KEY, "", std::string);
DEF_ENV_PARAM_2(XLNX_USE_CACHE_DIR, "", std::string);
DEF_ENV_PARAM_2(XLNX_ENABLE_OLD_QDQ, "", std::string);
DEF_ENV_PARAM_2(XLNX_ENCRYPTION_KEY, "", std::string);
DEF_ENV_PARAM_2(XLNX_ENABLE_PY3_ROUND, "", std::string);
DEF_ENV_PARAM_2(VITISAI_EP_JSON_CONFIG, "", std::string);
DEF_ENV_PARAM_2(XLNX_SESSION_COUNT, "1", int);
#define CHECK(expr, msg)                                                       \
  do {                                                                         \
    if (!(expr)) {                                                             \
      std::cerr << __FILE__ << ":" << __LINE__ << (msg) << std::endl;          \
      std::abort();                                                            \
    }                                                                          \
  } while (0)
#define CHECK_STATUS_OK(expr)                                                  \
  do {                                                                         \
    Status _tmp_status = (expr);                                               \
    CHECK(_tmp_status.IsOK()) << _tmp_status;                                  \
  } while (0)

static int calculate_product(const std::vector<int64_t>& v) {
  int total = 1;
  for (auto& i : v)
    total *= (int)i;
  return total;
}

std::vector<std::string> split(const std::string& str, char delimiter) {
  std::vector<std::string> result;
  std::string temp;
  for (char ch : str) {
    if (ch == delimiter) {
      if (!temp.empty()) {
        result.push_back(temp);
        temp.clear();
      }
    } else {
      temp += ch;
    }
  }
  if (!temp.empty()) {
    result.push_back(temp);
  }
  return result;
}

static void usage() {
  std::cout << "usage: test_onnx_runner [-b <batch_number>] [-n "
               "'enable_ep'] [-e 'encryption_key] "
               "[-p: enable onnx profiler] [-i: provider options] "
               "[-C: session config] <onnx model> [<input_file>] "
               "[<input_file> ...] \n"
            << std::endl;
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

Ort::SessionOptions
get_session_option(std::string& external_session_configs,
                   const std::vector<std::string>& customops,
                   const std::filesystem::path& model_path, int ort_opt_level,
                   bool enable_profiler) {
  auto session_options = Ort::SessionOptions();
  auto ext_sess_configs = split(external_session_configs, ',');
  for (auto ext_sess_config : ext_sess_configs) {
    auto content = split(ext_sess_config, '|');
    CHECK(content.size() == 2,
          std::string("external config entry format error: ") +
              ext_sess_config);
    session_options.AddConfigEntry(content[0].c_str(), content[1].c_str());
    std::cout << "external session_config set " << content[0] << " = "
              << content[1] << std::endl;
  }
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

  if (enable_profiler) {
    auto profile_name = std::string("profile_test_onnx_runner");
#if _WIN32
    session_options.EnableProfiling(
        std::wstring(profile_name.begin(), profile_name.end()).c_str());
#else
    session_options.EnableProfiling(profile_name.c_str());
#endif
  }

  if (!ENV_PARAM(CACHE_CONTEXT_FILE_PATH).empty()) {
    session_options.AddConfigEntry("ep.context_file_path",
                                   ENV_PARAM(CACHE_CONTEXT_FILE_PATH).c_str());
  }
  if (ENV_PARAM(XLNX_ENABLE_CACHE_CONTEXT)) {
    std::cout << "Enable cache context" << std::endl;
    session_options.AddConfigEntry("ep.context_enable", "1");
    session_options.AddConfigEntry(
        "ep.context_embed_mode", ENV_PARAM(CACHE_CONTEXT_EMBEDED_MODE).c_str());
    session_options.AddConfigEntry("session.workload_type",
                                   ENV_PARAM(WORKLOAD_TYPE).c_str());
    if (!ENV_PARAM(XLNX_ENABLE_EP_SHARED_CONTEXT).empty()) {
      session_options.AddConfigEntry(
          "ep.share_ep_contexts",
          ENV_PARAM(XLNX_ENABLE_EP_SHARED_CONTEXT).c_str());
    }
  }
  return session_options;
}

std::unordered_map<std::string, std::string>
get_provider_options(const std::string& encryption_key,
                     const std::string& external_provider_options) {
  auto options = std::unordered_map<std::string, std::string>{};

  // parse external provider options
  auto ext_opts = split(external_provider_options, ' ');
  for (auto ext_opt : ext_opts) {
    auto content = split(ext_opt, '|');
    CHECK(content.size() == 2,
          std::string("external provider options format error: ") + ext_opt);
    options[content[0]] = content[1];
    std::cout << "provider_option set " << content[0] << " = " << content[1]
              << std::endl;
  }

  std::string config_file = ENV_PARAM(VITISAI_EP_JSON_CONFIG);
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
  } else if (!ENV_PARAM(XLNX_ENCRYPTION_KEY).empty()) {
    options["encryptionKey"] = ENV_PARAM(XLNX_ENCRYPTION_KEY);
  }

  if (!ENV_PARAM(XLNX_CONFIG_TARGET_NAME).empty()) {
    options["xlnx_target"] = ENV_PARAM(XLNX_CONFIG_TARGET_NAME);
  }

  if (!ENV_PARAM(ENABLE_CACHE_FILE_IO_IN_MEM).empty()) {
    options["enable_cache_file_io_in_mem"] =
        ENV_PARAM(ENABLE_CACHE_FILE_IO_IN_MEM);
  }
  if (ENV_PARAM(XLNX_ENABLE_OLD_QDQ) != "") {
    options["xlnx_enable_old_qdq"] = ENV_PARAM(XLNX_ENABLE_OLD_QDQ);
  }
  if (ENV_PARAM(XLNX_ENABLE_PY3_ROUND) != "") {
    options["xlnx_enable_py3_round"] = ENV_PARAM(XLNX_ENABLE_PY3_ROUND);
  }
  if (ENV_PARAM(XLNX_AIE_TOOL_PATH) != "") {
    options["xlnx_aie_tool_path"] = ENV_PARAM(XLNX_AIE_TOOL_PATH);
  }
  if (ENV_PARAM(XLNX_TARGET_NAME) != "") {
    options["xlnx_target_name"] = ENV_PARAM(XLNX_TARGET_NAME);
  }
  return options;
}

void del_ctx_model(const std::filesystem::path& model_path) {
  // delete EP cache_context onnx file when exit
  try {
    if (!ENV_PARAM(CACHE_CONTEXT_FILE_PATH).empty()) {
      std::filesystem::remove(ENV_PARAM(CACHE_CONTEXT_FILE_PATH));
    } else {
      auto ctx_path_name =
          model_path.stem().string() + "_ctx" + model_path.extension().string();
      auto ctx_path = model_path.parent_path() / ctx_path_name;
      std::filesystem::remove(ctx_path);
    }
  } catch (std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
  }
}

std::unique_ptr<Ort::Session>
create_session(const std::filesystem::path& model_path, Ort::Env& env,
               const Ort::SessionOptions& session_options) {
  std::unique_ptr<Ort::Session> p_session;
  auto create_session_start_time = std::chrono::steady_clock::now();
  try {
    if (ENV_PARAM(XLNX_USE_MEMORY_MODEL)) {
      auto model_data = ReadBinaryFile(model_path.string());
      p_session = std::make_unique<Ort::Session>(
          env, model_data.data(), model_data.size(), session_options);
    } else {
#if _WIN32
      p_session = std::make_unique<Ort::Session>(
          env, model_path.wstring().data(), session_options);
#else
      p_session = std::make_unique<Ort::Session>(
          env, model_path.u8string().data(), session_options);
#endif
    }
  } catch (Ort::Exception& e) {
    std::cout << "Catched Ort Exception: " << e.what() << std::endl;
    exit(1);
  }
  auto create_session_end_time = std::chrono::steady_clock::now();
  std::cout << " ONNXRuntime session create :  "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   create_session_end_time - create_session_start_time)
                   .count()
            << " us" << std::endl;
  return p_session;
}

void run_session(const std::filesystem::path& model_path, Ort::Env& env,
                 const Ort::SessionOptions& session_options, int batch_number) {
  Ort::AllocatorWithDefaultOptions allocator;
  std::unique_ptr<Ort::Session> p_session =
      create_session(model_path, env, session_options);

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
      CHECK(output_tensors[i].IsTensor(), output_names[i]);
      auto output_tensor_shape =
          output_tensors[i].GetTensorTypeAndShapeInfo().GetShape();
    }
  } catch (const Ort::Exception& exception) {
    std::cout << "ERROR running model inference: " << exception.what()
              << std::endl;
    exit(-1);
  }
}
int main(int argc, char* argv[]) {
  {
    int opt = 0;
    int64_t batch_number = 1;
    bool enable_profiler = false;
    bool enable_ep = true;
    int ort_opt_level = -1;
    std::string encryption_key = "";
    std::vector<std::string> customops;
    std::string external_provider_options = "";
    std::string external_session_configs = "";

    while ((opt = getopt(argc, argv, "b:pns:c:l:e:i:C:")) != -1) {
      switch (opt) {
      case 'b':
        batch_number = std::stoi(optarg);
        break;
      case 'p':
        enable_profiler = true;
        break;
      case 'n':
        enable_ep = false;
        break;
      case 's':
        rng = std::mt19937(std::stoul(optarg));
        break;
      case 'l':
        ort_opt_level = std::stoi(optarg);
        break;
      case 'e':
        encryption_key = optarg;
        break;
      case 'c':
        customops.emplace_back(optarg);
        break;
      case 'i':
        external_provider_options = optarg;
        break;
      case 'C':
        external_session_configs = optarg;
        break;
      default:
        usage();
        exit(1);
      }
    }
    if (optind >= argc) {
      usage();
      exit(1);
    }
    if (!ENV_PARAM(XLNX_EXTERNAL_SESSION_CONFIG).empty()) {
      external_session_configs = ENV_PARAM(XLNX_EXTERNAL_SESSION_CONFIG);
    }
    if (!ENV_PARAM(XLNX_EXTERNAL_PROVIDER_OPTION).empty()) {
      external_provider_options = ENV_PARAM(XLNX_EXTERNAL_PROVIDER_OPTION);
    }
    // bool enable_optimize = ENV_PARAM(ENABLE_OPT);
    auto model_name = std::string(argv[optind]);

    Ort::Env env(ORT_LOGGING_LEVEL_INFO, "test_onnx_runner");
    auto model_path = std::filesystem::path(model_name);

    auto session_options =
        get_session_option(external_session_configs, customops, model_path,
                           ort_opt_level, enable_profiler);

    if (enable_ep) {
      auto options =
          get_provider_options(encryption_key, external_provider_options);

      session_options.AppendExecutionProvider_VitisAI(options);
      // CheckStatus(OrtSessionOptionsAppendExecutionProvider_VITISAI(
      //     session_options, json_config.c_str()));
    }

    for (int i = 0; i < ENV_PARAM(XLNX_SESSION_COUNT); ++i) {
      del_ctx_model(model_path);
      run_session(model_path, env, session_options, batch_number);
    }
  }
  std::cout << "BYE" << std::endl;
  return 0;
}
