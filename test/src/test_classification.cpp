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

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
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

#include "util/env_config.hpp"

DEF_ENV_PARAM_2(MORPHIZEN_EP_JSON_CONFIG, "", std::string);
DEF_ENV_PARAM_2(XLNX_USE_CACHE_KEY, "", std::string);
DEF_ENV_PARAM_2(XLNX_USE_CACHE_DIR, "", std::string);
DEF_ENV_PARAM(XLNX_ENABLE_CACHE_CONTEXT, "0");
DEF_ENV_PARAM_2(CACHE_CONTEXT_EMBEDED_MODE, "1", std::string);
DEF_ENV_PARAM_2(XLNX_ENABLE_EP_SHARED_CONTEXT, "", std::string);
DEF_ENV_PARAM_2(ENABLE_CACHE_FILE_IO_IN_MEM, "", std::string);
DEF_ENV_PARAM_2(MORPHIZEN_EP_DLL, "onnxruntime_morphizen_ep.dll", std::string);
DEF_ENV_PARAM_2(MORPHIZEN_ORT_BRIDGE_BACKEND, "mlir-backend", std::string);
DEF_ENV_PARAM_2(ORT_LOG_LEVEL, "error", std::string);
DEF_ENV_PARAM_2(EP_KREGISTERATIONNAME, "MorphiZenExecutionProvider", std::string);

#define CHECK(expr, msg)                                                       \
  do {                                                                         \
    if (!(expr)) {                                                             \
      std::cerr << __FILE__ << ":" << __LINE__ << " " << (msg) << std::endl;   \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

struct ClassificationResult {
  int class_id;
  float confidence;
  std::string label;
};

static void usage() {
  std::cout << "Usage: test_classification [options] <model.onnx> <input_file>\n"
            << "Options:\n"
            << "  -k <num>     Top-K results to display (default: 5)\n"
            << "  -n           Disable MorphiZen EP (use CPU)\n"
            << "  -p           Enable ONNX profiler\n"
            << "  -l <file>    Label file (one label per line)\n"
            << "  -h           Show this help message\n"
            << std::endl;
}

std::vector<std::string> load_labels(const std::string& label_file) {
  std::vector<std::string> labels;
  std::ifstream file(label_file);
  if (!file.is_open()) {
    std::cerr << "Warning: Could not open label file: " << label_file << std::endl;
    return labels;
  }
  
  std::string line;
  while (std::getline(file, line)) {
    line.erase(line.find_last_not_of(" \n\r\t") + 1);
    labels.push_back(line);
  }
  
  std::cout << "Loaded " << labels.size() << " labels from " << label_file << std::endl;
  return labels;
}

static std::vector<float> softmax(float* data, int64_t size) {
  float max = -100000;
  for (int i = 0; i < size; i++) {
    max = max > data[i] ? max : data[i];
  }
  auto f = [max](float f) -> float {
    return expf(f - max);
  };
  auto output = std::vector<float>(size);
  std::transform(data, data + size, output.begin(), f);
  auto sum =
      std::accumulate(output.begin(), output.end(), 0.0f, std::plus<float>());
  std::transform(output.begin(), output.end(), output.begin(),
                 [sum](float v) { return v / sum; });
  return output;
}

static std::vector<std::pair<int, float>> topk(const std::vector<float>& score,
                                               int K) {
  auto indices = std::vector<int>(score.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::partial_sort(indices.begin(), indices.begin() + K, indices.end(),
                    [&score](int a, int b) { return score[a] > score[b]; });
  auto ret = std::vector<std::pair<int, float>>(K);
  std::transform(
      indices.begin(), indices.begin() + K, ret.begin(),
      [&score](int index) { return std::make_pair(index, score[index]); });
  return ret;
}

static const char* lookup(int index) {
  static const char* table[] = {
#include "word_list.inc"
  };

  if (index < 0) {
    return "";
  } else {
    return table[index];
  }
}

static void print_topk(const std::vector<std::pair<int, float>>& topk_results,
                      const std::vector<std::string>& labels) {
  for (const auto& v : topk_results) {
    std::string label_text;
    if (!labels.empty() && v.first < static_cast<int>(labels.size())) {
      label_text = labels[v.first];
    } else {
      label_text = lookup(v.first);
    }
    
    std::cout << std::setiosflags(std::ios::left) << std::setw(11)
              << "score[" + std::to_string(v.first) + "]"
              << " =  " << std::setw(12) << v.second
              << " text: " << label_text << ","
              << std::resetiosflags(std::ios::left) << std::endl;
  }
}

static void preprocess(const std::string& input_file,
                      std::vector<float>& input_tensor_values,
                      const std::vector<int64_t>& input_shape) {
  auto size = input_shape[1] * input_shape[2] * input_shape[3] * sizeof(float);
  CHECK(
      std::ifstream(input_file, std::ios::binary)
          .read((char*)input_tensor_values.data(), size)
          .good(),
      std::string("fail to read! filename=") + input_file);
}

static int calculate_product(const std::vector<int64_t>& v) {
  int total = 1;
  for (auto& i : v)
    total *= (int)i;
  return total;
}

void run_classification(const std::filesystem::path& model_path,
                       const std::string& input_file,
                       int top_k,
                       bool enable_ep,
                       bool enable_profiler,
                       const std::vector<std::string>& labels) {
  
  const std::string kRegistrationName = ENV_PARAM(EP_KREGISTERATIONNAME);
  std::cout << "=================" << kRegistrationName;
  std::cout << "enable_ep = " << (enable_ep ? "true" : "false") << std::endl;
  
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test_classification");
  std::vector<Ort::ConstEpDevice> selected_devices = {};
  
  
  auto library_path = std::filesystem::u8path(ENV_PARAM(MORPHIZEN_EP_DLL));
  if (!std::filesystem::exists(library_path)) {
    std::cerr << "Execution provider library not found: " << library_path
              << std::endl;
    abort();
  }
  std::cout << "Using ORT API 2.0 create session with MorphiZen EP, "
             "RegisterExecutionProviderLibrary: "
            << ENV_PARAM(MORPHIZEN_EP_DLL) << " ort_bridge_backend: " 
            << ENV_PARAM(MORPHIZEN_ORT_BRIDGE_BACKEND) << std::endl;
  auto status_0 = Ort::GetApi().RegisterExecutionProviderLibrary(
      env, kRegistrationName.c_str(), library_path.c_str());
  CHECK(status_0 == nullptr,
        std::string("RegisterExecutionProviderLibrary failed: status = ") +
        Ort::GetApi().GetErrorMessage(status_0));

  for (const auto& device : env.GetEpDevices()) {
    if (device.EpName() == kRegistrationName) {
      std::cout << "-----Selected EP device: " << device.EpName()
                << " from vendor: " << device.EpVendor() << std::endl;
      selected_devices.emplace_back(device);
    }
  }
  CHECK(!selected_devices.empty(),
        "No devices found for EP: " + kRegistrationName);
  
  
  Ort::SessionOptions session_options;
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  
  if (enable_profiler) {
    auto profile_name = std::string("profile_test_classification");
#if _WIN32
    session_options.EnableProfiling(
        std::wstring(profile_name.begin(), profile_name.end()).c_str());
#else
    session_options.EnableProfiling(profile_name.c_str());
#endif
  }
  
  if (enable_ep) {
    std::unordered_map<std::string, std::string> provider_options;

    std::string config_file = ENV_PARAM(MORPHIZEN_EP_JSON_CONFIG);
    if (!config_file.empty()) {
      provider_options["config_file"] = config_file;
      std::cout << "Using MorphiZen EP config: " << config_file << std::endl;
    }
    
    if (!ENV_PARAM(XLNX_USE_CACHE_KEY).empty()) {
      provider_options["cacheKey"] = ENV_PARAM(XLNX_USE_CACHE_KEY);
    }
    if (!ENV_PARAM(XLNX_USE_CACHE_DIR).empty()) {
      provider_options["cacheDir"] = ENV_PARAM(XLNX_USE_CACHE_DIR);
    }
    if (!ENV_PARAM(ENABLE_CACHE_FILE_IO_IN_MEM).empty()) {
      provider_options["enable_cache_file_io_in_mem"] =
          ENV_PARAM(ENABLE_CACHE_FILE_IO_IN_MEM);
    }
    
    if (ENV_PARAM(XLNX_ENABLE_CACHE_CONTEXT)) {
      session_options.AddConfigEntry("ep.context_enable", "1");
      session_options.AddConfigEntry("ep.context_embed_mode", 
                                     ENV_PARAM(CACHE_CONTEXT_EMBEDED_MODE).c_str());
      if (!ENV_PARAM(XLNX_ENABLE_EP_SHARED_CONTEXT).empty()) {
        session_options.AddConfigEntry("ep.share_ep_contexts",
                                       ENV_PARAM(XLNX_ENABLE_EP_SHARED_CONTEXT).c_str());
      }
    }
    
   
    session_options.AppendExecutionProvider_V2(env, selected_devices,
                                                 provider_options);
    
  }
  
  std::unique_ptr<Ort::Session> session;
#if _WIN32
  session = std::make_unique<Ort::Session>(env, model_path.wstring().c_str(), session_options);
#else
  session = std::make_unique<Ort::Session>(env, model_path.u8string().c_str(), session_options);
#endif
  
  Ort::AllocatorWithDefaultOptions allocator;
  
  auto input_name = session->GetInputNameAllocated(0, allocator);
  auto input_shape = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  
  auto output_name = session->GetOutputNameAllocated(0, allocator);
  auto output_shape = session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  
  int total_number_elements = calculate_product(input_shape);
  std::vector<float> input_tensor_values(total_number_elements);
  preprocess(input_file, input_tensor_values, input_shape);
  
  std::vector<Ort::Value> input_tensors;
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  input_tensors.push_back(Ort::Value::CreateTensor<float>(
      memory_info, input_tensor_values.data(), input_tensor_values.size(),
      input_shape.data(), input_shape.size()));
  
  const char* input_names[] = {input_name.get()};
  const char* output_names[] = {output_name.get()};
  
  std::cout << "Running model..." << std::endl;
  
  try {
    auto output_tensors = session->Run(Ort::RunOptions{nullptr}, 
                                       input_names, input_tensors.data(), 1,
                                       output_names, 1);
    
    std::cout << "done" << std::endl;
    
    auto output_tensor_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    auto batch = output_tensor_shape[0];
    auto channel = output_tensor_shape[1];
    auto output_tensor_ptr = output_tensors[0].GetTensorMutableData<float>();
    
    for (auto index = 0; index < batch; ++index) {
      auto softmax_output = softmax(output_tensor_ptr + channel * index, channel);
      auto tb_topk = topk(softmax_output, top_k);
      std::cout << "batch_index: " << index << std::endl;
      print_topk(tb_topk, labels);
    }
    
  } catch (const Ort::Exception& exception) {
    std::cout << "ERROR running model inference: " << exception.what() << std::endl;
    exit(-1);
  }
  
  
  auto status = (Ort::GetApi().UnregisterExecutionProviderLibrary(
      env, kRegistrationName.c_str()));
  (void)status;
  std::cout << "Unregistered EP library: " << ENV_PARAM(MORPHIZEN_EP_DLL) << std::endl;
  
}

int main(int argc, char* argv[]) {
  int opt = 0;
  int top_k = 5;
  bool enable_ep = true;
  bool enable_profiler = false;
  std::string label_file;
  
  while ((opt = getopt(argc, argv, "k:npl:h")) != -1) {
    switch (opt) {
    case 'k':
      top_k = std::stoi(optarg);
      break;
    case 'n':
      enable_ep = false;
      break;
    case 'p':
      enable_profiler = true;
      break;
    case 'l':
      label_file = optarg;
      break;
    case 'h':
      usage();
      return 0;
    default:
      usage();
      return 1;
    }
  }
  
  if (optind + 2 > argc) {
    std::cerr << "Error: Missing required arguments" << std::endl;
    usage();
    return 1;
  }
  
  std::string model_path = argv[optind];
  std::string input_file = argv[optind + 1];
  
  std::vector<std::string> labels;
  if (!label_file.empty()) {
    labels = load_labels(label_file);
  }
  
  try {
    run_classification(model_path, input_file, top_k, enable_ep, enable_profiler, labels);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  
  return 0;
}
