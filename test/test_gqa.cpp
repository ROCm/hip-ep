/*
 * Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Test GQA (Grouped Query Attention) layer inference
 * Runs a single GQA layer ONNX model with random input data
 */
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
#include <random>
#include <unordered_map>
#include <cmath>
#include <cstdlib>

#if _WIN32
#  include <codecvt>
#  include <locale>
using convert_t = std::codecvt_utf8<wchar_t>;
std::wstring_convert<convert_t, wchar_t> strconverter;
#endif

// Simple environment variable helper
static std::string get_env(const char* name, const std::string& default_value = "") {
  const char* value = std::getenv(name);
  return value ? value : default_value;
}

#define CHECK(expr, msg)                                                       \
  do {                                                                         \
    if (!(expr)) {                                                             \
      std::cerr << __FILE__ << ":" << __LINE__ << " " << (msg) << std::endl;   \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

static void usage() {
  std::cout << "Usage: test_gqa [options] <model.onnx>\n"
            << "Options:\n"
            << "  -b <batch_size>   Batch size (default: 1)\n"
            << "  -s <seq_len>      Sequence length (default: 128)\n"
            << "  -i <iterations>   Number of iterations (default: 10)\n"
            << "  -w <warmup>       Warmup iterations (default: 3)\n"
            << "  -n                Disable MorphiZen EP (use CPU only)\n"
            << "  -p                Enable ONNX profiler\n"
            << "  -v                Verbose output\n"
            << "  -h                Show this help message\n"
            << "\nEnvironment Variables:\n"
            << "  MORPHIZEN_EP_JSON_CONFIG  Path to MorphiZen config file\n"
            << "  MORPHIZEN_EP_DLL          Path to MorphiZen EP library\n"
            << "  MORPHIZEN_DEBUG_GQA       Debug level for GQA (0-3)\n"
            << std::endl;
}

struct GQAConfig {
  int batch_size = 1;
  int seq_len = 128;
  int iterations = 10;
  int warmup = 3;
  bool enable_ep = true;
  bool enable_profiler = false;
  bool verbose = false;
};

// Convert ONNX element type to string
static std::string element_type_to_string(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "float32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "float16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "int64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return "int32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return "int8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return "uint8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return "bool";
    default: return "unknown";
  }
}

// Calculate total number of elements
static int64_t calculate_product(const std::vector<int64_t>& shape) {
  int64_t total = 1;
  for (auto dim : shape) {
    if (dim > 0) total *= dim;
  }
  return total;
}

// Generate random float data
static void fill_random_float(float* data, size_t size, float min_val = -1.0f, float max_val = 1.0f) {
  static std::mt19937 rng(42);  // Fixed seed for reproducibility
  std::uniform_real_distribution<float> dist(min_val, max_val);
  for (size_t i = 0; i < size; i++) {
    data[i] = dist(rng);
  }
}

// Generate random int64 data
static void fill_random_int64(int64_t* data, size_t size, int64_t min_val = 0, int64_t max_val = 100) {
  static std::mt19937 rng(42);
  std::uniform_int_distribution<int64_t> dist(min_val, max_val);
  for (size_t i = 0; i < size; i++) {
    data[i] = dist(rng);
  }
}

// Generate random int32 data
static void fill_random_int32(int32_t* data, size_t size, int32_t min_val = 0, int32_t max_val = 100) {
  static std::mt19937 rng(42);
  std::uniform_int_distribution<int32_t> dist(min_val, max_val);
  for (size_t i = 0; i < size; i++) {
    data[i] = dist(rng);
  }
}

class GQATester {
public:
  GQATester(const std::filesystem::path& model_path, GQAConfig& config)
      : config_(config), model_path_(model_path) {
    
    const std::string kRegistrationName = get_env("EP_KREGISTERATIONNAME", "MorphiZenExecutionProvider");
    
    std::cout << "=== GQA Layer Test ===" << std::endl;
    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Batch size: " << config_.batch_size << std::endl;
    std::cout << "Sequence length: " << config_.seq_len << std::endl;
    std::cout << "Iterations: " << config_.iterations << std::endl;
    std::cout << "Warmup: " << config_.warmup << std::endl;
    std::cout << "EP enabled: " << (config_.enable_ep ? "yes" : "no") << std::endl;
    std::cout << std::endl;
    
    // Create ONNX Runtime environment
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "test_gqa");
    
    // Register MorphiZen EP if enabled
    if (config_.enable_ep) {
      auto ep_dll = get_env("MORPHIZEN_EP_DLL", "onnxruntime_morphizen_ep.dll");
      auto library_path = std::filesystem::u8path(ep_dll);
      if (!std::filesystem::exists(library_path)) {
        std::cerr << "Warning: EP library not found: " << library_path << std::endl;
        std::cerr << "Continuing with CPU-only execution" << std::endl;
        config_.enable_ep = false;
      } else {
        auto status = Ort::GetApi().RegisterExecutionProviderLibrary(
            *env_, kRegistrationName.c_str(), library_path.c_str());
        
        if (status != nullptr) {
          std::cerr << "Warning: Failed to register EP: " 
                    << Ort::GetApi().GetErrorMessage(status) << std::endl;
          config_.enable_ep = false;
        } else {
          std::cout << "Registered MorphiZen EP for GQA acceleration" << std::endl;
        }
      }
    }
    
    // Create session options
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    
    if (config_.enable_profiler) {
#if _WIN32
      session_options.EnableProfiling(L"profile_test_gqa");
#else
      session_options.EnableProfiling("profile_test_gqa");
#endif
    }
    
    // Configure EP if enabled
    if (config_.enable_ep) {
      std::vector<Ort::ConstEpDevice> selected_devices;
      for (const auto& device : env_->GetEpDevices()) {
        if (device.EpName() == kRegistrationName) {
          selected_devices.emplace_back(device);
        }
      }
      
      if (!selected_devices.empty()) {
        std::unordered_map<std::string, std::string> provider_options;
        std::string config_file = get_env("MORPHIZEN_EP_JSON_CONFIG", "");
        if (!config_file.empty()) {
          provider_options["config_file"] = config_file;
          std::cout << "Using config: " << config_file << std::endl;
        }
        
        session_options.AppendExecutionProvider_V2(*env_, selected_devices, provider_options);
        std::cout << "Using MorphiZen EP (GQA on GPU)" << std::endl;
      }
    }
    
    // Create session
    std::cout << "Loading model..." << std::endl;
#if _WIN32
    session_ = std::make_unique<Ort::Session>(*env_, model_path.wstring().c_str(), session_options);
#else
    session_ = std::make_unique<Ort::Session>(*env_, model_path.u8string().c_str(), session_options);
#endif
    
    // Analyze model inputs/outputs
    AnalyzeModel();
  }
  
  ~GQATester() {
    const std::string kRegistrationName = get_env("EP_KREGISTERATIONNAME", "MorphiZenExecutionProvider");
    if (config_.enable_ep && env_) {
      Ort::GetApi().UnregisterExecutionProviderLibrary(*env_, kRegistrationName.c_str());
    }
  }
  
  void AnalyzeModel() {
    Ort::AllocatorWithDefaultOptions allocator;
    
    num_inputs_ = session_->GetInputCount();
    num_outputs_ = session_->GetOutputCount();
    
    std::cout << "\n=== Model Info ===" << std::endl;
    std::cout << "Inputs: " << num_inputs_ << std::endl;
    
    for (size_t i = 0; i < num_inputs_; i++) {
      auto name = session_->GetInputNameAllocated(i, allocator);
      auto type_info = session_->GetInputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      auto shape = tensor_info.GetShape();
      auto elem_type = tensor_info.GetElementType();
      
      input_names_.push_back(name.get());
      input_shapes_.push_back(shape);
      input_types_.push_back(elem_type);
      
      std::cout << "  [" << i << "] " << name.get() 
                << " : " << element_type_to_string(elem_type) << " [";
      for (size_t j = 0; j < shape.size(); j++) {
        if (j > 0) std::cout << ", ";
        std::cout << shape[j];
      }
      std::cout << "]" << std::endl;
    }
    
    std::cout << "Outputs: " << num_outputs_ << std::endl;
    
    for (size_t i = 0; i < num_outputs_; i++) {
      auto name = session_->GetOutputNameAllocated(i, allocator);
      auto type_info = session_->GetOutputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      auto shape = tensor_info.GetShape();
      auto elem_type = tensor_info.GetElementType();
      
      output_names_.push_back(name.get());
      output_shapes_.push_back(shape);
      output_types_.push_back(elem_type);
      
      std::cout << "  [" << i << "] " << name.get()
                << " : " << element_type_to_string(elem_type) << " [";
      for (size_t j = 0; j < shape.size(); j++) {
        if (j > 0) std::cout << ", ";
        std::cout << shape[j];
      }
      std::cout << "]" << std::endl;
    }
    std::cout << std::endl;
  }
  
  // Resolve dynamic dimensions in shape
  std::vector<int64_t> ResolveShape(const std::vector<int64_t>& shape, const std::string& name) {
    std::vector<int64_t> resolved = shape;
    
    for (size_t i = 0; i < resolved.size(); i++) {
      if (resolved[i] <= 0) {
        // Try to infer from common dimension names
        if (i == 0) {
          // First dimension is usually batch size
          resolved[i] = config_.batch_size;
        } else if (name.find("past") != std::string::npos && i == 2) {
          // Past KV cache sequence dimension - start with 0 or small value
          resolved[i] = 0;
        } else {
          // Default to seq_len for other dynamic dimensions
          resolved[i] = config_.seq_len;
        }
        
        if (config_.verbose) {
          std::cout << "  Resolved " << name << " dim[" << i << "] to " << resolved[i] << std::endl;
        }
      }
    }
    
    return resolved;
  }
  
  void Run() {
    std::cout << "=== Preparing Input Data ===" << std::endl;
    
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    // Prepare input data buffers
    std::vector<std::vector<float>> float_buffers;
    std::vector<std::vector<int64_t>> int64_buffers;
    std::vector<std::vector<int32_t>> int32_buffers;
    std::vector<Ort::Value> input_tensors;
    std::vector<const char*> input_name_ptrs;
    std::vector<const char*> output_name_ptrs;
    
    for (size_t i = 0; i < num_inputs_; i++) {
      auto resolved_shape = ResolveShape(input_shapes_[i], input_names_[i]);
      int64_t num_elements = calculate_product(resolved_shape);
      
      if (config_.verbose) {
        std::cout << "Input " << input_names_[i] << ": " << num_elements << " elements" << std::endl;
      }
      
      switch (input_types_[i]) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
          float_buffers.emplace_back(num_elements);
          fill_random_float(float_buffers.back().data(), num_elements);
          input_tensors.push_back(Ort::Value::CreateTensor<float>(
              memory_info, float_buffers.back().data(), num_elements,
              resolved_shape.data(), resolved_shape.size()));
          break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
          // For FP16, we allocate as uint16_t
          std::vector<uint16_t> fp16_data(num_elements);
          // Initialize with zero (proper FP16 conversion would be needed for real data)
          std::fill(fp16_data.begin(), fp16_data.end(), 0);
          input_tensors.push_back(Ort::Value::CreateTensor(
              memory_info, fp16_data.data(), num_elements * sizeof(uint16_t),
              resolved_shape.data(), resolved_shape.size(),
              ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
          break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
          int64_buffers.emplace_back(num_elements);
          // For seqlens inputs, use appropriate values
          if (input_names_[i].find("seqlens") != std::string::npos) {
            std::fill(int64_buffers.back().begin(), int64_buffers.back().end(), config_.seq_len);
          } else if (input_names_[i].find("total_sequence_length") != std::string::npos) {
            std::fill(int64_buffers.back().begin(), int64_buffers.back().end(), config_.seq_len);
          } else {
            fill_random_int64(int64_buffers.back().data(), num_elements, 0, 100);
          }
          input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
              memory_info, int64_buffers.back().data(), num_elements,
              resolved_shape.data(), resolved_shape.size()));
          break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
          int32_buffers.emplace_back(num_elements);
          if (input_names_[i].find("seqlens") != std::string::npos) {
            std::fill(int32_buffers.back().begin(), int32_buffers.back().end(), config_.seq_len);
          } else if (input_names_[i].find("total_sequence_length") != std::string::npos) {
            std::fill(int32_buffers.back().begin(), int32_buffers.back().end(), config_.seq_len);
          } else {
            fill_random_int32(int32_buffers.back().data(), num_elements, 0, 100);
          }
          input_tensors.push_back(Ort::Value::CreateTensor<int32_t>(
              memory_info, int32_buffers.back().data(), num_elements,
              resolved_shape.data(), resolved_shape.size()));
          break;
        }
        default:
          std::cerr << "Unsupported input type: " << element_type_to_string(input_types_[i]) << std::endl;
          // Create empty tensor with float type as fallback
          float_buffers.emplace_back(1, 0.0f);
          input_tensors.push_back(Ort::Value::CreateTensor<float>(
              memory_info, float_buffers.back().data(), 1,
              resolved_shape.data(), resolved_shape.size()));
          break;
      }
      
      input_name_ptrs.push_back(input_names_[i].c_str());
    }
    
    for (size_t i = 0; i < num_outputs_; i++) {
      output_name_ptrs.push_back(output_names_[i].c_str());
    }
    
    std::cout << "\n=== Running Inference ===" << std::endl;
    
    // Warmup runs
    std::cout << "Warmup (" << config_.warmup << " iterations)..." << std::endl;
    for (int i = 0; i < config_.warmup; i++) {
      try {
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_name_ptrs.data(), input_tensors.data(), num_inputs_,
            output_name_ptrs.data(), num_outputs_);
        
        if (config_.verbose) {
          std::cout << "  Warmup " << (i + 1) << " completed" << std::endl;
        }
      } catch (const Ort::Exception& e) {
        std::cerr << "Warmup error: " << e.what() << std::endl;
        return;
      }
    }
    
    // Benchmark runs
    std::cout << "Benchmark (" << config_.iterations << " iterations)..." << std::endl;
    
    std::vector<double> latencies;
    latencies.reserve(config_.iterations);
    
    for (int i = 0; i < config_.iterations; i++) {
      auto start = std::chrono::high_resolution_clock::now();
      
      try {
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_name_ptrs.data(), input_tensors.data(), num_inputs_,
            output_name_ptrs.data(), num_outputs_);
        
        auto end = std::chrono::high_resolution_clock::now();
        double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();
        latencies.push_back(latency_ms);
        
        if (config_.verbose) {
          std::cout << "  Iteration " << (i + 1) << ": " << std::fixed 
                    << std::setprecision(2) << latency_ms << " ms" << std::endl;
        }
        
        // Print output info on first iteration
        if (i == 0) {
          std::cout << "\n=== Output Info ===" << std::endl;
          for (size_t j = 0; j < output_tensors.size(); j++) {
            auto output_shape = output_tensors[j].GetTensorTypeAndShapeInfo().GetShape();
            std::cout << "  " << output_names_[j] << ": [";
            for (size_t k = 0; k < output_shape.size(); k++) {
              if (k > 0) std::cout << ", ";
              std::cout << output_shape[k];
            }
            std::cout << "]" << std::endl;
            
            // Print sample output values
            if (output_types_[j] == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
              float* data = output_tensors[j].GetTensorMutableData<float>();
              int64_t num_elements = calculate_product(output_shape);
              std::cout << "    First 5 values: ";
              for (int64_t k = 0; k < std::min(int64_t(5), num_elements); k++) {
                std::cout << std::fixed << std::setprecision(4) << data[k] << " ";
              }
              std::cout << std::endl;
            }
          }
          std::cout << std::endl;
        }
        
      } catch (const Ort::Exception& e) {
        std::cerr << "Inference error: " << e.what() << std::endl;
        return;
      }
    }
    
    // Calculate statistics
    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double mean = sum / latencies.size();
    
    std::sort(latencies.begin(), latencies.end());
    double min_latency = latencies.front();
    double max_latency = latencies.back();
    double median = latencies[latencies.size() / 2];
    double p90 = latencies[static_cast<size_t>(latencies.size() * 0.9)];
    double p99 = latencies[static_cast<size_t>(latencies.size() * 0.99)];
    
    // Variance and std dev
    double sq_sum = 0.0;
    for (double lat : latencies) {
      sq_sum += (lat - mean) * (lat - mean);
    }
    double std_dev = std::sqrt(sq_sum / latencies.size());
    
    std::cout << "=== Benchmark Results ===" << std::endl;
    std::cout << "  Iterations: " << config_.iterations << std::endl;
    std::cout << "  Mean latency: " << std::fixed << std::setprecision(2) << mean << " ms" << std::endl;
    std::cout << "  Std dev: " << std::fixed << std::setprecision(2) << std_dev << " ms" << std::endl;
    std::cout << "  Min latency: " << std::fixed << std::setprecision(2) << min_latency << " ms" << std::endl;
    std::cout << "  Max latency: " << std::fixed << std::setprecision(2) << max_latency << " ms" << std::endl;
    std::cout << "  Median: " << std::fixed << std::setprecision(2) << median << " ms" << std::endl;
    std::cout << "  P90: " << std::fixed << std::setprecision(2) << p90 << " ms" << std::endl;
    std::cout << "  P99: " << std::fixed << std::setprecision(2) << p99 << " ms" << std::endl;
    std::cout << "  Throughput: " << std::fixed << std::setprecision(2) 
              << (1000.0 / mean) << " infer/sec" << std::endl;
  }
  
private:
  GQAConfig& config_;
  std::filesystem::path model_path_;
  
  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;
  
  size_t num_inputs_;
  size_t num_outputs_;
  
  std::vector<std::string> input_names_;
  std::vector<std::vector<int64_t>> input_shapes_;
  std::vector<ONNXTensorElementDataType> input_types_;
  
  std::vector<std::string> output_names_;
  std::vector<std::vector<int64_t>> output_shapes_;
  std::vector<ONNXTensorElementDataType> output_types_;
};

// Simple command line argument parsing (no getopt dependency)
int main(int argc, char* argv[]) {
  GQAConfig config;
  std::string model_path;
  
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    
    if (arg == "-h" || arg == "--help") {
      usage();
      return 0;
    } else if (arg == "-b" && i + 1 < argc) {
      config.batch_size = std::stoi(argv[++i]);
    } else if (arg == "-s" && i + 1 < argc) {
      config.seq_len = std::stoi(argv[++i]);
    } else if (arg == "-i" && i + 1 < argc) {
      config.iterations = std::stoi(argv[++i]);
    } else if (arg == "-w" && i + 1 < argc) {
      config.warmup = std::stoi(argv[++i]);
    } else if (arg == "-n") {
      config.enable_ep = false;
    } else if (arg == "-p") {
      config.enable_profiler = true;
    } else if (arg == "-v") {
      config.verbose = true;
    } else if (arg[0] != '-') {
      model_path = arg;
    } else {
      std::cerr << "Unknown option: " << arg << std::endl;
      usage();
      return 1;
    }
  }
  
  if (model_path.empty()) {
    std::cerr << "Error: Missing model path" << std::endl;
    usage();
    return 1;
  }
  
  if (!std::filesystem::exists(model_path)) {
    std::cerr << "Error: Model file not found: " << model_path << std::endl;
    return 1;
  }
  
  try {
    GQATester tester(model_path, config);
    tester.Run();
  } catch (const Ort::Exception& e) {
    std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  
  return 0;
}
