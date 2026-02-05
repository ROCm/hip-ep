// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

/**
 * Model Verifier - Compare CPU vs GPU (HipDNN EP) outputs for any ONNX model
 * 
 * Usage:
 *   model_verifier.exe <model_path> [options]
 * 
 * Options:
 *   --tolerance <float>   Tolerance for comparison (default: 0.1)
 *   --iterations <int>    Number of iterations (default: 1)
 *   --seed <int>          Random seed for input generation (default: 42)
 *   --verbose             Enable verbose output
 * 
 * Environment:
 *   MORPHIZEN_DEBUG_ROCM=2  Enable debug output from HipDNN EP
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <random>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <numeric>

#ifndef ORT_API_MANUAL_INIT
#define ORT_API_MANUAL_INIT
#endif
#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
inline std::wstring ToWideString(const char* str) {
  int len = MultiByteToWideChar(CP_UTF8, 0, str, -1, nullptr, 0);
  std::wstring result(len - 1, 0);
  MultiByteToWideChar(CP_UTF8, 0, str, -1, &result[0], len);
  return result;
}
#endif

#ifndef HIPDNN_EP_LIB_PATH
#ifdef _WIN32
// The DLL name is set by morphizen's CMAKE output_name
#define HIPDNN_EP_LIB_PATH "onnxruntime_morphizen_ep.dll"
#else
#define HIPDNN_EP_LIB_PATH "./libonnxruntime_morphizen_ep.so"
#endif
#endif

// Configuration
struct Config {
  std::string model_path;
  float tolerance = 0.1f;
  int iterations = 1;
  int seed = 42;
  bool verbose = false;
};

// Statistics
struct ComparisonStats {
  float max_diff = 0.0f;
  float max_diff_cpu_val = 0.0f;  // CPU value that caused max_diff
  float max_diff_gpu_val = 0.0f;  // GPU value that caused max_diff
  size_t max_diff_idx = 0;        // Index where max_diff occurred
  float mean_diff = 0.0f;
  float std_diff = 0.0f;
  size_t mismatch_count = 0;
  size_t total_elements = 0;
  size_t nan_count = 0;
  size_t inf_count = 0;
};

void print_usage(const char* prog_name) {
  std::cout << "Model Verifier - Compare CPU vs GPU (HipDNN EP) outputs\n\n";
  std::cout << "Usage: " << prog_name << " <model_path> [options]\n\n";
  std::cout << "Options:\n";
  std::cout << "  --tolerance <float>   Tolerance for comparison (default: 0.1)\n";
  std::cout << "  --iterations <int>    Number of iterations (default: 1)\n";
  std::cout << "  --seed <int>          Random seed for input generation (default: 42)\n";
  std::cout << "  --verbose             Enable verbose output\n";
  std::cout << "\nEnvironment:\n";
  std::cout << "  MORPHIZEN_DEBUG_ROCM=2  Enable debug output from HipDNN EP\n";
}

Config parse_args(int argc, char** argv) {
  Config config;
  
  if (argc < 2) {
    print_usage(argv[0]);
    exit(1);
  }
  
  config.model_path = argv[1];
  
  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--tolerance" && i + 1 < argc) {
      config.tolerance = std::stof(argv[++i]);
    } else if (arg == "--iterations" && i + 1 < argc) {
      config.iterations = std::stoi(argv[++i]);
    } else if (arg == "--seed" && i + 1 < argc) {
      config.seed = std::stoi(argv[++i]);
    } else if (arg == "--verbose") {
      config.verbose = true;
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      exit(0);
    }
  }
  
  return config;
}

std::string shape_to_string(const std::vector<int64_t>& shape) {
  std::string s = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    s += std::to_string(shape[i]);
    if (i < shape.size() - 1) s += ", ";
  }
  s += "]";
  return s;
}

ComparisonStats compare_outputs(const std::vector<float>& cpu, 
                                 const std::vector<float>& gpu,
                                 float tolerance,
                                 bool verbose) {
  ComparisonStats stats;
  stats.total_elements = cpu.size();
  
  if (cpu.size() != gpu.size()) {
    std::cerr << "ERROR: Size mismatch! CPU=" << cpu.size() << ", GPU=" << gpu.size() << std::endl;
    return stats;
  }
  
  std::vector<float> diffs(cpu.size());
  float sum_diff = 0.0f;
  
  for (size_t i = 0; i < cpu.size(); ++i) {
    if (std::isnan(gpu[i])) {
      stats.nan_count++;
    }
    if (std::isinf(gpu[i])) {
      stats.inf_count++;
    }
    
    float diff = std::abs(cpu[i] - gpu[i]);
    diffs[i] = diff;
    sum_diff += diff;
    
    // Track max difference and the values that caused it
    if (diff > stats.max_diff) {
      stats.max_diff = diff;
      stats.max_diff_cpu_val = cpu[i];
      stats.max_diff_gpu_val = gpu[i];
      stats.max_diff_idx = i;
    }
    
    if (diff > tolerance) {
      stats.mismatch_count++;
      if (verbose && stats.mismatch_count <= 10) {
        std::cout << "  MISMATCH[" << i << "]: CPU=" << cpu[i] 
                  << ", GPU=" << gpu[i] << ", diff=" << diff << std::endl;
      }
    }
  }
  
  stats.mean_diff = sum_diff / cpu.size();
  
  // Calculate standard deviation
  float sum_sq = 0.0f;
  for (size_t i = 0; i < diffs.size(); ++i) {
    float d = diffs[i] - stats.mean_diff;
    sum_sq += d * d;
  }
  stats.std_diff = std::sqrt(sum_sq / cpu.size());
  
  return stats;
}

void print_stats(const ComparisonStats& stats, float tolerance) {
  std::cout << "\n=== Comparison Statistics ===" << std::endl;
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "  Total elements:     " << stats.total_elements << std::endl;
  std::cout << "  Max difference:     " << stats.max_diff 
            << " (at index " << stats.max_diff_idx 
            << ": CPU=" << stats.max_diff_cpu_val 
            << ", GPU=" << stats.max_diff_gpu_val << ")" << std::endl;
  std::cout << "  Mean difference:    " << stats.mean_diff << std::endl;
  std::cout << "  Std difference:     " << stats.std_diff << std::endl;
  std::cout << "  Tolerance:          " << tolerance << std::endl;
  std::cout << "  Mismatches:         " << stats.mismatch_count 
            << " (" << (100.0f * stats.mismatch_count / stats.total_elements) << "%)" << std::endl;
  
  if (stats.nan_count > 0) {
    std::cout << "  WARNING: NaN values: " << stats.nan_count << std::endl;
  }
  if (stats.inf_count > 0) {
    std::cout << "  WARNING: Inf values: " << stats.inf_count << std::endl;
  }
  
  std::cout << "\n=== Result ===" << std::endl;
  if (stats.mismatch_count == 0 && stats.nan_count == 0 && stats.inf_count == 0) {
    std::cout << "  [PASS] All " << stats.total_elements << " elements within tolerance" << std::endl;
  } else {
    std::cout << "  [FAIL] " << stats.mismatch_count << " elements exceed tolerance" << std::endl;
  }
}

int main(int argc, char** argv) {
  std::cout << "\n" << std::string(70, '=') << std::endl;
  std::cout << "  HipDNN EP Model Verifier - CPU vs GPU Comparison" << std::endl;
  std::cout << std::string(70, '=') << "\n" << std::endl;
  
  Config config = parse_args(argc, argv);
  
  // Check if model exists
  std::ifstream f(config.model_path);
  if (!f.good()) {
    std::cerr << "ERROR: Model file not found: " << config.model_path << std::endl;
    return 1;
  }
  f.close();
  
  std::cout << "Configuration:" << std::endl;
  std::cout << "  Model:      " << config.model_path << std::endl;
  std::cout << "  Tolerance:  " << config.tolerance << std::endl;
  std::cout << "  Iterations: " << config.iterations << std::endl;
  std::cout << "  Seed:       " << config.seed << std::endl;
  std::cout << "  Verbose:    " << (config.verbose ? "yes" : "no") << std::endl;
  std::cout << std::endl;
  
  // Initialize ORT C++ API
  Ort::InitApi(OrtGetApiBase()->GetApi(ORT_API_VERSION));
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ModelVerifier");
  
  // Register HipDNN EP using C++ API
  const char* lib_path = HIPDNN_EP_LIB_PATH;
  std::cout << "Loading EP library: " << lib_path << std::endl;
  
  try {
#ifdef _WIN32
    auto lib_path_w = ToWideString(lib_path);
    env.RegisterExecutionProviderLibrary("MorphiZen", lib_path_w);
#else
    env.RegisterExecutionProviderLibrary("MorphiZen", lib_path);
#endif
  } catch (const Ort::Exception& ex) {
    std::cerr << "ERROR: Failed to register HipDNN EP: " << ex.what() << std::endl;
    return 1;
  }
  std::cout << "HipDNN EP registered successfully from: " << lib_path << std::endl;
  
  // Get model info using CPU session
  std::cout << "\n--- Loading Model ---" << std::endl;
  
  Ort::SessionOptions cpu_options;
  cpu_options.SetIntraOpNumThreads(1);
  
#ifdef _WIN32
  auto model_path_w = ToWideString(config.model_path.c_str());
  Ort::Session cpu_session(env, model_path_w.c_str(), cpu_options);
#else
  Ort::Session cpu_session(env, config.model_path.c_str(), cpu_options);
#endif
  
  // Get input/output info
  Ort::AllocatorWithDefaultOptions allocator;
  
  size_t num_inputs = cpu_session.GetInputCount();
  size_t num_outputs = cpu_session.GetOutputCount();
  
  std::cout << "Model inputs:  " << num_inputs << std::endl;
  std::cout << "Model outputs: " << num_outputs << std::endl;
  
  // Prepare inputs
  std::vector<std::string> input_names;
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::vector<float>> input_data;
  
  std::mt19937 rng(config.seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  
  std::cout << "\n--- Input Information ---" << std::endl;
  for (size_t i = 0; i < num_inputs; ++i) {
    auto name = cpu_session.GetInputNameAllocated(i, allocator);
    input_names.push_back(name.get());
    
    auto type_info = cpu_session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape();
    
    // Handle dynamic dimensions
    for (auto& dim : shape) {
      if (dim < 0) dim = 1;  // Default dynamic dims to 1
    }
    input_shapes.push_back(shape);
    
    // Calculate size and generate random data
    size_t size = 1;
    for (auto d : shape) size *= static_cast<size_t>(d);
    
    std::vector<float> data(size);
    for (size_t j = 0; j < size; ++j) {
      data[j] = dist(rng);
    }
    input_data.push_back(data);
    
    std::cout << "  Input " << i << ": " << input_names[i] 
              << " " << shape_to_string(shape) 
              << " (" << size << " elements)" << std::endl;
  }
  
  // Get output names
  std::vector<std::string> output_names;
  std::cout << "\n--- Output Information ---" << std::endl;
  for (size_t i = 0; i < num_outputs; ++i) {
    auto name = cpu_session.GetOutputNameAllocated(i, allocator);
    output_names.push_back(name.get());
    
    auto type_info = cpu_session.GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape();
    
    std::cout << "  Output " << i << ": " << output_names[i] 
              << " " << shape_to_string(shape) << std::endl;
  }
  
  // Convert names to char pointers
  std::vector<const char*> input_name_ptrs;
  for (const auto& n : input_names) input_name_ptrs.push_back(n.c_str());
  std::vector<const char*> output_name_ptrs;
  for (const auto& n : output_names) output_name_ptrs.push_back(n.c_str());
  
  // Run iterations
  bool all_passed = true;
  
  for (int iter = 0; iter < config.iterations; ++iter) {
    std::cout << "\n" << std::string(70, '-') << std::endl;
    std::cout << "  Iteration " << (iter + 1) << "/" << config.iterations << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    // Regenerate input data for each iteration (except first)
    if (iter > 0) {
      for (size_t i = 0; i < input_data.size(); ++i) {
        for (size_t j = 0; j < input_data[i].size(); ++j) {
          input_data[i][j] = dist(rng);
        }
      }
    }
    
    // Create input tensors
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> input_tensors;
    for (size_t i = 0; i < num_inputs; ++i) {
      input_tensors.push_back(Ort::Value::CreateTensor<float>(
          memory_info, input_data[i].data(), input_data[i].size(),
          input_shapes[i].data(), input_shapes[i].size()));
    }
    
    // Run CPU inference
    std::cout << "\n--- CPU Inference ---" << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    
    auto cpu_outputs = cpu_session.Run(Ort::RunOptions{}, 
                                        input_name_ptrs.data(), 
                                        input_tensors.data(), 
                                        num_inputs,
                                        output_name_ptrs.data(), 
                                        num_outputs);
    
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count();
    std::cout << "  Time: " << (cpu_time / 1000.0f) << " ms" << std::endl;
    
    // Store CPU outputs
    std::vector<std::vector<float>> cpu_output_data;
    std::vector<std::vector<int64_t>> output_shapes;
    for (size_t i = 0; i < cpu_outputs.size(); ++i) {
      const float* data = cpu_outputs[i].GetTensorData<float>();
      size_t size = cpu_outputs[i].GetTensorTypeAndShapeInfo().GetElementCount();
      auto shape = cpu_outputs[i].GetTensorTypeAndShapeInfo().GetShape();
      
      cpu_output_data.push_back(std::vector<float>(data, data + size));
      output_shapes.push_back(shape);
      
      if (config.verbose) {
        std::cout << "  Output " << i << " " << shape_to_string(shape) 
                  << ": first=" << data[0] << ", last=" << data[size-1] << std::endl;
      }
    }
    
    // Try GPU inference with HipDNN EP
    std::cout << "\n--- GPU Inference (HipDNN EP) ---" << std::endl;
    
    std::vector<Ort::ConstEpDevice> devices = env.GetEpDevices();
    const OrtEpDevice* hipdnn_device = nullptr;
    
    for (const auto& device : devices) {
      std::string ep_name = device.EpName();
      if (ep_name == "MorphiZen" || ep_name == "MorphiZenExecutionProvider") {
        hipdnn_device = static_cast<const OrtEpDevice*>(device);
        break;
      }
    }
    
    if (hipdnn_device == nullptr) {
      std::cout << "  WARNING: HipDNN EP V2 device not available" << std::endl;
      std::cout << "  Skipping GPU comparison" << std::endl;
      continue;
    }
    
    Ort::SessionOptions gpu_options;
    
    // Enable profiling if verbose mode to see which nodes are run on GPU
    if (config.verbose) {
#ifdef _WIN32
      gpu_options.EnableProfiling(L"hipdnn_profile");
#else
      gpu_options.EnableProfiling("hipdnn_profile");
#endif
    }
    
    // Append EP using C++ style API
    try {
      std::vector<Ort::ConstEpDevice> selected_devices;
      selected_devices.push_back(Ort::ConstEpDevice(hipdnn_device));
      gpu_options.AppendExecutionProvider_V2(env, selected_devices, {});
    } catch (const Ort::Exception& ex) {
      std::cerr << "  ERROR: Failed to add HipDNN EP: " << ex.what() << std::endl;
      all_passed = false;
      continue;
    }
    
    try {
#ifdef _WIN32
      Ort::Session gpu_session(env, model_path_w.c_str(), gpu_options);
#else
      Ort::Session gpu_session(env, config.model_path.c_str(), gpu_options);
#endif
      
      // Show model metadata - which nodes are handled by which EP
      if (config.verbose) {
        std::cout << "\n--- Node Execution Providers ---" << std::endl;
        // Note: ORT doesn't expose per-node EP info directly via C++ API
        // The profiling file will contain this information
        std::cout << "  (Check hipdnn_profile*.json for detailed EP assignments)" << std::endl;
      }
      
      // Recreate input tensors for GPU session
      std::vector<Ort::Value> gpu_input_tensors;
      for (size_t i = 0; i < num_inputs; ++i) {
        gpu_input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, input_data[i].data(), input_data[i].size(),
            input_shapes[i].data(), input_shapes[i].size()));
      }
      
      auto gpu_start = std::chrono::high_resolution_clock::now();
      
      auto gpu_outputs = gpu_session.Run(Ort::RunOptions{},
                                          input_name_ptrs.data(),
                                          gpu_input_tensors.data(),
                                          num_inputs,
                                          output_name_ptrs.data(),
                                          num_outputs);
      
      auto gpu_end = std::chrono::high_resolution_clock::now();
      auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start).count();
      std::cout << "  Time: " << (gpu_time / 1000.0f) << " ms" << std::endl;
      std::cout << "  Speedup: " << (static_cast<float>(cpu_time) / gpu_time) << "x" << std::endl;
      
      // Compare outputs
      std::cout << "\n--- Output Comparison ---" << std::endl;
      for (size_t i = 0; i < gpu_outputs.size(); ++i) {
        const float* data = gpu_outputs[i].GetTensorData<float>();
        size_t size = gpu_outputs[i].GetTensorTypeAndShapeInfo().GetElementCount();
        
        std::vector<float> gpu_output(data, data + size);
        
        std::cout << "\nOutput " << i << " (" << output_names[i] << ") "
                  << shape_to_string(output_shapes[i]) << ":" << std::endl;
        
        if (config.verbose) {
          std::cout << "  CPU: first=" << cpu_output_data[i][0] 
                    << ", last=" << cpu_output_data[i].back() << std::endl;
          std::cout << "  GPU: first=" << gpu_output[0] 
                    << ", last=" << gpu_output.back() << std::endl;
        }
        
        auto stats = compare_outputs(cpu_output_data[i], gpu_output, config.tolerance, config.verbose);
        print_stats(stats, config.tolerance);
        
        if (stats.mismatch_count > 0 || stats.nan_count > 0) {
          all_passed = false;
        }
      }
      
    } catch (const Ort::Exception& ex) {
      std::cerr << "  ERROR: " << ex.what() << std::endl;
      all_passed = false;
    }
  }
  
  // Final summary
  std::cout << "\n" << std::string(70, '=') << std::endl;
  std::cout << "  FINAL RESULT" << std::endl;
  std::cout << std::string(70, '=') << std::endl;
  
  if (all_passed) {
    std::cout << "\n  [PASS] All comparisons passed!\n" << std::endl;
    return 0;
  } else {
    std::cout << "\n  [FAIL] Some comparisons failed!\n" << std::endl;
    return 1;
  }
}
