/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#ifndef ORT_API_MANUAL_INIT
#define ORT_API_MANUAL_INIT
#endif
#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
inline std::wstring ToWideString(const char *str) {
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

// Per-element diff entry for top-N tracking
struct DiffEntry {
  size_t idx;
  float cpu_val;
  float gpu_val;
  float diff;
};

// Statistics
struct ComparisonStats {
  float max_diff = 0.0f;
  float max_diff_cpu_val = 0.0f; // CPU value that caused max_diff
  float max_diff_gpu_val = 0.0f; // GPU value that caused max_diff
  size_t max_diff_idx = 0;       // Index where max_diff occurred
  float mean_diff = 0.0f;
  float std_diff = 0.0f;
  size_t mismatch_count = 0;
  size_t total_elements = 0;
  size_t nan_count = 0;
  size_t inf_count = 0;
  std::vector<DiffEntry> top_diffs; // Top N largest differences
};

void print_usage(const char *prog_name) {
  std::cout << "Model Verifier - Compare CPU vs GPU (HipDNN EP) outputs\n\n";
  std::cout << "Usage: " << prog_name << " <model_path> [options]\n\n";
  std::cout << "Options:\n";
  std::cout
      << "  --tolerance <float>   Tolerance for comparison (default: 0.1)\n";
  std::cout << "  --iterations <int>    Number of iterations (default: 1)\n";
  std::cout << "  --seed <int>          Random seed for input generation "
               "(default: 42)\n";
  std::cout << "  --verbose             Enable verbose output\n";
  std::cout << "\nEnvironment:\n";
  std::cout << "  MORPHIZEN_DEBUG_ROCM=2  Enable debug output from HipDNN EP\n";
}

Config parse_args(int argc, char **argv) {
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

std::string shape_to_string(const std::vector<int64_t> &shape) {
  std::string s = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    s += std::to_string(shape[i]);
    if (i < shape.size() - 1)
      s += ", ";
  }
  s += "]";
  return s;
}

ComparisonStats compare_outputs(const std::vector<float> &cpu,
                                const std::vector<float> &gpu, float tolerance,
                                bool verbose) {
  ComparisonStats stats;
  stats.total_elements = cpu.size();

  if (cpu.size() != gpu.size()) {
    std::cerr << "ERROR: Size mismatch! CPU=" << cpu.size()
              << ", GPU=" << gpu.size() << std::endl;
    return stats;
  }

  constexpr size_t TOP_N = 20;
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

    // Maintain top-N largest diffs (insertion sort into small vector)
    if (stats.top_diffs.size() < TOP_N || diff > stats.top_diffs.back().diff) {
      DiffEntry entry{i, cpu[i], gpu[i], diff};
      if (stats.top_diffs.size() >= TOP_N)
        stats.top_diffs.back() = entry;
      else
        stats.top_diffs.push_back(entry);
      // Bubble up to maintain descending order
      for (size_t j = stats.top_diffs.size() - 1; j > 0; --j) {
        if (stats.top_diffs[j].diff > stats.top_diffs[j - 1].diff)
          std::swap(stats.top_diffs[j], stats.top_diffs[j - 1]);
        else
          break;
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

void print_stats(const ComparisonStats &stats, float tolerance,
                 const std::vector<float> &cpu, const std::vector<float> &gpu) {
  std::cout << "\n=== Comparison Statistics ===" << std::endl;
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "  Total elements:     " << stats.total_elements << std::endl;
  std::cout << "  Max difference:     " << stats.max_diff << " (at index "
            << stats.max_diff_idx << ": CPU=" << stats.max_diff_cpu_val
            << ", GPU=" << stats.max_diff_gpu_val << ")" << std::endl;
  std::cout << "  Mean difference:    " << stats.mean_diff << std::endl;
  std::cout << "  Std difference:     " << stats.std_diff << std::endl;
  std::cout << "  Tolerance:          " << tolerance << std::endl;
  std::cout << "  Mismatches:         " << stats.mismatch_count << " ("
            << (100.0f * stats.mismatch_count / stats.total_elements) << "%)"
            << std::endl;

  if (stats.nan_count > 0) {
    std::cout << "  WARNING: NaN values: " << stats.nan_count << std::endl;
  }
  if (stats.inf_count > 0) {
    std::cout << "  WARNING: Inf values: " << stats.inf_count << std::endl;
  }

  // Print top-20 largest differences
  if (!stats.top_diffs.empty()) {
    std::cout << "\n=== Top " << stats.top_diffs.size()
              << " Largest Differences ===" << std::endl;
    std::cout << "  " << std::setw(8) << "Index"
              << "  " << std::setw(12) << "CPU"
              << "  " << std::setw(12) << "GPU"
              << "  " << std::setw(12) << "Diff" << std::endl;
    std::cout << "  " << std::string(50, '-') << std::endl;
    for (const auto &d : stats.top_diffs) {
      std::cout << "  " << std::setw(8) << d.idx << "  " << std::setw(12)
                << d.cpu_val << "  " << std::setw(12) << d.gpu_val << "  "
                << std::setw(12) << d.diff << std::endl;
    }
  }

  // Print a window of values around the max diff index
  if (stats.total_elements > 0 && !cpu.empty() && !gpu.empty()) {
    std::cout << "\n=== Values Around Max Diff (index " << stats.max_diff_idx
              << ") ===" << std::endl;
    std::cout << "  " << std::setw(8) << "Index"
              << "  " << std::setw(12) << "CPU"
              << "  " << std::setw(12) << "GPU"
              << "  " << std::setw(12) << "Diff" << std::endl;
    std::cout << "  " << std::string(50, '-') << std::endl;
    int64_t center = static_cast<int64_t>(stats.max_diff_idx);
    int64_t start = std::max(int64_t(0), center - 5);
    int64_t end =
        std::min(static_cast<int64_t>(stats.total_elements), center + 6);
    for (int64_t i = start; i < end; ++i) {
      float diff = std::abs(cpu[i] - gpu[i]);
      std::cout << (static_cast<size_t>(i) == stats.max_diff_idx ? "  >" : "  ")
                << std::setw(7) << i << "  " << std::setw(12) << cpu[i] << "  "
                << std::setw(12) << gpu[i] << "  " << std::setw(12) << diff
                << std::endl;
    }
  }

  std::cout << "\n=== Result ===" << std::endl;
  if (stats.mismatch_count == 0 && stats.nan_count == 0 &&
      stats.inf_count == 0) {
    std::cout << "  [PASS] All " << stats.total_elements
              << " elements within tolerance" << std::endl;
  } else {
    std::cout << "  [FAIL] " << stats.mismatch_count
              << " elements exceed tolerance" << std::endl;
  }
}

// Helper: convert fp16 (uint16_t) to float32
inline float fp16_to_float(uint16_t h) {
  uint32_t sign = (h >> 15) & 0x1;
  uint32_t exponent = (h >> 10) & 0x1f;
  uint32_t mantissa = h & 0x3ff;

  uint32_t f;
  if (exponent == 0) {
    if (mantissa == 0) {
      f = sign << 31; // +/- zero
    } else {
      // Subnormal fp16 -> normal fp32
      exponent = 1;
      while (!(mantissa & 0x400)) {
        mantissa <<= 1;
        exponent--;
      }
      mantissa &= 0x3ff;
      f = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    // Inf or NaN
    f = (sign << 31) | 0x7f800000 | (mantissa << 13);
  } else {
    f = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  }

  float result;
  std::memcpy(&result, &f, sizeof(float));
  return result;
}

// Helper: convert float32 to fp16 (uint16_t) with rounding
inline uint16_t float_to_fp16(float value) {
  uint32_t f;
  std::memcpy(&f, &value, sizeof(float));

  uint32_t sign = (f >> 31) & 0x1;
  int32_t exponent = ((f >> 23) & 0xff) - 127;
  uint32_t mantissa = f & 0x7fffff;

  uint16_t h;
  if (exponent > 15) {
    h = (sign << 15) | 0x7c00; // Inf
  } else if (exponent > -15) {
    h = (sign << 15) | ((exponent + 15) << 10) | (mantissa >> 13);
  } else {
    h = (sign << 15); // Zero (flush subnormals)
  }
  return h;
}

// Helper: get element type name string
std::string elem_type_name(ONNXTensorElementDataType type) {
  switch (type) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    return "float32";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    return "float16";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    return "int32";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    return "int64";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    return "uint8";
  default:
    return "type_" + std::to_string(static_cast<int>(type));
  }
}

// Helper: extract output tensor data as float vector (handles fp16 conversion)
std::vector<float> extract_output_as_float(const Ort::Value &tensor) {
  auto type_info = tensor.GetTensorTypeAndShapeInfo();
  auto elem_type = type_info.GetElementType();
  size_t count = type_info.GetElementCount();

  std::vector<float> result(count);

  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    const float *data = tensor.GetTensorData<float>();
    std::copy(data, data + count, result.begin());
  } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    const uint16_t *data = tensor.GetTensorData<uint16_t>();
    for (size_t i = 0; i < count; ++i) {
      result[i] = fp16_to_float(data[i]);
    }
  } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    const int32_t *data = tensor.GetTensorData<int32_t>();
    for (size_t i = 0; i < count; ++i) {
      result[i] = static_cast<float>(data[i]);
    }
  } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    const int64_t *data = tensor.GetTensorData<int64_t>();
    for (size_t i = 0; i < count; ++i) {
      result[i] = static_cast<float>(data[i]);
    }
  } else {
    std::cerr << "WARNING: Unsupported output type "
              << static_cast<int>(elem_type) << ", treating as float"
              << std::endl;
    const float *data = tensor.GetTensorData<float>();
    std::copy(data, data + count, result.begin());
  }

  return result;
}

int main(int argc, char **argv) {
  std::cout << "\n" << std::string(70, '=') << std::endl;
  std::cout << "  HipDNN EP Model Verifier - CPU vs GPU Comparison"
            << std::endl;
  std::cout << std::string(70, '=') << "\n" << std::endl;

  Config config = parse_args(argc, argv);

  // Check if model exists
  std::ifstream f(config.model_path);
  if (!f.good()) {
    std::cerr << "ERROR: Model file not found: " << config.model_path
              << std::endl;
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

  // Determine log level from environment variable ORT_LOG_LEVEL
  OrtLoggingLevel ort_log_level = ORT_LOGGING_LEVEL_WARNING;
  const char *log_level_env = std::getenv("ORT_LOG_LEVEL");
  if (log_level_env != nullptr) {
    std::string log_level_str(log_level_env);
    if (log_level_str == "info") {
      ort_log_level = ORT_LOGGING_LEVEL_INFO;
    } else if (log_level_str == "warning") {
      ort_log_level = ORT_LOGGING_LEVEL_WARNING;
    } else if (log_level_str == "error") {
      ort_log_level = ORT_LOGGING_LEVEL_ERROR;
    }
  }
  Ort::Env env(ort_log_level, "ModelVerifier");

  // Register HipDNN EP using C++ API
  const char *lib_path = HIPDNN_EP_LIB_PATH;
  std::cout << "Loading EP library: " << lib_path << std::endl;

  try {
#ifdef _WIN32
    auto lib_path_w = ToWideString(lib_path);
    env.RegisterExecutionProviderLibrary("MorphiZen", lib_path_w);
#else
    env.RegisterExecutionProviderLibrary("MorphiZen", lib_path);
#endif
  } catch (const Ort::Exception &ex) {
    std::cerr << "ERROR: Failed to register HipDNN EP: " << ex.what()
              << std::endl;
    return 1;
  }
  std::cout << "HipDNN EP registered successfully from: " << lib_path
            << std::endl;

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

  // Prepare inputs (supports mixed data types: float32, float16, int32)
  std::vector<std::string> input_names;
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<ONNXTensorElementDataType> input_types;

  // Raw byte storage for each input (supports any data type)
  std::vector<std::vector<uint8_t>> input_raw_data;
  // Element counts for each input
  std::vector<size_t> input_elem_counts;

  std::mt19937 rng(config.seed);
  std::normal_distribution<float> dist(0.0f,
                                       0.5f); // Smaller range for fp16 safety
  std::uniform_int_distribution<int32_t> int_dist(0, 255);

  // First pass: collect all input metadata to determine GQA-specific values
  std::cout << "\n--- Input Information ---" << std::endl;
  for (size_t i = 0; i < num_inputs; ++i) {
    auto name = cpu_session.GetInputNameAllocated(i, allocator);
    input_names.push_back(name.get());

    auto type_info = cpu_session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape();
    auto elem_type = tensor_info.GetElementType();
    input_types.push_back(elem_type);

    // Handle dynamic dimensions:
    // - Negative dims (symbolic) -> default to 1
    // - Zero dims are VALID and preserved (e.g., [1, 8, 0, 128] for empty KV
    // cache)
    for (auto &dim : shape) {
      if (dim < 0)
        dim = 1; // Default symbolic dims to 1
      // NOTE: dim == 0 is intentionally preserved (zero-sized tensor)
    }
    input_shapes.push_back(shape);

    // Calculate element count (may be 0 for zero-sized dims)
    size_t size = 1;
    for (auto d : shape)
      size *= static_cast<size_t>(d);
    input_elem_counts.push_back(size);

    std::cout << "  Input " << i << ": " << input_names[i] << " "
              << shape_to_string(shape) << " " << elem_type_name(elem_type)
              << " (" << size << " elements)" << std::endl;
  }

  // Detect GQA-specific inputs and derive proper values:
  // - past_key shape [B, nhead_k, past_seqlen, hdim] -> past_seq_len
  // - query shape [B, seqlen_q, hidden] -> current query seq_len
  int64_t past_seq_len = 0;
  int64_t query_seq_len = 1; // default
  for (size_t i = 0; i < num_inputs; ++i) {
    const auto &name = input_names[i];
    const auto &shape = input_shapes[i];
    if (name.find("past_key") != std::string::npos && shape.size() >= 3) {
      past_seq_len = shape[2]; // [B, nhead, past_seqlen, hdim]
    }
    if (name == "query" && shape.size() >= 2) {
      query_seq_len = shape[1]; // [B, seqlen_q, hidden]
    }
  }
  int64_t total_seq_len_val = past_seq_len + query_seq_len;

  if (config.verbose) {
    std::cout << "\n  GQA detected: past_seq_len=" << past_seq_len
              << ", query_seq_len=" << query_seq_len
              << ", total_seq_len=" << total_seq_len_val << std::endl;
  }

  // Second pass: generate input data
  for (size_t i = 0; i < num_inputs; ++i) {
    auto elem_type = input_types[i];
    size_t size = input_elem_counts[i];
    const auto &name = input_names[i];

    // Determine element byte size
    size_t elem_bytes = 4; // default float32
    switch (elem_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      elem_bytes = 2;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      elem_bytes = 4;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      elem_bytes = 8;
      break;
    default:
      elem_bytes = 4;
      break;
    }

    std::vector<uint8_t> raw(size * elem_bytes, 0);

    if (size > 0) {
      // Check for GQA-specific semantic inputs
      bool is_seqlens = (name.find("seqlens") != std::string::npos);
      bool is_total_seq = (name.find("total_seq") != std::string::npos ||
                           name.find("total_sequence") != std::string::npos);

      if (is_seqlens || is_total_seq) {
        // GQA semantic values:
        //   seqlens_k = total_sequence_length - 1  (ORT CPU uses: total_seqlen
        //   = seqlens_k + 1) total_seq_len = past + query
        // Ref: onnxruntime/contrib_ops/cpu/bert/gqa_attention_base.h
        int64_t fill_val =
            is_seqlens ? (total_seq_len_val - 1) : total_seq_len_val;

        if (config.verbose) {
          std::cout << "  Setting " << name << " = " << fill_val << std::endl;
        }

        if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
          int32_t *data = reinterpret_cast<int32_t *>(raw.data());
          for (size_t j = 0; j < size; ++j)
            data[j] = static_cast<int32_t>(fill_val);
        } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
          int64_t *data = reinterpret_cast<int64_t *>(raw.data());
          for (size_t j = 0; j < size; ++j)
            data[j] = fill_val;
        }
      } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        float *data = reinterpret_cast<float *>(raw.data());
        for (size_t j = 0; j < size; ++j)
          data[j] = dist(rng);
      } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        uint16_t *data = reinterpret_cast<uint16_t *>(raw.data());
        for (size_t j = 0; j < size; ++j)
          data[j] = float_to_fp16(dist(rng));
      } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        int32_t *data = reinterpret_cast<int32_t *>(raw.data());
        for (size_t j = 0; j < size; ++j)
          data[j] = int_dist(rng);
      } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        int64_t *data = reinterpret_cast<int64_t *>(raw.data());
        for (size_t j = 0; j < size; ++j)
          data[j] = static_cast<int64_t>(int_dist(rng));
      } else {
        // Fallback: treat as float32
        float *data = reinterpret_cast<float *>(raw.data());
        for (size_t j = 0; j < size; ++j)
          data[j] = dist(rng);
      }
    }
    input_raw_data.push_back(std::move(raw));
  }

  // Get output names and types
  std::vector<std::string> output_names;
  std::vector<ONNXTensorElementDataType> output_types;
  std::cout << "\n--- Output Information ---" << std::endl;
  for (size_t i = 0; i < num_outputs; ++i) {
    auto name = cpu_session.GetOutputNameAllocated(i, allocator);
    output_names.push_back(name.get());

    auto type_info = cpu_session.GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape();
    auto elem_type = tensor_info.GetElementType();
    output_types.push_back(elem_type);

    std::cout << "  Output " << i << ": " << output_names[i] << " "
              << shape_to_string(shape) << " " << elem_type_name(elem_type)
              << std::endl;
  }

  // Convert names to char pointers
  std::vector<const char *> input_name_ptrs;
  for (const auto &n : input_names)
    input_name_ptrs.push_back(n.c_str());
  std::vector<const char *> output_name_ptrs;
  for (const auto &n : output_names)
    output_name_ptrs.push_back(n.c_str());

  // Run iterations
  bool all_passed = true;

  for (int iter = 0; iter < config.iterations; ++iter) {
    std::cout << "\n" << std::string(70, '-') << std::endl;
    std::cout << "  Iteration " << (iter + 1) << "/" << config.iterations
              << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    // Regenerate input data for each iteration (except first)
    if (iter > 0) {
      for (size_t i = 0; i < input_raw_data.size(); ++i) {
        auto elem_type = input_types[i];
        size_t size = input_elem_counts[i];
        if (size == 0)
          continue; // Skip zero-sized tensors

        if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
          float *data = reinterpret_cast<float *>(input_raw_data[i].data());
          for (size_t j = 0; j < size; ++j)
            data[j] = dist(rng);
        } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
          uint16_t *data =
              reinterpret_cast<uint16_t *>(input_raw_data[i].data());
          for (size_t j = 0; j < size; ++j)
            data[j] = float_to_fp16(dist(rng));
        } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
          int32_t *data = reinterpret_cast<int32_t *>(input_raw_data[i].data());
          for (size_t j = 0; j < size; ++j)
            data[j] = int_dist(rng);
        }
      }
    }

    // Helper: create ORT input tensors from raw data (reused for CPU and GPU)
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto create_input_tensors = [&]() -> std::vector<Ort::Value> {
      std::vector<Ort::Value> tensors;
      for (size_t i = 0; i < num_inputs; ++i) {
        auto elem_type = input_types[i];
        size_t elem_count = input_elem_counts[i];

        if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
          tensors.push_back(Ort::Value::CreateTensor<float>(
              memory_info, reinterpret_cast<float *>(input_raw_data[i].data()),
              elem_count, input_shapes[i].data(), input_shapes[i].size()));
        } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
          tensors.push_back(Ort::Value::CreateTensor(
              memory_info, input_raw_data[i].data(), input_raw_data[i].size(),
              input_shapes[i].data(), input_shapes[i].size(),
              ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
        } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
          tensors.push_back(Ort::Value::CreateTensor<int32_t>(
              memory_info,
              reinterpret_cast<int32_t *>(input_raw_data[i].data()), elem_count,
              input_shapes[i].data(), input_shapes[i].size()));
        } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
          tensors.push_back(Ort::Value::CreateTensor<int64_t>(
              memory_info,
              reinterpret_cast<int64_t *>(input_raw_data[i].data()), elem_count,
              input_shapes[i].data(), input_shapes[i].size()));
        } else {
          // Fallback: treat as float32
          tensors.push_back(Ort::Value::CreateTensor<float>(
              memory_info, reinterpret_cast<float *>(input_raw_data[i].data()),
              elem_count, input_shapes[i].data(), input_shapes[i].size()));
        }
      }
      return tensors;
    };

    std::vector<Ort::Value> input_tensors = create_input_tensors();

    // Run CPU inference
    std::cout << "\n--- CPU Inference ---" << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();

    auto cpu_outputs = cpu_session.Run(
        Ort::RunOptions{}, input_name_ptrs.data(), input_tensors.data(),
        num_inputs, output_name_ptrs.data(), num_outputs);

    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(
                        cpu_end - cpu_start)
                        .count();
    std::cout << "  Time: " << (cpu_time / 1000.0f) << " ms" << std::endl;

    // Store CPU outputs (convert to float for comparison)
    std::vector<std::vector<float>> cpu_output_data;
    std::vector<std::vector<int64_t>> output_shapes;
    for (size_t i = 0; i < cpu_outputs.size(); ++i) {
      auto shape = cpu_outputs[i].GetTensorTypeAndShapeInfo().GetShape();
      auto out_elem_type =
          cpu_outputs[i].GetTensorTypeAndShapeInfo().GetElementType();

      // Extract as float regardless of actual type
      auto float_data = extract_output_as_float(cpu_outputs[i]);
      cpu_output_data.push_back(float_data);
      output_shapes.push_back(shape);

      if (config.verbose && !float_data.empty()) {
        std::cout << "  Output " << i << " " << shape_to_string(shape) << " "
                  << elem_type_name(out_elem_type)
                  << ": first=" << float_data[0]
                  << ", last=" << float_data.back() << std::endl;
      }
    }

    // Try GPU inference with HipDNN EP
    std::cout << "\n--- GPU Inference (HipDNN EP) ---" << std::endl;

    std::vector<Ort::ConstEpDevice> devices = env.GetEpDevices();
    const OrtEpDevice *hipdnn_device = nullptr;

    for (const auto &device : devices) {
      std::string ep_name = device.EpName();
      if (ep_name == "MorphiZen" || ep_name == "MorphiZenExecutionProvider") {
        hipdnn_device = static_cast<const OrtEpDevice *>(device);
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
    } catch (const Ort::Exception &ex) {
      std::cerr << "  ERROR: Failed to add HipDNN EP: " << ex.what()
                << std::endl;
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
        std::cout
            << "  (Check hipdnn_profile*.json for detailed EP assignments)"
            << std::endl;
      }

      // Recreate input tensors for GPU session (same data, separate ORT
      // objects)
      std::vector<Ort::Value> gpu_input_tensors = create_input_tensors();

      auto gpu_start = std::chrono::high_resolution_clock::now();

      auto gpu_outputs = gpu_session.Run(
          Ort::RunOptions{}, input_name_ptrs.data(), gpu_input_tensors.data(),
          num_inputs, output_name_ptrs.data(), num_outputs);

      auto gpu_end = std::chrono::high_resolution_clock::now();
      auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(
                          gpu_end - gpu_start)
                          .count();
      std::cout << "  Time: " << (gpu_time / 1000.0f) << " ms" << std::endl;
      std::cout << "  Speedup: " << (static_cast<float>(cpu_time) / gpu_time)
                << "x" << std::endl;

      // Compare outputs
      std::cout << "\n--- Output Comparison ---" << std::endl;
      for (size_t i = 0; i < gpu_outputs.size(); ++i) {
        auto gpu_elem_type =
            gpu_outputs[i].GetTensorTypeAndShapeInfo().GetElementType();

        // Extract GPU output as float (handles fp16 conversion)
        auto gpu_output = extract_output_as_float(gpu_outputs[i]);

        std::cout << "\nOutput " << i << " (" << output_names[i] << ") "
                  << shape_to_string(output_shapes[i]) << " "
                  << elem_type_name(gpu_elem_type) << ":" << std::endl;

        if (config.verbose && !gpu_output.empty() &&
            !cpu_output_data[i].empty()) {
          std::cout << "  CPU: first=" << cpu_output_data[i][0]
                    << ", last=" << cpu_output_data[i].back() << std::endl;
          std::cout << "  GPU: first=" << gpu_output[0]
                    << ", last=" << gpu_output.back() << std::endl;
        }

        if (cpu_output_data[i].empty() && gpu_output.empty()) {
          std::cout << "  (zero-sized output, skipping comparison)"
                    << std::endl;
          continue;
        }

        auto stats = compare_outputs(cpu_output_data[i], gpu_output,
                                     config.tolerance, config.verbose);
        print_stats(stats, config.tolerance, cpu_output_data[i], gpu_output);

        if (stats.mismatch_count > 0 || stats.nan_count > 0) {
          all_passed = false;
        }
      }

    } catch (const Ort::Exception &ex) {
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
