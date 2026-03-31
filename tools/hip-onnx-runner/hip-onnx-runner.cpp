/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- hip-onnx-runner.cpp - Run an ONNX model via MorphiZen EP ----------===//
//
// Loads an ONNX model, generates random inputs, runs one inference via the
// MorphiZen execution provider, and reports timing.
//
// Usage:
//   hip-onnx-runner <model.onnx> [options]
//
// Options:
//   -n, --no-ep                  Skip EP registration, use CPU only
//   -h, --help                   Show this help
//
//===----------------------------------------------------------------------===//

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#if _WIN32
#include <codecvt>
#include <locale>
#endif

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static int64_t calculate_product(const std::vector<int64_t> &shape) {
  int64_t n = 1;
  for (auto d : shape)
    n *= d;
  return n;
}

static float fp16_to_float(const char *p, int64_t idx) {
  uint16_t h;
  memcpy(&h, p + idx * 2, 2);
  uint32_t sign = (h >> 15) & 1;
  uint32_t exp = (h >> 10) & 0x1f;
  uint32_t mant = h & 0x3ff;
  uint32_t f;
  if (exp == 0)
    f = (sign << 31) | (mant << 13);
  else if (exp == 31)
    f = (sign << 31) | (0xff << 23) | (mant << 13);
  else
    f = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
  float result;
  memcpy(&result, &f, 4);
  return result;
}

static float tensor_elem(const char *p, ONNXTensorElementDataType t,
                         int64_t idx) {
  switch (t) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    return reinterpret_cast<const float *>(p)[idx];
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    return fp16_to_float(p, idx);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    return static_cast<float>(reinterpret_cast<const int64_t *>(p)[idx]);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    return static_cast<float>(reinterpret_cast<const int32_t *>(p)[idx]);
  default:
    return 0.0f;
  }
}

static size_t element_byte_size(ONNXTensorElementDataType t) {
  switch (t) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    return 4;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    return 2;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    return 8;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    return 4;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    return 2;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    return 1;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    return 1;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    return 2;
  default:
    std::cerr << "Unsupported element type: " << (int)t
              << " -- assuming 2 bytes\n";
    return 2;
  }
}

// ---------------------------------------------------------------------------
// Command-line parsing (no Boost dependency)
// ---------------------------------------------------------------------------

struct Options {
  std::string model_path;
  bool no_ep = false;
  bool skip_cpu = false;
};

static void print_usage(const char *argv0) {
  std::cout << "Usage: " << argv0 << " <model.onnx> [options]\n"
            << "\nOptions:\n"
            << "  -n, --no-ep                 CPU only, skip EP\n"
            << "  -s, --skip-cpu              Skip CPU comparison\n"
            << "  -h, --help                  Show this help\n";
}

static Options parse_args(int argc, char *argv[]) {
  Options opts;
  bool model_set = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      std::exit(0);
    } else if (arg == "-n" || arg == "--no-ep") {
      opts.no_ep = true;
    } else if (arg == "-s" || arg == "--skip-cpu") {
      opts.skip_cpu = true;
    } else if (arg[0] != '-' && !model_set) {
      opts.model_path = arg;
      model_set = true;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      print_usage(argv[0]);
      std::exit(1);
    }
  }

  if (!model_set) {
    std::cerr << "Error: model path is required.\n";
    print_usage(argv[0]);
    std::exit(1);
  }
  return opts;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[]) {
  Options opts = parse_args(argc, argv);

  std::mt19937 rng(42);

  // ORT environment
  Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "hip-onnx-runner");

  // EP registration
  const std::string kEpName = "MorphiZenExecutionProvider";
  const std::string ep_dll = "onnxruntime_morphizen_ep.dll";
  const std::string ep_config = "morphizen_config.json";

  if (!opts.no_ep) {
    auto lib_path = std::filesystem::u8path(ep_dll);
    if (!std::filesystem::exists(lib_path)) {
      std::cerr << "EP library not found: " << ep_dll << "\n"
                << "Set MORPHIZEN_VITISAI_EP or use --no-ep.\n";
      return 1;
    }
    std::cout << "Registering EP: " << ep_dll << "\n";
    auto *status = Ort::GetApi().RegisterExecutionProviderLibrary(
        env, kEpName.c_str(), lib_path.c_str());
    if (status) {
      std::cerr << "RegisterExecutionProviderLibrary failed: "
                << Ort::GetApi().GetErrorMessage(status) << "\n";
      return 1;
    }
  }

  // Session options
  Ort::SessionOptions session_opts;
  session_opts.SetLogSeverityLevel(ORT_LOGGING_LEVEL_ERROR);

  if (!opts.no_ep) {
    // Collect devices for this EP
    std::vector<Ort::ConstEpDevice> devices;
    for (const auto &dev : env.GetEpDevices())
      if (dev.EpName() == kEpName)
        devices.emplace_back(dev);

    if (devices.empty()) {
      std::cerr << "No devices found for EP: " << kEpName << "\n";
      return 1;
    }
    std::cout << "Found " << devices.size() << " EP device(s)\n";

    std::unordered_map<std::string, std::string> ep_opts;
    ep_opts["config_file"] = ep_config;
    session_opts.AppendExecutionProvider_V2(env, devices, ep_opts);
  }

  // Create session
  auto model_path = std::filesystem::path(opts.model_path);
  std::cout << "Loading model: " << model_path.string() << "\n";

  std::unique_ptr<Ort::Session> session;
  {
    auto t0 = std::chrono::steady_clock::now();
    try {
#if _WIN32
      session = std::make_unique<Ort::Session>(
          env, model_path.wstring().c_str(), session_opts);
#else
      session = std::make_unique<Ort::Session>(
          env, model_path.u8string().c_str(), session_opts);
#endif
    } catch (const Ort::Exception &e) {
      std::cerr << "Session creation failed: " << e.what() << "\n";
      return 1;
    }
    auto t1 = std::chrono::steady_clock::now();
    std::cout << "Session created in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
                     .count()
              << " ms\n";
  }

  Ort::AllocatorWithDefaultOptions allocator;

  // Collect input info
  size_t input_count = session->GetInputCount();
  std::vector<std::string> input_names_str;
  std::vector<const char *> input_names;
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<ONNXTensorElementDataType> input_types;

  for (size_t i = 0; i < input_count; ++i) {
    auto name_ptr = session->GetInputNameAllocated(i, allocator);
    input_names_str.push_back(name_ptr.get());
    auto info = session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
    input_shapes.push_back(info.GetShape());
    input_types.push_back(info.GetElementType());
  }
  for (auto &s : input_names_str)
    input_names.push_back(s.c_str());

  // Collect output info
  size_t output_count = session->GetOutputCount();
  std::vector<std::string> output_names_str;
  std::vector<const char *> output_names;

  for (size_t i = 0; i < output_count; ++i) {
    auto name_ptr = session->GetOutputNameAllocated(i, allocator);
    output_names_str.push_back(name_ptr.get());
  }
  for (auto &s : output_names_str)
    output_names.push_back(s.c_str());

  std::cout << "Inputs: " << input_count << "  Outputs: " << output_count
            << "\n";

  // Build input tensors with random data
  std::vector<std::vector<char>> input_buffers(input_count);
  std::vector<Ort::Value> input_tensors;
  std::uniform_real_distribution<float> dist(-256.0f, 255.0f);

  for (size_t i = 0; i < input_count; ++i) {
    auto shape = input_shapes[i];
    if (!shape.empty() && shape[0] == -1)
      shape[0] = 1;

    size_t elem_size = element_byte_size(input_types[i]);
    int64_t n_elems = calculate_product(shape);
    input_buffers[i].resize(n_elems * elem_size, 0);

    auto dt = input_types[i];
    if (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      auto *p = reinterpret_cast<int64_t *>(input_buffers[i].data());
      std::uniform_int_distribution<int64_t> idist(0, 100);
      for (int64_t j = 0; j < n_elems; ++j)
        p[j] = idist(rng);
    } else if (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      auto *p = reinterpret_cast<int32_t *>(input_buffers[i].data());
      std::uniform_int_distribution<int32_t> idist(0, 100);
      for (int64_t j = 0; j < n_elems; ++j)
        p[j] = idist(rng);
    } else if (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      auto *p = reinterpret_cast<float *>(input_buffers[i].data());
      for (int64_t j = 0; j < n_elems; ++j)
        p[j] = dist(rng);
    } else if (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      auto *p = reinterpret_cast<uint16_t *>(input_buffers[i].data());
      for (int64_t j = 0; j < n_elems; ++j) {
        float v = dist(rng);
        uint32_t fbits;
        memcpy(&fbits, &v, 4);
        uint32_t sign = (fbits >> 16) & 0x8000;
        int32_t exp = ((fbits >> 23) & 0xff) - 127 + 15;
        uint32_t mant = (fbits >> 13) & 0x3ff;
        if (exp <= 0) {
          p[j] = (uint16_t)sign;
        } else if (exp >= 31) {
          p[j] = (uint16_t)(sign | 0x7c00);
        } else {
          p[j] = (uint16_t)(sign | (exp << 10) | mant);
        }
      }
    }

    Ort::MemoryInfo mem =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    input_tensors.push_back(Ort::Value::CreateTensor(
        mem, input_buffers[i].data(), input_buffers[i].size(), shape.data(),
        shape.size(), input_types[i]));
  }

  // Run inference with EP
  std::cout << "Running EP inference...\n";
  std::vector<std::vector<char>> ep_output_bufs;
  std::vector<std::vector<int64_t>> ep_output_shapes;
  {
    auto t0 = std::chrono::steady_clock::now();
    try {
      auto outputs = session->Run(Ort::RunOptions{}, input_names.data(),
                                  input_tensors.data(), input_count,
                                  output_names.data(), output_count);
      auto t1 = std::chrono::steady_clock::now();
      std::cout << "EP Inference: "
                << std::chrono::duration_cast<std::chrono::microseconds>(t1 -
                                                                         t0)
                       .count()
                << " us\n";
      std::cout << "OK - " << outputs.size() << " output tensor(s)\n";

      for (size_t i = 0; i < outputs.size(); ++i) {
        if (!outputs[i].IsTensor()) {
          std::cout << "  output " << i << ": not a tensor\n";
          continue;
        }
        auto info = outputs[i].GetTensorTypeAndShapeInfo();
        auto dtype = info.GetElementType();
        auto shape = info.GetShape();
        std::cout << "  " << output_names_str[i] << ": type=" << (int)dtype
                  << " shape=[";
        for (auto d : shape)
          std::cout << d << ",";
        std::cout << "]\n";

        ep_output_shapes.push_back(shape);
        size_t elem_sz = element_byte_size(dtype);
        size_t byte_count = calculate_product(shape) * elem_sz;
        auto *data = static_cast<const char *>(outputs[i].GetTensorRawData());
        ep_output_bufs.emplace_back(data, data + byte_count);
      }
    } catch (const Ort::Exception &e) {
      std::cerr << "EP Inference failed: " << e.what() << "\n";
      return 1;
    }
  }

  session.reset();
  if (!opts.no_ep) {
    Ort::GetApi().UnregisterExecutionProviderLibrary(env, kEpName.c_str());
  }

  // Run CPU-only inference for accuracy comparison
  if (!opts.no_ep && !opts.skip_cpu) {
    std::cout << "\nRunning CPU inference for comparison...\n";
    Ort::SessionOptions cpu_opts;
    cpu_opts.SetLogSeverityLevel(ORT_LOGGING_LEVEL_ERROR);
    std::unique_ptr<Ort::Session> cpu_session;
    try {
#if _WIN32
      cpu_session = std::make_unique<Ort::Session>(
          env, model_path.wstring().c_str(), cpu_opts);
#else
      cpu_session = std::make_unique<Ort::Session>(
          env, model_path.u8string().c_str(), cpu_opts);
#endif
    } catch (const Ort::Exception &e) {
      std::cerr << "CPU session failed: " << e.what() << "\n";
      return 1;
    }

    try {
      auto cpu_outputs = cpu_session->Run(Ort::RunOptions{}, input_names.data(),
                                          input_tensors.data(), input_count,
                                          output_names.data(), output_count);
      std::cout << "CPU OK - " << cpu_outputs.size() << " output tensor(s)\n";

      std::cout << "\n=== Output Values (first 5) ===\n";
      for (size_t i = 0; i < cpu_outputs.size(); ++i) {
        auto info = cpu_outputs[i].GetTensorTypeAndShapeInfo();
        auto dtype = info.GetElementType();
        int64_t n = calculate_product(info.GetShape());
        int64_t show = std::min(n, (int64_t)5);
        const auto *ep_raw =
            reinterpret_cast<const char *>(ep_output_bufs[i].data());
        const auto *cpu_raw =
            static_cast<const char *>(cpu_outputs[i].GetTensorRawData());
        std::cout << "  " << output_names_str[i] << " (n=" << n << "):\n";
        std::cout << "    CPU: ";
        for (int64_t j = 0; j < show; ++j)
          std::cout << std::fixed << std::setprecision(4)
                    << tensor_elem(cpu_raw, dtype, j) << " ";
        std::cout << "\n    EP:  ";
        for (int64_t j = 0; j < show; ++j)
          std::cout << std::fixed << std::setprecision(4)
                    << tensor_elem(ep_raw, dtype, j) << " ";
        std::cout << "\n";
      }

      std::cout << "\n=== Accuracy Comparison (EP vs CPU) ===\n";
      bool all_pass = true;
      for (size_t i = 0; i < cpu_outputs.size(); ++i) {
        auto info = cpu_outputs[i].GetTensorTypeAndShapeInfo();
        auto dtype = info.GetElementType();
        int64_t n = calculate_product(info.GetShape());

        const auto *ep_raw =
            reinterpret_cast<const char *>(ep_output_bufs[i].data());
        const auto *cpu_raw =
            static_cast<const char *>(cpu_outputs[i].GetTensorRawData());

        double max_abs_diff = 0.0;
        double max_cpu_abs = 0.0;
        double dot = 0.0, norm_ep = 0.0, norm_cpu = 0.0;
        int64_t nan_count = 0, inf_count = 0, mismatch_nan = 0;

        for (int64_t j = 0; j < n; ++j) {
          float ev = tensor_elem(ep_raw, dtype, j);
          float cv = tensor_elem(cpu_raw, dtype, j);
          if (std::isnan(ev) || std::isnan(cv)) {
            ++nan_count;
            if (std::isnan(ev) != std::isnan(cv))
              ++mismatch_nan;
            continue;
          }
          if (std::isinf(ev) || std::isinf(cv)) {
            ++inf_count;
            continue;
          }
          double diff = std::abs((double)ev - (double)cv);
          if (diff > max_abs_diff)
            max_abs_diff = diff;
          if (std::abs((double)cv) > max_cpu_abs)
            max_cpu_abs = std::abs((double)cv);
          dot += (double)ev * (double)cv;
          norm_ep += (double)ev * (double)ev;
          norm_cpu += (double)cv * (double)cv;
        }

        double denom = std::sqrt(norm_ep) * std::sqrt(norm_cpu);
        double cosine =
            (denom > 1e-30) ? dot / denom : (max_abs_diff < 1e-6 ? 1.0 : 0.0);
        double rel_diff = max_abs_diff / (max_cpu_abs + 1e-10);
        bool pass = (cosine > 0.95 || max_abs_diff < 1e-6) && rel_diff < 0.5 &&
                    mismatch_nan == 0;
        if (!pass)
          all_pass = false;

        std::cout << "  " << output_names_str[i]
                  << ": max_abs=" << std::scientific << std::setprecision(4)
                  << max_abs_diff << " rel=" << rel_diff
                  << " cosine=" << std::fixed << std::setprecision(4) << cosine;
        if (nan_count)
          std::cout << " nan=" << nan_count;
        if (inf_count)
          std::cout << " inf=" << inf_count;
        std::cout << " [" << (pass ? "PASS" : "FAIL") << "]\n";
      }
      std::cout << "\nResult: " << (all_pass ? "ALL PASS" : "FAIL") << "\n";
      if (!all_pass)
        return 1;
    } catch (const Ort::Exception &e) {
      std::cerr << "CPU inference failed: " << e.what() << "\n";
      return 1;
    }
  }

  return 0;
}
