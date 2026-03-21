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
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
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
    std::cerr << "Unsupported element type: " << t << "\n";
    std::exit(1);
  }
}

// ---------------------------------------------------------------------------
// Command-line parsing (no Boost dependency)
// ---------------------------------------------------------------------------

struct Options {
  std::string model_path;
  bool no_ep = false;
};

static void print_usage(const char *argv0) {
  std::cout << "Usage: " << argv0 << " <model.onnx> [options]\n"
            << "\nOptions:\n"
            << "  -n, --no-ep                 CPU only, skip EP\n"
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
  const std::string kEpName = "VitisAIExecutionProvider";
  const std::string ep_dll = "onnxruntime_morphizen_ep.dll";

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

    session_opts.AppendExecutionProvider_V2(env, devices, {});
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
    input_buffers[i].resize(n_elems * elem_size);

    // Fill as float (reinterpreted for other types — sufficient for random
    // test)
    auto *fdata = reinterpret_cast<float *>(input_buffers[i].data());
    std::generate(fdata, fdata + (input_buffers[i].size() / sizeof(float)),
                  [&] { return dist(rng); });

    Ort::MemoryInfo mem =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    input_tensors.push_back(Ort::Value::CreateTensor(
        mem, input_buffers[i].data(), input_buffers[i].size(), shape.data(),
        shape.size(), input_types[i]));
  }

  // Run inference
  std::cout << "Running inference...\n";
  {
    auto t0 = std::chrono::steady_clock::now();
    try {
      auto outputs = session->Run(Ort::RunOptions{}, input_names.data(),
                                  input_tensors.data(), input_count,
                                  output_names.data(), output_count);
      auto t1 = std::chrono::steady_clock::now();
      std::cout << "Inference: "
                << std::chrono::duration_cast<std::chrono::microseconds>(t1 -
                                                                         t0)
                       .count()
                << " us\n";
      std::cout << "OK - " << outputs.size() << " output tensor(s)\n";
    } catch (const Ort::Exception &e) {
      std::cerr << "Inference failed: " << e.what() << "\n";
      return 1;
    }
  }

  // Unregister EP
  if (!opts.no_ep) {
    Ort::GetApi().UnregisterExecutionProviderLibrary(env, kEpName.c_str());
  }

  return 0;
}
