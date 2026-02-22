/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

/**
 * Simple test to create an ORT session WITHOUT MorphiZen EP
 * to verify the basic ORT integration works.
 */

#include <iostream>
#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#  define NOMINMAX
#  include <windows.h>
inline std::wstring ToWideString(const char* str) {
  int len = MultiByteToWideChar(CP_UTF8, 0, str, -1, nullptr, 0);
  std::wstring result(len - 1, 0);
  MultiByteToWideChar(CP_UTF8, 0, str, -1, &result[0], len);
  return result;
}
#endif

#ifndef CONV_TEST_MODEL_PATH
#  define CONV_TEST_MODEL_PATH "./conv_model.onnx"
#endif

int main() {
  std::cout << "=== Simple ORT Session Test ===" << std::endl;

  try {
    // Create ORT environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SimpleTest");
    std::cout << "[1] Created ORT environment" << std::endl;

    // Create session options (NO EP)
    Ort::SessionOptions session_options;
    std::cout << "[2] Created session options" << std::endl;

    // Create session with CPU EP only
    std::cout << "[3] Creating session with model: " << CONV_TEST_MODEL_PATH
              << std::endl;
#ifdef _WIN32
    auto model_path_w = ToWideString(CONV_TEST_MODEL_PATH);
    Ort::Session session(env, model_path_w.c_str(), session_options);
#else
    Ort::Session session(env, CONV_TEST_MODEL_PATH, session_options);
#endif
    std::cout << "[4] Session created successfully!" << std::endl;

    return 0;
  } catch (const Ort::Exception& ex) {
    std::cerr << "ERROR: " << ex.what() << std::endl;
    return 1;
  }
}
