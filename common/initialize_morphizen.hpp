/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifdef __GNUC__
#  pragma GCC diagnostic ignored "-Wpedantic"
#  pragma GCC diagnostic ignored "-Wconversion"
#  pragma GCC diagnostic ignored "-Wsign-compare"
#  pragma GCC diagnostic ignored "-Wunused-variable"
#  pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#include "morphizen/onnxruntime_api.hpp"
#include <exception>
#include <iostream>

void initialize_morphizen() {
  try {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test_onnx_runner");
    Ort::SessionOptions().AppendExecutionProvider_VitisAI();
  } catch (const std::exception& e) {
    std::cerr << "exception occurs : " << e.what() << "\n";
  }
}

// Deprecated VAIP alias for backward compatibility
[[deprecated("Use initialize_morphizen()")]] inline void initialize_vaip() {
  initialize_morphizen();
}
