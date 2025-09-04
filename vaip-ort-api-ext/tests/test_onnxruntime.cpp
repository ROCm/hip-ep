/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include <core/session/onnxruntime_c_api.h>
#include <iostream>
#include <vaip/vaip_ort_api.h>

int main() {
  Ort::InitApi();
  std::cout << "ONNXRuntime integration test successful!" << std::endl;
  std::cout << "API Version: " << ORT_API_VERSION << std::endl;
  std::cout << "VAIP_ORT_API Version: " << VAIP_ORT_API_MAJOR << "."
            << VAIP_ORT_API_MINOR << "." << VAIP_ORT_API_PATCH << std::endl;
  return 0;
}
