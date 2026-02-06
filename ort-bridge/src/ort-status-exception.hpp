/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "./morphizen/morphizen-ort-api-ext.hpp"
#include <exception>
#include <onnxruntime_c_api.h>
namespace morphizen {

struct OrtStatusException : public std::exception {
  OrtStatusException(const OrtApi& api, OrtStatus* status);

  // Move constructor
  OrtStatusException(OrtStatusException&& other) noexcept;

  // Move assignment operator
  OrtStatusException& operator=(OrtStatusException&& other) noexcept;

  ~OrtStatusException();

  const char* what() const noexcept override;

  // Prevent copying to avoid double-free of status
  OrtStatusException(const OrtStatusException&) = delete;
  OrtStatusException& operator=(const OrtStatusException&) = delete;

private:
  const OrtApi& ort_api_;
  OrtStatus* status_;
  const char* error_msg_ = nullptr;
};

} // namespace morphizen
