/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include <exception>
#define ORT_API_MANUAL_INIT
#ifdef __GNUC__
#  pragma GCC diagnostic ignored "-Wpedantic"
#  pragma GCC diagnostic ignored "-Wconversion"
#  pragma GCC diagnostic ignored "-Wsign-compare"
#  pragma GCC diagnostic ignored "-Wunused-variable"
#  pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#include <core/session/onnxruntime_c_api.h>
#include <core/session/onnxruntime_cxx_api.h>
#include <core/session/onnxruntime_lite_custom_op.h>
#undef ORT_API_MANUAL_INIT

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
