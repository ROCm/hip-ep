/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "./ort-status-exception.hpp"
#include <string>

namespace morphizen {

// Forward declarations
struct OrtGraphWrapper;

struct ApiPtrs {
  const OrtApi& ort_api;
  const OrtEpApi& ep_api; // Method declarations
  void throw_if_error(OrtStatus* status) const;
  void throw_error(const std::string& message) const;
};
} // namespace morphizen
