/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include "onnxruntime_c_api.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace morphizen::detail {

// CustomOp::Compute is a stable void-returning plugin ABI. Implementations
// report execution failures by throwing; the ORT callback must translate every
// exception into an OrtStatus so none can cross the C ABI boundary.
template <typename Compute>
OrtStatus *translateComputeExceptions(const OrtApi &api,
                                      Compute &&compute) noexcept {
  try {
    std::forward<Compute>(compute)();
    return nullptr;
  } catch (const std::exception &e) {
    return api.CreateStatus(ORT_RUNTIME_EXCEPTION, e.what());
  } catch (...) {
    return api.CreateStatus(ORT_RUNTIME_EXCEPTION,
                            "CustomOp::Compute failed with unknown exception");
  }
}

[[noreturn]] inline void throwComputeFailure(const char *entryPoint, int code) {
  throw std::runtime_error(std::string(entryPoint) +
                           " failed with code: " + std::to_string(code));
}

} // namespace morphizen::detail
