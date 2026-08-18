/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include "onnxruntime_c_api.h"

#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

namespace morphizen::detail {

// Carries the first exception out of a noexcept C callback. The owner rethrows
// only after the foreign call returns and its generated cleanup has run.
class DeferredComputeException {
public:
  void captureCurrent() noexcept {
    if (!exception_)
      exception_ = std::current_exception();
  }

  [[nodiscard]] bool hasValue() const noexcept {
    return static_cast<bool>(exception_);
  }

  void rethrow() const {
    if (exception_)
      std::rethrow_exception(exception_);
  }

private:
  std::exception_ptr exception_;
};

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
