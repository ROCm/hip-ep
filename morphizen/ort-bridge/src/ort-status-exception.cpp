/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./ort-status-exception.hpp"

namespace morphizen {

OrtStatusException::OrtStatusException(const OrtApi &api, OrtStatus *status)
    : ort_api_{api}, status_{status} {
  // Get error message from status
  error_msg_ = ort_api_.GetErrorMessage(status_);
}

// Move constructor
OrtStatusException::OrtStatusException(OrtStatusException &&other) noexcept
    : ort_api_{other.ort_api_}, status_{other.status_},
      error_msg_{other.error_msg_} {
  other.status_ = nullptr; // Transfer ownership
  other.error_msg_ = nullptr;
}

// Move assignment operator
OrtStatusException &
OrtStatusException::operator=(OrtStatusException &&other) noexcept {
  if (this != &other) {
    // Clean up current resources
    if (status_) {
      ort_api_.ReleaseStatus(status_);
    }
    // Transfer ownership
    status_ = other.status_;
    error_msg_ = other.error_msg_;
    other.status_ = nullptr;
    other.error_msg_ = nullptr;
  }
  return *this;
}

OrtStatusException::~OrtStatusException() {
  if (status_) {
    ort_api_.ReleaseStatus(status_);
  }
}

const char *OrtStatusException::what() const noexcept {
  return error_msg_ ? error_msg_ : "Unknown ORT error";
}

} // namespace morphizen
