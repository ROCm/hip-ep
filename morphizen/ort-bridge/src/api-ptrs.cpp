/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./api-ptrs.hpp"
#include <glog/logging.h>
#include <string>

namespace morphizen {

void ApiPtrs::throw_if_error(OrtStatus *status) const {
  if (status != nullptr) {
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 5272) // throwing non-copyable exception type
#endif
    LOG(INFO) << "Error in ORT API: " << ort_api.GetErrorCode(status)
              << ", message: " << ort_api.GetErrorMessage(status);
    throw OrtStatusException(ort_api, status);
#ifdef _MSC_VER
#pragma warning(pop)
#endif
  }
}

void ApiPtrs::throw_error(const std::string &message) const {
  throw_if_error(ort_api.CreateStatus(ORT_FAIL, message.c_str()));
}

} // namespace morphizen
