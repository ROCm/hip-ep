/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./ort-api-version.hpp"

#include <atomic>
#include <onnxruntime_c_api.h>

namespace morphizen {

namespace {

std::atomic<uint32_t> g_negotiated_ort_api_version{ORT_API_VERSION};

uint32_t RuntimeApiVersionFromString(const char *version) noexcept {
  if (version == nullptr) {
    return 0;
  }
  const char *p = version;
  while (*p != '\0' && *p != '.') {
    ++p;
  }
  if (*p != '.') {
    return 0;
  }
  ++p;
  uint32_t minor = 0;
  bool any_digit = false;
  for (; *p >= '0' && *p <= '9'; ++p) {
    minor = minor * 10 + static_cast<uint32_t>(*p - '0');
    any_digit = true;
  }
  return any_digit ? minor : 0;
}

} // namespace

uint32_t NegotiatedOrtApiVersion() noexcept {
  return g_negotiated_ort_api_version.load(std::memory_order_relaxed);
}

const OrtApi *NegotiateOrtApi(const OrtApiBase &ort_api_base,
                              uint32_t min_version) noexcept {
  uint32_t ceiling = ORT_API_VERSION;
  if (const uint32_t runtime_version =
          RuntimeApiVersionFromString(ort_api_base.GetVersionString());
      runtime_version != 0 && runtime_version < ceiling) {
    ceiling = runtime_version;
  }

  for (uint32_t version = ceiling; version >= min_version; --version) {
    if (const OrtApi *ort_api = ort_api_base.GetApi(version)) {
      g_negotiated_ort_api_version.store(version, std::memory_order_relaxed);
      return ort_api;
    }
  }

  g_negotiated_ort_api_version.store(0, std::memory_order_relaxed);
  return nullptr;
}

const OrtApi *FallbackOrtApiForStatus(const OrtApiBase &ort_api_base) noexcept {
  uint32_t ceiling =
      RuntimeApiVersionFromString(ort_api_base.GetVersionString());
  if (ceiling == 0 || ceiling > ORT_API_VERSION) {
    ceiling = ORT_API_VERSION;
  }
  for (uint32_t version = ceiling; version >= 1; --version) {
    if (const OrtApi *ort_api = ort_api_base.GetApi(version)) {
      return ort_api;
    }
  }
  return nullptr;
}

} // namespace morphizen
