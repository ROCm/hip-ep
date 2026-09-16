/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include <cstdint>

struct OrtApi;
struct OrtApiBase;

namespace morphizen {

inline constexpr uint32_t kMinOrtApiVersion{24};

uint32_t NegotiatedOrtApiVersion() noexcept;

const OrtApi *NegotiateOrtApi(const OrtApiBase &ort_api_base,
                              uint32_t min_version) noexcept;

const OrtApi *FallbackOrtApiForStatus(const OrtApiBase &ort_api_base) noexcept;

} // namespace morphizen
