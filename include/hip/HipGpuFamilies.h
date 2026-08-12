/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace hip_ep {

// Member-to-family map. Keep in sync with cmake/hip_families.cmake.
inline std::optional<std::string_view> familyForGpuArch(std::string_view arch) {
  static constexpr struct {
    std::string_view family;
    std::string_view member;
  } kMembers[] = {
      {"gfx115X-all", "gfx1150"},
      {"gfx115X-all", "gfx1151"},
      {"gfx115X-all", "gfx1152"},
      {"gfx115X-all", "gfx1153"},
  };
  for (const auto &entry : kMembers) {
    if (entry.member == arch)
      return entry.family;
  }
  return std::nullopt;
}

// Basenames to try when dlopen'ing custom kernels (no directory/extension).
// Per-arch DLL first, then the family fatbin if the arch belongs to one.
inline std::vector<std::string>
customKernelLibraryBasenames(std::string_view arch) {
  std::vector<std::string> names;
  names.emplace_back(std::string("custom_kernels_") + std::string(arch));
  if (auto family = familyForGpuArch(arch))
    names.emplace_back(std::string("custom_kernels_") + std::string(*family));
  return names;
}

} // namespace hip_ep
