/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./morphizen-hip-device-select.hpp"

#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace morphizen {

bool IsOwnMemoryInfoName(const char *name) {
  return name != nullptr &&
         (std::strcmp(name, kHipGpuMemoryInfoName) == 0 ||
          std::strcmp(name, kHipHostAccessibleMemoryInfoName) == 0);
}

int SelectAllocatorHipDevice(int requested, bool own_memory_info,
                             int hip_device_count) {
  if (requested < 0 || hip_device_count <= 0) {
    return -1;
  }
  if (own_memory_info) {
    return requested < hip_device_count ? requested : -1;
  }
  return hip_device_count == 1 ? 0 : -1;
}

std::string DescribeHipDeviceFilterEnv(const EnvReader &read_env) {
  std::string out;
  auto add = [&](const char *name) {
    const std::string value = read_env(name);
    if (value.empty()) {
      return;
    }
    if (!out.empty()) {
      out += ", ";
    }
    out += name;
    out += '=';
    out += value;
  };
  const bool hip_visible_set = !read_env("HIP_VISIBLE_DEVICES").empty();
  add("HIP_VISIBLE_DEVICES");
  if (!hip_visible_set) {
    add("CUDA_VISIBLE_DEVICES");
  }
  add("ROCR_VISIBLE_DEVICES");
  add("GPU_DEVICE_ORDINAL");
  return out;
}

std::string DescribeHipDeviceFilterEnv() {
  return DescribeHipDeviceFilterEnv([](const char *name) -> std::string {
#ifdef _WIN32
    // Not std::getenv: a static-CRT DLL's CRT keeps an environment snapshot
    // that misses variables the host process sets after load.
    char buf[1024];
    const DWORD n = GetEnvironmentVariableA(name, buf, sizeof(buf));
    return (n > 0 && n < sizeof(buf)) ? std::string(buf, n) : std::string();
#else
    const char *v = std::getenv(name);
    return v ? std::string(v) : std::string();
#endif
  });
}

} // namespace morphizen
