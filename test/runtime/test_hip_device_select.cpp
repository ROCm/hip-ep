/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// GPU-free unit test for the MorphiZen EP allocator's HIP device selection
// (morphizen/ort-bridge/src/morphizen-hip-device-select.cpp).
//
// The case that motivated it: the AMD GPU umbrella EP forwards memory infos
// whose device id is the DXGI adapter number. With an NVIDIA GPU driving the
// primary display the AMD iGPU is DXGI adapter 1, while HIP (AMD devices only)
// numbers it 0; hipSetDevice(1) then fails.
//===----------------------------------------------------------------------===//

#include "morphizen-hip-device-select.hpp"

#include <cstdio>
#include <map>
#include <string>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
      ++g_failures;                                                            \
    }                                                                          \
  } while (0)

using morphizen::SelectAllocatorHipDevice;

void TestOwnMemoryInfo() {
  // The factory's own memory infos carry a HIP ordinal.
  CHECK(SelectAllocatorHipDevice(0, true, 1) == 0);
  CHECK(SelectAllocatorHipDevice(1, true, 2) == 1);
  // Out of range: keep the current device instead of failing hipSetDevice.
  CHECK(SelectAllocatorHipDevice(1, true, 1) == -1);
}

void TestParentEpMemoryInfo() {
  // A parent EP's device id is a DXGI adapter number. With one HIP device it
  // can only mean that device, whatever the DXGI numbering.
  CHECK(SelectAllocatorHipDevice(0, false, 1) == 0);
  CHECK(SelectAllocatorHipDevice(1, false, 1) == 0);
  CHECK(SelectAllocatorHipDevice(3, false, 1) == 0);
  // With several HIP devices the number cannot be mapped to an ordinal, so the
  // thread's current device (the one the runtime selected) is kept.
  CHECK(SelectAllocatorHipDevice(0, false, 2) == -1);
  CHECK(SelectAllocatorHipDevice(1, false, 2) == -1);
}

void TestNothingToSelect() {
  CHECK(SelectAllocatorHipDevice(-1, true, 1) == -1);
  CHECK(SelectAllocatorHipDevice(-1, false, 1) == -1);
  CHECK(SelectAllocatorHipDevice(0, true, 0) == -1);
  CHECK(SelectAllocatorHipDevice(1, false, 0) == -1);
}

void TestMemoryInfoNames() {
  CHECK(morphizen::IsOwnMemoryInfoName(morphizen::kHipGpuMemoryInfoName));
  CHECK(morphizen::IsOwnMemoryInfoName(
      morphizen::kHipHostAccessibleMemoryInfoName));
  // amdgpu-ep's memory info names.
  CHECK(!morphizen::IsOwnMemoryInfoName("default"));
  CHECK(!morphizen::IsOwnMemoryInfoName("pinned"));
  CHECK(!morphizen::IsOwnMemoryInfoName(""));
  CHECK(!morphizen::IsOwnMemoryInfoName(nullptr));
}

std::string Describe(const std::map<std::string, std::string> &env) {
  return morphizen::DescribeHipDeviceFilterEnv(
      [&](const char *name) -> std::string {
        auto it = env.find(name);
        return it == env.end() ? std::string() : it->second;
      });
}

void TestFilterEnv() {
  CHECK(Describe({}).empty());
  CHECK(Describe({{"CUDA_VISIBLE_DEVICES", "1"}}) == "CUDA_VISIBLE_DEVICES=1");
  // HIP ignores CUDA_VISIBLE_DEVICES once HIP_VISIBLE_DEVICES is set.
  CHECK(Describe({{"HIP_VISIBLE_DEVICES", "0"},
                  {"CUDA_VISIBLE_DEVICES", "1"}}) == "HIP_VISIBLE_DEVICES=0");
  CHECK(Describe({{"HIP_VISIBLE_DEVICES", "0"}, {"GPU_DEVICE_ORDINAL", "2"}}) ==
        "HIP_VISIBLE_DEVICES=0, GPU_DEVICE_ORDINAL=2");
  CHECK(Describe({{"ROCR_VISIBLE_DEVICES", "-1"}}) ==
        "ROCR_VISIBLE_DEVICES=-1");
  CHECK(Describe({{"UNRELATED", "1"}}).empty());
}

} // namespace

int main() {
  TestOwnMemoryInfo();
  TestParentEpMemoryInfo();
  TestNothingToSelect();
  TestMemoryInfoNames();
  TestFilterEnv();
  if (g_failures) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("test-hip-device-select: all checks passed\n");
  return 0;
}
