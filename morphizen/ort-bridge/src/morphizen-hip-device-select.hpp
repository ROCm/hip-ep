/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

// HIP device selection for the MorphiZen EP's ORT allocator. Kept free of HIP
// and ORT headers so it can be unit tested without a device. See
// docs/design/hip-device-selection.md.

#include <functional>
#include <string>

namespace morphizen {

// Names of the OrtMemoryInfo instances the factory registers on its
// OrtEpDevices. A memory info with one of these names carries a HIP ordinal as
// its device id; any other name was created by a parent EP.
inline constexpr const char *kHipGpuMemoryInfoName = "MorphiZen";
inline constexpr const char *kHipHostAccessibleMemoryInfoName =
    "MorphiZen host accessible";

bool IsOwnMemoryInfoName(const char *name);

// Returns the HIP ordinal an allocator should make current, or -1 to leave the
// calling thread's current device alone.
//
// `requested` is the OrtMemoryInfo device id (-1 when unreadable). A parent EP
// such as the AMD GPU umbrella (amdgpu-ep) registers its own memory infos with
// the DXGI adapter number as device id and forwards them to this EP's
// CreateAllocator. That number is not a HIP ordinal: HIP enumerates AMD
// devices only, while DXGI numbers every adapter and puts the one driving the
// primary display first. A foreign id is therefore never used as an ordinal.
int SelectAllocatorHipDevice(int requested, bool own_memory_info,
                             int hip_device_count);

using EnvReader = std::function<std::string(const char *name)>;

// Lists the set environment variables the HIP runtime filters devices by, as
// "NAME=value" joined with ", "; empty when none applies. CUDA_VISIBLE_DEVICES
// is reported only while HIP_VISIBLE_DEVICES is unset, because that is when
// the HIP runtime honors it as an alias.
std::string DescribeHipDeviceFilterEnv(const EnvReader &read_env);

// Same, reading the process environment.
std::string DescribeHipDeviceFilterEnv();

} // namespace morphizen
