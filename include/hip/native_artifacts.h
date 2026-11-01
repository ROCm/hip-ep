/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_NATIVE_ARTIFACTS_H
#define HIP_NATIVE_ARTIFACTS_H

// Set by the top-level CMake option HIPDNN_EP_ENABLE_NATIVE_ARTIFACTS
// (default OFF). When unset, native artifact support is off.
#ifndef HIPDNN_EP_ENABLE_NATIVE_ARTIFACTS
#define HIPDNN_EP_ENABLE_NATIVE_ARTIFACTS 0
#endif

namespace hipdnn {

inline constexpr bool nativeArtifactsEnabled() {
  return HIPDNN_EP_ENABLE_NATIVE_ARTIFACTS != 0;
}

inline constexpr const char kNativeArtifactsDisabledMessage[] =
    "artifact_format=NATIVE is not available in this build; configure with "
    "-DHIPDNN_EP_ENABLE_NATIVE_ARTIFACTS=ON";

} // namespace hipdnn

#endif // HIP_NATIVE_ARTIFACTS_H
