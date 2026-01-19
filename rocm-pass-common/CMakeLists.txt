# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.

# rocm-pass-common: Shared utilities for ROCm Level-2 passes
# This is a header-only library that provides common functionality

add_library(rocm-pass-common INTERFACE)
add_library(morphizen::rocm-pass-common ALIAS rocm-pass-common)

target_include_directories(rocm-pass-common INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>
)

# Dependencies - consumers of this library need these
target_link_libraries(rocm-pass-common INTERFACE
  morphizen::morphizen-core-static
  rocm_proto
  glog::glog
)
