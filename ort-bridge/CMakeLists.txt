##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
include(${CMAKE_CURRENT_SOURCE_DIR}/../vaip-core/cmake/find_onnxruntime.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/check-ort-version.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/ort-bridge.cmake)
if(morphizen_ENABLE_UNIT_TEST AND morphizen_ENABLE_MORPHIZEN_CORE_DYNAMIC)
  add_subdirectory (test)
endif()
