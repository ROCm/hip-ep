##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Find installed ONNX Runtime package instead of using source tree.
# The installed package provides onnxruntime::onnxruntime target with
# include directories and libraries configured via INTERFACE properties.
#
# Targets should link against onnxruntime::onnxruntime directly:
#   target_link_libraries(my_target PUBLIC onnxruntime::onnxruntime)
# Include directories propagate automatically - no manual management needed.

if(NOT TARGET onnxruntime::onnxruntime)
  find_package(onnxruntime REQUIRED)
  message(STATUS "ONNX Runtime found via find_package")
  message(STATUS "  Target: onnxruntime::onnxruntime")
endif()

# Search for MorphiZen provider headers in vendored 3rd-party directory
# (allows building without source tree)
find_path(ORT_CORE_PROVIDERS_MORPHIZEN_INCLUDE_DIR
  NAMES morphizen/morphizen_ort_api.h
  PATHS
    "${CMAKE_CURRENT_SOURCE_DIR}/../3rd-party/onnxruntime-morphizen-headers"
    "${CMAKE_SOURCE_DIR}/3rd-party/onnxruntime-morphizen-headers"
  NO_DEFAULT_PATH
  NO_CMAKE_FIND_ROOT_PATH
)

if(NOT ORT_CORE_PROVIDERS_MORPHIZEN_INCLUDE_DIR)
  message(FATAL_ERROR "Cannot find morphizen_ort_api.h. Searched in:\n"
    "  - ${CMAKE_CURRENT_SOURCE_DIR}/../3rd-party/onnxruntime-morphizen-headers\n"
    "  - ${CMAKE_SOURCE_DIR}/3rd-party/onnxruntime-morphizen-headers\n"
    "Please ensure the vendored MorphiZen headers are present.")
else()
  # Set to the directory containing morphizen/ subdirectory with headers
  set(MORPHIZEN_ORT_API_DIR "${ORT_CORE_PROVIDERS_MORPHIZEN_INCLUDE_DIR}")
  message(STATUS "ORT_CORE_PROVIDERS_MORPHIZEN_INCLUDE_DIR: ${ORT_CORE_PROVIDERS_MORPHIZEN_INCLUDE_DIR}")
  message(STATUS "MORPHIZEN_ORT_API_DIR: ${MORPHIZEN_ORT_API_DIR}")
endif()
