##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Search for VitisAI provider headers in multiple locations:
# 1. Vendored headers in 3rd-party (allows building without source tree)
# 2. ONNXRuntime source tree (for development with source)
find_path(ORT_CORE_PROVIDERS_VITISAI_INCLUDE_DIR
  NAMES vaip/vaip_ort_api.h
  PATHS
    "${CMAKE_SOURCE_DIR}/3rd-party/onnxruntime-vitisai-headers"
    "${ONNXRUNTIME_SOURCE_TREE_DIR}/onnxruntime/core/providers/vitisai/include"
    "${onnxruntime_SOURCE_DIR}/../onnxruntime/core/providers/vitisai/include"
  NO_DEFAULT_PATH
  NO_CMAKE_FIND_ROOT_PATH
)

if(NOT ORT_CORE_PROVIDERS_VITISAI_INCLUDE_DIR)
  message(FATAL_ERROR "Cannot find vaip_ort_api.h. Searched in:\n"
    "  - ${CMAKE_SOURCE_DIR}/3rd-party/onnxruntime-vitisai-headers\n"
    "  - ${ONNXRUNTIME_SOURCE_TREE_DIR}/onnxruntime/core/providers/vitisai/include\n"
    "Either the vendored headers are missing or ONNXRuntime source tree is not available.")
else()
  # get directory of vaip_ort_api.h parent
  get_filename_component(VAIP_ORT_API_DIR "${ORT_CORE_PROVIDERS_VITISAI_INCLUDE_DIR}/.." DIRECTORY)
  message(STATUS "ORT_CORE_PROVIDERS_VITISAI_INCLUDE_DIR: ${ORT_CORE_PROVIDERS_VITISAI_INCLUDE_DIR}")
endif()
