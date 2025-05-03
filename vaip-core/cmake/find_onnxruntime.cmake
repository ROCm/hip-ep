##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
find_path(ORT_CORE_PROVIDERS_VITISAI_INCLUDE_DIR
  NAMES vaip/vaip_ort_api.h
  PATHS "${ONNXRUNTIME_SOURCE_TREE_DIR}/onnxruntime/core/providers/vitisai/include"
  PATHS "${onnxruntime_SOURCE_DIR}/../onnxruntime/core/providers/vitisai/include"
  NO_DEFAULT_PATH
  NO_CMAKE_FIND_ROOT_PATH
)

if(NOT ORT_CORE_PROVIDERS_VITISAI_INCLUDE_DIR)
  message(FATAL_ERROR "cannot find vaip_ort_api.h in ${ONNXRUNTIME_SOURCE_TREE_DIR}/onnxruntime/core/providers/vitisai/include")
else()
  # get directory of ${ONNXRUNTIME_SOURCE_TREE_DIR}/onnxruntime/core/providers/vitisai/include/vaip/vaip_ort_api.h
  get_filename_component(VAIP_ORT_API_DIR "${ORT_CORE_PROVIDERS_VITISAI_INCLUDE_DIR}/.." DIRECTORY)
  message(STATUS "ORT_CORE_PROVIDERS_VITISAI_INCLUDE_DIR: ${ORT_CORE_PROVIDERS_VITISAI_INCLUDE_DIR}")
endif()
