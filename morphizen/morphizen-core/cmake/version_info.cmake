##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
add_custom_command (
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/morphizen_version_info.hpp.inc ${CMAKE_CURRENT_BINARY_DIR}/version.rc
  COMMAND ${CMAKE_COMMAND} -E echo
  "PROJECT_GIT_COMMIT_ID=${PROJECT_GIT_COMMIT_ID}"
  "ORT_CORE_PROVIDERS_MORPHIZEN_INCLUDE_DIR=${ORT_CORE_PROVIDERS_MORPHIZEN_INCLUDE_DIR}"
  "morphizen_OUTPUT_NAME=${morphizen_OUTPUT_NAME}.dll"
  $<TARGET_FILE:Python3::Interpreter>  ${CMAKE_CURRENT_SOURCE_DIR}/src/morphizen_version_info.hpp.inc.py "${MORPHIZEN_VERSION_INFO_FILE}"
  COMMAND ${CMAKE_COMMAND} -E env
  "PROJECT_GIT_COMMIT_ID=${PROJECT_GIT_COMMIT_ID}"
  "ORT_CORE_PROVIDERS_MORPHIZEN_INCLUDE_DIR=${ORT_CORE_PROVIDERS_MORPHIZEN_INCLUDE_DIR}"
  "morphizen_OUTPUT_NAME=${morphizen_OUTPUT_NAME}.dll"
  $<TARGET_FILE:Python3::Interpreter>  ${CMAKE_CURRENT_SOURCE_DIR}/src/morphizen_version_info.hpp.inc.py "${MORPHIZEN_VERSION_INFO_FILE}"
  DEPENDS ${MORPHIZEN_VERSION_INFO_FILE}
)
