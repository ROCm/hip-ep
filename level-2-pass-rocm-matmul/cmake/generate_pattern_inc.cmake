##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_custom_command (
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/matmul_pattern_json.hpp
  COMMAND ${CMAKE_COMMAND} -E env
    $<TARGET_FILE:Python3::Interpreter> ${morphizen_SOURCE_DIR}/tools/xxd.py
    "--column" 16
    "--var" matmul_json
    --output ${CMAKE_CURRENT_BINARY_DIR}/matmul_pattern_json.hpp
    "${CMAKE_CURRENT_SOURCE_DIR}/../patterns/matmul.json"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../patterns/matmul.json" "${CMAKE_CURRENT_LIST_FILE}"
)
