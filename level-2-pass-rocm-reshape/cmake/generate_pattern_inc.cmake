##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_custom_command (
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/reshape_pattern_json.hpp
  COMMAND ${CMAKE_COMMAND} -E env
    $<TARGET_FILE:Python3::Interpreter> ${CMAKE_SOURCE_DIR}/cmake/xxd.py
    "--column" 16
    "--var" reshape_json
    --output ${CMAKE_CURRENT_BINARY_DIR}/reshape_pattern_json.hpp
    "${CMAKE_CURRENT_SOURCE_DIR}/../patterns/reshape.json"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../patterns/reshape.json" "${CMAKE_CURRENT_LIST_FILE}"
)
