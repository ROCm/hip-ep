##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
include(${CMAKE_CURRENT_LIST_DIR}/generated_gtest_targets.cmake OPTIONAL)
add_custom_target(
    generate_gtest_targets_cmake
    COMMAND
      $<TARGET_FILE:Python3::Interpreter> ${CMAKE_CURRENT_LIST_DIR}/generate_gtest_targets_for_debugging.py
      $<TARGET_FILE:${TEST_EXE_NAME}>
    COMMENT "Generating ${CMAKE_CURRENT_BINARY_DIR}/generated_gtest_targets.cmake"
    DEPENDS ${TEST_EXE_NAME} ${CMAKE_CURRENT_LIST_DIR}/generate_gtest_targets.py
)
set_target_properties(generate_gtest_targets_cmake PROPERTIES
    FOLDER morphizen/unit-tests/cmake
)
