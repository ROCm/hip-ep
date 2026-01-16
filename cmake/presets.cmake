##
# ** Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
find_package(Python3 REQUIRED Interpreter)
add_custom_command(
    OUTPUT ${CMAKE_SOURCE_DIR}/CMakePresets.json
    COMMAND
        ${CMAKE_COMMAND} -E env
        VAI_RT_BUILD_DIR=${CMAKE_BUILD_DIR}
        VAI_RT_PREFIX=${CMAKE_INSTALL_PREFIX}
        $<TARGET_FILE:Python3::Interpreter>
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/initialize-cmake-preset.py
)
add_custom_target(
    generate_cmake_presets
    DEPENDS ${CMAKE_SOURCE_DIR}/CMakePresets.json
)
