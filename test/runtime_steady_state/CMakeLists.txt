##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Native (non-bitcode) build of module_registry.cpp + smoke test driver.
# No HIP / MIOpen / hipBLASLt dependency. CTest target: RuntimeSteadyState.

cmake_minimum_required(VERSION 3.18)

add_executable(test-runtime-steady-state
    runtime_steady_state.cpp
    ${CMAKE_SOURCE_DIR}/lib/Runtime/module_registry.cpp)

target_include_directories(test-runtime-steady-state PRIVATE
    ${CMAKE_SOURCE_DIR}/lib/Runtime)

target_compile_features(test-runtime-steady-state PRIVATE cxx_std_17)

add_test(NAME RuntimeSteadyState
    COMMAND test-runtime-steady-state)

set_target_properties(test-runtime-steady-state PROPERTIES
    FOLDER "runtime-tests")
