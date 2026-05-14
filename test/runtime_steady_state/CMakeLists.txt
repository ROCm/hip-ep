##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

#==============================================================================
# Module Registry Smoke Test
#==============================================================================
# Standalone unit test for lib/Runtime/module_registry.{h,cpp}. Builds the
# registry as native C++ (not bitcode) and links a tiny test driver against
# it. Runs as a CTest target alongside the LIT + E2E suites.
#
# Why native build:
#   The production runtime is compiled to LLVM bitcode and embedded into each
#   generated model.dll. Native compilation here lets us:
#     1. Run the test on the build machine without spinning up a full
#        model compile + GPU execution.
#     2. Exercise the registry's lifecycle paths (init / destroy /
#        begin_compute) without coupling to the rest of the runtime
#        (no HIP / MIOpen / hipBLASLt dependency).
#
# The test stubs `hipdnn_ep::get_module_registry(::RuntimeState*)` locally,
# so it does NOT link against hipdnn_ep_runtime_state.cpp. The registry
# implementation under test is the same .cpp that compiles to bitcode for
# the real runtime.
#==============================================================================

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
