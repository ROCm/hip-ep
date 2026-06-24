##
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
##

#=============================================================================
# Unified Memory Manager unit tests (GPU-free)
#=============================================================================
# Compiles the MM source files natively against lib/Runtime/mock/ types.
# HIPDNN_EP_MM_MOCK_HAL replaces all HIP calls with malloc/free/memcpy so
# no HIP SDK or GPU is required. Pattern follows test_output_allocator.cpp.
#=============================================================================

cmake_minimum_required(VERSION 3.18)

# ---------------------------------------------------------------------------
# Single executable covering all MM unit tests.
# ---------------------------------------------------------------------------
add_executable(test-memory-manager
    test_hal.cpp
    test_memory_manager.cpp
    # Source files under test — compiled natively, NOT as bitcode.
    ${CMAKE_SOURCE_DIR}/lib/Runtime/mm/hal_apu.cpp
    ${CMAKE_SOURCE_DIR}/lib/Runtime/mm/hal_discrete.cpp
    ${CMAKE_SOURCE_DIR}/lib/Runtime/mm/hal_factory.cpp
    ${CMAKE_SOURCE_DIR}/lib/Runtime/mm/memory_manager.cpp
)

target_compile_features(test-memory-manager PRIVATE cxx_std_17)

# Activate the mock HIP path in all MM source files and tests.
target_compile_definitions(test-memory-manager PRIVATE
    HIPDNN_EP_MM_MOCK_HAL        # replaces hipMalloc/hipHostMalloc/… with malloc/free
    HIPDNN_EP_RT_NO_EXPORT       # drops __declspec(dllexport) on runtime entry points
    _CRT_SECURE_NO_WARNINGS      # suppress MSVC secure-CRT warnings
)

# Include paths:
#   lib/Runtime       -> mm/hal.h, mm/memory_manager.h, hipdnn_ep_runtime.h, …
#   lib/Runtime/mock  -> runtime_types.h resolves to GPU-free mock typedefs
target_include_directories(test-memory-manager PRIVATE
    ${CMAKE_SOURCE_DIR}/lib/Runtime
    ${CMAKE_SOURCE_DIR}/lib/Runtime/mock
)

# Register with CTest. Both test executables in this directory report under
# the same test name so `ctest -R MemoryManagerUnitTests` runs them all.
add_test(NAME MemoryManagerUnitTests COMMAND test-memory-manager)

set_target_properties(test-memory-manager PROPERTIES FOLDER "runtime-tests")
