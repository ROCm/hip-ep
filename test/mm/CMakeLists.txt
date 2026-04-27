##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# UMM unit tests.

cmake_minimum_required(VERSION 3.20)

enable_testing()

# When LLVM is loaded it forces CMAKE_MSVC_RUNTIME_LIBRARY to MultiThreaded
# (/MT). The MemoryManager static lib inherits that, so tests must match.
if(MSVC AND CMAKE_MSVC_RUNTIME_LIBRARY)
    set(CMAKE_MSVC_RUNTIME_LIBRARY ${CMAKE_MSVC_RUNTIME_LIBRARY})
endif()

add_executable(test_mm_types test_mm_types.cpp)
target_link_libraries(test_mm_types PRIVATE MemoryManager)
add_test(NAME UMM_TypesCompile COMMAND test_mm_types)

if(MM_USE_MOCK_HAL)
    add_executable(test_hal_mock test_hal_mock.cpp)
    target_link_libraries(test_hal_mock PRIVATE MemoryManager)
    add_test(NAME UMM_HalMock COMMAND test_hal_mock)
endif()

add_executable(test_handle_table test_handle_table.cpp)
target_link_libraries(test_handle_table PRIVATE MemoryManager)
# Internal header (handle_table.h) lives in lib/MemoryManager/
target_include_directories(test_handle_table PRIVATE
    ${CMAKE_SOURCE_DIR}/lib/MemoryManager)
add_test(NAME UMM_HandleTable COMMAND test_handle_table)

add_executable(test_mm_core test_mm_core.cpp)
target_link_libraries(test_mm_core PRIVATE MemoryManager)
add_test(NAME UMM_Core COMMAND test_mm_core)

add_executable(test_mm_pool test_mm_pool.cpp)
target_link_libraries(test_mm_pool PRIVATE MemoryManager)
add_test(NAME UMM_Pool COMMAND test_mm_pool)
