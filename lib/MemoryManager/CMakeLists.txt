##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# MemoryManager — Unified Memory Manager (UMM) foundation library.
# Standalone static library compiled with MSVC (not bitcode).

cmake_minimum_required(VERSION 3.20)

option(MM_USE_MOCK_HAL "Build UMM with mock HAL (no GPU required)" ON)

set(MM_SOURCES
    mm_core.cpp
    mm_error.cpp
    mm_pool.cpp
    handle_table.cpp
)

if(MM_USE_MOCK_HAL)
    list(APPEND MM_SOURCES hal_mock.cpp)
    message(STATUS "[UMM] Building with MOCK HAL (no GPU required)")
else()
    list(APPEND MM_SOURCES hal_rocm.cpp)
    message(STATUS "[UMM] Building with ROCm HAL")
endif()

add_library(MemoryManager STATIC ${MM_SOURCES})

target_include_directories(MemoryManager PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_features(MemoryManager PUBLIC cxx_std_17)

if(MM_USE_MOCK_HAL)
    target_compile_definitions(MemoryManager PRIVATE MM_USE_MOCK_HAL)
else()
    if(THEROCK_DIST)
        target_include_directories(MemoryManager PRIVATE ${THEROCK_DIST}/include)
        target_compile_definitions(MemoryManager PRIVATE __HIP_PLATFORM_AMD__)
    endif()

    find_library(AMDHIP64_LIB amdhip64
        PATHS ${THEROCK_DIST}/lib
        NO_DEFAULT_PATH)
    if(AMDHIP64_LIB)
        target_link_libraries(MemoryManager PRIVATE ${AMDHIP64_LIB})
    endif()
endif()

if(MSVC)
    target_compile_options(MemoryManager PRIVATE /W4)
else()
    target_compile_options(MemoryManager PRIVATE -Wall -Wextra)
endif()

install(TARGETS MemoryManager
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
