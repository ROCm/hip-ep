##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Unified Memory Manager library

set(MM_SOURCES
    mm_hal.cpp
    mm_hal_host.cpp
    mm_config.cpp
    mm_core.cpp
    mm_pool.cpp
    mm_arena.cpp
    mm_handle_table.cpp
    mm_runtime_bridge.cpp
    mm_free_list.cpp
    mm_kv.cpp
    mm_block_table.cpp
)

set(MM_HEADERS
    mm_types.h
    mm_hal.h
    mm_config.h
    mm_core.h
    mm_pool.h
    mm_arena.h
    mm_internal.h
    mm_handle_table.h
)

add_library(HipDnnEpMemoryManager STATIC ${MM_SOURCES} ${MM_HEADERS})

target_include_directories(HipDnnEpMemoryManager
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}
)

# HIP backend (conditional on real runtime)
if(NOT BUILD_MOCK_RUNTIME)
    find_package(hip QUIET)
    if(TARGET hip::host)
        target_sources(HipDnnEpMemoryManager PRIVATE mm_hal_hip.cpp)
        target_compile_definitions(HipDnnEpMemoryManager PUBLIC MM_HAS_HIP)
        target_link_libraries(HipDnnEpMemoryManager PRIVATE hip::host)
    else()
        message(WARNING "Memory Manager: hip::host not found, building without HIP backend")
    endif()
endif()

# C++17 required for std::shared_mutex, structured bindings
target_compile_features(HipDnnEpMemoryManager PUBLIC cxx_std_17)

message(STATUS "Memory Manager: enabled (mock=${BUILD_MOCK_RUNTIME})")
