##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

find_package(hip QUIET CONFIG)

set(MM_SOURCES
    mm_config.cpp
    mm_handle_table.h
    mm_handle_table.cpp
    mm_hal.h
    mm_hal.cpp
    mm_activation.h
    mm_activation.cpp
    mm_kv_manager.h
    mm_kv_manager.cpp
    mm_core.cpp)

add_library(mm STATIC ${MM_SOURCES})

target_include_directories(mm
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include)

if(hip_FOUND)
  target_link_libraries(mm PUBLIC hip::host)
else()
  message(STATUS "[MemoryManager] HIP not found — building with stubs only")
endif()

if(MSVC)
  target_compile_options(mm PRIVATE /Zc:preprocessor)
endif()

install(TARGETS mm
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
