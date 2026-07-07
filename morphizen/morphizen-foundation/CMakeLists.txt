##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

cmake_minimum_required(VERSION 3.22)

project(
  morphizen-foundation
  VERSION 1.0.0
  LANGUAGES C CXX)

# ============================================================================
# CRITICAL DEPENDENCY CHECK
# ============================================================================
# morphizen-foundation MUST have ZERO MorphiZen dependencies
# Only external libraries allowed: glog, GSL, zlib, OpenSSL
# See README.md for detailed dependency rules
# ============================================================================

# Find optional dependencies
find_package(ZLIB QUIET)
set(ENABLE_COMPRESSION "0")
if(TARGET ZLIB::ZLIB)
  set(ENABLE_COMPRESSION "1")
endif()

find_package(OpenSSL QUIET)
set(WITH_OPENSSL "0")
if(TARGET OpenSSL::Crypto)
  set(WITH_OPENSSL "1")
endif()

# Generate mem_binary embedded resources
set(MEM_BINARY_CONTENT_FILE mem_binary_file.hpp.inc)
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${MEM_BINARY_CONTENT_FILE}"
  COMMAND
    ${CMAKE_COMMAND} -E env "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../tools"
    $<TARGET_FILE:Python3::Interpreter>
    ${CMAKE_CURRENT_SOURCE_DIR}/src/compress_binary.py
    "${ENABLE_COMPRESSION}"
    "${CMAKE_CURRENT_BINARY_DIR}/${MEM_BINARY_CONTENT_FILE}"
    "${MORPHIZEN_EMBEDDED_RESOURCE_PATH}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/compress_binary.py"
  COMMENT "Generating embedded resource file for mem_binary")

set_source_files_properties(
  "${CMAKE_CURRENT_BINARY_DIR}/${MEM_BINARY_CONTENT_FILE}" PROPERTIES GENERATED TRUE)

configure_file(src/mem_binary.cpp "${CMAKE_CURRENT_BINARY_DIR}/mem_binary.cpp" @ONLY)

# Source files
set(MORPHIZEN_FOUNDATION_SOURCES
    src/env_config.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/mem_binary.cpp
    src/encryption.cpp
)

set(MORPHIZEN_FOUNDATION_HEADERS
    include/morphizen-foundation/env_config.hpp
    include/morphizen-foundation/parse_value.hpp
    include/morphizen-foundation/mem_binary.hpp
    include/morphizen-foundation/encryption.hpp
)

# Create library
add_library(morphizen-foundation STATIC
    ${MORPHIZEN_FOUNDATION_SOURCES}
    ${MORPHIZEN_FOUNDATION_HEADERS}
    "${CMAKE_CURRENT_BINARY_DIR}/${MEM_BINARY_CONTENT_FILE}"
)

add_library(morphizen::foundation ALIAS morphizen-foundation)

# Include directories
target_include_directories(morphizen-foundation
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Compiler options
target_compile_options(morphizen-foundation PRIVATE ${MORPHIZEN_COMPILER_OPTIONS})

# Link external libraries ONLY (NO MorphiZen dependencies)
target_link_libraries(morphizen-foundation
    PUBLIC
        glog::glog
        Microsoft.GSL::GSL
)

if(TARGET ZLIB::ZLIB)
  target_link_libraries(morphizen-foundation PRIVATE ZLIB::ZLIB)
  target_compile_definitions(morphizen-foundation PRIVATE ENABLE_COMPRESSION=1)
endif()

if(TARGET OpenSSL::Crypto)
  target_link_libraries(morphizen-foundation PRIVATE OpenSSL::Crypto)
  target_compile_definitions(morphizen-foundation PRIVATE WITH_OPENSSL=1)
  if(WIN32)
    target_link_libraries(morphizen-foundation PRIVATE WS2_32.LIB GDI32.LIB ADVAPI32.LIB CRYPT32.LIB USER32.LIB)
  endif()
endif()

# Verification: Ensure no MorphiZen dependencies
# This comment serves as documentation - actual enforcement via code review
# If needed in future, add CMake check here to fail build on morphizen-* dependencies
