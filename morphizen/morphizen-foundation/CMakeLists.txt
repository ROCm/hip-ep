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

# OpenSSL backs only EPContext encryption (the encryption_key provider option).
# A static OpenSSL leaves the EP DLL with no OpenSSL runtime dependency, so a
# build runs on machines without one. A static libcrypto.a is not always
# position-independent on Linux, so static is the default on Windows only.
option(HIPEP_WITH_OPENSSL "Link OpenSSL for EPContext encryption" ON)
if(WIN32)
  set(_hipep_openssl_static_default ON)
else()
  set(_hipep_openssl_static_default OFF)
endif()
option(HIPEP_OPENSSL_STATIC
       "Prefer a static OpenSSL; otherwise ship its DLL beside the EP"
       ${_hipep_openssl_static_default})
# The OpenSSL DLL the EP needs at run time, when only a DLL OpenSSL was found;
# installed beside the EP and packaged into the wheel (python/CMakeLists.txt).
set(HIPEP_OPENSSL_RUNTIME_DLL "" CACHE INTERNAL "OpenSSL DLL the EP loads")

set(WITH_OPENSSL "0")
if(HIPEP_WITH_OPENSSL)
  if(HIPEP_OPENSSL_STATIC)
    set(OPENSSL_USE_STATIC_LIBS TRUE)
    if(MSVC AND CMAKE_MSVC_RUNTIME_LIBRARY MATCHES "^MultiThreaded(Debug)?$")
      set(OPENSSL_MSVC_STATIC_RT TRUE)
    endif()
  endif()
  find_package(OpenSSL QUIET)
  if(NOT TARGET OpenSSL::Crypto AND HIPEP_OPENSSL_STATIC)
    # No static OpenSSL: fall back to a shared one rather than drop encryption.
    unset(OPENSSL_USE_STATIC_LIBS)
    unset(OPENSSL_MSVC_STATIC_RT)
    find_package(OpenSSL QUIET)
  endif()
  if(TARGET OpenSSL::Crypto)
    set(WITH_OPENSSL "1")
  endif()
endif()

# On Windows a static library and a DLL import library can share a name
# (libcrypto.lib), so ask lib.exe: an import library's members name its DLL.
if(WITH_OPENSSL AND MSVC AND CMAKE_AR)
  if(LIB_EAY_RELEASE)
    set(_hipep_crypto_lib "${LIB_EAY_RELEASE}")
  else()
    list(GET OPENSSL_CRYPTO_LIBRARY -1 _hipep_crypto_lib)
  endif()
  execute_process(COMMAND "${CMAKE_AR}" /NOLOGO /LIST "${_hipep_crypto_lib}"
                  OUTPUT_VARIABLE _hipep_crypto_members
                  ERROR_QUIET RESULT_VARIABLE _hipep_list_rc)
  if(_hipep_list_rc EQUAL 0
     AND _hipep_crypto_members MATCHES "([A-Za-z0-9_.+-]+\\.[Dd][Ll][Ll])")
    set(_hipep_crypto_dll_name "${CMAKE_MATCH_1}")
    get_filename_component(_hipep_crypto_dir "${_hipep_crypto_lib}" DIRECTORY)
    unset(_hipep_crypto_dll CACHE)
    find_file(_hipep_crypto_dll NAMES "${_hipep_crypto_dll_name}"
              HINTS "${_hipep_crypto_dir}/../bin" "${_hipep_crypto_dir}")
    if(_hipep_crypto_dll)
      set(HIPEP_OPENSSL_RUNTIME_DLL "${_hipep_crypto_dll}"
          CACHE INTERNAL "OpenSSL DLL the EP loads")
      set(_hipep_bin_dir "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
      if(NOT _hipep_bin_dir)
        set(_hipep_bin_dir "${CMAKE_BINARY_DIR}/bin")
      endif()
      file(COPY "${_hipep_crypto_dll}" DESTINATION "${_hipep_bin_dir}")
      install(FILES "${_hipep_crypto_dll}" DESTINATION bin)
      message(WARNING
        "Only a DLL OpenSSL was found (${_hipep_crypto_lib} imports "
        "${_hipep_crypto_dll_name}): the EP needs that DLL at run time. It is "
        "copied beside the EP in bin/ and the install tree. For a "
        "self-contained EP, point OPENSSL_ROOT_DIR at a static OpenSSL, or "
        "set HIPEP_WITH_OPENSSL=OFF if EPContext encryption is not needed.")
    else()
      message(WARNING
        "Only a DLL OpenSSL was found (${_hipep_crypto_lib} imports "
        "${_hipep_crypto_dll_name}), and ${_hipep_crypto_dll_name} was not "
        "found to ship beside the EP; it must be on the loader path at run "
        "time. Point OPENSSL_ROOT_DIR at a static OpenSSL, or set "
        "HIPEP_WITH_OPENSSL=OFF if EPContext encryption is not needed.")
    endif()
  endif()
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
