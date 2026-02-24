##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
set(FETCHCONTENT_QUIET TRUE CACHE BOOL "enable fetchcontent quiet")
include(FetchContent)
file(STRINGS ${CMAKE_CURRENT_LIST_DIR}/deps.txt MORPHIZEN_DEPS_LIST)
file(READ "${CMAKE_CURRENT_LIST_DIR}/dep.h.inc.in" MORPHIZEN_DEP_H_INC_IN)
set(MORPHIZEN_DEP_H_INC "")
foreach(MORPHIZEN_DEP IN LISTS MORPHIZEN_DEPS_LIST)
  # Lines start with "#" are comments
  if(NOT MORPHIZEN_DEP MATCHES "^#")
    message(STATUS "MORPHIZEN_DEP = ${MORPHIZEN_DEP}")
    # The first column is name
    list(POP_FRONT MORPHIZEN_DEP MORPHIZEN_DEP_NAME)
    # The second column is URL
    # The URL below may be a local file path or an HTTPS URL
    list(POP_FRONT MORPHIZEN_DEP MORPHIZEN_DEP_URL)
    set(DEP_URL_${MORPHIZEN_DEP_NAME} ${MORPHIZEN_DEP_URL})
    # The third column is SHA1 hash value
    set(DEP_SHA1_${MORPHIZEN_DEP_NAME} ${MORPHIZEN_DEP})
    string(CONFIGURE "${MORPHIZEN_DEP_H_INC_IN}" _tmp @ONLY)
    string(APPEND MORPHIZEN_DEP_H_INC "${_tmp}")
  endif()
endforeach()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/morphizen_deps.inc.h" "${MORPHIZEN_DEP_H_INC}")

find_package(Microsoft.GSL QUIET)
if(NOT TARGET Microsoft.GSL::GSL)
  message(STATUS "cannot find_package(Microsoft.GSL), fetch it from ${DEP_URL_microsoft_gsl}")
  FetchContent_Declare(
    Microsoft.GSL
    URL ${DEP_URL_microsoft_gsl}
    URL_HASH SHA1=${DEP_SHA1_microsoft_gsl}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    EXCLUDE_FROM_ALL
    OVERRIDE_FIND_PACKAGE
  )
  find_package(Microsoft.GSL REQUIRED)
endif()

if(morphizen_ENABLE_UNIT_TEST)
  find_package(GTest CONFIG QUIET)
  if(TARGET GTest::gtest)
    get_target_property(TMP GTest::gtest INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "found find_package(GTest) at ${TMP}")
  else()
    message(STATUS "fetch GTest from ${DEP_URL_GTest}")
    FetchContent_Declare(
      GTest
      GIT_REPOSITORY ${DEP_URL_GTest}
      GIT_TAG ${DEP_SHA1_GTest}
      GIT_SHALLOW TRUE
      CMAKE_ARGS -Dgtest_force_shared_crt=ON
      EXCLUDE_FROM_ALL
      OVERRIDE_FIND_PACKAGE
    )
    find_package(GTest REQUIRED)
  endif()
  if(morphizen_ENABLE_BOOST)
    set(Boost_USE_STATIC_LIBS ON CACHE BOOL "use static Boost libraries")
    set(Boost_USE_STATIC_RUNTIME ON CACHE BOOL "use static Boost runtime")
    find_package(Boost CONFIG COMPONENTS process filesystem program_options QUIET)
    if(TARGET Boost::headers)
      # for old version of Boost, we don't have target Boost::process for some reasons.
      get_target_property(TMP Boost::headers INTERFACE_INCLUDE_DIRECTORIES)
      message(STATUS "found Boost::headers at ${TMP}")
    else()
      message(STATUS "cannot find_package(Boost::process), fetch it from ${DEP_URL_Boost}")
      list(APPEND BOOST_INCLUDE_LIBRARIES "process" "program_options")
      # disable Boost.Context execution context, there is a build error like MASM
      set(BOOST_CONTEXT_NO_EXECUTION_CONTEXT ON CACHE BOOL "disable Boost.Context execution context")
      FetchContent_Declare(
        Boost
        URL ${DEP_URL_Boost}
        URL_MD5 ${DEP_SHA1_Boost}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        EXCLUDE_FROM_ALL
        OVERRIDE_FIND_PACKAGE
      )
      #find_package(Boost REQUIRED COMPONENTS process)
      fetchcontent_MakeAvailable(Boost)
    endif()
  endif()
endif()

set(WITH_GFLAGS OFF CACHE BOOL "disable WITH_GFLAGS for glog")
# Save current BUILD_TESTING value and disable it for glog
set(_SAVED_BUILD_TESTING ${BUILD_TESTING})
set(BUILD_TESTING OFF CACHE BOOL "disable glog tests to avoid runtime library mismatch" FORCE)
# Force glog to build as a static library and enable internal symbols
# glog v0.7.1 GetStackTrace/Symbolize are internal APIs (GLOG_NO_EXPORT).
# MorphiZen uses these internal APIs in morphizen_compile_model.cpp.
# Building as static library with GLOG_STATIC_DEFINE makes these symbols available.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build glog as static library")
find_package(glog CONFIG QUIET)
if(TARGET glog::glog)
  get_target_property(TMP glog::glog INTERFACE_INCLUDE_DIRECTORIES)
  message(STATUS "found glog at ${TMP} ")
else()
  message(STATUS "cannot find_package(glog), fetch it from ${DEP_URL_glog}")
  FetchContent_Declare(
    glog
    URL ${DEP_URL_glog}
    URL_HASH SHA1=${DEP_SHA1_glog}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    EXCLUDE_FROM_ALL
    SYSTEM  # Treat glog headers as system headers to suppress warnings
    OVERRIDE_FIND_PACKAGE
  )
  find_package(glog REQUIRED)
endif()
# Add GLOG_STATIC_DEFINE to enable internal symbols for static linking
# glog::glog is an alias, so we need to apply the definition to the actual target
if(TARGET glog)
  target_compile_definitions(glog INTERFACE GLOG_STATIC_DEFINE)
endif()
# Restore BUILD_SHARED_LIBS and BUILD_TESTING after glog configuration
set(BUILD_SHARED_LIBS ON CACHE BOOL "Restore default shared library setting")
if(DEFINED _SAVED_BUILD_TESTING)
  set(BUILD_TESTING ${_SAVED_BUILD_TESTING} CACHE BOOL "Restore BUILD_TESTING to previous value" FORCE)
  unset(_SAVED_BUILD_TESTING)
endif()



if(NOT CMAKE_MSVC_RUNTIME_LIBRARY)
  set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE BOOL "use dynamic msvc runtime for protobuf by default, /MD")
elseif(${CMAKE_MSVC_RUNTIME_LIBRARY} MATCHES ".*DLL.*")
  set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE BOOL "use dynamic msvc runtime for protobuf, /MD")
else()
  set(protobuf_MSVC_STATIC_RUNTIME ON CACHE BOOL "use static msvc runtime for protobuf, /MT")
endif()
set(protobuf_BUILD_TESTS OFF CACHE BOOL "disable protobuf tests")
set(protobuf_WITH_ZLIB OFF CACHE BOOL "disable zlib for protobuf")
set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "disable protobuf build shared libs")
set(protobuf_BUILD_EXAMPLES OFF CACHE BOOL "disable protobuf examples")
# Enable -fPIC for protobuf static lib to link into shared library on Linux
set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "enable PIC for protobuf")
## it is error-prone to use MODULE mode to find protobuf,
## Protobuf_USE_STATIC_LIBS must be defined.
if(NOT Protobuf_FOUND)
  find_package(Protobuf CONFIG QUIET)
endif()
if(TARGET protobuf::libprotobuf)
  get_target_property(TMP protobuf::libprotobuf INTERFACE_INCLUDE_DIRECTORIES)
  message(STATUS "found protobuf at ${TMP}")
else()
  message(STATUS "cannot find_package(Protobuf), fetch it from ${DEP_URL_protobuf}")
  FetchContent_Declare(
    Protobuf
    URL ${DEP_URL_protobuf}
    URL_HASH SHA1=${DEP_SHA1_protobuf}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    CMAKE_ARGS
    EXCLUDE_FROM_ALL
    SYSTEM  # Treat protobuf headers as system headers to suppress warnings
    OVERRIDE_FIND_PACKAGE)
  find_package(Protobuf REQUIRED)
endif()

## in order to build it, we need to run some python scripts to
## generate some source code.
if(BUILD_PYTHON)
  message(STATUS "find_package(Python3) both for  python scripts and for python embedding")
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
  find_package(pybind11 REQUIRED CONFIG)
  if(TARGET pybind11::embed)
    get_target_property(TMP pybind11::embed INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "found pybind11 at ${TMP}")
  else()
    message(STATUS "cannot find_package(pybind11), fetch it from ${DEP_URL_pybind11}")
    FetchContent_Declare(
      pybind11
      URL ${DEP_URL_pybind11}
      URL_HASH SHA1=${DEP_SHA1_pybind11}
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
      OVERRIDE_FIND_PACKAGE)
    find_package(pybind11 REQUIRED)
  endif()
else()
  message(STATUS "find_package(Python3) for python scripts")
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
endif()

if(NOT TARGET Python3::Interpreter)
  message(FATAL_ERROR "Python3::Interpreter not found")
endif(NOT TARGET Python3::Interpreter)

## find_package(onnx)
# Two independent features need ONNX: onnx-ir-imp backend and node_with_named_args pattern API
# Download ONNX if either feature is enabled to avoid redundant downloads
if(morphizen_ENABLE_ONNX_BACKEND OR morphizen_ENABLE_ONNX_SCHEMA_SUPPORT)
  set(ONNX_USE_MSVC_STATIC_RUNTIME ON CACHE BOOL "use static onnx")
  set(ONNX_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
  set(BUILD_SHARED_LIBS OFF)
  ## we have to create our own namespace to avoid name confliction with ORT
  set(ONNX_NAMESPACE "morphizen_onnx" CACHE STRING "onnx namespace")
  if(TARGET onnx::onnx)
    get_target_property(TMP onnx::onnx INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "found onnx at ${TMP}")
  else()
    message(STATUS "cannot find_package(onnx), fetch it from ${DEP_URL_onnx}")
    FetchContent_Declare(
      onnx
      URL ${DEP_URL_onnx}
      URL_HASH SHA1=${DEP_SHA1_onnx}
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
      CMAKE_ARGS
      EXCLUDE_FROM_ALL
      OVERRIDE_FIND_PACKAGE)
    find_package(onnx REQUIRED)
  endif()
  set(BUILD_SHARED_LIBS ${ONNX_BUILD_SHARED_LIBS})
else()
  message(STATUS "ONNX backend is disabled, skipping ONNX find/fetch")
endif()

##
## hash-library dependency (small header-mostly hash library for MD5/SHA)
## Note: No find_package fallback - upstream doesn't provide CMake config files
##
message(STATUS "Fetching hash-library from ${DEP_URL_hash-library}")
FetchContent_Declare(
  hash-library
  GIT_REPOSITORY ${DEP_URL_hash-library}
  GIT_TAG ${DEP_SHA1_hash-library}
  GIT_SHALLOW TRUE
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(hash-library)

##
## LLVM/MLIR dependency (when morphizen_ENABLE_MLIR_BACKEND=ON)
## Three-tier fallback: pre-installed → local source → GitHub download
## Based on proven morphizen-mlir pattern
##
if(morphizen_ENABLE_MLIR_BACKEND)
  message(STATUS "Configuring LLVM/MLIR for MorphiZen")

  # LLVM configuration options (applied when building from source)
  set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "LLVM projects to build")
  set(LLVM_TARGETS_TO_BUILD "host" CACHE STRING "LLVM targets to build")
  set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "Enable LLVM assertions")
  set(LLVM_ENABLE_RTTI OFF CACHE BOOL "Disable RTTI in LLVM")
  set(LLVM_ENABLE_LIBEDIT OFF CACHE BOOL "Disable libedit in LLVM")
  set(LLVM_BUILD_TOOLS ON CACHE BOOL "Build LLVM tools")
  set(LLVM_INSTALL_UTILS OFF CACHE BOOL "Install LLVM utilities")
  set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "Build LLVM tests")
  set(LLVM_DISABLE_ASSEMBLY_FILES OFF CACHE BOOL "disable assembly")
  set(LLVM_ENABLE_ZLIB OFF CACHE BOOL "Enable zlib compression")
  set(LLVM_ENABLE_ZSTD OFF CACHE BOOL "Enable zstd compression")

  # Tier 1: Try to find pre-installed MLIR/LLVM first
  # MLIR implies LLVM exists (prevents incomplete LLVM installations)
  find_package(MLIR QUIET CONFIG)

  if(MLIR_FOUND)
    # MLIR found, now find LLVM (which must exist if MLIR exists)
    find_package(LLVM REQUIRED CONFIG)
    message(STATUS "Found pre-installed LLVM and MLIR")
    message(STATUS "LLVM_DIR: ${LLVM_DIR}")
    message(STATUS "MLIR_DIR: ${MLIR_DIR}")
    set(MORPHIZEN_LLVM_PREINSTALLED ON CACHE BOOL "Using pre-installed LLVM" FORCE)
  else()
    # MLIR not found, try local source or download
    # Do NOT call find_package(LLVM) to avoid importing incomplete installations
    message(STATUS "LLVM/MLIR not found in CMAKE_PREFIX_PATH, will use FetchContent")
    set(MORPHIZEN_LLVM_PREINSTALLED OFF CACHE BOOL "Using FetchContent LLVM" FORCE)

    # Tier 2: Try to find LLVM source in local directories
    find_path(LOCAL_LLVM
      NAMES CMakeLists.txt
      PATHS
        "${CMAKE_SOURCE_DIR}/../llvm-project/llvm"
        "${CMAKE_SOURCE_DIR}/llvm-project/llvm"
        "${CMAKE_SOURCE_DIR}/3rd-party/llvm-project/llvm"
      NO_DEFAULT_PATH)

    if(LOCAL_LLVM)
      # Found local LLVM source
      message(STATUS "Found LLVM source in local directory")
      message(STATUS "LLVM SOURCE_DIR: ${LOCAL_LLVM}")
      FetchContent_Declare(
        llvm-project
        SOURCE_DIR ${LOCAL_LLVM}/..
        EXCLUDE_FROM_ALL
        SOURCE_SUBDIR llvm)
    else()
      # Tier 3: Download LLVM from GitHub
      message(STATUS "Cannot find LLVM in local directories")
      message(STATUS "Fetching LLVM source from GitHub")
      message(STATUS "WARNING: This will download and build LLVM (~20GB, 30-60 min)")
      FetchContent_Declare(
        llvm-project
        GIT_REPOSITORY ${DEP_URL_llvm}
        GIT_TAG ${DEP_SHA1_llvm}
        GIT_SUBMODULES_RECURSE
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        EXCLUDE_FROM_ALL
        SOURCE_SUBDIR llvm
      )
    endif()

    # Make LLVM available - this adds LLVM as a subdirectory
    FetchContent_MakeAvailable(llvm-project)

    # Set include directories for downstream targets
    set(LLVM_INCLUDE_DIRS
      "${llvm-project_SOURCE_DIR}/llvm/include"
      "${llvm-project_BINARY_DIR}/include"
      CACHE PATH "LLVM include directories" FORCE)
    set(MLIR_INCLUDE_DIRS
      "${llvm-project_SOURCE_DIR}/mlir/include"
      "${llvm-project_BINARY_DIR}/tools/mlir/include"
      CACHE PATH "MLIR include directories" FORCE)

    # Make include directories globally available for all targets
    # This is necessary because subdirectory builds don't automatically propagate
    # MLIR source includes to targets that use find_package(MLIR)
    include_directories(SYSTEM
      "${llvm-project_SOURCE_DIR}/llvm/include"
      "${llvm-project_BINARY_DIR}/include"
      "${llvm-project_SOURCE_DIR}/mlir/include"
      "${llvm-project_BINARY_DIR}/tools/mlir/include")

    # Note: In a subdirectory build, MLIR config files are in tools/mlir/cmake/modules/CMakeFiles/
    set(LLVM_DIR "${llvm-project_BINARY_DIR}/lib/cmake/llvm" CACHE PATH "" FORCE)
    set(MLIR_DIR "${llvm-project_BINARY_DIR}/tools/mlir/cmake/modules/CMakeFiles" CACHE PATH "" FORCE)

    message(STATUS "LLVM source dir: ${llvm-project_SOURCE_DIR}")
    message(STATUS "LLVM binary dir: ${llvm-project_BINARY_DIR}")
    message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
    message(STATUS "MLIR_INCLUDE_DIRS: ${MLIR_INCLUDE_DIRS}")
    message(STATUS "LLVM_DIR: ${LLVM_DIR}")
    message(STATUS "MLIR_DIR: ${MLIR_DIR}")
  endif()

  message(STATUS "LLVM/MLIR configuration complete")
endif()
