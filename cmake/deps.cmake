##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
include(FetchContent)


message(STATUS "Configuring LLVM/MLIR for onnx-hipdnn-ep")

# LLVM configuration options
set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "LLVM projects to build")
set(LLVM_TARGETS_TO_BUILD "host" CACHE STRING "LLVM targets to build")
set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "Enable LLVM assertions")
set(LLVM_ENABLE_RTTI OFF CACHE BOOL "Disable RTTI in LLVM")
set(LLVM_ENABLE_LIBEDIT OFF CACHE BOOL "Enable libedit in LLVM")
set(LLVM_BUILD_TOOLS ON CACHE BOOL "Build LLVM tools")
set(LLVM_INSTALL_UTILS OFF CACHE BOOL "Install LLVM utilities")
set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "Build LLVM tests")
set(LLVM_DISABLE_ASSEMBLY_FILES OFF CACHE BOOL "disable assembly")
set(LLVM_ENABLE_ZLIB OFF CACHE BOOL "Enable zlib compression")
set(LLVM_ENABLE_ZSTD OFF CACHE BOOL "Enable zstd compression")

# Try to find MLIR first (MLIR implies LLVM exists)
# This prevents importing incomplete LLVM installations (e.g., LLVM without MLIR)
find_package(MLIR QUIET CONFIG)

if(MLIR_FOUND)
  # MLIR found, now find LLVM (which must exist if MLIR exists)
  find_package(LLVM REQUIRED CONFIG)
  message(STATUS "Found pre-installed LLVM and MLIR")
  message(STATUS "LLVM_DIR: ${LLVM_DIR}")
  message(STATUS "MLIR_DIR: ${MLIR_DIR}")
  set(MORPHIZEN_LLVM_PREINSTALLED ON CACHE BOOL "Using pre-installed LLVM" FORCE)
else()
  # MLIR not found, build LLVM+MLIR from source
  # Do NOT call find_package(LLVM) to avoid importing incomplete LLVM installations
  message(STATUS "LLVM/MLIR not found in CMAKE_PREFIX_PATH, will use FetchContent and build inline")
  set(MORPHIZEN_LLVM_PREINSTALLED OFF CACHE BOOL "Using FetchContent LLVM" FORCE)

  # Try to find LLVM in local directories first
  find_path(LOCAL_LLVM
    NAMES CMakeLists.txt
    PATHS
    "${CMAKE_SOURCE_DIR}/../llvm/llvm"
    "${CMAKE_SOURCE_DIR}/llvm/llvm"
    "${CMAKE_SOURCE_DIR}/3rd-party/llvm/llvm"
    "${CMAKE_SOURCE_DIR}/../llvm-project/llvm"
    "${CMAKE_SOURCE_DIR}/llvm-project/llvm"
    "${CMAKE_SOURCE_DIR}/3rd-party/llvm-project/llvm"
    NO_DEFAULT_PATH)

  if(LOCAL_LLVM)
    message(STATUS "Found LLVM source in local working directory")
    message(STATUS "LLVM SOURCE_DIR: ${LOCAL_LLVM}")
    FetchContent_Declare(
      llvm-project
      SOURCE_DIR ${LOCAL_LLVM}/..
      EXCLUDE_FROM_ALL
      SOURCE_SUBDIR llvm)
  else()
    message(STATUS "Cannot find LLVM in local directories")
    message(STATUS "Fetching LLVM source from GitHub")
    message(STATUS "WARNING: This will download and build LLVM, which may take a long time")
    FetchContent_Declare(
      llvm-project
      GIT_REPOSITORY https://github.com/llvm/llvm-project.git
      GIT_TAG f8cb7987c64dcffb72414a40560055cb717dbf74
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

# Function to get git version info for a component
function(vaip_add_version_info)
  set(options)
  set(oneValueArgs COMPONENT DIR)
  set(multiValueArgs)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})
  if(EXISTS "${ARG_DIR}/.git" AND IS_DIRECTORY "${ARG_DIR}/.git")
    execute_process(
      COMMAND "git" rev-parse HEAD
      WORKING_DIRECTORY ${ARG_DIR}
      OUTPUT_VARIABLE TMP_GIT_COMMIT
      ERROR_VARIABLE error
      RESULT_VARIABLE resultVar
      OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
    execute_process(
      COMMAND "git" describe --tags --abbrev=1 HEAD
      WORKING_DIRECTORY ${ARG_DIR}
      OUTPUT_VARIABLE TMP_VERSION
      ERROR_VARIABLE error
      RESULT_VARIABLE resultVar
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()
  set(COMP_GIT_COMMIT ${TMP_GIT_COMMIT} PARENT_SCOPE)
  set(COMP_VERSION ${TMP_VERSION} PARENT_SCOPE)
  message(STATUS "FindPackage Version info: ${ARG_COMPONENT}=${TMP_GIT_COMMIT} ${TMP_VERSION}")
endfunction()

# Collect version info for onnx-hipdnn-ep
set(VERSION_LIST
    onnx-hipdnn-ep=onnx-hipdnn-ep)
set(VERSION_INFO "")
foreach(COMP_PAIR IN LISTS VERSION_LIST)
  string(FIND "${COMP_PAIR}" "=" pos)
  string(SUBSTRING "${COMP_PAIR}" 0 ${pos} COMP)
  math(EXPR COMP_BEG "${pos} + 1")
  string(SUBSTRING "${COMP_PAIR}" ${COMP_BEG} -1 DIR)
  set(BUILD_DIR "${CMAKE_SOURCE_DIR}/../${DIR}/")

  string(TOLOWER COMP ${COMP})
  set(FETCH_SRC_DIR "${${COMP}_SOURCE_DIR}")
  if(EXISTS "${FETCH_SRC_DIR}" AND IS_DIRECTORY "${FETCH_SRC_DIR}")
    vaip_add_version_info(COMPONENT ${COMP} DIR ${FETCH_SRC_DIR})
  elseif(EXISTS ${BUILD_DIR} AND IS_DIRECTORY ${BUILD_DIR})
    vaip_add_version_info(COMPONENT ${COMP} DIR ${BUILD_DIR})
  endif()
  if (NOT DEFINED COMP_GIT_COMMIT OR NOT DEFINED COMP_VERSION)
    set(COMP_GIT_COMMIT "N/A")
    set(COMP_VERSION "N/A")
  endif()
  string(APPEND VERSION_INFO "${COMP};${COMP_GIT_COMMIT};${COMP_VERSION}\n")
  unset(COMP_GIT_COMMIT)
  unset(COMP_VERSION)
endforeach()

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/version.txt" "${VERSION_INFO}")
set(MORPHIZEN_VERSION_INFO_FILE "${CMAKE_CURRENT_BINARY_DIR}/version.txt")

## Use morphizen from git submodule
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/3rd-party/morphizen/CMakeLists.txt")
  message(FATAL_ERROR "MorphiZen submodule not found. Run: git submodule update --init --recursive")
endif()
message(STATUS "Using MorphiZen from git submodule: 3rd-party/morphizen")

set(morphizen_ENABLE_UNIT_TEST ON CACHE BOOL "enable morphizen unit test or not")
if(morphizen_ENABLE_UNIT_TEST)
  include(CTest)
  enable_testing()
endif()

set(MORPHIZEN_JSON_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/etc/morphizen_config.json")

# MorphiZen options (only non-default values)
set(morphizen_ENABLE_RYZENAI_BIN_METADATA OFF CACHE BOOL "Disable ryzenai_bin_metadata submodule" FORCE)
set(morphizen_OUTPUT_NAME "onnxruntime_morphizen_ep" CACHE STRING "Output name of MorphiZen library" FORCE)

# Add morphizen subdirectory (after all options are set)
add_subdirectory(3rd-party/morphizen)
