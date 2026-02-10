##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
include(FetchContent)

# Function to get git version info for a component
function(morphizen_add_version_info)
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
    morphizen_add_version_info(COMPONENT ${COMP} DIR ${FETCH_SRC_DIR})
  elseif(EXISTS ${BUILD_DIR} AND IS_DIRECTORY ${BUILD_DIR})
    morphizen_add_version_info(COMPONENT ${COMP} DIR ${BUILD_DIR})
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

# Force static linking for glog to avoid runtime library conflicts
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
set(GLOG_BUILD_SHARED OFF CACHE BOOL "Build glog shared library" FORCE)

# Enable ORT bridge and MLIR backend (same as onnx-hipdnn-ep)
set(morphizen_ENABLE_ORT_BRIDGE ON CACHE BOOL "Enable ORT bridge" FORCE)
option(morphizen_ENABLE_MLIR_BACKEND "Enable MLIR backend in MorphiZen" ON)
set(morphizen_ENABLE_ONNX_BACKEND OFF CACHE BOOL "Enable ONNX backend" FORCE)
set(morphizen_ENABLE_ONNX_SCHEMA_SUPPORT OFF CACHE BOOL "Enable ONNX schema support" FORCE)
set(morphizen_ENABLE_RYZENAI_BIN_METADATA OFF CACHE BOOL "Disable ryzenai_bin_metadata" FORCE)
set(morphizen_ENABLE_BOOST OFF CACHE BOOL "Disable Boost dependency" FORCE)
set(morphizen_OUTPUT_NAME "onnxruntime_morphizen_ep" CACHE STRING "Set output name" FORCE)
set(MORPHIZEN_JSON_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/etc/morphizen_config.json" CACHE FILEPATH "Path to morphizen config file" FORCE)

# Silence MLIR std::complex<APFloat> deprecation warning (MSVC)
if(MSVC)
  add_compile_definitions(_SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING)
endif()

# Ensure ONNX uses dynamic runtime to match CMAKE_MSVC_RUNTIME_LIBRARY setting
set(ONNX_USE_MSVC_STATIC_RUNTIME OFF CACHE BOOL "Use static MSVC runtime" FORCE)

set(MORPHIZEN_VERSEION_INFO_FILE "${CMAKE_CURRENT_BINARY_DIR}/version.txt")
set(MORPHIZEN_JSON_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/etc/morphizen_config.json")

# Add morphizen subdirectory (after all options are set)
add_subdirectory(3rd-party/morphizen)
