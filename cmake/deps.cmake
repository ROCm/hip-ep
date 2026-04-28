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

# Force static linking for glog to avoid runtime library conflicts
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
set(GLOG_BUILD_SHARED OFF CACHE BOOL "Build glog shared library" FORCE)

# Enable ORT bridge and MLIR backend (same as onnx-hipdnn-ep)
set(morphizen_ENABLE_RYZENAI_BIN_METADATA OFF CACHE BOOL "Disable ryzenai_bin_metadata" FORCE)
set(morphizen_OUTPUT_NAME "onnxruntime_morphizen_ep" CACHE STRING "Set output name" FORCE)
set(MORPHIZEN_VERSEION_INFO_FILE "${CMAKE_CURRENT_BINARY_DIR}/version.txt")
set(MORPHIZEN_JSON_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/etc/morphizen_config.json")

# Compile a hipHostMalloc(Mapped|Coherent)-based OrtAllocator + DataTransfer
# into the EP factory so that all EP consumers (hip-onnx-runner, perf_test,
# OGA via ORT) place tensors in GPU-mapped memory rather than host RAM. This
# is what lets OGA recognise MorphiZen EP as a GPU device and skip the full
# H2D/D2H copy of the KV cache on every decode step.
#
# Mock builds (BUILD_MOCK_RUNTIME=ON, the CMake default) intentionally don't
# require ROCm/TheRock, so the morphizen ort-bridge `find_package(hip)` would
# fail. Leave the option at its standalone-MorphiZen default (OFF) for those
# builds. For real builds we also need to seed HIP_PLATFORM here, before the
# add_subdirectory(3rd-party/morphizen) below triggers find_package(hip),
# because TheRock's hip-config.cmake errors out on "Unexpected HIP_PLATFORM"
# otherwise. (3rd-party/custom_kernels/cmake/hip_utils.cmake also seeds it,
# but that subdir is added later in the top-level CMakeLists.txt.)
#
# This deps.cmake is included from the top-level CMakeLists.txt before its
# own `option(BUILD_MOCK_RUNTIME ...)` declaration, so we re-declare the
# same option here (option() is idempotent — keeps the value if already set
# on the command line via -DBUILD_MOCK_RUNTIME=...). Keep the default in
# sync with the top-level declaration.
option(BUILD_MOCK_RUNTIME "Build mock runtime (no GPU required)" ON)

if(NOT BUILD_MOCK_RUNTIME)
  if(NOT DEFINED HIP_PLATFORM)
    set(HIP_PLATFORM "amd" CACHE STRING "HIP platform (amd or nvidia)")
  endif()
  set(morphizen_ENABLE_HIP_GPU_ALLOCATOR ON CACHE BOOL "enable hipHostMalloc-based OrtAllocator + DataTransfer in MorphiZen EP factory (requires HIP runtime)" FORCE)
endif()

# Add morphizen subdirectory (after all options are set)
add_subdirectory(3rd-party/morphizen)
